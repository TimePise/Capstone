"""
모션 CSV(24차원 포즈 + z속도)를 학습에 바로 사용할 수 있도록
시퀀스 데이터셋과 하이브리드 TemporalConv+BiGRU 모델을 정의하는 스크립트입니다.

사용 예시:
python models_효상.py \
    --csv ./pose_data_combined.csv \
    --label-csv ./fall_intervals.csv \
    --epochs 20 --seq-len 45 --stride 5

label-csv 포맷은 start,end,fall,part(정수) 컬럼을 가진 간단한 CSV입니다.
start/end는 merged CSV 상의 행 인덱스 구간(포함/제외)입니다.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

from fall.model_gru import FallTemporalHybridNet

SELECTED_IDX = [0, 10, 15, 16, 23, 24]
PART_LABELS = ["머리", "손목", "골반", "기타"]


@dataclass
class EventInterval:
    start: int
    end: int
    fall: int
    part: int


def load_event_intervals(path: Path) -> List[EventInterval]:
    intervals: List[EventInterval] = []
    if not path.exists():
        raise FileNotFoundError(f"label csv not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intervals.append(
                EventInterval(
                    start=int(row["start"]),
                    end=int(row["end"]),
                    fall=int(row["fall"]),
                    part=int(row["part"]),
                )
            )
    return intervals


JOINT_COLUMNS = [
    ("nose", ["nose_x", "nose_y", "nose_z", "nose_speed_z"]),
    ("lm10", ["lm10_x", "lm10_y", "lm10_z", "lm10_speed_z"]),
    ("right_wrist", ["right_wrist_x", "right_wrist_y", "right_wrist_z", "right_wrist_speed_z"]),
    ("left_wrist", ["left_wrist_x", "left_wrist_y", "left_wrist_z", "left_wrist_speed_z"]),
    ("right_hip", ["right_hip_x", "right_hip_y", "right_hip_z", "right_hip_speed_z"]),
    ("left_hip", ["left_hip_x", "left_hip_y", "left_hip_z", "left_hip_speed_z"]),
]
FEATURE_COLUMNS = [col for _, cols in JOINT_COLUMNS for col in cols]
FALL_COLUMN = "fall_label"
PART_COLUMN = "part_label"
FEATURES_PER_JOINT = 5
JOINT_NAME_TO_IDX = {name: idx for idx, (name, _) in enumerate(JOINT_COLUMNS)}
HORIZONTAL_MIRROR_PAIRS = [
    (JOINT_NAME_TO_IDX["right_wrist"], JOINT_NAME_TO_IDX["left_wrist"]),
    (JOINT_NAME_TO_IDX["right_hip"], JOINT_NAME_TO_IDX["left_hip"]),
]


class PoseSequenceDataset(Dataset):
    """
    pose_data_combined.csv 형식 (N x 24)을 받아 시퀀스 단위로 잘라주는 Dataset.
    features: 각 관절별 [x, y, z, speed_z] + 자동 계산한 [Δx, Δy] → 6개 joint × 6특징 = 36차원.
    """

    def __init__(
        self,
        csv_path: Path,
        seq_len: int = 45,
        stride: int = 5,
        intervals: Optional[List[EventInterval]] = None,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.normalize = normalize
        self.intervals = intervals or []
        self.sample_starts: List[int] = []

        try:
            df = pd.read_csv(csv_path, engine="c", low_memory=False)
        except Exception:
            df = pd.read_csv(csv_path, engine="python")
        missing = [col for col in FEATURE_COLUMNS + [FALL_COLUMN, PART_COLUMN] if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 24:
            raise ValueError("CSV must have exactly 24 feature columns")
        joints = raw.reshape(-1, 6, 4)
        coords = joints[..., :3]

        deltas = np.zeros_like(coords)
        deltas[1:] = coords[1:] - coords[:-1]
        # Δz는 원래 speed_z 열을 활용
        deltas[..., 2:3] = joints[..., 3:4]

        features = np.concatenate([coords, deltas[..., :2]], axis=2)
        self.feature_dim = features.shape[2] * features.shape[1]
        self.frames_raw = features.reshape(features.shape[0], -1).astype(np.float32)
        self.frames = self.frames_raw.copy()
        self.mean = None
        self.std = None

        self.fall_labels = df[FALL_COLUMN].astype(np.int64).to_numpy()
        self.part_labels = df[PART_COLUMN].astype(np.int64).to_numpy()
        for interval in self.intervals:
            self.fall_labels[interval.start : interval.end] = interval.fall
            self.part_labels[interval.start : interval.end] = interval.part

        self.samples: List[Tuple[np.ndarray, int, int]] = []
        self.sample_fall_labels: List[int] = []
        self.sample_part_labels: List[int] = []

        if normalize:
            mean = self.frames_raw.mean(axis=0, keepdims=True)
            std = self.frames_raw.std(axis=0, keepdims=True) + 1e-6
            self.apply_normalization(mean, std)
        else:
            self._build_samples()

    def _build_samples(self):
        self.samples = []
        self.sample_fall_labels = []
        self.sample_part_labels = []
        self.sample_starts = []
        total = len(self.frames)
        for start in range(0, total - self.seq_len + 1, self.stride):
            end = start + self.seq_len
            seq = self.frames[start:end]
            fall = int(np.round(self.fall_labels[start:end].mean()))
            part = int(np.round(self.part_labels[start:end].mean()))
            self.samples.append((seq.astype(np.float32), fall, part))
            self.sample_fall_labels.append(fall)
            self.sample_part_labels.append(part)
            self.sample_starts.append(start)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, fall, part = self.samples[idx]
        return {
            "sequence": torch.from_numpy(seq),
            "fall": torch.tensor(fall, dtype=torch.long),
            "part": torch.tensor(part, dtype=torch.long),
        }

    def apply_normalization(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.frames = (self.frames_raw - self.mean) / self.std
        self._build_samples()

    def get_frame_indices_for_samples(self, sample_indices: Iterable[int]) -> np.ndarray:
        frame_indices = set()
        total = len(self.frames)
        for sample_idx in sample_indices:
            start = self.sample_starts[sample_idx]
            end = min(start + self.seq_len, total)
            frame_indices.update(range(start, end))
        if not frame_indices:
            return np.arange(total)
        return np.array(sorted(frame_indices), dtype=np.int64)


def _apply_augmentations(
    seq: torch.Tensor,
    noise_std: float,
    scale_jitter: float,
    dropout_prob: float,
    temporal_mask_prob: float,
    horizontal_flip_prob: float,
) -> torch.Tensor:
    if noise_std > 0:
        seq = seq + torch.randn_like(seq) * noise_std
    if scale_jitter > 0:
        low = max(0.0, 1.0 - scale_jitter)
        high = 1.0 + scale_jitter
        scales = torch.empty(seq.size(0), 1, 1, device=seq.device).uniform_(low, high)
        seq = seq * scales
    if dropout_prob > 0:
        mask = torch.rand_like(seq).lt(dropout_prob)
        seq = seq.masked_fill(mask, 0.0)
    if temporal_mask_prob > 0:
        frame_mask = torch.rand(seq.size(0), seq.size(1), 1, device=seq.device) < temporal_mask_prob
        seq = seq.masked_fill(frame_mask, 0.0)
    if horizontal_flip_prob > 0:
        flip_mask = torch.rand(seq.size(0), device=seq.device) < horizontal_flip_prob
        if flip_mask.any():
            seq[flip_mask] = _horizontal_flip_batch(seq[flip_mask])
    return seq


def _horizontal_flip_batch(batch_seq: torch.Tensor) -> torch.Tensor:
    batch, seq_len, dim = batch_seq.shape
    if dim % FEATURES_PER_JOINT != 0:
        raise ValueError(
            f"Feature dimension {dim} is not divisible by {FEATURES_PER_JOINT}; cannot apply horizontal flip."
        )
    joint_dim = dim // FEATURES_PER_JOINT
    view = batch_seq.reshape(batch, seq_len, joint_dim, FEATURES_PER_JOINT).clone()
    view[..., 0] = 1.0 - view[..., 0]
    view[..., 3] = -view[..., 3]
    for a_idx, b_idx in HORIZONTAL_MIRROR_PAIRS:
        view[..., [a_idx, b_idx], :] = view[..., [b_idx, a_idx], :]
    return view.reshape_as(batch_seq)


class MirrorAugmentedSubset(Dataset):
    """Subset wrapper that deterministically includes mirrored copies of every sequence."""

    def __init__(self, subset: Dataset):
        self.subset = subset

    def __len__(self):
        return len(self.subset) * 2

    def __getitem__(self, idx: int):
        base_idx = idx // 2
        mirror = idx % 2 == 1
        sample = self.subset[base_idx]
        if not mirror:
            return sample
        seq = sample["sequence"]
        flipped = _horizontal_flip_batch(seq.unsqueeze(0)).squeeze(0)
        return {
            "sequence": flipped,
            "fall": sample["fall"],
            "part": sample["part"],
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    fall_criterion: nn.Module,
    part_criterion: nn.Module,
    device: torch.device,
    part_weight: float = 0.3,
    augment: bool = False,
    augment_cfg: Optional[dict] = None,
    epoch_idx: int = 1,
    total_epochs: int = 1,
) -> float:
    model.train()
    decay = 1.0
    if total_epochs > 0:
        progress = min(max(epoch_idx - 1, 0), total_epochs) / total_epochs
        decay = max(0.05, 1.0 - progress)
    total_loss = 0.0
    for batch in loader:
        seq = batch["sequence"].to(device, non_blocking=True)
        fall_label = batch["fall"].to(device, non_blocking=True)
        part_label = batch["part"].to(device, non_blocking=True)
        if augment and augment_cfg:
            noise_std = augment_cfg.get("noise_std", 0.0) * decay
            scale_jitter = augment_cfg.get("scale_jitter", 0.0) * decay
            dropout_prob = augment_cfg.get("dropout_prob", 0.0) * decay
            temporal_mask_prob = augment_cfg.get("temporal_mask_prob", 0.0) * decay
            horizontal_flip_prob = augment_cfg.get("horizontal_flip_prob", 0.0)
            seq = _apply_augmentations(
                seq,
                noise_std=noise_std,
                scale_jitter=scale_jitter,
                dropout_prob=dropout_prob,
                temporal_mask_prob=temporal_mask_prob,
                horizontal_flip_prob=horizontal_flip_prob,
            )

        optimizer.zero_grad()
        fall_logits, part_logits = model(seq)
        fall_loss = fall_criterion(fall_logits, fall_label)
        part_loss = part_criterion(part_logits, part_label)
        loss = fall_loss + part_weight * part_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    fall_criterion: nn.Module,
    part_criterion: nn.Module,
    device: torch.device,
    part_weight: float = 0.3,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    for batch in loader:
        seq = batch["sequence"].to(device, non_blocking=True)
        fall_label = batch["fall"].to(device, non_blocking=True)
        part_label = batch["part"].to(device, non_blocking=True)
        fall_logits, part_logits = model(seq)
        fall_loss = fall_criterion(fall_logits, fall_label)
        part_loss = part_criterion(part_logits, part_label)
        loss = fall_loss + part_weight * part_loss
        total_loss += loss.item()
        total_correct += (fall_logits.argmax(dim=1) == fall_label).sum().item()
        total += fall_label.size(0)
        fall_probs = torch.softmax(fall_logits, dim=1)[:, 1]
        diff = fall_probs - fall_label.float()
        mse_sum += torch.sum(diff * diff).item()
        mae_sum += torch.sum(diff.abs()).item()
    acc = total_correct / max(total, 1)
    mse = mse_sum / max(total, 1)
    mae = mae_sum / max(total, 1)
    return total_loss / max(len(loader), 1), acc, mse, mae


def plot_training_curves(history: dict, out_path: Path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if not epochs:
        print("⚠️ 저장할 학습 기록이 없습니다.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="학습손실")
    axes[0].plot(epochs, history["val_loss"], label="검증손실")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("손실 추이")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="검증정확도")
    axes[1].plot(epochs, history["val_mse"], label="MSE")
    axes[1].plot(epochs, history["val_mae"], label="MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("정확도 / 오차 추이")
    axes[1].legend()

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 학습 곡선 저장 완료: {out_path}")


def build_loaders(
    csv_path: Path,
    label_csv: Optional[Path],
    seq_len: int,
    stride: int,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int,
    train_mirror_duplicate: bool = False,
) -> Tuple[DataLoader, DataLoader, int, List[int]]:
    intervals = load_event_intervals(label_csv) if label_csv else None
    dataset = PoseSequenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        stride=stride,
        intervals=intervals,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training samples generated. Adjust seq_len/stride.")

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    if len(train_set) == 0:
        raise RuntimeError("Train split is empty; adjust val_ratio.")
    train_frame_idx = dataset.get_frame_indices_for_samples(train_set.indices)
    train_frames = dataset.frames_raw[train_frame_idx]
    mean = train_frames.mean(axis=0, keepdims=True)
    std = train_frames.std(axis=0, keepdims=True) + 1e-6
    dataset.apply_normalization(mean, std)
    if train_mirror_duplicate:
        train_set = MirrorAugmentedSubset(train_set)

    loader_kwargs = dict(
        num_workers=max(0, num_workers),
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, dataset.feature_dim, dataset.sample_fall_labels


def detect_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device_name: str) -> torch.device:
    name = device_name.lower()
    if name == "auto":
        return torch.device(detect_default_device())
    if name == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없어 CPU로 대체합니다.")
        name = "cpu"
    if name == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("⚠️ MPS를 사용할 수 없어 CPU로 대체합니다.")
            name = "cpu"
    return torch.device(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Fall detection trainer for pose_data CSV.")
    default_csv = Path(__file__).resolve().parent / "pose_data_combined.csv"
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"병합된 pose_data CSV 경로 (기본: {default_csv})",
    )
    parser.add_argument("--label-csv", type=Path, help="낙상 구간 라벨 start,end,fall,part CSV")
    parser.add_argument("--seq-len", type=int, default=45)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--part-weight", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 수 (기본 0)")
    parser.add_argument(
        "--device",
        type=str,
        default=detect_default_device(),
        choices=["auto", "cpu", "cuda", "mps"],
        help="훈련 디바이스 (기본값: 자동 감지)",
    )
    parser.add_argument("--no-augment", action="store_true", help="데이터 증강 비활성화")
    parser.add_argument("--augment-noise-std", type=float, default=0.02, help="가우시안 노이즈 표준편차")
    parser.add_argument("--augment-scale-jitter", type=float, default=0.1, help="전체 스케일 변화 비율")
    parser.add_argument("--augment-dropout-prob", type=float, default=0.05, help="특징별 드롭 확률")
    parser.add_argument("--augment-temporal-mask", type=float, default=0.05, help="프레임 드롭 확률")
    parser.add_argument("--augment-horizontal-flip", type=float, default=0.5, help="좌우 반전 적용 확률")
    parser.add_argument("--train-mirror-duplicate", action="store_true", help="훈련 세트에 좌우 반전 시퀀스를 한 번 더 포함")
    parser.add_argument("--plot-file", type=Path, default=Path("training_curves.png"), help="학습 곡선 저장 경로")
    parser.add_argument("--no-plot", action="store_true", help="학습 곡선 그래프 저장 생략")
    parser.add_argument("--seed", type=int, default=42, help="데이터 분할을 위한 시드")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    train_loader, val_loader, feature_dim, sample_fall_labels = build_loaders(
        csv_path=csv_path,
        label_csv=args.label_csv,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        train_mirror_duplicate=args.train_mirror_duplicate,
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    fall_counts = np.bincount(sample_fall_labels or [0], minlength=2)
    fall_weights = fall_counts.max() / (fall_counts + 1e-6)
    fall_weight_tensor = torch.tensor(fall_weights, dtype=torch.float32, device=device)

    model = FallTemporalHybridNet(
        input_dim=feature_dim,
        temporal_channels=96,
        hidden_dim=160,
        num_layers=2,
        fall_classes=2,
        part_classes=len(PART_LABELS),
        include_velocity=False,  # 이미 Δx, Δy를 포함했으므로 추가 연산 생략
    ).to(device)

    fall_criterion = nn.CrossEntropyLoss(weight=fall_weight_tensor)
    part_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    augment_enabled = not args.no_augment
    augment_cfg = {
        "noise_std": max(0.0, args.augment_noise_std),
        "scale_jitter": max(0.0, args.augment_scale_jitter),
        "dropout_prob": min(max(0.0, args.augment_dropout_prob), 1.0),
        "temporal_mask_prob": min(max(0.0, args.augment_temporal_mask), 1.0),
        "horizontal_flip_prob": min(max(0.0, args.augment_horizontal_flip), 1.0),
    }

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_mse": [],
        "val_mae": [],
    }
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            fall_criterion,
            part_criterion,
            device,
            args.part_weight,
            augment=augment_enabled,
            augment_cfg=augment_cfg,
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_acc, val_mse, val_mae = evaluate(
            model,
            val_loader,
            fall_criterion,
            part_criterion,
            device,
            args.part_weight,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        print(
            "  "
            f"학습손실(모델이 학습 데이터에서 틀린 정도)={train_loss:.4f} "
            f"검증손실(새 데이터에서 낙상/부위 예측이 틀린 정도)={val_loss:.4f} "
            f"검증낙상정확도(낙상/정상 분류 정확도)={val_acc:.3f} "
            f"MSE(낙상 확률과 정답 차이의 제곱 평균)={val_mse:.4f} "
            f"MAE(낙상 확률과 정답 차이의 절대 평균)={val_mae:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            out_path = Path("fall") / "fall_temporal_hybrid_best.pth"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_path))
            print(f"  ✅ Saved best model (acc={best_acc:.3f})")

    if not args.no_plot:
        try:
            plot_training_curves(history, args.plot_file)
        except Exception as e:
            print(f"⚠️ 학습 곡선 저장 실패: {e}")


if __name__ == "__main__":
    main()
