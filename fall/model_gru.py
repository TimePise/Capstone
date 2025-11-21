import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)


class FallBiGRUAttentionNet(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, fall_classes=2, part_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.fall_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, fall_classes)
        )

        self.part_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, part_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.attn(out)
        out = self.dropout(out)
        return self.fall_head(out), self.part_head(out)


class TemporalFeatureEncoder(nn.Module):
    """Convolutional temporal encoder that augments pose features with velocity."""

    def __init__(self, input_dim: int, temporal_channels: int = 64, include_velocity: bool = True):
        super().__init__()
        self.include_velocity = include_velocity
        conv_in_dim = input_dim * (2 if include_velocity else 1)
        self.net = nn.Sequential(
            nn.Conv1d(conv_in_dim, temporal_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_channels),
            nn.GELU(),
            nn.Conv1d(
                temporal_channels,
                temporal_channels,
                kernel_size=3,
                padding=1,
                groups=temporal_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.include_velocity:
            vel = torch.zeros_like(x)
            vel[:, 1:] = x[:, 1:] - x[:, :-1]
            seq = torch.cat([x, vel], dim=2)
        else:
            seq = x
        seq = seq.transpose(1, 2)  # (B, C, T)
        seq = self.net(seq)
        return seq.transpose(1, 2)  # (B, T, C')


class FallTemporalHybridNet(nn.Module):
    """
    Hybrid temporal model that injects local temporal patterns (Conv1d) before GRU.
    Keeps the 기존 어텐션/멀티헤드 구조를 사용하면서 x/y/z 속도까지 자동으로 학습합니다.
    """

    def __init__(
        self,
        input_dim=24,
        temporal_channels=64,
        hidden_dim=128,
        num_layers=2,
        fall_classes=2,
        part_classes=4,
        include_velocity=True,
    ):
        super().__init__()
        self.temporal_encoder = TemporalFeatureEncoder(
            input_dim=input_dim,
            temporal_channels=temporal_channels,
            include_velocity=include_velocity,
        )
        self.gru = nn.GRU(
            temporal_channels,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.fall_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, fall_classes),
        )

        self.part_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, part_classes),
        )

    def forward(self, x: torch.Tensor):
        seq = self.temporal_encoder(x)
        out, _ = self.gru(seq)
        out = self.attn(out)
        out = self.dropout(out)
        return self.fall_head(out), self.part_head(out)
