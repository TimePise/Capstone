from django.http import StreamingHttpResponse, JsonResponse
import torch
import numpy as np
import cv2
import mediapipe as mp
import os
import threading
import json
from collections import deque
from django.conf import settings
from .models import FallAlert
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .model_gru import FallTemporalHybridNet
import pygame
import time
from datetime import datetime

# 상태 플래그 및 공유 변수
pose_thread_started = False
shared_frame = None  # ✅ 스트리밍용 공유 프레임
last_visible_frame = None
privacy_lock = threading.Lock()
frame_lock = threading.Lock()
landmark_lock = threading.Lock()

# 전역 상태
privacy_mode = False
last_fall_label = "정상입니다"
last_fall_pred = 0
alarm_cooldown = 0
ALARM_INTERVAL = 5  # 최소 알림 간격 (초)
FRAME_DELAY = 0.05
POSE_CONFIDENCE = 0.6
VISIBILITY_THRESHOLD = 0.45
SEQUENCE_WINDOW = 45
MIN_SEQUENCE_LENGTH = 30
FALL_SCORE_THRESHOLD = 0.5
FALL_CONFIRMATION_WINDOW = 4
FALL_CONFIRMATION_THRESHOLD = 2
PART_CONF_THRESHOLD = 0.55
PART_LABELS = ["머리", "손목", "골반", "기타"]
FALL_LEVEL_BY_PART = {
    "머리": "고위험",
    "골반": "중위험",
    "손목": "저위험",
    "기타": "저위험",
}
LANDMARK_LABELS = {
    0: "코",
    10: "오른눈",
    15: "왼손목",
    16: "오른손목",
    23: "왼엉덩",
    24: "오른엉덩",
}
landmark_text_lines = ["좌표 정보를 수집 중입니다..."]

FEATURE_DIM = 30
FEATURE_STATS_PATH = os.path.join(settings.BASE_DIR, 'fall', 'feature_stats.json')
try:
    with open(FEATURE_STATS_PATH, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    feature_mean = np.array(stats.get('mean', []), dtype=np.float32)
    feature_std = np.array(stats.get('std', []), dtype=np.float32)
    if feature_mean.shape[0] != FEATURE_DIM or feature_std.shape[0] != FEATURE_DIM:
        raise ValueError("feature_stats dimension mismatch")
except Exception as e:
    print("⚠️ feature_stats 불러오기 실패:", e)
    feature_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    feature_std = np.ones(FEATURE_DIM, dtype=np.float32)

HIP_DROP_THRESHOLD = 0.12
HIP_BASELINE_ALPHA = 0.02
hip_baseline = None

# 모델 준비
SELECTED_IDX = [0, 10, 15, 16, 23, 24]
model = FallTemporalHybridNet(
    input_dim=FEATURE_DIM,
    temporal_channels=96,
    hidden_dim=160,
    num_layers=2,
    fall_classes=2,
    part_classes=len(PART_LABELS),
    include_velocity=False,
)
model_path = os.path.join(settings.BASE_DIR, 'fall', 'fall_temporal_hybrid_best.pth')
try:
    state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
except Exception as e:
    print("❌ 하이브리드 모델 로드 실패:", e)
model.eval()

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 알람 설정
ALARM_PATH = os.path.join(settings.BASE_DIR, 'fall', 'fall_alert.mp3')

def play_alarm():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.play()
    except Exception as e:
        print("❌ 알람 실패:", e)

def _set_privacy_mode(enabled: bool):
    global privacy_mode, shared_frame, last_visible_frame
    with privacy_lock:
        privacy_mode = enabled
        if privacy_mode and shared_frame is not None:
            shared_frame = np.zeros_like(shared_frame)
        elif not privacy_mode:
            with frame_lock:
                if last_visible_frame is not None:
                    shared_frame = last_visible_frame.copy()
        return privacy_mode

def get_privacy_mode_state():
    with privacy_lock:
        return privacy_mode

def _update_landmark_text(landmarks):
    lines = ["Pose 좌표 (정규화 기준)"]
    for idx in SELECTED_IDX:
        lm = landmarks[idx]
        label = LANDMARK_LABELS.get(idx, f"LM{idx}")
        lines.append(f"{label:<6} X:{lm.x:.3f}  Y:{lm.y:.3f}  Z:{lm.z:.3f}")
    with landmark_lock:
        global landmark_text_lines
        landmark_text_lines = lines

def _get_landmark_text():
    with landmark_lock:
        return list(landmark_text_lines)

def _update_hip_baseline(current_y):
    global hip_baseline
    if current_y is None:
        return
    if hip_baseline is None:
        hip_baseline = current_y
    else:
        hip_baseline = (1 - HIP_BASELINE_ALPHA) * hip_baseline + HIP_BASELINE_ALPHA * current_y

def _reset_hip_baseline():
    global hip_baseline
    hip_baseline = None

def toggle_privacy_mode(request):
    new_state = _set_privacy_mode(not get_privacy_mode_state())
    return JsonResponse({'privacy_mode': new_state})

def fall_status(request):
    return JsonResponse({
        'label': last_fall_label,
        'fall': last_fall_pred == 1
    })

def reset_alert_lock(request):
    return JsonResponse({'status': 'reset complete'})

# ✅ 감지 백그라운드 스레드 실행 함수
def start_pose_thread_once():
    global pose_thread_started
    if not pose_thread_started:
        print("📡 낙상 감지 쓰레드 시작됨")
        t = threading.Thread(target=generate_pose_estimation, daemon=True)
        t.start()
        pose_thread_started = True

# ✅ 낙상 감지 루프 (프레임 저장 포함)
def generate_pose_estimation():
    global last_fall_label, last_fall_pred, alarm_cooldown, shared_frame, last_visible_frame, hip_baseline
    sequence = deque(maxlen=SEQUENCE_WINDOW)
    prev_coords = None
    fall_votes = deque(maxlen=FALL_CONFIRMATION_WINDOW)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 카메라 연결 실패")
        return

    try:
        pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=POSE_CONFIDENCE,
            min_tracking_confidence=POSE_CONFIDENCE,
        )
    except Exception as e:
        print(f"❌ Pose 초기화 실패: {e}")
        cap.release()
        return

    try:
        while cap.isOpened():
            ret, original_frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(original_frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            label = "정상입니다"
            fall_pred = 0

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                if all(landmarks[idx].visibility >= VISIBILITY_THRESHOLD for idx in SELECTED_IDX):
                    _update_landmark_text(landmarks)
                    hip_center_y = (landmarks[23].y + landmarks[24].y) / 2.0
                    coords = np.array(
                        [[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] for idx in SELECTED_IDX],
                        dtype=np.float32,
                    )
                    if prev_coords is None:
                        deltas = np.zeros_like(coords)
                    else:
                        deltas = coords - prev_coords
                    prev_coords = coords
                    feature_vec = np.concatenate([coords, deltas[:, :2]], axis=1).reshape(-1)
                    normalized = ((feature_vec - feature_mean) / feature_std).astype(np.float32)
                    sequence.append(normalized)

                    if len(sequence) >= MIN_SEQUENCE_LENGTH:
                        input_seq = np.array(list(sequence)[-MIN_SEQUENCE_LENGTH:], dtype=np.float32)
                        input_tensor = torch.from_numpy(input_seq).unsqueeze(0)
                        with torch.no_grad():
                            fall_out, part_out = model(input_tensor)
                            fall_probs = torch.softmax(fall_out, dim=1)
                            fall_score = float(fall_probs[0, 1])
                            raw_pred = 1 if fall_score >= FALL_SCORE_THRESHOLD else 0
                            fall_votes.append(raw_pred)
                            fall_candidate = 1 if sum(fall_votes) >= FALL_CONFIRMATION_THRESHOLD else 0
                            if fall_candidate and hip_baseline is not None:
                                hip_drop = hip_center_y - hip_baseline
                                if hip_drop < HIP_DROP_THRESHOLD:
                                    fall_candidate = 0
                                    if fall_votes:
                                        fall_votes[-1] = 0
                            fall_pred = fall_candidate

                            if fall_pred == 1:
                                current_time = time.time()
                                if current_time - alarm_cooldown >= ALARM_INTERVAL:
                                    alarm_cooldown = current_time

                                    part_probs = torch.softmax(part_out, dim=1)
                                    part_idx = torch.argmax(part_probs, dim=1).item()
                                    part_score = float(part_probs[0, part_idx])
                                    part = PART_LABELS[part_idx] if part_idx < len(PART_LABELS) else "기타"

                                    if part_score < PART_CONF_THRESHOLD:
                                        z_parts = {
                                            "머리": min(landmarks[i].z for i in [0, 10]),
                                            "손목": min(landmarks[i].z for i in [15, 16]),
                                            "골반": min(landmarks[i].z for i in [23, 24]),
                                        }
                                        part = min(z_parts, key=z_parts.get)

                                    label = f"{part} 중심 낙상 발생"
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    fall_level = FALL_LEVEL_BY_PART.get(part, "저위험")

                                    threading.Thread(target=play_alarm, daemon=True).start()

                                    alert = FallAlert.objects.create(
                                        message=label,
                                        part=part,
                                        fall_level=fall_level,
                                        name="환자A",
                                        room_number="101호",
                                        is_read=False
                                    )

                                    _reset_hip_baseline()

                                    channel_layer = get_channel_layer()
                                    async_to_sync(channel_layer.group_send)(
                                        "fall_alert_group",
                                        {
                                            "type": "send_alert",
                                            "id": alert.id,
                                            "message": label,
                                            "name": "환자A",
                                            "room_number": "101호",
                                            "fall_level": fall_level,
                                            "part": part,
                                            "timestamp": timestamp
                                        }
                                    )
                            else:
                                _update_hip_baseline(hip_center_y)
                    else:
                        _update_hip_baseline(hip_center_y)
                else:
                    prev_coords = None
                    _reset_hip_baseline()
            else:
                prev_coords = None
                _reset_hip_baseline()

            last_fall_label = label
            last_fall_pred = fall_pred

            privacy_active = get_privacy_mode_state()

            if result.pose_landmarks and not privacy_active:
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

            with frame_lock:
                last_visible_frame = frame.copy()

            if privacy_active:
                frame[:] = (0, 0, 0)
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )

            shared_frame = frame.copy()  # ✅ 최신 프레임 저장

            time.sleep(FRAME_DELAY)

    except Exception as e:
        print(f"❌ 통합 루프 오류 발생: {e}")
    finally:
        pose.close()
        cap.release()
        print("📷 카메라 자원 해제 완료")

# ✅ 프레임만 보여주는 스트리밍 함수
def pose_estimation_feed(request):
    def stream_shared_frame():
        try:
            while True:
                if shared_frame is not None:
                    _, buffer = cv2.imencode('.jpg', shared_frame)
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    )
                time.sleep(0.05)
        except (GeneratorExit, ConnectionAbortedError, BrokenPipeError):
            return
    return StreamingHttpResponse(stream_shared_frame(), content_type='multipart/x-mixed-replace; boundary=frame')

# ✅ SSE 스트리밍 알림
def fall_alert_stream(request):
    def event_stream():
        last_sent = None
        try:
            while True:
                alert = FallAlert.objects.filter(is_read=False).order_by('-timestamp').first()
                if alert and alert.timestamp != last_sent:
                    last_sent = alert.timestamp
                    payload = {
                        "id": alert.id,
                        "message": alert.message,
                        "name": alert.name,
                        "room_number": alert.room_number,
                        "fall_level": alert.fall_level,
                        "part": alert.part,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                time.sleep(1)
        except (GeneratorExit, ConnectionAbortedError, BrokenPipeError):
            return
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
