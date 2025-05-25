from django.http import StreamingHttpResponse, JsonResponse
import torch
import numpy as np
import cv2
import mediapipe as mp
import os
import threading
from django.conf import settings
from .models import FallAlert
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .model_gru import FallBiGRUAttentionNet
import pygame
import time
from datetime import datetime

pose_thread_started = False

def start_pose_thread_once():
    global pose_thread_started
    if not pose_thread_started:
        print("📡 낙상 감지 쓰레드 시작됨")
        t = threading.Thread(target=generate_pose_estimation, daemon=True)
        t.start()
        pose_thread_started = True

# 전역 상태
privacy_mode = False
last_fall_label = "정상입니다"
last_fall_pred = 0
alarm_cooldown = 0
ALARM_INTERVAL = 5  # 최소 알림 간격 (초 단위)

# 모델 준비
SELECTED_IDX = [0, 10, 15, 16, 23, 24]
model = FallBiGRUAttentionNet(input_dim=24, hidden_dim=128, num_layers=2, fall_classes=2, part_classes=4)
model_path = os.path.join(settings.BASE_DIR, 'fall', 'fall_bigru_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
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

def toggle_privacy_mode(request):
    global privacy_mode
    privacy_mode = not privacy_mode
    return JsonResponse({'privacy_mode': privacy_mode})

def fall_status(request):
    return JsonResponse({
        'label': last_fall_label,
        'fall': last_fall_pred == 1
    })

def reset_alert_lock(request):
    return JsonResponse({'status': 'reset complete'})

def generate_pose_estimation():
    global privacy_mode, last_fall_label, last_fall_pred, alarm_cooldown
    from .models import FallAlert

    sequence = []
    prev_zs = None
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 카메라 연결 실패")
        return

    try:
        while cap.isOpened():
            ret, original_frame = cap.read()
            if not ret:
                break

            original_frame = cv2.flip(original_frame, 1)
            rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            frame = np.zeros_like(original_frame) if privacy_mode else original_frame.copy()

            label = "정상입니다"
            color = (0, 255, 0)
            fall_pred = 0

            if result.pose_landmarks:
                keypoints = []
                current_zs = []

                for idx in SELECTED_IDX:
                    lm = result.pose_landmarks.landmark[idx]
                    current_zs.append(lm.z)

                for i, idx in enumerate(SELECTED_IDX):
                    lm = result.pose_landmarks.landmark[idx]
                    z_now = lm.z
                    z_prev = prev_zs[i] if prev_zs else z_now
                    speed_z = z_now - z_prev
                    keypoints.extend([lm.x, lm.y, lm.z, speed_z])

                prev_zs = current_zs
                sequence.append(keypoints)

                if len(sequence) >= 30:
                    input_seq = np.array(sequence[-30:])
                    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        fall_out, _ = model(input_tensor)
                        fall_pred = torch.argmax(fall_out, dim=1).item()

                        if fall_pred == 1:
                            current_time = time.time()
                            if current_time - alarm_cooldown >= ALARM_INTERVAL:
                                alarm_cooldown = current_time

                                z_parts = {
                                    "머리": min(result.pose_landmarks.landmark[i].z for i in [0, 10]),
                                    "손목": min(result.pose_landmarks.landmark[i].z for i in [15, 16]),
                                    "골반": min(result.pose_landmarks.landmark[i].z for i in [23, 24]),
                                }
                                part = min(z_parts, key=z_parts.get)
                                label = f"{part} 중심 낙상 발생"
                                color = (0, 0, 255)
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                threading.Thread(target=play_alarm, daemon=True).start()

                                FallAlert.objects.create(
                                    message=label,
                                    part=part,
                                    fall_level="심각",
                                    name="환자A",
                                    room_number="101호",
                                    is_read=False
                                )

                                channel_layer = get_channel_layer()
                                async_to_sync(channel_layer.group_send)(
                                    "fall_alert_group",
                                    {
                                        "type": "send_alert",
                                        "message": label,
                                        "name": "환자A",
                                        "room_number": "101호",
                                        "fall_level": "심각",
                                        "part": part,
                                        "timestamp": timestamp
                                    }
                                )
                        else:
                            label = "정상입니다"
                            color = (0, 255, 0)

            last_fall_label = label
            last_fall_pred = fall_pred

            if result.pose_landmarks:
                landmark_spec = mp_drawing.DrawingSpec(
                    color=(255, 255, 255) if privacy_mode else (0, 255, 255),
                    thickness=4, circle_radius=4
                )
                connection_spec = mp_drawing.DrawingSpec(
                    color=(255, 255, 255) if privacy_mode else (0, 255, 255),
                    thickness=3
                )
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec
                )

            _, buffer = cv2.imencode('.jpg', frame)

            try:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                )
                time.sleep(0.05)
            except GeneratorExit:
                print("🛑연결 종료됨")
                break
            except Exception as e:
                print(f"❌ 스트리밍 전송 오류: {e}")
                break

    except Exception as e:
        print(f"❌ 루프 내부 오류 발생: {e}")
    finally:
        cap.release()
        print("📷 카메라 자원 해제 완료")

def pose_estimation_feed(request):
    try:
        return StreamingHttpResponse(
            generate_pose_estimation(),
            content_type='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"❌ 스트리밍 오류: {e}")
        return JsonResponse({"error": "스트리밍 실패"}, status=500)

def fall_alert_stream(request):
    def event_stream():
        last_sent = None
        while True:
            alert = FallAlert.objects.filter(is_read=False).order_by('-timestamp').first()
            if alert and alert.timestamp != last_sent:
                last_sent = alert.timestamp
                yield f"data: {alert.message}\n\n"
            time.sleep(1)  # 1초마다 확인

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')