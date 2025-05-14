from django.http import StreamingHttpResponse, JsonResponse
import torch
import numpy as np
import cv2
import mediapipe as mp
import os
import threading
from django.conf import settings
from .model_gru import FallGRU
import pygame

# 전역 상태
privacy_mode = False
last_fall_label = "정상입니다"
last_fall_pred = 0

# 모델 준비
SELECTED_IDX = [0, 10, 15, 16, 23, 24]
model = FallGRU(input_dim=24, hidden_dim=128, num_layers=2, fall_classes=2, part_classes=4)
model_path = os.path.join(settings.BASE_DIR, 'fall', 'fall_gru_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 알람 설정
ALARM_PATH = os.path.join(settings.BASE_DIR, 'fall', 'fall_alert.mp3')
alarm_played = False

def play_alarm():
    global alarm_played
    if not alarm_played:
        alarm_played = True
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(ALARM_PATH)
            pygame.mixer.music.play()
        except Exception as e:
            print("❌ 알람 실패:", e)

# 보호모드 토글 API
def toggle_privacy_mode(request):
    global privacy_mode
    privacy_mode = not privacy_mode
    return JsonResponse({'privacy_mode': privacy_mode})

# 낙상 상태 API
def fall_status(request):
    return JsonResponse({
        'label': last_fall_label,
        'fall': last_fall_pred == 1
    })

# 영상 스트리밍
def generate_pose_estimation():
    global alarm_played, privacy_mode, last_fall_label, last_fall_pred
    sequence = []
    prev_zs = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 연결 실패")
        return

    while cap.isOpened():
        ret, original_frame = cap.read()
        if not ret:
            break

        original_frame = cv2.flip(original_frame, 1)
        rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        # ✅ 보호모드 여부에 따라 배경 설정
        if privacy_mode:
            frame = np.zeros_like(original_frame)  # 검정 배경
        else:
            frame = original_frame.copy()  # 일반 배경

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
                        z_parts = {
                            "머리": min(result.pose_landmarks.landmark[i].z for i in [0, 10]),
                            "손목": min(result.pose_landmarks.landmark[i].z for i in [15, 16]),
                            "골반": min(result.pose_landmarks.landmark[i].z for i in [23, 24]),
                        }
                        part = min(z_parts, key=z_parts.get)
                        label = f"{part} 중심 낙상 발생"
                        color = (0, 0, 255)
                        threading.Thread(target=play_alarm, daemon=True).start()
                    else:
                        label = "정상입니다"
                        color = (0, 255, 0)
                        alarm_played = False

        # ✅ 상태 저장
        last_fall_label = label
        last_fall_pred = fall_pred

        # ✅ 랜드마크는 항상 표시
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3)
            )

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()




# 스트리밍 엔드포인트
def pose_estimation_feed(request):
    return StreamingHttpResponse(generate_pose_estimation(), content_type='multipart/x-mixed-replace; boundary=frame')
