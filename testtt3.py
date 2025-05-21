
import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("fall_detection_model_0415.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

sequence = []
SEQUENCE_LENGTH = 30

fall_timer = 0
last_prediction = "Normal"

def extract_features_from_landmarks_seq(sequence):
    sequence = np.array(sequence)

    def safe_dist(a, b, frame):
        try:
            return np.linalg.norm(frame[a] - frame[b])
        except:
            return 0.0

    shoulder_dists = []
    head_hip_dists = []
    shoulder_angles = []
    center_ys = []
    lw_la_dists = []
    rw_ra_dists = []

    for frame in sequence:
        try:
            shoulder_dists.append(safe_dist(11, 12, frame))
            mid_hip = (frame[23] + frame[24]) / 2 if 23 < len(frame) and 24 < len(frame) else np.array([0.0, 0.0])
            head_hip_dists.append(np.linalg.norm(frame[0] - mid_hip) if 0 < len(frame) else 0.0)
            shoulder_angles.append(np.arctan2(frame[11][1] - frame[12][1], frame[11][0] - frame[12][0]) if 11 < len(frame) and 12 < len(frame) else 0.0)
            center_ys.append(mid_hip[1])
            lw_la_dists.append(safe_dist(15, 27, frame))
            rw_ra_dists.append(safe_dist(16, 28, frame))
        except:
            shoulder_dists.append(0.0)
            head_hip_dists.append(0.0)
            shoulder_angles.append(0.0)
            center_ys.append(0.0)
            lw_la_dists.append(0.0)
            rw_ra_dists.append(0.0)

    try:
        delta_ys = np.diff(center_ys)
    except:
        delta_ys = [0.0]

    features = [
        np.mean(shoulder_dists), np.std(shoulder_dists),
        np.mean(head_hip_dists), np.std(head_hip_dists),
        np.mean(shoulder_angles), np.std(shoulder_angles),
        np.mean(np.abs(delta_ys)), np.max(np.abs(delta_ys)),
        np.mean(lw_la_dists), np.mean(rw_ra_dists),
        0  # risk_score placeholder
    ]

    return np.array(features[:-1])  # 10 features only

cap = cv2.VideoCapture(0)
print("🔄 Real-time Fall Detection started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark])
        sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            features = extract_features_from_landmarks_seq(sequence).reshape(1, -1)
            try:
                prediction = model.predict(features)[0]
                if prediction != "Normal" and prediction != "정상":
                    fall_timer = 90
                    last_prediction = prediction
                elif prediction == "정상":
                    last_prediction = "Normal"
            except Exception as e:
                last_prediction = "Prediction error"
            sequence = []

    if fall_timer > 0:
        label = f"Fall Detected: {last_prediction}"
        color = (0, 0, 255)
        fall_timer -= 1
    else:
        label = "Normal"
        color = (0, 255, 0)

    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Fall Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
