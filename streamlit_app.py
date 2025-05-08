import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Webcam
cap = st.image(frame, channels="BGR")

# Timer
start_time = time.time()
duration = 60  # seconds

# Red dot
dot_x, dot_y = np.random.randint(100, 500), np.random.randint(100, 400)
last_dot_move = time.time()

# Trackers
tracking_errors = []
nose_positions = []
jaw_tension_scores = []

print("NanoScan running... Follow the red dot with your eyes and try to stay still.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    elapsed = time.time() - start_time
    if elapsed > duration:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Eye center (landmarks 33 and 263)
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_center = ((left_eye.x + right_eye.x) / 2 * w,
                      (left_eye.y + right_eye.y) / 2 * h)

        # Track distance from red dot
        dist = np.linalg.norm(np.array([dot_x, dot_y]) - np.array(eye_center))
        tracking_errors.append(dist)
        cv2.circle(frame, (int(eye_center[0]), int(eye_center[1])), 5, (0, 255, 0), -1)

        # Nose (for head movement)
        nose = landmarks[1]
        nose_pos = (nose.x * w, nose.y * h)
        nose_positions.append(nose_pos)

        # Jaw tension (jaw corners 234 & 454)
        jaw_diff = np.linalg.norm(np.array([landmarks[234].x, landmarks[234].y]) -
                                  np.array([landmarks[454].x, landmarks[454].y]))
        jaw_tension_scores.append(jaw_diff)

    # Move red dot every 3 seconds
    if time.time() - last_dot_move > 3:
        dot_x, dot_y = np.random.randint(100, 500), np.random.randint(100, 400)
        last_dot_move = time.time()

    # Draw red dot and timer
    cv2.circle(frame, (dot_x, dot_y), 10, (0, 0, 255), -1)
    cv2.putText(frame, f"Time Left: {int(duration - elapsed)}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("NanoScan (Tracking Accuracy)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

# -------------------------
# Score Computation
# -------------------------
def std_dev(coords):
    return np.std(coords, axis=0) if len(coords) > 1 else (0, 0)

# Eye tracking accuracy
if tracking_errors:
    avg_error = np.mean(tracking_errors)
    max_error = 300  # Define a max "bad" error in pixels
    norm_error = min(avg_error / max_error, 1.0)
    eye_score = int((1 - norm_error) * 100)
else:
    eye_score = 0

# Head stillness
head_jitter = std_dev(nose_positions)
head_score = max(0, 100 - int(head_jitter[0] + head_jitter[1]) // 2)

# Facial tension
jaw_std = np.std(jaw_tension_scores) if jaw_tension_scores else 0
tension_score = max(0, 100 - int(jaw_std * 1000))

# Weighted final score
final_score = int((eye_score * 0.5 + head_score * 0.3 + tension_score * 0.2))

# Report
print("\n--- NanoScan Report ---")
print(f"Eye Tracking Accuracy Score: {eye_score}/100")
print(f"Head Stillness Score: {head_score}/100")
print(f"Facial Tension Score: {tension_score}/100")
print(f"Final Score: {final_score}/100")
