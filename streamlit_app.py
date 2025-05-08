import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image

# Setup Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# App config
st.set_page_config(page_title="NanoScan Live", layout="centered")
st.title("🧠 NanoScan Live (Webcam Anxiety Tracker)")
st.markdown("Follow the red dot with your eyes and try to stay still.")

run_scan = st.button("Start Scan")
duration = 60  # seconds

# Video frame placeholder
frame_placeholder = st.empty()

# Metrics
eye_error_vals = []
nose_positions = []
jaw_tension_vals = []

def std_dev(coords):
    return np.std(coords, axis=0) if len(coords) > 1 else (0, 0)

if run_scan:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    dot_x, dot_y = np.random.randint(100, 500), np.random.randint(100, 400)
    last_dot_time = time.time()

    st.info("Running scan... Press 'q' in webcam window to quit early.")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Blue background only
        blank_frame = np.zeros_like(frame)
        blank_frame[:] = (128, 0, 0)  # Navy blue background

        # Draw dot
        cv2.circle(blank_frame, (dot_x, dot_y), 10, (0, 0, 255), -1)

        # Move dot every 3s
        if time.time() - last_dot_time > 3:
            dot_x, dot_y = np.random.randint(100, 500), np.random.randint(100, 400)
            last_dot_time = time.time()

        # Detect face
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Eyes
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            eye_center = ((left_eye.x + right_eye.x) / 2 * w, (left_eye.y + right_eye.y) / 2 * h)
            dist = np.linalg.norm(np.array([dot_x, dot_y]) - np.array(eye_center))
            eye_error_vals.append(dist)

            # Head movement (nose tip)
            nose = landmarks[1]
            nose_pos = (nose.x * w, nose.y * h)
            nose_positions.append(nose_pos)

            # Jaw tension (width variation)
            jaw_diff = np.linalg.norm(np.array([landmarks[234].x, landmarks[234].y]) -
                                      np.array([landmarks[454].x, landmarks[454].y]))
            jaw_tension_vals.append(jaw_diff)

        # Display frame
        display = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(display, channels="RGB")

    cap.release()
    st.success("Scan complete!")

    # Scoring
    eye_score = 100 - int(min(np.mean(eye_error_vals) / 3, 100)) if eye_error_vals else 0
    jitter = std_dev(nose_positions)
    head_score = max(0, 100 - int(jitter[0] + jitter[1]) // 2)
    tension_std = np.std(jaw_tension_vals) if jaw_tension_vals else 0
    tension_score = max(0, 100 - int(tension_std * 1000))
    final_score = int((eye_score * 0.5 + head_score * 0.3 + tension_score * 0.2))

    st.subheader("📊 NanoScan Results")
    st.metric("Eye Tracking Accuracy", f"{eye_score}/100")
    st.metric("Head Stillness", f"{head_score}/100")
    st.metric("Facial Tension", f"{tension_score}/100")
    st.metric("Final Score", f"{final_score}/100")
