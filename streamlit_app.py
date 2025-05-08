import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

st.title("NanoScan: Eye Tracking and Facial Analysis")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    # Tracking variables
    tracking_errors = []
    nose_positions = []
    jaw_tension_scores = []

    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        h, w, _ = frame.shape

        # Create a navy blue background
        navy_blue = np.zeros_like(frame)
        navy_blue[:] = (128, 0, 0)  # BGR for navy blue

        # Red dot position
        dot_x, dot_y = np.random.randint(100, w - 100), np.random.randint(100, h - 100)
        cv2.circle(navy_blue, (dot_x, dot_y), 10, (0, 0, 255), -1)

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
            cv2.circle(navy_blue, (int(eye_center[0]), int(eye_center[1])), 5, (0, 255, 0), -1)

            # Nose (for head movement)
            nose = landmarks[1]
            nose_pos = (nose.x * w, nose.y * h)
            nose_positions.append(nose_pos)

            # Jaw tension (jaw corners 234 & 454)
            jaw_diff = np.linalg.norm(np.array([landmarks[234].x, landmarks[234].y]) -
                                      np.array([landmarks[454].x, landmarks[454].y]))
            jaw_tension_scores.append(jaw_diff)

        stframe.image(navy_blue, channels="BGR")

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
    st.subheader("NanoScan Report")
    st.write(f"**Eye Tracking Accuracy Score:** {eye_score}/100")
    st.write(f"**Head Stillness Score:** {head_score}/100")
    st.write(f"**Facial Tension Score:** {tension_score}/100")
    st.write(f"**Final Score:** {final_score}/100")

