import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

class NanoScanProcessor(VideoTransformerBase):
    def __init__(self):
        self.dot_x = np.random.randint(100, 500)
        self.dot_y = np.random.randint(100, 400)
        self.last_dot_move = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Navy blue background
        canvas = np.zeros_like(img)
        canvas[:] = (128, 0, 0)

        # Move red dot every 3 seconds
        if cv2.getTickCount() / cv2.getTickFrequency() - self.last_dot_move > 3:
            self.dot_x = np.random.randint(100, w - 100)
            self.dot_y = np.random.randint(100, h - 100)
            self.last_dot_move = cv2.getTickCount() / cv2.getTickFrequency()

        # Draw red dot
        cv2.circle(canvas, (self.dot_x, self.dot_y), 10, (0, 0, 255), -1)

        # Face mesh
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            eye_center = ((left_eye.x + right_eye.x) / 2 * w, (left_eye.y + right_eye.y) / 2 * h)

            cv2.circle(canvas, (int(eye_center[0]), int(eye_center[1])), 5, (0, 255, 0), -1)

        return canvas

st.title("NanoScan Live")
webrtc_streamer(key="nanoscanner", video_processor_factory=NanoScanProcessor)

