import cv2
import mediapipe as mp
import numpy as np
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model = joblib.load("lsf_model.pkl")

cap = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 1

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            features = []
            for lm in hand:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features)

            prediction = model.predict([features])[0]

            cv2.putText(frame, f"Signe: {prediction}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("LSF Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()
