import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

try:
    with GestureRecognizer.create_from_options(options) as recognizer:
        prev_time = 0
        timestamp = 0
        while cap.isOpened():
            curr_time = cv2.getTickCount()
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = recognizer.recognize_for_video(mp_image, timestamp)
            timestamp += 1

            gesture_text = ""
            if result.gestures:
                gesture_text = result.gestures[0][0].category_name

            if prev_time != 0:
                fps = cv2.getTickFrequency() / (curr_time - prev_time)
                cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Gesture: {gesture_text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            prev_time = curr_time

            cv2.imshow('Sign Language Translator', frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break

    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f'Erreur: {e}')
    print(f'Erreur: {e}')
