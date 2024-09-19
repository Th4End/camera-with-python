import cv2
import mediapipe as mp 

cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_hands_draw = mp.solutions.drawing_utils

try:
    with mp_hands.Hands(
        max_num_hands = 4,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.4
    ) as hands:
        width, height = 640,480

        while cam.isOpened():
            ret, frame = cam.read()
            
            frame = cv2.resize(frame, (width, height))
            rgb_frame = cv2.cvtColor(frame, (cv2.COLOR_RGB2BGR))
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks: 
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_hands_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
            cv2.imshow('camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break
    cam.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f'Erreur : {e}')
