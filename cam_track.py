import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
) as hands,mp_face_detection.FaceDetection(
    min_detection_confidence=0.7
) as face_detection :
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        face_results = face_detection.process(frame_rgb)

        if hand_results.multi_hand_landmarks :
            for landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
        if face_results.detections :
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,255), 2)
        
        cv2.imshow('track & face detect', frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()
        