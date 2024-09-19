import cv2
import mediapipe as mp 


cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

try:
    with mp_hands.Hands(
        max_num_hands = 4,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.4
    ) as hands, mp_face_detection.FaceDetection(
        min_detection_confidence = 0.7
    ) as face_detection :
        while cap.isOpened():
            ret, frame = cap.read()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            face_results = face_detection.process(frame_rgb)

            if hand_results.multi_hand_landmarks :
                for landmarks in hand_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            if face_results.detections :
                for detection in face_results.detections:
                    box = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break
            cv2.imshow('cam',frame)

    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f'Erreur: {e}')
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     faces = face_cascade.detectMultiScale(
#         frame
#     )

#     for(x, y, w, h) in faces:
#         cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,255))
    
#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         break


# cap.release()
# cv2.destroyAllWindows()