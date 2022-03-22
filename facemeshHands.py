import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.flip(frame,1)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)                
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Face
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(0,260,255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0,223,255), thickness=1, circle_radius=1))
                            
        
        #Primera mano
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(52,144,0), thickness=2, circle_radius=2))
        

        #Segunda mano
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
             
        cv2.imshow(' Hands-Facemesh ', image)
        if cv2.waitKey(10) & 0xFF == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()