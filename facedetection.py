import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.flip(frame,1)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)                
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
       
        
        #Detectando Poses
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3) )
            

          
        cv2.imshow('Facedetection y pose', image)
        if cv2.waitKey(10) & 0xFF == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()