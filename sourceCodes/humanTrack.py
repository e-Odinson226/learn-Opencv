import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic()


mpDrawingUtils = mp.solutions.drawing_utils
mpDrawingSpec = mpDrawingUtils.DrawingSpec((0, 250, 230) ,thickness = 1, circle_radius = 1)

mpDrawingStyles = mp.solutions.drawing_styles

pTime = 0
while True:
    success, frame = cap.read()
    frame.flags.writeable = False
    frame = cv2.flip(frame, 1)
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frameRgb)
    frame.flags.writeable = True
    #print(result)
    
    if result.pose_landmarks:
        
        #for poseLandmark in result.pose_landmarks.landmark:
        
        mpDrawingUtils.draw_landmarks(
            frame,
            result.face_landmarks,
            connections = mpHolistic.FACEMESH_CONTOURS,
            landmark_drawing_spec = mpDrawingSpec,
            connection_drawing_spec = mpDrawingSpec
            
            )
        
        mpDrawingUtils.draw_landmarks(
            frame,
            result.pose_landmarks,
            connections = mpHolistic.POSE_CONNECTIONS,
            landmark_drawing_spec = mpDrawingStyles.get_default_pose_landmarks_style()
            )
        
    
    #else:
    #    pass
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(frame, str(f'fps:{int(fps)}'),(5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (50, 100, 50), 2)
    cv2.imshow("Legion Eye", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break