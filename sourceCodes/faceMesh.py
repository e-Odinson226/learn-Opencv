import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

mpDrawingUtils = mp.solutions.drawing_utils
mpDrawingSpec = mpDrawingUtils.DrawingSpec((0, 0, 230) ,thickness = 1, circle_radius = 1)

mpDrawingStyles = mp.solutions.drawing_styles

pTime = 0
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(frameRgb)
    
    if result.multi_face_landmarks:
        for faceLandmark in result.multi_face_landmarks:
            
            mpDrawingUtils.draw_landmarks(
                frame,
                faceLandmark,
                connections = mpFaceMesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = mpDrawingSpec,
                connection_drawing_spec = mpDrawingSpec
                )
            
            mpDrawingUtils.draw_landmarks(
                frame,
                faceLandmark,
                connections = mpFaceMesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec = mpDrawingSpec,
                connection_drawing_spec = mpDrawingSpec
                )
            
            #mpDrawingUtils.draw_landmarks(
            #    frame,
            #    faceLandmark,
            #    connections = mpFaceMesh.FACEMESH_IRISES,
            #    #landmark_drawing_spec = mpDrawingSpec,
            #    connection_drawing_spec = 
            #    mpDrawingStyles.get_default_face_mesh_iris_connections_style()
            #    )
    
    #else:
    #    pass
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(frame, str(f'fps:{int(fps)}'),(5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (50, 100, 50), 2)
    cv2.imshow("Legion Eye", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break