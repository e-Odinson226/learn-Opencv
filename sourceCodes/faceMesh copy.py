import cv2
import mediapipe as mp
import time


mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


mpDrawingUtils = mp.solutions.drawing_utils
mpDrawingSpec = mpDrawingUtils.DrawingSpec((0, 250, 230), thickness=1, circle_radius=1)

mpDrawingStyles = mp.solutions.drawing_styles

while True:
    frame = cv2.imread(
        # "/home/erfan/Downloads/drive-download-20230503T191557Z-001/IMG_20230503_223908_507.jpg"
        #'/home/erfan/Downloads/drive-download-20230503T191557Z-001/IMG_20230503_223908_545.jpg'
        "/home/erfan/Downloads/drive-download-20230503T191557Z-001/IMG_20230503_223908_843.jpg"
        # "/home/erfan/Downloads/drive-download-20230503T191557Z-001/IMG_20230503_223908_892.jpg"
        # "/home/erfan/Downloads/drive-download-20230503T191557Z-001/IMG_20230503_223908_973.jpg"
    )
    # frame = cv2.flip(frame, 1)
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(frameRgb)

    if result.multi_face_landmarks:
        for faceLandmark in result.multi_face_landmarks:
            mpDrawingUtils.draw_landmarks(
                frame,
                faceLandmark,
                connections=mpFaceMesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mpDrawingSpec,
                connection_drawing_spec=mpDrawingSpec,
            )

    # else:
    #    pass

    cTime = time.time()

    pTime = cTime

    cv2.imshow("Legion Eye", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
