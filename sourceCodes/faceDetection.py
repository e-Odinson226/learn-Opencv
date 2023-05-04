import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetector = mpFaceDetection.FaceDetection()
pTime = 0

while True:
    res, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetector.process(frameRgb)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            y, x, c = frameRgb.shape
            bbox = int(bboxC.xmin*x), int(bboxC.ymin*y), int(bboxC.width*x), int(bboxC.height*y)
            
            #----Tarsim moraba baray chehre ha.
            cv2.rectangle(frame, bbox, (255, 0, 255),
                          2, cv2.FONT_HERSHEY_COMPLEX)
            
            #----darj darsad movafaghiat dar tashkhis chehre ha.
            cv2.putText(frame, str(f' { int(detection.score[0]*100) }%'),
                        (bbox[0]-20, bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break