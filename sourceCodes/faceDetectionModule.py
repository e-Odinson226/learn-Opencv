from logging import NullHandler
from unicodedata import name
import cv2
import mediapipe as mp

class FaceDetection():
    def __init__(self, min_detection_confidence=0.5):
        self.detectionConfident = min_detection_confidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetector = self.mpFaceDetection.FaceDetection()
        self.pTime = 0
        

    def findFaces(self, frame, draw=False):
        self.frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetector.process(self.frameRgb)
        if self.results.detections:
            bboxs = []
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                score = detection.score[0]
                y, x, c = self.frameRgb.shape
                bbox = int(bboxC.xmin*x), int(bboxC.ymin*y), int(bboxC.width*x), int(bboxC.height*y)
                bboxs.append([id, bbox, score])
                if draw:
                    #----Tarsim moraba baray chehre ha.
                    cv2.rectangle(frame, bbox, (255, 0, 255),
                                2, cv2.FONT_HERSHEY_COMPLEX)

                    #----darj darsad movafaghiat dar tashkhis chehre ha.
                    cv2.putText(frame, str(f' { int(score*100) }%'),
                                (bbox[0]-20, bbox[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            return frame
        else:
            return None
        


    

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    detector = FaceDetection()
    
    while True:
        res, frame = cap.read()
        tframe = cv2.flip(frame, 1)
        frame = detector.findFaces(tframe, True)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
if __name__ == "__main__":
    main()
    
    
    