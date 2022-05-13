from traceback import print_tb
from unittest import result
import cv2
import mediapipe as mp

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetector = mpFaceDetection.FaceDetection()

pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
while True:
    res, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetector.process(frameRgb)
    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)
    cv2.imshow("Frame", frameRgb)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break