from re import T
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

while True:
    res, img = cap.read()
    cv2.imshow('Image', img)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break