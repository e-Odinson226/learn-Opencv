from telnetlib import Telnet
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)
cap.set(10, 100)

while True:
    success, img = cap.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 5)
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = grayImg[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    
    
    cv2.imshow("Img" ,img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break