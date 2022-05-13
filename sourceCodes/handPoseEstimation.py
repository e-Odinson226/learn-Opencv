import time
import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
hands = mpHands.Hands()

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10, 100)

currentTime = 0
nextFrameTime = 0

while True:
    res, img = cap.read()
    img = cv2.flip(img, 1)
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgbImg)
    
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            mpDrawing.draw_landmarks(img, handLm, mpHands.HAND_CONNECTIONS)
    
    
    currentTime = time.time()
    fps = (1/(currentTime-nextFrameTime))
    nextFrameTime = currentTime
    
    cv2.putText(
        img, str(int(fps)), (15, 80), cv2.FONT_HERSHEY_PLAIN, 5, (0, 250, 0), 10)
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break