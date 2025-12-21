import cv2 
import numpy 
from ultralytics import YOLO


## Video Input Setup 
cap = cv2.VideoCapture("./Public/1.mp4")

while True:
    succes,img = cap.read()
    cv2.imshow("Output",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break