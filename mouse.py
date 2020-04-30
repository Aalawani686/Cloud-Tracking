import cv2
import numpy as np
from math import sqrt

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image,((int)(x),(int)(y)),20,(255,0,0),-1)

image = np.zeros((512,512,3), np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image",draw_circle)

while True:
    cv2.imshow("image",image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
