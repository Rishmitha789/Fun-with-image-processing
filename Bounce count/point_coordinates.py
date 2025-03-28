import cv2
import numpy as np

p1 = (0, 0)
pts = []

def mouseClick(event, xpos, ypos, flags, params):
    global dp1

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (xpos, ypos)
        p1 = [p1[0], p1[1]]
        pts.append(p1)

path = "./frame.png"
frame = cv2.imread(path)

cv2.namedWindow('FRAME')

cv2.setMouseCallback('FRAME', mouseClick)

while True:
    cv2.imshow('FRAME', frame)
    if cv2.waitkey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()