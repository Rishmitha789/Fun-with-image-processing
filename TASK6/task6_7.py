import cv2
import numpy as np

p1 = (0, 0)

def MouseClick(event, xpos, ypos, flags, params):
    global dp1

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (xpos, ypos)
        print(p1)

path = "../card.jpg"
frame = cv2.imread(path)

cv2.namedWindow('FRAME')

cv2.setMouseCallback('FRAME', MouseClick)

while True:
    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()