import cv2
import numpy as np

p1 = (0, 0)
pts = []

path = "./frame.png"
frame = cv2.imread(path)
frame = cv2.resize(frame, (700, 600))

def mouseClick(event, xpos, ypos, flags, params):
    global dp1

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (xpos, ypos)
        cv2.circle(frame, (xpos, ypos), radius = 0, color = (0, 0, 255), thickness = 5)
        p1 = [p1[0], p1[1]]
        pts.append(p1)
        print(f"Point clicked: {p1}")


cv2.namedWindow('FRAME')


cv2.setMouseCallback('FRAME', mouseClick)

while True:
    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
print("Collected points:", pts)