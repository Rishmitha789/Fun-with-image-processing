import cv2
import numpy as np

p1 = (0, 0)
pts = []

def MouseClick(event, xpos, ypos, flags, params):
    global dp1

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (xpos, ypos)
        p1 = [p1[0], p1[1]]
        print(p1)
        pts.append(p1)

path = "../card.jpg"
frame = cv2.imread(path)

cv2.namedWindow('FRAME')

cv2.setMouseCallback('FRAME', MouseClick)

while True:
    cv2.imshow('FRAME', frame)

    if len(pts) >= 4:
        pts_src = np.float32(pts[:4])
        pts_dst = np.float32([[ 0 , 0 ] , [ 200 , 0 ] , [ 200 , 400 ] , [ 0 , 400 ]])
        matrix =  cv2.getPerspectiveTransform(pts_src,pts_dst)
        warped =  cv2.warpPerspective(frame,matrix,( 200 , 400 ))   

        cv2.imshow('WARPED', warped)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()