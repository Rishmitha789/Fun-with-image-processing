# importing required libraries
import cv2
import numpy as np

# Global variables shared between the mouseClick function and rest of the code
draw = False
reset = False
# initially p1 and p2 = 0
p1 = (0,0) # First point of line segment
p2 = p1 # Second point of line segment

# Mouse callback function
def mouseClick(event,xPos,yPos,flags,param):
 # print(event,xPos,yPos,flags,param)

 # Global variables shared between the mouseClick function and rest of the code
    global draw,reset,p1,p2 

 # if left click press event, enable drawing and p1 and p2 as current position
    if event==cv2.EVENT_LBUTTONDOWN:
        draw = True
        reset = False
        p1 = (xPos,yPos)
        p2 = p1

 # Continuously update p2 on mouse movement and left mouse press
    if event==cv2.EVENT_MOUSEMOVE and draw:
        p2 = (xPos,yPos)

 # if left click release, stop drawing
    if event==cv2.EVENT_LBUTTONUP:
        draw = False

 # if right click, reset the frame/canvas, and p1 and p2 to 0
    if event==cv2.EVENT_RBUTTONDOWN:
        reset = True
        p1 = (0,0)
        p2 = p1

# Creating a black image/frame (0 pixel value) of 500x500 size
frame = np.zeros((500,500,3), np.uint8)

# Creating an window to display image/frame
cv2.namedWindow('FRAME') 
# This function detects every new events and triggers the "mouseClick"function
cv2.setMouseCallback('FRAME',mouseClick)


while True:
 cv2.line(frame,p1,p2,(0,255,0),2)
 p1 = p2 # swapping points for next line segment (p2 copies to p1 and p2 updats to latest position)
 cv2.imshow('FRAME',frame)
 if reset:
    frame = np.zeros((500,500,3), np.uint8) # renew black frame on right click

 if cv2.waitKey(1) & 0xff == ord('q'): # to quit press 'q'
    break
cv2.destroyAllWindows()