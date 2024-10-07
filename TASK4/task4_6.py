# importing required libraries
import cv2
import numpy as np

# Global variables shared between the mouseClick function and rest of the code
draw = False
p1 = (0,0) # top left cornor point
p2 = p1 # bottom right cornor point

new_img_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\TASK4\\task4.output.png"

# Mouse callback function
def mouseClick(event,xPos,yPos,flags,param):
 # print(event,xPos,yPos,flags,param)

 # Global variables shared between the mouseClick function and rest of the code
 global draw,p1,p2 

 # if left click press event, start drawing with p1 as top left cornor point coordinates
 if event == cv2.EVENT_LBUTTONDOWN:
    draw = True
    p1 = (xPos,yPos)
    p2 = p1

 # Continuously update bottom right cornor point (p2) of rectangle on mouse move event
 if event == cv2.EVENT_MOUSEMOVE and draw:
    p2 = (xPos,yPos)

 # if left click release, stop drawing
 if event == cv2.EVENT_LBUTTONUP:
    draw = False
    cropped_area = frame[min(p1[1], p2[1]):max(p1[1], p2[1]), min(p1[0], p2[0]):max(p1[0], p2[0])]
    cv2.imshow('Cropped Area', cropped_area)

    cv2.imwrite(new_img_path, cropped_area)

    

img_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\tom.jpeg"
frame = cv2.imread(img_path)

# Creating an window to display image/frame
cv2.namedWindow('FRAME') 

# This function detects every new events and triggers the "mouseClick" function
cv2.setMouseCallback('FRAME',mouseClick)

while True:
    frame = cv2.imread(img_path)
    cv2.rectangle(frame,p1,p2,(0,255,0),2)
    cv2.imshow('FRAME',frame)

    if cv2.waitKey(1) & 0xff == ord('q'): # to quit press 'q'
        break
cv2.destroyAllWindows()