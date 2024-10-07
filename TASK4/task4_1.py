# importing required libraries
import cv2
import numpy as np

# Mouse callback function
def mouseClick(event,xPos,yPos,flags,param):
 print(event,xPos,yPos,flags,param)

# Creating a black image/frame (0 pixel value) of 500x500 size
frame = np.zeros((500,500), np.uint8)

# Creating an window to display image/frame
cv2.namedWindow('FRAME') 

# This function detects every new events and triggers the "mouseClick" function
cv2.setMouseCallback('FRAME',mouseClick)

while True:
 cv2.imshow('FRAME',frame) 
 if cv2.waitKey(1) & 0xff == ord('q'): # to quit press 'q'
    break

cv2.destroyAllWindows()