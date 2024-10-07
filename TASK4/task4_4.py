# importing required libraries
import cv2
import numpy as np

# Variables for trackbar value
val = 0

# callback function for the slider
def trackbar_1(value):
 global val # changes in red trackbar position updates "value" which is
 val = value
 print('Value: ',val)

# Creating an window for trackbar
cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars',500,50)

# Creating a trackbar with the above callback function
cv2.createTrackbar('Value','Trackbars',val,100,trackbar_1)

# press any key to quit the program
cv2.waitKey()
cv2.destroyAllWindows()