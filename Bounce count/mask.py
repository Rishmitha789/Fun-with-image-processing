import cv2
import numpy as np

def onTrack1(val):
    global hueLow
    hueLow = val

def onTrack2(val):
    global hueHigh
    hueHigh = val

def onTrack3(val):
    global satLow
    satLow = val

def onTrack4(val):
    global satHigh
    satHigh = val

def onTrack5(val):
    global valLow
    valLow = val

def onTrack6(val):
    global valHigh
    valHigh = val

cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 400, 300)



cv2.createTrackbar('Hue Low', 'Trackbars', 25, 40, onTrack1)
cv2.createTrackbar('Hue High', 'Trackbars', 40, 179, onTrack2)
cv2.createTrackbar('Sat Low', 'Trackbars', 80, 151, onTrack3)
cv2.createTrackbar('Sat High', 'Trackbars', 151, 255, onTrack4)
cv2.createTrackbar('Val Low', 'Trackbars', 134, 255, onTrack5)
cv2.createTrackbar('Val High', 'Trackbars', 255, 255, onTrack6)

input_path = "./input.mov"
cam = cv2.VideoCapture(input_path)

if not cam.isOpened():
    print("Error: Could not open video file. Check the file path and format.")
    exit()

while True:
    ret, image = cam.read()

    if not ret:
        # If the video ends, reset the video capture to the beginning
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, image = cam.read()
        if not ret:
            print("Error: Could not read frame after resetting video.")
            break

    frameHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([hueLow, satLow, valLow])  # lower and upper boundary for color range in HSV
    upperBound = np.array([hueHigh, satHigh, valHigh])
    mask = cv2.inRange(frameHSV, lowerBound, upperBound)  # Creating Mask using the color range
    masked = cv2.bitwise_and(image, image, mask=mask)

    mask = cv2.resize(mask, (700, 600))
    image = cv2.resize(image, (700, 600))
    masked = cv2.resize(masked, (700, 600))

    cv2.imshow('mask', mask)
    cv2.imshow('Ball', image)
    cv2.imshow('masked', masked)

    print("lowerBound: ", lowerBound)
    print("upperBound: ", upperBound)

    if cv2.waitKey(1) & 0xff == ord('q'):  # to quit the camera press 'q'
        break

cam.release()
cv2.destroyAllWindows()