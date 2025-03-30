import cv2
import numpy as np

hueLow_warped = 20  # Initial guess, adjust as needed
hueHigh_warped = 50 # Initial guess, adjust as needed
satLow_warped = 70  # Initial guess, adjust as needed
satHigh_warped = 255 # Initial guess, adjust as needed
valLow_warped = 100  # Initial guess, adjust as needed
valHigh_warped = 255 # Initial guess, adjust as needed

def onTrackWarped1(val):
    global hueLow_warped
    hueLow_warped = val

def onTrackWarped2(val):
    global hueHigh_warped
    hueHigh_warped = val

def onTrackWarped3(val):
    global satLow_warped
    satLow_warped = val

def onTrackWarped4(val):
    global satHigh_warped
    satHigh_warped = val

def onTrackWarped5(val):
    global valLow_warped
    valLow_warped = val

def onTrackWarped6(val):
    global valHigh_warped
    valHigh_warped = val

cv2.namedWindow('Trackbars - Warped')
cv2.resizeWindow('Trackbars - Warped', 400, 300)
cv2.createTrackbar('Hue Low', 'Trackbars - Warped', hueLow_warped, 179, onTrackWarped1)
cv2.createTrackbar('Hue High', 'Trackbars - Warped', hueHigh_warped, 179, onTrackWarped2)
cv2.createTrackbar('Sat Low', 'Trackbars - Warped', satLow_warped, 255, onTrackWarped3)
cv2.createTrackbar('Sat High', 'Trackbars - Warped', satHigh_warped, 255, onTrackWarped4)
cv2.createTrackbar('Val Low', 'Trackbars - Warped', valLow_warped, 255, onTrackWarped5)
cv2.createTrackbar('Val High', 'Trackbars - Warped', valHigh_warped, 255, onTrackWarped6)

input_path = "./input.mov"
cam = cv2.VideoCapture(input_path)

if not cam.isOpened():
    print("Error: Could not open video file. Check the file path and format.")
    exit()

pts = np.array([[10, 243], [194, 109], [329, 11], [482, 71], [687, 147], [577, 307], [415, 555], [178, 370]], dtype=np.int32)
pts_src = np.float32(pts[:8:2])
pts_dst = np.float32([[0, 0], [200, 0], [200, 400], [0, 400]])
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

while True:
    ret, image = cam.read()

    if not ret:
        # If the video ends, reset the video capture to the beginning
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, image = cam.read()
        if not ret:
            print("Error: Could not read frame after resetting video.")
            break

    frame_resized = cv2.resize(image, (700, 600))
    warped = cv2.warpPerspective(frame_resized, matrix, (200, 400))
    warpedHSV = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lowerBound_warped_arr = np.array([hueLow_warped, satLow_warped, valLow_warped])
    upperBound_warped_arr = np.array([hueHigh_warped, satHigh_warped, valHigh_warped])
    mask_warped = cv2.inRange(warpedHSV, lowerBound_warped_arr, upperBound_warped_arr)
    masked_warped = cv2.bitwise_and(warped, warped, mask=mask_warped)

    cv2.imshow('Mask - Warped', mask_warped)
    cv2.imshow('Warped Frame', warped)
    cv2.imshow('Masked Warped', masked_warped)
    cv2.imshow('Trackbars - Warped', np.zeros((1, 400, 3), np.uint8)) # Dummy window to keep trackbars visible

    print("Warped Lower Bound: ", lowerBound_warped_arr)
    print("Warped Upper Bound: ", upperBound_warped_arr)

    if cv2.waitKey(1) & 0xff == ord('q'):  # to quit the camera press 'q'
        break

cam.release()
cv2.destroyAllWindows()