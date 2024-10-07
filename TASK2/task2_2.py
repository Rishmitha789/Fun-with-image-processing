# import required libraries here
import cv2

# video capture object where 0 is the camera number for a usb camera (or webcam)
# if 0 doesn't work, you might need to change the camera number to get the right camera you want to access
cam = cv2.VideoCapture(0)

width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cam.get(cv2.CAP_PROP_FPS)
count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
format = cam.get(cv2.CAP_PROP_FORMAT)

while True:
    _ , frame = cam.read() # reading one frame from the camera object
    cv2.imshow('Webcam', frame) # display the current frame in a window named 'Webcam'

    print('resolution:',width, '|  x:', height, '| frames per second:', fps, '|  count: ', count, '|  format: ', format)


    # Waits for 1ms and check for the pressed key
    if cv2.waitKey(1) & 0xff == ord('q'): # press q to quit the camera (getout of loop)
        break
cam.release() # close the camera
cv2.destroyAllWindows() # Close all the active windows