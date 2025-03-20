import cv2

#video capture object where 0 is the camera number for a usb camera (or webcam)
# if 0 doesn't work, you might need to change the camera number to get the right camera you want to access
cam = cv2.VideoCapture(0)

# Changing video capture property 
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cam.set(cv2.CAP_PROP_FPS,15)


# Getting camera feed width and height after modifying them
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cam.get(cv2.CAP_PROP_FPS)
codec = cam.get(cv2.CAP_PROP_FOURCC )

while True:
    i, frame = cam.read() # reading one frame from the camera object
    cv2.imshow('Webcam', frame) # display the current frame in a window named 'Webcam'
    print('resolution:',width, '|  x', height, '| frames per second:', fps)
    print(codec)
    # Waits for 1ms and check for the pressed key
    if cv2.waitKey(1) & 0xff == ord('q'): # press q to quit the camera (get out of the loop)
        break
cam.release() # close the camera
cv2.destroyAllWindows() # Close all the active windows
