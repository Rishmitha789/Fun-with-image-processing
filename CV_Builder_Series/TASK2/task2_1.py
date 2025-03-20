# import required libraries here
import cv2
# video capture object where 0 is the camera number for a usb camera (or webcam)
# if 0 doesn't work, you might need to change the camera number to get the right camera you want to access
cam = cv2.VideoCapture(0)
# # for video file, use:
# cam = cv2.VideoCapture('video_file_path')
# # for IP camera, use:
# cam = cv2.VideoCapture('IP_Address')
while True:
 _ , frame = cam.read() # reading one frame from the camera object
 cv2.imshow('Webcam', frame) # display the current frame in a window named 'Webcam'
 # Waits for 1ms and check for the pressed key
 if cv2.waitKey(1) & 0xff == ord('q'): # press q to quit the camera (getout of loop)
    break
cam.release() # close the camera
cv2.destroyAllWindows() # Close all the active windows