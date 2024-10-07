# import required libraries here
import cv2
# video capture object where 0 is the camera number for a usb camera (or webcam)
# if 0 doesn't work, you might need to change the camera number to get the right camera you want to access
cam = cv2.VideoCapture(0)

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # convert to integer
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)
output_file = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\TASK2\\recording.MP4" # file location + name
output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

while True:

    _ , frame = cam.read() # reading one frame from the camera object
    cv2.imshow('Webcam', frame) # display the current frame in a window named 'Webcam'
    output.write(frame)
    # Waits for 1ms and check for the pressed key
    if cv2.waitKey(1) & 0xff == ord('q'): # press q to quit the camera (get out of the loop)
        break

cam.release() # close the camera
output.release() # close video writer
cv2.destroyAllWindows() # Close all the active windows