# Importing OpenCV Library
import cv2
# Relative or absolute path of the input image file
path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\robert.jpeg"

# reading image (by default the flag is 1 if not specidied)
image = cv2.imread(path)
# Display image in a window
cv2.imshow("Output",image)
# Wait until any key press (press any key to close the window)
cv2.waitKey()
# kill all the windows
cv2.destroyAllWindows()