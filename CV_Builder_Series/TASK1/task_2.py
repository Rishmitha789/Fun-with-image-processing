# Importing OpenCV Library
import cv2
# Relative or absolute path of the input image file
img_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\robert.jpeg"

# reading image (by default the flag is 1 if not specidied)
image = cv2.imread(img_path)

img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img_hue =  cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
img_rgb =  cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


# Display image in a window
cv2.imshow("Output1",image)
cv2.imshow("Output2",img_gray)
cv2.imshow("Output3",img_hue)
cv2.imshow("Output4",img_rgb)
# Wait until any key press (press any key to close the window)
cv2.waitKey()
# kill all the windows
cv2.destroyAllWindows()