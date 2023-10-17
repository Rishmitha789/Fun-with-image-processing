import cv2
import numpy as np

img1_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\TASK1\\tom.jpeg"
img2_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\TASK1\\robert.jpeg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

height = min(img1.shape[0], img2.shape[0])
width = min(img1.shape[1], img2.shape[1])
img1 = cv2.resize(img1, (width, height))
img2 = cv2.resize(img2, (width, height))

img = np.hstack((img1, img2))

gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

gray_img = np.hstack((gray_img1, gray_img2))
new_gray_img = np.dstack([gray_img, gray_img, gray_img])

# h, w = img.shape[:2]
# img = cv2.resize(img, (w, h))
# gray_img = cv2.resize(gray_img, (w, h))

new_img = np.vstack((img, new_gray_img))

cv2.imshow("Output", new_img)

# Save the combined image as "A1_solution.jpg" with the proper file extension
cv2.imwrite('A1_solution.jpg', new_img)

cv2.waitKey()
cv2.destroyAllWindows()
