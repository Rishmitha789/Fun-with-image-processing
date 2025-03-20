import cv2

path = "../ball.png"

img = cv2.imread(path)
size = img.shape
w = int(size[0]/2)
h = int(size[1]/2)

img_resize = cv2.resize(img, (h, w))

cv2.imshow("Actual", img)
cv2.imshow("Resized", img_resize)

cv2.waitKey()
cv2.destroyAllWindows()