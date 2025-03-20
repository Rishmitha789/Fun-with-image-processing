import cv2

path = "../ball.png"

img = cv2.imread(path)
size = img.shape
w = size[0]
h = size[1]

center = (int(h/2), int(w/2))
M = cv2.getRotationMatrix2D(center, 45, 1) 
img_rotate = cv2.warpAffine(img, M, (h, w)) 

cv2.imshow("Actual", img)
cv2.imshow("Rotated", img_rotate)

cv2.waitKey()
cv2.destroyAllWindows()