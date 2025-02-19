import cv2

path = "../ball.png"

img = cv2.imread(path)


img_rotate = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow("Actual", img)
cv2.imshow("Rotated", img_rotate)

cv2.waitKey()
cv2.destroyAllWindows()