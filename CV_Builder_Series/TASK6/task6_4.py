import cv2

path = "../ball.png"

img = cv2.imread(path)


edge = cv2.Canny(img, 200, 300)

cv2.imshow("Actual", img)
cv2.imshow("Rotated", edge)

cv2.waitKey()
cv2.destroyAllWindows()