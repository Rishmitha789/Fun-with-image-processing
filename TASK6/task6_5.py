import cv2

path = "../ball.png"

img = cv2.imread(path)


edge = cv2.Canny(img, 200, 300)

contours, heirarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contoured = img.copy()
cv2.drawContours(contoured, contours, 0, (0, 255, 0), 5)

cv2.imshow("Actual", img)
cv2.imshow("Rotated", contoured)

cv2.waitKey()
cv2.destroyAllWindows()