import cv2
import numpy as np

image = np.ones((500, 1000, 3), dtype=np.uint8) * 255

cv2.line(image, (0, 350), (999, 350), (0, 0, 0), 2)
cv2.rectangle(image, (200, 300), (200, 350),(0,0,0),8)
cv2.putText(image, 'G A M E  O V E R', (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
cv2.putText(image, 'HI 00041 00041', (710, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 3)
cv2.circle(image, (185, 290), 10, (0, 0, 0), 5)



cv2.imshow('image', image)

img_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\TASK3\\task3_output.png"
cv2.imwrite(img_path,image)

cv2.waitKey() # Wait untill a key press
cv2.destroyAllWindows() # Close all the active windows

