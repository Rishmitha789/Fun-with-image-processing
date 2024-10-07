import cv2
import numpy as np

image = np.zeros((300, 300, 3), dtype = np.uint8)
line = image.copy()
arrow = image.copy()
polyLine = image.copy()
rectangle = image.copy()
circle = image.copy()
text = image.copy()

# Test points
p1 = [100,100]
p2 = [200,200]
p3 = [200,100]
p4 = [100,200]
points = np.array([p1,p2,p3,p4])

# Drawing functions
cv2.line(line,p1,p2,(0,255,0),2)
cv2.arrowedLine(arrow,p1,p2,(0, 0, 255),3)
cv2.polylines(polyLine,[points],False,(255, 0, 0),4)
cv2.rectangle(rectangle,p1,p2,(0,255,0),2)
cv2.circle(circle,(150,150),50,(0,255,0),2)
cv2.putText(text,'sample_text', p4, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
(0,255,0))

# Display the drawn objects
cv2.imshow('line',line)
cv2.imshow('arrow',arrow)
cv2.imshow('polyLine',polyLine)
cv2.imshow('rectangle',rectangle)
cv2.imshow('circle',circle)
cv2.imshow('text',text)

cv2.waitKey() # Wait untill a key press
cv2.destroyAllWindows() # Close all the active windows
