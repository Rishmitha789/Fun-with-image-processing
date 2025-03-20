import cv2 
 
path =  "../ball.png" 


image = cv2.imread(path)

median = image.copy() 
gaussian = image.copy() 

median = cv2.medianBlur(median, 7) 
gaussian = cv2.GaussianBlur(gaussian, (7, 7), cv2.BORDER_DEFAULT) 

cv2.imshow( "Original" ,image) 
cv2.imshow( "Median Blur", median) 
cv2.imshow( "Gaussian Blur" ,gaussian)  
 
cv2.waitKey() 
 
cv2.destroyAllWindows() 