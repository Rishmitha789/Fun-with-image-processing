import cv2
import numpy as np

img_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\robert.jpeg"

image = cv2.imread(img_path)

print(image.shape)
print('1 pixel', image[5, 5])

new_image = image[:10,:10]
# B,G,R = cv2.split(new_image)
B = new_image[:,:,0]
G = new_image[:,:,1]
R = new_image[:,:,2]
print('Blue Channel')
print(B)
print('Green Channel')
print(G)
print('Red Channel')
print(R)

image_merged = cv2.merge((B, G, R))
print(image_merged)

# cv2.imshow('Image', image)
# cv2.imshow('Part', new_image)

#image[:,:] = (0,0,0)
# image[:,:,1] = 0
image[100:200,100:200] = (0, 0, 0)


cv2.imshow('Output', image)


cv2.waitKey()
cv2.destroyAllWindows()