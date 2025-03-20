import cv2

img_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\TASK2\\square.jpg"
img = cv2.imread(img_path)
h, w, d = img.shape
square_size = 184

img[:, :] = (255, 255, 255)

for i in range(0, h, square_size):
    for j in range(0, w, square_size):
        if (i // square_size + j // square_size) % 2 == 0:
            img[i:i + square_size, j:j + square_size] = (0, 0, 0)  # Set to black

fps = 2
width = img.shape[1]
height = img.shape[0]
output_file = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\CV_Builder_Series\\TASK2\\checkerboard.mp4"

output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

for i in range(2 * fps):
    output.write(img)

    img = cv2.bitwise_not(img)

output.release()

cap = cv2.VideoCapture(output_file)

while True:
    _, frame = cap.read()
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1000 // fps) & 0xFF == ord('q'): # Press 'q' to quit the camera (get out of the loop)
        break

cap.release()
cv2.destroyAllWindows()
