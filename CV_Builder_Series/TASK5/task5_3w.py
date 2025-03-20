import cv2
import numpy as np

path1 = "../green_screen.mp4"  # Path to input video
path2 = "../bg.jpg"  # Path to background image
output_file = "./green.mp4"  # Output video file path

# Initialize video capture and background image
vid = cv2.VideoCapture(path1)
img = cv2.imread(path2)

# Get video properties (width, height, and fps)
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object to save the output video with the same properties as input video
output = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

while True:
    ret, frame = vid.read()
    if not ret:
        break  # Break the loop if video ends

    # Convert the frame to HSV color space to detect the green color
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the green color range in HSV space
    lowerBound = np.array([35, 50, 50])  # Lower bound of green
    upperBound = np.array([85, 255, 255])  # Upper bound of green

    # Create a mask to identify green areas
    mask = cv2.inRange(frameHSV, lowerBound, upperBound)

    # Invert the mask (so green areas become black, everything else becomes white)
    mask_inv = cv2.bitwise_not(mask)

    # Extract the foreground (everything except the green screen)
    foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Resize the background image to match the frame size
    img_resize = cv2.resize(img, (frame.shape[1], frame.shape[0]))

    # Extract the background using the green screen mask
    img_only = cv2.bitwise_and(img_resize, img_resize, mask=mask)

    # Combine the foreground and the background
    combined = cv2.add(foreground, img_only)

    # Show the result
    cv2.imshow("Green Screen Removal", combined)

    # Write the frame to the output video file
    output.write(combined)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
output.release()
cv2.destroyAllWindows()
