# import cv2

# # Create a mouse callback function to capture click events
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Coordinates: ({x}, {y})")

# # Create a blank image (you can load an image if needed)
# image = cv2.imread("C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\Bounce count\\screen_shot.jpg")  # Replace with your image path

# # Display the image
# cv2.imshow("Image", image)

# # Set a mouse callback function for the image window
# cv2.setMouseCallback("Image", click_event)

# # Wait for the user to click on the image and press any key to exit
# cv2.waitKey(0)

# # Close all OpenCV windows
# cv2.destroyAllWindows()
import cv2
import csv

# Function to collect point coordinates by mouse click
def collect_coordinates(event, x, y, flags, param):
    global points, grid_image

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(grid_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Grid Image", grid_image)

# Create a blank grid image
grid_image = cv2.imread("C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\Bounce count\\screen_shot.jpg")  # Replace with your screenshot file
points = []

# Create a window for collecting coordinates
cv2.imshow("Grid Image", grid_image)
cv2.setMouseCallback("Grid Image", collect_coordinates)

# Wait for user to collect points and save to CSV
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Print the number of points collected for debugging
print(f"Number of points collected: {len(points)}")

# Save collected coordinates to a CSV file
with open("point_coordinates.csv", "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["X", "Y"])
    csv_writer.writerows(points)

# Close OpenCV window
cv2.destroyAllWindows()
