import cv2
import numpy as np

# Function to get the lower and upper bounds for a color in an image
def get_color_bounds(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the region of interest (ROI) for color sampling
    roi = hsv_image[100:200, 100:200]  # Adjust the coordinates as needed

    # Calculate the lower and upper bounds for the color in the ROI
    lower_bound = np.min(roi, axis=(0, 1))
    upper_bound = np.max(roi, axis=(0, 1))

    return lower_bound, upper_bound

if __name__ == "__main__":
    image_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\Bounce count\\colour.png"  # Replace with your sample image path
    lower_bound, upper_bound = get_color_bounds(image_path)

    print(f"Lower Bound (H, S, V): {lower_bound}")
    print(f"Upper Bound (H, S, V): {upper_bound}")
