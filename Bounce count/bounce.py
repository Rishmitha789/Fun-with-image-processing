import cv2
import csv
import time
import numpy as np

# Function to track a ball based on color bounds
def track_ball(frame, lower_bound, upper_bound, quadrant_boundaries):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the ball's color
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find the contours of the ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Implement your ball tracking logic here and calculate ball_x and ball_y
    # For example:
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M['m00'] != 0:
            ball_x = int(M['m10'] / M['m00'])
            ball_y = int(M['m01'] / M['m00'])
        else:
            ball_x, ball_y = None, None
    else:
        ball_x, ball_y = None, None

    # Check if the ball is inside one of the quadrants
    for i, (x1, x2, y1, y2) in enumerate(quadrant_boundaries):
        if x1 <= ball_x <= x2 and y1 <= ball_y <= y2:
            return ball_x, ball_y, i + 1  # Return the quadrant number

    return None, None, "out_of_bound"  # Return if ball is outside quadrants

if __name__ == "__main__":
    # Load point coordinates from the CSV file
    with open("point_coordinates.csv", "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        points = [(int(row[0]), int(row[1])) for row in csv_reader]

    # Load the lower and upper bounds for the color from mask.py
    lower_bound = np.array([ 35, 176, 255])  # Replace with actual values
    upper_bound = np.array([ 35, 176, 255])  # Replace with actual values

    # Create video capture object and get video properties
    video_path = "C:\\Users\\rishm\\OneDrive\\Desktop\\BUILD\\Fun with image processing\\Bounce count\\Input.mov"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize variables
    bounce_count = 0
    prev_ball_y = 0
    prev_time = None

    with open("point_coordinates.csv", "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    points = [(int(row[0]), int(row[1])) for row in csv_reader]


    # Automatically generate boundaries for four quadrants based on collected points
    if len(points) == 4:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        x4, y4 = points[3]
        x5, y5 = points[4]
        x6, y6 = points[5]
        x7, y7 = points[6]
        x8, y8 = points[7]

        # Automatically define boundaries for the quadrants as a big square with 4 symmetric squares inside
        xmid = (x4 + x5)//2
        ymid = (y4 + y5)//2
        quadrant_boundaries = [
            (x1, x2, mid, x4),      # Top-left quadrant
            (x2, x3, x5, mid),      # Top-right quadrant
            (x4, mid, x7, x6),      # Bottom-left quadrant
            (mid, x5, x8, x7)       # Bottom-right quadrant
        ]
    else:
        print("Error: You should collect coordinates for exactly 4 points to define the quadrants.")

    # Open a CSV file to write bounce information
    with open("bounce_info.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Bounce Number", "Time of Bounce", "Quadrant of Bounce", "Frame Number"])

        # Enter the loop to process each frame of the video
        frame_number = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Implement ball tracking logic here
            ball_x, ball_y, quadrant = track_ball(frame, lower_bound, upper_bound, quadrant_boundaries)  # Pass the color bounds and quadrant boundaries

            # Detect when the ball touches the ground and calculate time elapsed
            if ball_y is not None and ball_y > prev_ball_y:
                if prev_time is not None:
                    current_time = time.time()
                    time_elapsed = current_time - prev_time

                    # Increment the bounce count
                    bounce_count += 1

                    # Print and write the bounce information to the CSV file
                    print(f"{bounce_count}, {time_elapsed:.3f}, {quadrant}, {frame_number}")
                    csv_writer.writerow([bounce_count, time_elapsed, quadrant, frame_number])

                prev_time = time.time()

            prev_ball_y = ball_y
            frame_number += 1

            # Display frame with bounce count
            cv2.putText(frame, f"Bounce Count: {bounce_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Bounce Detection", frame)

            # Press 'q' to exit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
