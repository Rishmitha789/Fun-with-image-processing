import cv2
import numpy as np
import csv

path = "./input.mov"

vid = cv2.VideoCapture(path)
fps = int(vid.get(cv2.CAP_PROP_FPS))

bounce_count = 0
bounces = [] 
previous_y = None
region = "N/A"  # Default value if ball is not detected
bounce_time = 0.0  # Default time if no bounce occurs

# Points for drawing the grid and for perspective transformation
pts = np.array([[10, 244], [195, 110], [328, 11], [481, 71],
                [685, 147], [574, 308], [415, 555], [180, 369]], dtype=np.int32)

# Use every second point for the source and define destination points
pts_src = np.float32(pts[:8:2])
pts_dst = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Lower and upper bounds for the ball color in HSV
lowerBound = np.array([30, 75, 132])
upperBound = np.array([45, 255, 255])

# Function for ball detection
def ballDetection(frame, frameHSV):
    mask = cv2.inRange(frameHSV, lowerBound, upperBound)
    ballContours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_locations = []
    for ballContour in ballContours:
        area = cv2.contourArea(ballContour)
        if area > 230:
            x, y, w, h = cv2.boundingRect(ballContour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            center_x = x + w // 2
            center_y = y + h // 2
            detected_locations.append(((center_x, center_y), ballContour))
    return detected_locations

# Function to draw grid lines on the frame
def drawGrid(frame, points):
    cv2.line(frame, tuple(points[0]), tuple(points[2]), (0, 255, 0), 2)
    cv2.line(frame, tuple(points[2]), tuple(points[4]), (0, 255, 0), 2)
    cv2.line(frame, tuple(points[4]), tuple(points[6]), (0, 255, 0), 2)
    cv2.line(frame, tuple(points[6]), tuple(points[0]), (0, 255, 0), 2)
    cv2.line(frame, tuple(points[1]), tuple(points[5]), (0, 255, 0), 2)
    cv2.line(frame, tuple(points[3]), tuple(points[7]), (0, 255, 0), 2)

# Function to get region based on warped coordinates
def getRegion(x, y):
    x_mid = 400 // 2  # Midpoint of x in warped image
    y_mid = 400 // 2  # Midpoint of y in warped image
    if x < x_mid and y < y_mid:
        return "Top Left (1)"
    elif x >= x_mid and y < y_mid:
        return "Top Right (2)"
    elif x < x_mid and y >= y_mid:
        return "Bottom Left (3)"
    elif x >= x_mid and y >= y_mid:
        return "Bottom Right (4)"
    return "Out of Bounds"

# Initialize variable to track downward movement
was_moving_down = False

while True:
    ret, frame = vid.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 600))
    drawGrid(frame, pts)

    # Convert to HSV for ball detection
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ballPosition = ballDetection(frame, frameHSV)

    # Generate warped frame using perspective transform
    warped = cv2.warpPerspective(frame, matrix, (400, 400))
    # Note: We now transform coordinates rather than detecting on warped frame

    if ballPosition:
        # Get the y-coordinate from the original detection
        y = ballPosition[0][0][1]

        if previous_y is not None:
            if y < previous_y and was_moving_down:
                # Bounce detected: ball moving upward after moving downward
                # Transform the ball's coordinate from the original frame to the warped space
                ball_center = np.array([[ballPosition[0][0]]], dtype=np.float32)  # shape (1,1,2)
                ball_center_warped = cv2.perspectiveTransform(ball_center, matrix)
                x_warped, y_warped = ball_center_warped[0][0]
                region = getRegion(x_warped, y_warped)

                bounce_count += 1
                bounce_time = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000
                frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))

                bounces.append([bounce_count, bounce_time, region, frame_num])
                print(f"Bounce {bounce_count}: Time={bounce_time:.3f}s, Region={region}, Frame={frame_num}")

                # Reset the movement flag
                was_moving_down = False

            elif y > previous_y:
                was_moving_down = True

        previous_y = y  # Update the previous y-coordinate

    text_y_offset = 20
    cv2.putText(frame, f"Count: {bounce_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Time: {vid.get(cv2.CAP_PROP_POS_MSEC)/1000:.3f} s", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Bounce detected at {bounce_time:.2f}s on region {region}", (70, 550 + text_y_offset * 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Video Feed", frame)
    cv2.imshow("Warped", warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Optionally save results to CSV
# with open("bounce_data.csv", "w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Bounce Number", "Time (s)", "Quadrant", "Frame Number"])
#     writer.writerows(bounces)

vid.release()
cv2.destroyAllWindows()
