import cv2
import numpy as np
import csv

path = "./input.mov"

vid = cv2.VideoCapture(path)

fps = int(vid.get(cv2.CAP_PROP_FPS))

bounce_count = 0
bounces = [] 
previous_y = None
last_bounce_region = "N/A"
last_bounce_time = 0.0


pts = np.array([[10, 243], [194, 109], [329, 11], [482, 71], [687, 147], [577, 307], [415, 555], [178, 370]], dtype=np.int32)

pts_src = np.float32(pts[:8:2])
pts_dst = np.float32([[0,0],[200,0],[200,400],[0,400]])
matrix = cv2.getPerspectiveTransform(pts_src,pts_dst)


# lower and upper bound for color from last program
lowerBound = np.array([30,75,132])#lower and upper boundary for color range in HSV
upperBound = np.array([45,255,255])


#Function for ball detection
def ballDetection(frame, frameHSV):
    mask = cv2.inRange(frameHSV, lowerBound, upperBound)#Creating Mask using the color range
    ballContours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # contours around mask
    # If multiple set of sontours available, iterate through each
    for ballContour in ballContours:
        area = cv2.contourArea(ballContour)
        if area > 500: # to filter out noise. This avoids very small contours that could be noise
            x,y,w,h = cv2.boundingRect(ballContour) # this function returns position and size of bounding box for tracking
            # print(x,y,w,h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),3) # draw rectangle arounf blue box
            return (x + w // 2, y + h)

 
# Function to draw grid lines on the frame
def drawGrid(frame, points):
    cv2.line(frame, tuple(points[0]), tuple(points[2]), (0, 255, 0), 2)  
    cv2.line(frame, tuple(points[2]), tuple(points[4]), (0, 255, 0), 2)  
    cv2.line(frame, tuple(points[4]), tuple(points[6]), (0, 255, 0), 2)  
    cv2.line(frame, tuple(points[6]), tuple(points[0]), (0, 255, 0), 2)  
    cv2.line(frame, tuple(points[1]), tuple(points[5]), (0, 255, 0), 2)  
    cv2.line(frame, tuple(points[3]), tuple(points[7]), (0, 255, 0), 2)  
    
#Function to get region
def getRegion(x, y):
    # Get midpoint values to divide into 2x2 regions
    x_mid = (pts[1][0] + pts[6][0]) // 2  # Midpoint of x-coordinates
    y_mid = (pts[2][1] + pts[5][1]) // 2  # Midpoint of y-coordinates

    # Assign region based on x and y position
    if x < x_mid and y < y_mid:
        return "Top Left (1)"
    elif x >= x_mid and y < y_mid:
        return "Top Right (2)"
    elif x < x_mid and y >= y_mid:
        return "Bottom Left (3)"
    elif x >= x_mid and y >= y_mid:
        return "Bottom Right (4)"
    return "Out of Bounds"

while True:
    ret, frame = vid.read()

    frame = cv2.resize(frame,(700, 600))
    resized_height, resized_width, _ = frame.shape

    # Draw the grid on the frame
    drawGrid(frame, pts)

    # converting to HSV for masking
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    warped = cv2.warpPerspective(frame,matrix,(200,400))
    ballDetection(frame, frameHSV)
    ballPosition = ballDetection(warped, frameHSV)

    region = "N/A"  # Default value if ball is not detected
    bounce_time = 0.0  # Default time if no bounce occurs
    
    if ballPosition:
        x, y = ballPosition
        region = getRegion(x, y)

        # Detect bounce
        if previous_y is not None and y > previous_y:
            bounce_count += 1
            bounce_time = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert ms to seconds
            frame_num = fps * bounce_time
            bounces.append([bounce_count, bounce_time, region, frame_num])

            print(f"Bounce {bounce_count}: Time={bounce_time}s, Region={region}, Frame={frame_num}")
            last_bounce_region = region
            last_bounce_time = bounce_time

        previous_y = y  # Update previous y-position

    else:
        # If no ball is detected, retain the last bounce information
        region = last_bounce_region
        bounce_time = last_bounce_time

    # Display frame
    text_y_offset = 20
    cv2.putText(frame, f"Count: {bounce_count}", (10, 30 + text_y_offset * 0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Time: {vid.get(cv2.CAP_PROP_POS_MSEC) / 1000:.3f}" + " s", (520, 30 + text_y_offset * 0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "Bounce detected at " + f"{bounce_time:.2f}s" + " on region " + f"{region}", (70, 550 + text_y_offset * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    cv2.imshow("Video Feed", frame)
    cv2.imshow("Warped", warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # Save results to CSV
# with open("bounce_data.csv", "w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Bounce Number", "Time (s)", "Quadrant", "Frame Number"])
#     writer.writerows(bounces)

# Release resources
vid.release()
cv2.destroyAllWindows()