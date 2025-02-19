import cv2

# Load image
path = "../ball.png"
img = cv2.imread(path)
original = img.copy()  # Store original image for transformations

new_img_path = "./cropped.png"

# Variables for cropping
draw = False
p1 = (0, 0) 
p2 = p1 

# Variables for trackbar settings
Filter = 0
Zoom = 0
Rotate = 0
Blur = 0
Sketch = 0

# Mouse callback function for cropping
def mouseClick(event, xPos, yPos, flags, param):
    global draw, p1, p2, img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        p1 = (xPos, yPos)
        p2 = p1

    if event == cv2.EVENT_MOUSEMOVE and draw:
        p2 = (xPos, yPos)

    if event == cv2.EVENT_LBUTTONUP:
        draw = False
        cropped_area = img[min(p1[1], p2[1]):max(p1[1], p2[1]), min(p1[0], p2[0]):max(p1[0], p2[0])]

        if cropped_area.size != 0:
            cv2.imshow('Cropped Area', cropped_area)
            cv2.imwrite(new_img_path, cropped_area)

# Trackbar functions
def Track1(val):
    global Filter
    Filter = val

def Track2(val):
    global Zoom
    Zoom = val   # Keep zoom between 1.0x to 6.0x

def Track3(val):
    global Rotate
    Rotate = val  # Rotation angle

def Track4(val):
    global Blur
    Blur = val if val % 2 == 1 else val + 1  # Ensure it's an odd number for medianBlur()

def Track5(val):
    global Sketch
    Sketch = val  # Toggle sketch effect (0 or 1)

# Create trackbars
cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 400, 300)

cv2.createTrackbar('Filter', 'Trackbars', 0, 22, Track1)
cv2.createTrackbar('Zoom', 'Trackbars', 1, 5, Track2)
cv2.createTrackbar('Rotate', 'Trackbars', 0, 360, Track3)
cv2.createTrackbar('Blur', 'Trackbars', 1, 20, Track4)
cv2.createTrackbar('Sketch', 'Trackbars', 50, 500, Track5)

cv2.namedWindow('FRAME')  # Ensure the window exists before setting the callback
cv2.setMouseCallback('FRAME', mouseClick)

# Main loop
while True:
    img = original.copy()  # Reset image every loop to apply fresh transformations

    # Apply filter
    if Filter > 0:
        img = cv2.applyColorMap(img, Filter - 1)

    # Apply zoom
    size = img.shape
    img = cv2.resize(img, (int(size[1] * Zoom), int(size[0] * Zoom)))

    # Apply rotation
    center = (size[1] // 2, size[0] // 2)
    M = cv2.getRotationMatrix2D(center, Rotate, 1)
    img = cv2.warpAffine(img, M, (size[1], size[0]))

    # Apply blur
    if Blur > 1:
        img = cv2.medianBlur(img, Blur)

    # Apply sketch effect
    if Sketch > 0:
        edges = cv2.Canny(img, Sketch, 150)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Draw cropping rectangle
    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

    # Show image
    cv2.imshow('FRAME', img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
