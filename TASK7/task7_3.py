# Import required libraries
import cv2
import mediapipe as mp
import face_recognition as fr
import os
import datetime
import csv

ATTENDANCE_FILE = "./attendance.csv"

# Load video file
cam = cv2.VideoCapture(0)

# Get frame width and height
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # Cast to int
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Cast to int

faces = mp.solutions.face_detection.FaceDetection()

image_directory = "../images/"

faceEncodings = {}
known_face_names = []  # Keep track of known names

# Loop through all files in the image directory
for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
        filepath = os.path.join(image_directory, filename)
        try:
            # Load the image
            image = fr.load_image_file(filepath)

            # Get face encodings for ALL faces in the image
            current_face_encodings = fr.face_encodings(image)

            # Check if any faces were detected
            if current_face_encodings:
                for face_encoding in current_face_encodings:
                    # Extract the name (or label) from the filename
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name) # Add name to the list

                    # Store the encoding with the name
                    if name not in faceEncodings:
                        faceEncodings[name] = []  # Create a list if it's the first encoding for the person
                    faceEncodings[name].append(face_encoding)

            else:
                print(f"No faces found in {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Initialize attendance dictionary (using known_face_names)
attendance = {name: {"status": "Absent", "date": "-", "time": "-"} for name in known_face_names}


# Create or open the CSV file
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Attendance", "Date", "Time"])  # Header row
else:
    try:  # Handle potential errors during file reading
        with open(ATTENDANCE_FILE, "r") as csvfile:  # Read existing data
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row.get("Name") #Handle missing name column
                if name in attendance:
                    attendance[name] = row
    except Exception as e:
        print(f"Error reading attendance file: {e}")


while True:
    _, frame = cam.read()  # Read one frame

    if _:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faceResults = faces.process(frameRGB)

        if faceResults.detections:  # Check if faces were detected
            for f in faceResults.detections:
                # Get bounding box coordinates
                bBox = f.location_data.relative_bounding_box
                x, y, w, h = (
                    int(bBox.xmin * width),
                    int(bBox.ymin * height),
                    int(bBox.width * width),
                    int(bBox.height * height),
                )

                cropped_face = frameRGB[y:y + h, x:x + w]  # getting cropped face

                if (cropped_face.shape[0] * cropped_face.shape[1]) > 0:  # Ensure valid crop
                    resized_face = cv2.resize(cropped_face, (150, 150))  # Resize for better detection
                    encoding_list = fr.face_encodings(resized_face)

                    if encoding_list:  # Ensure at least one encoding exists
                        encoding = encoding_list[0]

                        for name, known_encodings in faceEncodings.items():  # Iterate through known faces
                            for known_encoding in known_encodings:  # Iterate through encodings for each known face
                                match = fr.compare_faces([known_encoding], encoding)  # Corrected compare_faces parameters

                                if any(match):  # Check if any match is found
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                                    # Attendance marking logic (improved)
                                    now = datetime.datetime.now()
                                    date_str = now.strftime("%Y-%m-%d")
                                    time_str = now.strftime("%H:%M:%S")

                                    if attendance[name]["status"] == "Absent" or attendance[name]["date"] != date_str: #Check if already marked present today.
                                        attendance[name]["status"] = "Present"
                                        attendance[name]["date"] = date_str
                                        attendance[name]["time"] = time_str

                                        # Write updated attendance to CSV (append mode)
                                        try:
                                            with open(ATTENDANCE_FILE, "a", newline="") as csvfile: #Append mode
                                                writer = csv.writer(csvfile)
                                                writer.writerow([name, "Present", date_str, time_str])
                                        except Exception as e:
                                            print(f"Error writing to attendance file: {e}")

                                    break  # Inner loop
                            else:
                                continue  # Continue if the nested loop didn't break
                            break  # Outer loop

    cv2.imshow("Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()