import cv2
import face_recognition
import os
import datetime
import csv

# Directory containing template images
TEMPLATE_DIR = "templates"  # Create this directory and put your template images there
ATTENDANCE_FILE = "attendance.csv"

# Load template images and encodings
known_face_encodings = []
known_face_names = []

for filename in os.listdir(TEMPLATE_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Add more extensions if needed
        image_path = os.path.join(TEMPLATE_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]  # Get the encoding
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use filename without extension as name

# Initialize attendance dictionary
attendance = {name: {"status": "Absent", "date": "-", "time": "-"} for name in known_face_names}

# Create or open the CSV file
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Attendance", "Date", "Time"])  # Header row
else:
    with open(ATTENDANCE_FILE, "r") as csvfile:  # Read existing data
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Name"] in attendance:
                attendance[row["Name"]] = row

# Video capture
video_capture = cv2.VideoCapture(0)  # 0 for webcam, or specify video file path

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB (face_recognition needs RGB)
    rgb_frame = frame[:, :, ::-1]

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Update attendance if not already present
            if attendance[name]["status"] == "Absent":
                now = datetime.datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")

                attendance[name]["status"] = "Present"
                attendance[name]["date"] = date_str
                attendance[name]["time"] = time_str

                # Write updated attendance to CSV
                with open(ATTENDANCE_FILE, "w", newline="") as csvfile:
                    fieldnames = ["Name", "Attendance", "Date", "Time"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for person, data in attendance.items():
                        writer.writerow({"Name": person, "Attendance": data["status"], "Date": data["date"], "Time": data["time"]})

        # Draw rectangle and name on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

video_capture.release()
cv2.destroyAllWindows()