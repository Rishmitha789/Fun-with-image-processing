# Import required libraries
import cv2
import mediapipe as mp
import face_recognition as fr


# Load video file
cam = cv2.VideoCapture("../mrBean2.mp4")

# Get frame width and height
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Initialize face detection model
faces = mp.solutions.face_detection.FaceDetection()

# Load template face image
face = fr.load_image_file("../mrBean.png")



# Encode the template face (ensure at least one face is detected)
faceEncoding = fr.face_encodings(face)[0]


while True:
    _ , frame = cam.read()  # Read one frame

    if _:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faceResults = faces.process(frameRGB)
    

        if faceResults.detections != None:
            for f in faceResults.detections:
                # Get bounding box coordinates
                bBox = f.location_data.relative_bounding_box
                x, y, w, h = (
                    int(bBox.xmin * width),
                    int(bBox.ymin * height),
                    int(bBox.width * width),
                    int(bBox.height * height),
                )
                

                f = frameRGB[y:y+h,x:x+w] # getting cropped face 
                
                if (f.shape[0] * f.shape[1]) > 0:  # Ensure valid crop
                    f = cv2.resize(f, (150, 150))  # Resize for better detection
                    encoding_list = fr.face_encodings(f)

                    if encoding_list:  # Ensure at least one encoding exists
                        encoding = encoding_list[0]
                        match = fr.compare_faces([faceEncoding], encoding)

                        if True in match:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, 'Mr. Bean', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Webcam", frame)
        
        

        # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources

cam.release()
cv2.destroyAllWindows()
