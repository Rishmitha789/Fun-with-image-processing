import cv2
import mediapipe as mp

path = "../mrBean.mp4"
cam = cv2.VideoCapture(path)

w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

faces = mp.solutions.face_detection.FaceDetection()

while True:
    _ , frame = cam.read()

    if _:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faceResults = faces.process(frameRGB)

        if faceResults.detections != None:

            for face in faceResults.detections:
                bBox = face.location_data.relative_bounding_box

                x, y, wi, he = int(bBox.xmin*w),int(bBox.ymin*h),int(bBox.width*w),int(bBox.height*h)

                cv2.rectangle(frame, (x, y), (x+wi, y+he), (0, 0, 255), 3)
            cv2.imshow('Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

