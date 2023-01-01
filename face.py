import cv2

face_casecade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

while True:
    vid, frame = video.read()

    if vid == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_casecade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255),2)
        cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()