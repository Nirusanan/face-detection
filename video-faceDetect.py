import cv2

trained_Dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video  = cv2.VideoCapture('videos/test.mp4')
while True:
    success, frame = video.read()
    if success == True:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = trained_Dataset.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 0, 0), 2)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
    else:
        print("Video completed or Frame null")
        break