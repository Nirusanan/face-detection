import cv2

# Load the pre-trained cascade classifier
trained_Dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the image
img = cv2.imread('images/student.jpg')

# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load image")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = trained_Dataset.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

    # Check if any faces are detected
    if len(faces) == 0:
        print("No faces detected")
    else:
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (250, 0, 0), 2)

        # Display the image with detected faces
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
