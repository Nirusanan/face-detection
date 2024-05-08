import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Load pre-trained face detection model (consider using a more robust detector)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained emotion classification model
model = load_model("best_model.h5")


# Define emotions (assuming your model outputs these classes)
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame and store in 'frame'

    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.32, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region (ROI)
        roi = frame[y:y+h, x:x+w]

        # Resize the ROI to match your model's input size
        roi = cv2.resize(roi, (224, 224))

        # Preprocess the image (normalize pixel values)
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)  # Add a new dimension for batch processing

        # Predict emotion using your model
        prediction = model.predict(roi)[0]  # Assuming the model outputs a single prediction vector

        # Get the index of the emotion with the highest probability
        max_index = np.argmax(prediction)

        # Display the predicted emotion above the face
        cv2.putText(frame, emotions[max_index], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with detected faces and emotions
    cv2.imshow('Facial Emotion Analysis', cv2.resize(frame, (1000, 700)))

    if cv2.waitKey(10) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
