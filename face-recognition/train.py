import cv2
import os
import numpy as np


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(filename.split('-')[0])  # Assuming filename is label.extension
            print(labels)
        else:
            print(f"Failed to load image: {img_path}")
    if not images:
        print("No images loaded from the folder.")
    return images, labels

# Path to the dataset folder
dataset_folder = 'faceImg'

# Load images and labels from the dataset folder
images, labels = load_images_from_folder(dataset_folder)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create a mapping between unique labels and integer identifiers
label_mapping = {label: idx for idx, label in enumerate(set(labels))}

# Replace string labels with integer identifiers
integer_labels = [label_mapping[label] for label in labels]

# Convert labels to NumPy array
labels_np = np.array(integer_labels, dtype=np.int32)

# Convert images to grayscale
gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

# Train the recognizer
recognizer.train(gray_images, labels_np)


# Save the trained model
recognizer.save('trained_model.yml')

print("Training completed and model saved.")
