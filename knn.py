import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Function to preprocess images: Resize images to a fixed size
def preprocess_image(img_path, target_size=(128, 128)):
    abs_path = os.path.abspath(img_path)  # Get the absolute path
    print(f"Absolute path: {abs_path}")  # Debugging
    image = cv2.imread(abs_path)
    if image is None:
        print(f"Failed to load image: {abs_path}")  # Debugging
        return None
    image_resized = cv2.resize(image, target_size)  # Resize image
    return image_resized


# Function to generate binary mask for segmentation
def generate_mask(image, target_size=(128, 128)):
    # Assuming simple binary segmentation: non-background is 1, background is 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.resize(binary_mask, target_size)
    return binary_mask // 255  # Convert to binary (0, 1)


# Function to load data from directories
def load_data(cats_dir, dogs_dir, target_size=(128, 128)):
    image_paths = []
    labels = []

    # Check if directories exist
    if not os.path.exists(cats_dir):
        print(f"Error: Cats directory {cats_dir} does not exist.")
        return [], []

    if not os.path.exists(dogs_dir):
        print(f"Error: Dogs directory {dogs_dir} does not exist.")
        dogs_images = []  # No dog images to process
    else:
        dogs_images = [img for img in os.listdir(dogs_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Process cat images
    for img_name in os.listdir(cats_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(cats_dir, img_name)
            print(f"Loading image: {img_path}")  # Debugging
            image = preprocess_image(img_path, target_size)
            if image is None:
                continue
            mask = generate_mask(image, target_size)
            image_paths.append(image)  # Store image array
            labels.append(0)  # Label for cat

    # Process dog images (if any)
    for img_name in dogs_images:
        img_path = os.path.join(dogs_dir, img_name)
        print(f"Loading image: {img_path}")  # Debugging
        image = preprocess_image(img_path, target_size)
        if image is None:
            continue
        mask = generate_mask(image, target_size)
        image_paths.append(image)  # Store image array
        labels.append(1)  # Label for dog

    return image_paths, labels


# Function to extract features (flatten images for k-NN)
def extract_features(image_data):
    features = []
    for image in image_data:
        if isinstance(image, np.ndarray):  # Check if image is a valid array
            print(f"Extracting features for image: {image.shape}")  # Debugging
            extracted_features = image.flatten()  # Flatten the image to a 1D array
            features.append(extracted_features)
        else:
            print(f"Invalid image data: {image}")  # Debugging
    if not features:
        print("No valid images found for feature extraction.")
        return np.array([])  # Return an empty array if no features were collected
    return np.vstack(features)  # Stack all feature vectors vertically (to form a 2D array)


# Load images and prepare dataset
cats_dir = "../PetImages/Cat"
dogs_dir = "../PetImages/Dogs"

# Resize images to (128, 128) or any other desired size
image_data, labels = load_data(cats_dir, dogs_dir, target_size=(128, 128))
features = extract_features(image_data)

# Check if there are any features
if len(features) == 0 or len(labels) == 0:
    print("Error: No valid data for training")
else:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


    # Train k-NN model
    def train_knn(X_train, y_train, k=3):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        return knn


    knn_model = train_knn(X_train, y_train, k=3)


    # Evaluate model
    def evaluate_model(knn, X_test, y_test):
        predictions = knn.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy


    accuracy = evaluate_model(knn_model, X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


    # Visualize results for one test image
    def visualize_segmentation(image, predicted_label, target_size=(128, 128)):
        # Generate mask for segmentation
        mask = generate_mask(image, target_size)

        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
        plt.title(f"Original Image")

        # Display mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Predicted Mask\n{'Dog' if predicted_label == 1 else 'Cat'}")
        plt.show()


    test_image_path = '../PetImages/Test/12053.jpg'

    # Assuming 'test_image' is of shape (128, 128, 3)
    test_image = preprocess_image(test_image_path, target_size=(128, 128))

    # Check if the image was loaded correctly
    if test_image is None:
        print(f"Error: Could not load test image from {test_image_path}")
    else:
        # Flatten the test image into a 1D array of features
        test_features = test_image.reshape(1, -1)  # Reshape to (1, 49152) for prediction
        predicted_label = knn_model.predict(test_features)
        print(f"Predicted label: {predicted_label}")

    # Define the label mapping (adjust the labels as per your data)
    label_mapping = {
        0: "Cat",
        1: "Dog"
    }

    predicted_class_name = label_mapping[predicted_label[0]]

    # Print the class name
    print(f"Predicted label: {predicted_label[0]}, Predicted class: {predicted_class_name}")

    # Visualize the segmentation and prediction
    visualize_segmentation(test_image, predicted_label[0])

