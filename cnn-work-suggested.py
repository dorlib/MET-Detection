import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K  # Import Keras backend for custom loss functions
from tensorflow.keras.losses import binary_crossentropy  # Import binary cross-entropy loss
from tensorflow.keras.saving import register_keras_serializable

# Function to load and normalize NIfTI images
def load_nifti(file_path):
    """Load a NIfTI file and normalize its intensity."""
    nii = nib.load(file_path)
    image = nii.get_fdata()
    image = np.clip(image, 0, 255)  # Normalize intensity values to [0, 255]
    return image

def preprocess_nifti(image, mask=None, target_size=(128, 128, 64)):
    """Resize 3D NIfTI images and masks to a target size."""
    def resize_volume(volume, target_size):
        # Rescale each slice in the z-axis
        depth = target_size[2]
        resized_slices = []
        step = max(1, volume.shape[2] // depth)  # Decide the step size to get desired slices
        for i in range(0, volume.shape[2], step):
            if len(resized_slices) < depth:
                slice_resized = cv2.resize(volume[:, :, i], target_size[:2])  # Resize x, y dimensions
                resized_slices.append(slice_resized)
        resized_volume = np.stack(resized_slices, axis=-1)  # Stack slices in z-axis
        return resized_volume

    # Process image and add channel dimension
    image_resized = resize_volume(image, target_size)
    image_resized = np.expand_dims(image_resized, axis=-1)

    if mask is not None:
        mask_resized = resize_volume(mask, target_size)
        mask_resized = np.expand_dims(mask_resized, axis=-1)
        return image_resized, mask_resized

    return image_resized

# Function to build an improved 3D U-Net model
def build_3d_unet(input_shape=(128, 128, 64, 2)):  # Using two-channel input
    def conv_block(inputs, filters, dropout_rate=0.3):
        x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)  # Added dropout for regularization
        return x

    inputs = Input(shape=input_shape)
    conv1 = conv_block(inputs, 32, dropout_rate=0.3)  # Reduced filters, added dropout
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv_block(pool1, 64, dropout_rate=0.3)  # Reduced filters, added dropout
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv_block(pool2, 128, dropout_rate=0.3)  # Reduced filters, added dropout
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv_block(pool3, 256, dropout_rate=0.4)  # Reduced depth, added higher dropout
    up5 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    merge5 = concatenate([up5, conv3], axis=4)

    conv5 = conv_block(merge5, 128, dropout_rate=0.3)
    up6 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv2], axis=4)

    conv6 = conv_block(merge6, 64, dropout_rate=0.3)
    up7 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv1], axis=4)

    conv7 = conv_block(merge7, 32, dropout_rate=0.3)
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
    return Model(inputs, outputs)

# Custom IoU coefficient and IoU loss functions
@register_keras_serializable()
def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
def iou_loss(y_true, y_pred):
    return 1 - iou_coefficient(y_true, y_pred)

# Updated Combined Loss Function (IoU + BCE)
@register_keras_serializable()
def combined_iou_bce_loss(y_true, y_pred):
    iou = 1 - iou_coefficient(y_true, y_pred)
    return iou + binary_crossentropy(y_true, y_pred)

# Load NIfTI data
def load_nifti_data(root_dir, target_size=(128, 128, 64), with_masks=True):
    """Load NIfTI images (t1c and FLAIR) and segmentation masks (ET label) from the dataset."""
    images = []
    masks = [] if with_masks else None

    for case_dir in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_dir)
        if not os.path.isdir(case_path):
            continue
        t1c_path = os.path.join(case_path, f"{case_dir}-t1c.nii.gz")
        flair_path = os.path.join(case_path, f"{case_dir}-t2f.nii.gz")
        seg_path = f"{case_path}/masks/mask_value_3.0.nii" if with_masks else None

        if os.path.exists(t1c_path) and os.path.exists(flair_path):
            t1c_image = load_nifti(t1c_path)
            flair_image = load_nifti(flair_path)
            combined_image = np.concatenate(
                [preprocess_nifti(t1c_image, target_size=target_size),
                 preprocess_nifti(flair_image, target_size=target_size)], axis=-1)

            if with_masks and os.path.exists(seg_path):
                mask = load_nifti(seg_path)
                _, mask_resized = preprocess_nifti(t1c_image, mask, target_size=target_size)
                images.append(combined_image)
                masks.append(mask_resized)
            elif not with_masks:
                images.append(combined_image)
            else:
                if with_masks:
                    print(f"Mask missing for {case_dir}, skipping this case.")
        else:
            print(f"Skipping case {case_dir} due to missing t1c or flair file.")

    if with_masks:
        return np.array(images), np.array(masks)
    return np.array(images)

# Visualization function
def visualize_segmentation(image, predicted_mask, ground_truth_mask=None):
    """Visualize a slice of the 3D image with the predicted and ground truth masks."""
    thresholded_mask = (predicted_mask > 0.5).astype(np.float32)  # Apply thresholding
    slice_idx = image.shape[2] // 2
    if ground_truth_mask is not None:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image[:, :, slice_idx, 0], cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(thresholded_mask[:, :, slice_idx, 0], cmap='hot', alpha=0.7)
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth Mask")
        plt.imshow(ground_truth_mask[:, :, slice_idx, 0], cmap='hot', alpha=0.7)
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image[:, :, slice_idx, 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(thresholded_mask[:, :, slice_idx, 0], cmap='hot', alpha=0.7)
    plt.show()

# Main section for loading and training
if __name__ == "__main__":
    # Paths
    train_dir = "../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData"
    val_dir = "../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData"
    model_path = "unet_3d_model-t1c-flair-et-iou-2.keras"

    # Parameters
    target_size = (128, 128, 64)
    batch_size = 2
    epochs = 50
    learning_rate = 1e-5  # Reduced learning rate for better convergence

    # Load Training Data
    print("Loading training data...")
    train_images, train_masks = load_nifti_data(train_dir, target_size)

    if len(train_images) == 0 or len(train_masks) == 0:
        print("No valid training data found, exiting.")
        exit()

    # Normalize images
    train_images = train_images / 255.0

    # Binarize masks
    train_masks = (train_masks > 0).astype(np.float32)

    # Split Training Data for Validation
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

    # Load or Train Model
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}...")
        model = load_model(model_path, custom_objects={
            'iou_loss': iou_loss,
            'iou_coefficient': iou_coefficient,
            'combined_iou_bce_loss': combined_iou_bce_loss
        })
    else:
        print("Training a new model...")
        model = build_3d_unet(input_shape=(128, 128, 64, 2))  # Two-channel input (t1c + flair)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=combined_iou_bce_loss, metrics=[iou_coefficient])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Load Validation Data without masks
    print("Loading validation data...")
    val_images = load_nifti_data(val_dir, target_size, with_masks=False)
    val_images = val_images / 255.0

    # Predict and Visualize
    if len(val_images) == 0:
        print("No validation images found. Skipping visualization.")
    else:
        for i in range(len(val_images)):
            val_image = val_images[i:i + 1]  # Single image batch for prediction
            predicted_mask = model.predict(val_image)
            visualize_segmentation(val_image[0], predicted_mask[0])  # No ground truth mask

    print("Process completed.")