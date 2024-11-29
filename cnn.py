import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import MeanIoU
from matplotlib import cm


# Dice Loss
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1) / (denominator + 1)


# IoU Loss
def iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true + y_pred)
    union = total - intersection
    return 1 - (intersection + 1) / (union + 1)


# 3D U-Net Model
def build_3d_unet(input_shape=(128, 128, 64, 1)):
    def conv_block(inputs, filters):
        x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        return x

    inputs = Input(shape=input_shape)
    conv1 = conv_block(inputs, 32)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv_block(pool1, 64)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv_block(pool2, 128)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv_block(pool3, 256)
    up5 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    merge5 = concatenate([up5, conv3], axis=4)

    conv5 = conv_block(merge5, 128)
    up6 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv2], axis=4)

    conv6 = conv_block(merge6, 64)
    up7 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv1], axis=4)

    conv7 = conv_block(merge7, 32)
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
    return Model(inputs, outputs)


# Load and preprocess NIfTI data
def load_nifti(file_path):
    """Load a NIfTI file."""
    nifti = nib.load(file_path)
    return nifti.get_fdata()


def preprocess_nifti(image, target_size):
    """Resize a NIfTI image to the target size."""
    depth_resized = np.stack([
        cv2.resize(image[:, :, i], target_size[:2], interpolation=cv2.INTER_LINEAR)
        for i in range(image.shape[2])
    ], axis=-1)
    return np.expand_dims(depth_resized, axis=-1)


def load_nifti_data(root_dir, target_size=(128, 128, 64)):
    """Load and preprocess NIfTI images."""
    images = []
    for case_dir in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_dir)
        if not os.path.isdir(case_path):
            continue
        t1c_path = os.path.join(case_path, f"{case_dir}-t1c.nii.gz")
        if os.path.exists(t1c_path):
            image = load_nifti(t1c_path)
            image_resized = preprocess_nifti(image, target_size)
            images.append(image_resized)
    return np.array(images)


def load_separated_masks(root_dir, target_size=(128, 128, 64)):
    """Load and preprocess each mask in the segmentation file."""
    images, masks_dict = [], {}
    print(f"Loading data from {root_dir}")

    for case_dir in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_dir)
        print(f"Checking {case_path}")
        if not os.path.isdir(case_path):
            continue
        t1c_path = os.path.join(case_path, f"{case_dir}-t1c.nii.gz")
        seg_path = os.path.join(case_path, f"{case_dir}-seg.nii.gz")

        # Check if files exist
        if os.path.exists(t1c_path) and os.path.exists(seg_path):
            print(f"Loading: {case_dir}")
            image = load_nifti(t1c_path)
            mask = load_nifti(seg_path)

            # Preprocess image
            image_resized = preprocess_nifti(image, target_size)
            images.append(image_resized)

            # Process each unique mask in the segmentation file
            unique_values = np.unique(mask)
            unique_values = unique_values[unique_values != 0]  # Exclude background
            for value in unique_values:
                if value not in masks_dict:
                    masks_dict[value] = []
                mask_layer = (mask == value).astype(np.float32)
                mask_resized = preprocess_nifti(mask_layer, target_size)
                masks_dict[value].append(mask_resized)

    images = np.array(images)
    print(f"Loaded {len(images)} images")
    for value in masks_dict:
        masks_dict[value] = np.array(masks_dict[value])

    return images, masks_dict


# Visualization
def visualize_segmentation(image, predicted_mask, ground_truth_mask=None):
    """Visualize the middle slice."""
    slice_idx = image.shape[2] // 2
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image[:, :, slice_idx, 0], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask[:, :, slice_idx, 0], cmap='hot', alpha=0.7)

    if ground_truth_mask is not None:
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth Mask")
        plt.imshow(ground_truth_mask[:, :, slice_idx, 0], cmap='hot', alpha=0.7)

    plt.tight_layout()
    plt.show()


def train_and_save_model(mask_id, loss_fn, loss_name, train_images, train_mask, val_images, val_mask, target_size,
                         epochs, batch_size, model_dir):
    """
    Trains and saves the model for a given mask and loss function.
    """
    # Define model paths
    model_path = os.path.join(model_dir, f"model_mask_{mask_id}_{loss_name}.h5")
    history_path = os.path.join(model_dir, f"history_mask_{mask_id}_{loss_name}.npz")

    # Check if model already trained
    if os.path.exists(model_path):
        print(f"Model for Mask {mask_id} with {loss_name} Loss already trained. Loading...")
        model = tf.keras.models.load_model(model_path, custom_objects={loss_name: loss_fn})
    else:
        # Build and compile the model
        print(f"Training model for Mask {mask_id} with {loss_name} Loss...")
        model = build_3d_unet(input_shape=(*target_size, 1))
        model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_fn, metrics=['accuracy'])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True)
        ]

        # Confirm training has started
        print("Training started...")

        # Train the model
        history = model.fit(
            train_images, train_mask,
            validation_data=(val_images, val_mask),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        # Save training history
        print("Training complete, saving history...")
        np.savez(history_path, history.history)

    return model


def main():
    # Paths
    train_dir = "../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData"
    val_dir = "../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData"
    model_dir = "./trained_models"
    os.makedirs(model_dir, exist_ok=True)

    # Parameters
    target_size = (128, 128, 64)
    batch_size = 2
    epochs = 50

    # Load data
    print("Loading training data...")
    train_images, train_masks = load_separated_masks(train_dir, target_size)
    print(f"Loaded {len(train_images)} images and {len(train_masks)} masks for training")

    print("Loading validation data...")
    val_images, val_masks = load_separated_masks(val_dir, target_size)
    print(f"Loaded {len(val_images)} images and {len(val_masks)} masks for validation")

    # Train for each mask and loss function
    for mask_id in train_masks:
        print(f"Training for Mask {mask_id}...")
        train_mask = train_masks[mask_id]
        val_mask = val_masks.get(mask_id, None)

        if val_mask is not None:
            for loss_fn, loss_name in [(dice_loss, "Dice"), (iou_loss, "IoU")]:
                print(f"Training with {loss_name} loss...")
                model = train_and_save_model(mask_id, loss_fn, loss_name, train_images, train_mask, val_images, val_mask, target_size, epochs, batch_size, model_dir)

if __name__ == "__main__":
    main()
