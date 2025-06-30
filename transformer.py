import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Conv3D, Conv3DTranspose, concatenate, Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K  # Import Keras backend for custom loss functions
from tensorflow.keras.losses import binary_crossentropy  # Import binary cross-entropy loss
from tensorflow.keras.saving import register_keras_serializable

# Enable mixed precision training for better memory utilization
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Function to load and normalize NIfTI images
def load_nifti(file_path):
    """Load a NIfTI file and normalize its intensity."""
    nii = nib.load(file_path)
    image = nii.get_fdata()
    image = np.clip(image, 0, 255)  # Normalize intensity values to [0, 255]
    return image

def preprocess_nifti(image, mask=None, target_size=(32, 32, 16)):
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

# Function to build a Transformer-based encoder-decoder model
def build_transformer_model(input_shape=(32, 32, 16, 1)):
    inputs = Input(shape=input_shape)

    # Initial Convolution
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Transformer Encoder
    def transformer_encoder(x, num_heads=1, key_dim=8, ff_dim=32, dropout_rate=0.1):
        # Flatten the input for Multi-Head Attention
        x_flat = Reshape((-1, x.shape[-1]))(x)
        # Multi-Head Attention Layer
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_flat, x_flat)
        attn_output = Dropout(dropout_rate)(attn_output)
        x_flat = LayerNormalization(epsilon=1e-6)(x_flat + attn_output)

        # Feed Forward Network
        ffn_output = Dense(ff_dim, activation='relu')(x_flat)
        ffn_output = Dense(x_flat.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x_flat = LayerNormalization(epsilon=1e-6)(x_flat + ffn_output)
        x = Reshape((x.shape[1], x.shape[2], x.shape[3], x.shape[-1]))(x_flat)  # Reshape back to original shape
        return x

    # Apply the transformer encoder
    x = transformer_encoder(x, num_heads=1, key_dim=8, ff_dim=32, dropout_rate=0.1)

    # Transformer Decoder
    def transformer_decoder(x, encoder_output, num_heads=1, key_dim=8, ff_dim=32, dropout_rate=0.1):
        # Flatten the input for Multi-Head Attention
        x_flat = Reshape((-1, x.shape[-1]))(x)
        encoder_flat = Reshape((-1, encoder_output.shape[-1]))(encoder_output)
        # Multi-Head Attention Layer with Encoder Output as Key and Value
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_flat, encoder_flat)
        attn_output = Dropout(dropout_rate)(attn_output)
        x_flat = LayerNormalization(epsilon=1e-6)(x_flat + attn_output)

        # Feed Forward Network
        ffn_output = Dense(ff_dim, activation='relu')(x_flat)
        ffn_output = Dense(x_flat.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x_flat = LayerNormalization(epsilon=1e-6)(x_flat + ffn_output)
        x = Reshape((x.shape[1], x.shape[2], x.shape[3], x.shape[-1]))(x_flat)  # Reshape back to original shape
        return x

    # Apply the transformer decoder
    x = transformer_decoder(x, x, num_heads=1, key_dim=8, ff_dim=32, dropout_rate=0.1)

    # Final Convolution to produce output mask
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid', dtype='float32')(x)

    return Model(inputs, outputs)

# Custom IoU coefficient and IoU loss functions
@register_keras_serializable()
def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
def iou_loss(y_true, y_pred):
    return 1 - iou_coefficient(y_true, y_pred)

# Updated Combined Loss Function (IoU + Dice + BCE)
@register_keras_serializable()
def combined_iou_dice_bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    iou = 1 - iou_coefficient(y_true, y_pred)
    dice = 1 - iou_coefficient(y_true, y_pred)
    return iou + dice + binary_crossentropy(y_true, y_pred)

# Load NIfTI data
def load_nifti_data(root_dir, target_size=(32, 32, 16), with_masks=True):
    """Load NIfTI images (t1c) and segmentation masks (ET label) from the dataset."""
    images = []
    masks = [] if with_masks else None

    for case_dir in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_dir)
        if not os.path.isdir(case_path):
            continue
        t1c_path = os.path.join(case_path, f"{case_dir}-t1c.nii.gz")
        seg_path = f"{case_path}/masks/mask_value_3.0.nii" if with_masks else None

        if os.path.exists(t1c_path):
            t1c_image = load_nifti(t1c_path)
            t1c_image_resized = preprocess_nifti(t1c_image, target_size=target_size)

            if with_masks and os.path.exists(seg_path):
                mask = load_nifti(seg_path)
                _, mask_resized = preprocess_nifti(t1c_image, mask, target_size=target_size)
                images.append(t1c_image_resized)
                masks.append(mask_resized)
            elif not with_masks:
                images.append(t1c_image_resized)
            else:
                if with_masks:
                    print(f"Mask missing for {case_dir}, skipping this case.")
        else:
            print(f"Skipping case {case_dir} due to missing t1c file.")

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
    model_path = "unet_3d_model-t1c-transformer.keras"

    # Parameters
    target_size = (32, 32, 16)
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
            'combined_iou_dice_bce_loss': combined_iou_dice_bce_loss
        })
    else:
        print("Training a new model...")
        model = build_transformer_model(input_shape=(32, 32, 16, 1))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=combined_iou_dice_bce_loss, metrics=[iou_coefficient])
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