#!/usr/bin/env python3
# preprocess_t1c.py - Process BraTS MET data focusing only on t1c images

import os
import logging
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize scaler for intensity normalization
scaler = MinMaxScaler()

# Define directories
INPUT_DIR = "../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional"
OUTPUT_DIR = "../../MET-data/t1c_only"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

# Get sorted lists of t1c images and masks
t1c_list = sorted(glob.glob(os.path.join(INPUT_DIR, '*/*t1c.nii')))
mask_list = sorted(glob.glob(os.path.join(INPUT_DIR, '*/*seg.nii')))

logger.info(f"Found {len(t1c_list)} t1c images and {len(mask_list)} masks")

if len(t1c_list) != len(mask_list):
    logger.warning(f"Number of t1c images ({len(t1c_list)}) does not match number of masks ({len(mask_list)})")

# Process each image and its corresponding mask
for img_idx, (t1c_path, mask_path) in enumerate(zip(t1c_list, mask_list)):
    try:
        logger.info(f"Processing image {img_idx+1}/{len(t1c_list)}: {t1c_path}")
        
        # Extract case ID number for filename
        match = re.search(r"MET-(\d+)", t1c_path)
        if not match:
            logger.warning(f"Could not extract case ID from {t1c_path}, skipping")
            continue
        case_id = match.group(1)
        
        # Load t1c image and normalize
        t1c_image = nib.load(t1c_path).get_fdata()
        t1c_image = scaler.fit_transform(t1c_image.reshape(-1, t1c_image.shape[-1])).reshape(t1c_image.shape)
        
        # Load mask and convert class labels (if class 4 exists, convert to 3)
        mask = nib.load(mask_path).get_fdata()
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3  # Reassign mask values 4 to 3
        
        # Add channel dimension to the t1c image for model compatibility
        t1c_image = np.expand_dims(t1c_image, axis=-1)
        
        # Crop to a size to be divisible by 64 (for easy patch extraction)
        # Assuming original size is 240x240x155
        t1c_image = t1c_image[56:184, 56:184, 13:141]
        mask = mask[56:184, 56:184, 13:141]
        
        # Check if the mask contains enough useful information (non-background)
        val, counts = np.unique(mask, return_counts=True)
        if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with non-zero labels
            # One-hot encode the mask
            mask_categorical = to_categorical(mask, num_classes=4)
            
            # Save processed images and masks
            output_img_path = os.path.join(OUTPUT_IMAGES_DIR, f"image_{img_idx}_{case_id}.npy")
            output_mask_path = os.path.join(OUTPUT_MASKS_DIR, f"mask_{img_idx}_{case_id}.npy")
            
            np.save(output_img_path, t1c_image)
            np.save(output_mask_path, mask_categorical)
            
            logger.info(f"Saved processed image and mask for case {case_id}")
        else:
            logger.info(f"Skipping case {case_id} - insufficient mask data (< 1% useful volume)")
            
    except Exception as e:
        logger.error(f"Error processing {t1c_path}: {str(e)}")
        continue

logger.info("Preprocessing complete")