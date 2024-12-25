# https://youtu.be/oB35sV1npVI
"""
Use this code to get your BRATS 2020 dataset ready for semantic segmentation.
Code can be divided into a few parts....

#Combine
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize


https://pypi.org/project/nibabel/

All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings

T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.
"""
import logging

import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import re

scaler = MinMaxScaler()

t1c_list = sorted(glob.glob('../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional/*/*t1c.nii'))
t2w_list = sorted(glob.glob('../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional/*/*t2w.nii'))
# t1n_list = sorted(glob.glob('../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/*/*t1n.nii.gz'))
t2f_list = sorted(glob.glob('../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional/*/*t2f.nii'))
mask_list = sorted(glob.glob('../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional/*/*seg.nii'))

# Each volume generates 18 64x64x64x4 sub-volumes.
# Total 369 volumes = 6642 sub volumes

for img in range(len(t2w_list)):  # Using t2w_list as all lists are of same size
    try:
        print("Now preparing image and masks number: ", img)

        temp_image_t2 = nib.load(t2w_list[img]).get_fdata()
        temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
            temp_image_t2.shape)
        match = re.search(r"MET-(\d+)", t2w_list[img]).group(1)

        temp_image_t1ce = nib.load(t1c_list[img]).get_fdata()
        temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(
            temp_image_t1ce.shape)

        temp_image_flair = nib.load(t2f_list[img]).get_fdata()
        temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
            temp_image_flair.shape)

        temp_mask = nib.load(mask_list[img]).get_fdata()
        temp_mask = temp_mask.astype(np.uint8)
        temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3
        # print(np.unique(temp_mask))

        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

        # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
        # cropping x, y, and z
        temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]

        val, counts = np.unique(temp_mask, return_counts=True)

        if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
            print("Save Me")
            temp_mask = to_categorical(temp_mask, num_classes=4)
            np.save('../../MET-data/input_data/images/image_' + str(img + 165) + "_" + str(match) + '.npy', temp_combined_images)
            np.save('../../MET-data/input_data/masks/mask_' + str(img + 165) + "_" + str(match) + '.npy', temp_mask)

        else:
            print("I am useless")

    except Exception as e:
        problematic_file = t2w_list[img]
        logging.error(f"Error processing image {img} ({problematic_file}): {e}")
        continue  # Skip to the next image


