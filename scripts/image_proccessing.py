"""Provides some image processing functions."""

import datetime
import math
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.dft import pydicom
from pydicom import FileDataset, uid


def display_nifti(nifti_file_path, scan_type):
    """Display a single NIFTI file.

    Parameters:
    nifti_file_path (str): Path to the NIFTI file.
    """
    img = nib.load(nifti_file_path).get_fdata()

    print(f"The .nii files are stored in memory as numpy's: {type(img)}.")

    plt.style.use("default")
    fig, axes = plt.subplots(int(math.sqrt(img.shape[2])), int(math.sqrt(img.shape[2])), figsize=(12, 12))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(img[:, :, 1 + i])
    plt.title(scan_type)
    plt.show()


def apply_mask(matrix):
    """Applying mask on given matrix.

    Parameters:
    matrix : matrix to apply mask on
    """
    # Find unique values in the matrix.
    unique_values = np.unique(matrix)

    # Create a dictionary where each unique value has its own mask.
    masks = {value: (matrix == value) for value in unique_values}

    # Apply each mask to isolate submatrices.
    masked_images = {value: matrix * mask for value, mask in masks.items()}

    return masked_images


def display_nifti_with_slices_for_mask(nifti_file_path):
    """Display all NIFTI slices and separate plots for each unique value in the mask,
    also saves each unique value as a NIFTI file."""

    # Load NIFTI data
    img = nib.load(nifti_file_path).get_fdata()
    num_slices = img.shape[2]
    slice_range = range(num_slices)

    print(f"The .nii files are stored in memory as numpy's: {type(img)}.")
    print(f"Displaying slices from 0 to {num_slices - 1}")

    # Normalize the full image data to range [0, 1]
    img_normalized = (img - img.min()) / (img.max() - img.min())

    # Create a directory to save the individual NIFTI files
    nifti_dir = os.path.dirname(nifti_file_path)
    mask_save_dir = os.path.join(nifti_dir, 'masks-full')
    os.makedirs(mask_save_dir, exist_ok=True)

    # First plot: Original slices
    plt.style.use("default")
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in zip(slice_range, axes.ravel()):
        slice_data = img_normalized[:, :, i]
        colored_slice = cm.viridis(slice_data)  # Apply colormap
        ax.imshow(colored_slice)
        ax.set_title(f"Slice {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Second set of plots: Separate plot for each unique value in masks
    unique_values = np.unique(img)
    unique_values = unique_values[unique_values != 0]  # Exclude background value

    for value in unique_values:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        normalized_value = (value - img.min()) / (img.max() - img.min())
        fig.suptitle(f"Masked Regions for Value: {value} (Normalized: {normalized_value:.2f})", fontsize=16)

        # Create the mask for the current unique value
        mask = (img == value)

        # Save the mask as a new NIFTI file
        mask_nifti = nib.Nifti1Image(mask.astype(np.int16), affine=np.eye(4))
        mask_filename = os.path.join(mask_save_dir, f"mask_value_{value}.nii")
        nib.save(mask_nifti, mask_filename)
        print(f"Saved mask for value {value} as: {mask_filename}")

        for i, ax in zip(slice_range, axes.ravel()):
            slice_data = img[:, :, i]
            mask_slice = mask[:, :, i]

            # Apply mask and keep normalized intensity
            masked_slice = mask_slice * img_normalized[:, :, i]

            if masked_slice.any():  # Only plot if the mask is not empty
                colored_mask = cm.viridis(masked_slice)  # Directly use normalized values
                ax.imshow(colored_mask)
            else:
                ax.imshow(np.zeros_like(slice_data), cmap="gray")  # Blank plot if no mask

            ax.set_title(f"Slice {i}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def display_nifti_with_slices_for_mask_by_slices(nifti_file_path, num_slices=16):
    """Display NIFTI slices and separate plots for each unique value in the mask,
    also saves each unique value as a NIFTI file."""

    # Load NIFTI data
    img = nib.load(nifti_file_path).get_fdata()
    mid_slice = img.shape[2] // 2
    slice_range = range(mid_slice - num_slices // 2, mid_slice + num_slices // 2)

    print(f"The .nii files are stored in memory as numpy's: {type(img)}.")
    print(f"Displaying slices from {mid_slice - num_slices // 2} to {mid_slice + num_slices // 2}")

    # Normalize the full image data to range [0, 1]
    img_normalized = (img - img.min()) / (img.max() - img.min())

    # Create a directory to save the individual NIFTI files
    nifti_dir = os.path.dirname(nifti_file_path)
    mask_save_dir = os.path.join(nifti_dir, 'masks')
    os.makedirs(mask_save_dir, exist_ok=True)

    # First plot: Original slices
    plt.style.use("default")
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in zip(slice_range, axes.ravel()):
        slice_data = img_normalized[:, :, i]
        colored_slice = cm.viridis(slice_data)  # Apply colormap
        ax.imshow(colored_slice)
        ax.set_title(f"Slice {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Second set of plots: Separate plot for each unique value in masks
    unique_values = np.unique(img)
    unique_values = unique_values[unique_values != 0]  # Exclude background value

    for value in unique_values:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        normalized_value = (value - img.min()) / (img.max() - img.min())
        fig.suptitle(f"Masked Regions for Value: {value} (Normalized: {normalized_value:.2f})", fontsize=16)

        # Create the mask for the current unique value
        mask = (img == value)

        # Save the mask as a new NIFTI file
        mask_nifti = nib.Nifti1Image(mask.astype(np.int16), affine=np.eye(4))
        mask_filename = os.path.join(mask_save_dir, f"mask_value_{value}.nii")
        nib.save(mask_nifti, mask_filename)
        print(f"Saved mask for value {value} as: {mask_filename}")

        for i, ax in zip(slice_range, axes.ravel()):
            slice_data = img[:, :, i]
            mask_slice = mask[:, :, i]

            # Apply mask and keep normalized intensity
            masked_slice = mask_slice * img_normalized[:, :, i]

            if masked_slice.any():  # Only plot if the mask is not empty
                colored_mask = cm.viridis(masked_slice)  # Directly use normalized values
                ax.imshow(colored_mask)
            else:
                ax.imshow(np.zeros_like(slice_data), cmap="gray")  # Blank plot if no mask

            ax.set_title(f"Slice {i}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def process_folders(root_dir):
    """Process all the 'BraTS-MET-xxxxx-000' folders inside the root directory."""

    # Walk through the directory tree
    for subdir, dirs, files in os.walk(root_dir):
        # Check if the subdirectory matches the pattern for the BraTS-MET-* folder
        if 'BraTS-MET-' in os.path.basename(subdir) and os.path.isdir(subdir):
            print(f"Processing folder: {subdir}")

            # Process each NIFTI file that ends with 'seg.nii.gz'
            for file in files:
                if file.endswith('seg.nii.gz'):
                    nifti_file_path = os.path.join(subdir, file)
                    display_nifti_with_slices_for_mask(nifti_file_path)


def display_nifti_with_slices(nifti_file_path, num_slices=16):
    """Display slices of a NIFTI file in RGB using a colormap.

    Parameters:
    nifti_file_path (str): Path to the NIFTI file.
    num_slices (int): Number of slices to display.
    """
    img = nib.load(nifti_file_path).get_fdata()
    mid_slice = img.shape[2] // 2
    slice_range = range(mid_slice - num_slices // 2, mid_slice + num_slices // 2)

    print(f"The .nii files are stored in memory as numpy's: {type(img)}.")
    print(f"Displaying slices from {mid_slice - num_slices // 2} to {mid_slice + num_slices // 2}")

    plt.style.use("default")
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # Normalize image to range [0, 1] for RGB colormap application
    img = (img - img.min()) / (img.max() - img.min())

    for i, ax in zip(slice_range, axes.ravel()):
        slice_data = img[:, :, i]
        # Apply colormap and convert to RGB
        colored_slice = cm.viridis(img[:, :, i])  # Convert to RGB using 'viridis' colormap
        ax.imshow(colored_slice)
        ax.set_title(f"Slice {i}")
        ax.axis("off")

        non_zero_rgb_values = colored_slice[slice_data != 0]
        print(f"Non-zero values in slice {i}: {non_zero_rgb_values}")

    masked_images = apply_mask(img[:, :, 71])
    # Iterate over the masked images dictionary and display each masked region
    for value, masked_image in masked_images.items():  # Use .items() to get key-value pairs
        plt.imshow(masked_image)
        plt.title(f"Masked Value: {value}")
        plt.show()

    plt.tight_layout()
    plt.show()


def display_dicom(dicom_file_path):
    """Display a single DICOM file.

    Parameters:
    dicom_file_path (str): Path to the DICOM file.
    """
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Check if the file contains pixel data
    if hasattr(dicom_data, "pixel_array"):
        # Display the DICOM image
        plt.imshow(dicom_data.pixel_array, cmap="gray")
        plt.axis("off")  # Hide axis for a cleaner image display
        plt.title("DICOM Image")
        plt.show()
    else:
        print("The DICOM file does not contain any image data.")


def display_dicom_series(directory):
    """Display a group of DICOM files.

    Parameters:
    directory (str): Path to the DICOM directory.
    """
    # Get all DICOM files in the directory
    dicom_files = sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".dcm")
        ]
    )

    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread(dicom_file)
        plt.imshow(dicom_data.pixel_array, cmap="gray")
        plt.axis("off")
        plt.show()


def nifti_to_dicom(nifti_path, output_path):
    """Convert nifti file to DICOM.

    Parameters:
    nifti_path (str): Path to the nifti file
    output_path (str): Path to the DICOM file
    """
    # Load the NIfTI file
    img = nib.load(nifti_path)
    img_data = img.get_fdata()
    num_slices = img_data.shape[2]

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create File Meta Information
    file_meta = FileDataset(None, {}, preamble=b"\0" * 128)
    file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian

    # Create required DICOM metadata
    study_date = datetime.datetime.now().strftime("%Y%m%d")
    study_time = datetime.datetime.now().strftime("%H%M%S")

    for i in range(num_slices):
        # Create a new DICOM dataset
        dicom_file = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Set necessary DICOM metadata
        dicom_file.PatientName = "Test^Firstname"
        dicom_file.PatientID = "123456"
        dicom_file.Modality = "MR"
        dicom_file.StudyInstanceUID = uid.generate_uid()
        dicom_file.SeriesInstanceUID = uid.generate_uid()
        dicom_file.SOPInstanceUID = uid.generate_uid()
        dicom_file.StudyDate = study_date
        dicom_file.StudyTime = study_time
        dicom_file.SliceLocation = i
        dicom_file.InstanceNumber = i + 1
        dicom_file.PhotometricInterpretation = (
            "MONOCHROME2"  # Required for grayscale images
        )
        dicom_file.SamplesPerPixel = 1  # Single channel for grayscale images

        # Set pixel data and affine transformation
        slice_data = img_data[:, :, i]
        dicom_file.PixelData = slice_data.astype(np.int16).tobytes()
        dicom_file.Rows, dicom_file.Columns = slice_data.shape
        dicom_file.BitsAllocated = 16  # Typically 16 for medical images
        dicom_file.BitsStored = 16
        dicom_file.HighBit = 15
        dicom_file.PixelRepresentation = 1  # 1 for signed data (typical for MR)

        # Define the file path for this DICOM slice
        filename = os.path.join(output_path, f"slice_{i + 1:03}.dcm")

        # Save the DICOM file
        dicom_file.save_as(filename)

    print(f"DICOM series saved to {output_path}")


if __name__ == "__main__":
    # display_nifti("../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00025-000/BraTS-MET-00025-000-t1c.nii", "t1c")
    # nifti_to_dicom("../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00025-000/BraTS-MET-00025-000-t1n.nii",
    #                "../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00025-000/dicom")
    # display_dicom_series("../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/Brats-met-00025-000/dicom")
    # display_nifti_with_slices(
    #     "../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00025-000/BraTS-MET-00025-000-seg.nii")
    # display_nifti_with_slices_for_mask("../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00025-000/BraTS-MET-00025-000-seg.nii")
    # process_folders("../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData")
    display_nifti("../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/Brats-met-00025-000/masks/mask_value_1.0.nii", "cdscd")
