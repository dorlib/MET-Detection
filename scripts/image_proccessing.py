import datetime
import os

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from nibabel.dft import pydicom
from pydicom import Dataset, FileDataset, uid


def display_nifti(nifti_file_path):
    """
    Display a single NIFTI file.

    Parameters:
    nifti_file_path (str): Path to the NIFTI file.
    """
    img = nib.load(nifti_file_path).get_fdata()

    print(f"The .nii files are stored in memory as numpy's: {type(img)}.")

    plt.style.use('default')
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(img[:, :, 1 + i])
    plt.show()


def display_dicom(dicom_file_path):
    """
    Display a single DICOM file.

    Parameters:
    dicom_file_path (str): Path to the DICOM file.
    """
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Check if the file contains pixel data
    if hasattr(dicom_data, 'pixel_array'):
        # Display the DICOM image
        plt.imshow(dicom_data.pixel_array, cmap='gray')
        plt.axis('off')  # Hide axis for a cleaner image display
        plt.title("DICOM Image")
        plt.show()
    else:
        print("The DICOM file does not contain any image data.")


def display_dicom_series(directory):
    """
    Display a group of DICOM files.

    Parameters:
    directory (str): Path to the DICOM directory.
    """
    # Get all DICOM files in the directory
    dicom_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')])

    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread(dicom_file)
        plt.imshow(dicom_data.pixel_array, cmap="gray")
        plt.axis("off")
        plt.show()


def nifti_to_dicom(nifti_path, output_path):
    """
    convert nifti file to DICOM

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
    study_date = datetime.datetime.now().strftime('%Y%m%d')
    study_time = datetime.datetime.now().strftime('%H%M%S')

    for i in range(num_slices):
        # Create a new DICOM dataset
        dicom_file = FileDataset(
            None, {}, file_meta=file_meta, preamble=b"\0" * 128
        )

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
        dicom_file.PhotometricInterpretation = "MONOCHROME2"  # Required for grayscale images
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


if __name__ == '__main__':
    nifti_to_dicom(
        '../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00418-000/BraTS-MET-00418-000-t1n.nii',
                   '../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00418-000/dicom')
    display_dicom_series('../../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00418-000/dicom')
