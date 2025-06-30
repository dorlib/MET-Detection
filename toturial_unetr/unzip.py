import os
import gzip
import shutil


def decompress_nii_gz(base_dir, suffix='.nii.gz', verbose=True):
    """
    Traverses the base directory and its subdirectories to find and decompress
    all files with the specified suffix (e.g., '.nii.gz') to '.nii'.

    Parameters:
        base_dir (str): The base directory to start the search.
        suffix (str): The file suffix to look for. Default is '.nii.gz'.
        verbose (bool): If True, prints detailed logs. Default is True.
    """
    # Counter for tracking decompressed files
    decompressed_count = 0
    skipped_count = 0
    error_count = 0

    # Walk through the directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(suffix):
                gz_file_path = os.path.join(root, file)
                nii_file_name = file[:-3]  # Remove the '.gz' extension
                nii_file_path = os.path.join(root, nii_file_name)

                if os.path.exists(nii_file_path):
                    if verbose:
                        print(f"Skipping '{gz_file_path}' as '{nii_file_name}' already exists.")
                    skipped_count += 1
                    continue

                try:
                    if verbose:
                        print(f"Decompressing '{gz_file_path}' to '{nii_file_path}'...")

                    with gzip.open(gz_file_path, 'rb') as f_in:
                        with open(nii_file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    if verbose:
                        print(f"Successfully decompressed '{nii_file_path}'.")
                    decompressed_count += 1

                except Exception as e:
                    print(f"Error decompressing '{gz_file_path}': {e}")
                    error_count += 1

    # Summary of operations
    print("\nDecompression Completed.")
    print(f"Total files decompressed: {decompressed_count}")
    print(f"Total files skipped (already decompressed): {skipped_count}")
    print(f"Total errors encountered: {error_count}")


if __name__ == "__main__":
    # Example usage
    # Replace '/path/to/base/folder' with your actual base directory path
    base_directory = '../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData'

    # Call the decompression function
    decompress_nii_gz(base_directory, suffix='-t1c.nii.gz', verbose=True)
