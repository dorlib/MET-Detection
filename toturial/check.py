import os
import re
import numpy as np
import matplotlib.pyplot as plt


def check_and_plot_corresponding_files(directory):
    """
    Ensures that for every file of the format `mask_0.3_<number>_<another_number>.npy`
    there is a corresponding `mask_<number>_<another_number>.npy` and vice versa.
    Plots the corresponding masks for verification.

    Args:
        directory (str): Path to the directory containing the files.

    Returns:
        None
    """
    # Get all files in the directory
    all_files = os.listdir(directory)

    # Regular expressions to match both formats
    pattern_0_3 = re.compile(r"mask_0\.3_(\d+)_(\d+)\.npy")
    pattern_regular = re.compile(r"mask_(\d+)_(\d+)\.npy")

    # Parse filenames into sets
    files_0_3 = set()
    files_regular = set()

    for file in all_files:
        match_0_3 = pattern_0_3.match(file)
        match_regular = pattern_regular.match(file)

        if match_0_3:
            files_0_3.add((match_0_3.group(1), match_0_3.group(2)))
        elif match_regular:
            files_regular.add((match_regular.group(1), match_regular.group(2)))

    # Find mismatches
    missing_from_0_3 = files_regular - files_0_3
    missing_from_regular = files_0_3 - files_regular

    if missing_from_0_3 or missing_from_regular:
        if missing_from_0_3:
            print("The following files are missing a corresponding `mask_0.3_` version:")
            for item in missing_from_0_3:
                print(f"mask_0.3_{item[0]}_{item[1]}.npy")

        if missing_from_regular:
            print("The following files are missing a corresponding `mask_` version:")
            for item in missing_from_regular:
                print(f"mask_{item[0]}_{item[1]}.npy")

    # Plot corresponding masks for files that exist in both formats
    corresponding_files = files_0_3.intersection(files_regular)

    for number, another_number in corresponding_files:
        mask_0_3_path = os.path.join(directory, f"mask_0.3_{number}_{another_number}.npy")
        mask_regular_path = os.path.join(directory, f"mask_{number}_{another_number}.npy")

        # Load the masks
        mask_0_3 = np.load(mask_0_3_path)
        mask_regular = np.load(mask_regular_path)

        # Randomly select a slice for 3D masks
        if mask_0_3.ndim == 3 and mask_regular.ndim == 3:
            n_slice = mask_0_3.shape[2] // 2  # Middle slice for consistency

            plt.figure(figsize=(12, 6))

            # Plot mask_0.3
            plt.subplot(121)
            plt.imshow(mask_0_3[:, :, n_slice], cmap='gray')
            plt.title(f"mask_0.3_{number}_{another_number} - Slice {n_slice}")

            # Plot mask
            plt.subplot(122)
            plt.imshow(mask_regular[:, :, n_slice], cmap='gray')
            plt.title(f"mask_{number}_{another_number} - Slice {n_slice}")

            plt.show()
        else:
            print(f"Skipping non-3D masks: {mask_0_3_path} and {mask_regular_path}")


# Example usage
directory_path = "../../MET-data/input_data/masks/"  # Replace with your directory path
check_and_plot_corresponding_files(directory_path)
