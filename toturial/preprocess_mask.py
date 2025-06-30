import os
import re
import numpy as np
import matplotlib.pyplot as plt


def check_and_plot_4d_masks(directory):
    """
    Ensures corresponding files exist and plots slices or channels from 4D masks.

    Args:
        directory (str): Path to the directory containing the files.
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

    # Plot corresponding 4D masks for files that exist in both formats
    corresponding_files = files_0_3.intersection(files_regular)

    for number, another_number in corresponding_files:
        mask_0_3_path = os.path.join(directory, f"mask_0.3_{number}_{another_number}.npy")
        mask_regular_path = os.path.join(directory, f"mask_{number}_{another_number}.npy")

        # Load the masks
        mask_0_3 = np.load(mask_0_3_path)
        mask_regular = np.load(mask_regular_path)

        print(f"Loaded mask_0.3 shape: {mask_0_3.shape}, mask shape: {mask_regular.shape}")

        # Extract a specific channel (e.g., channel 0) for visualization
        channel = 0
        slice_index = mask_0_3.shape[2] // 2  # Middle slice

        plt.figure(figsize=(12, 6))

        # Plot mask_0.3
        plt.subplot(121)
        plt.imshow(mask_0_3[:, :, slice_index, channel])
        plt.title(f"mask_0.3_{number}_{another_number} - Slice {slice_index}, Channel {channel}")

        # Plot mask
        plt.subplot(122)
        plt.imshow(mask_regular[:, :, slice_index, channel])
        plt.title(f"mask_{number}_{another_number} - Slice {slice_index}, Channel {channel}")

        plt.show()


# Example usage
directory_path = "../../MET-data/input_data/masks/"  # Replace with your directory path
check_and_plot_4d_masks(directory_path)
