#!/usr/bin/env python3
"""
Script to display all values in a specific slice of a mask file.
This script loads a mask file and displays both the raw one-hot encoded values
and the decoded integer class values for a specific slice.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from glob import glob


def view_mask_slice(mask_path, slice_idx=None, axis=0, output_dir="./mask_slice_views"):
    """
    View all values in a specific slice of a mask.
    
    Parameters:
    -----------
    mask_path : str
        Path to the mask .npy file
    slice_idx : int, optional
        Index of the slice to view. If None, the middle slice is used.
    axis : int, optional
        Axis along which to take the slice (0=z, 1=y, 2=x)
    output_dir : str, optional
        Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mask data
    print(f"Loading mask from: {mask_path}")
    mask = np.load(mask_path)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(mask_path))[0]
    
    # Determine slice index if not provided
    if slice_idx is None:
        slice_idx = mask.shape[axis] // 2  # Middle slice
    
    # Ensure slice_idx is valid
    if slice_idx < 0 or slice_idx >= mask.shape[axis]:
        print(f"Invalid slice index {slice_idx}. Valid range is 0-{mask.shape[axis]-1}")
        slice_idx = mask.shape[axis] // 2
        print(f"Using middle slice: {slice_idx}")
    
    # Get the slice based on axis
    if axis == 0:  # z-axis (axial)
        axis_name = "axial"
        if mask.ndim == 4:  # One-hot encoded
            slice_data = mask[slice_idx, :, :, :]
            # Convert to class indices
            decoded_slice = np.argmax(slice_data, axis=-1)
        else:  # Already in class index form
            slice_data = mask[slice_idx, :, :]
            decoded_slice = slice_data
    elif axis == 1:  # y-axis (coronal)
        axis_name = "coronal"
        if mask.ndim == 4:  # One-hot encoded
            slice_data = mask[:, slice_idx, :, :]
            # Convert to class indices
            decoded_slice = np.argmax(slice_data, axis=-1)
        else:  # Already in class index form
            slice_data = mask[:, slice_idx, :]
            decoded_slice = slice_data
    else:  # x-axis (sagittal)
        axis_name = "sagittal"
        if mask.ndim == 4:  # One-hot encoded
            slice_data = mask[:, :, slice_idx, :]
            # Convert to class indices
            decoded_slice = np.argmax(slice_data, axis=-1)
        else:  # Already in class index form
            slice_data = mask[:, :, slice_idx]
            decoded_slice = slice_data
    
    # Print mask information
    print(f"\nMask shape: {mask.shape}")
    print(f"Viewing {axis_name} slice at index {slice_idx}")
    
    # For one-hot encoded masks
    if mask.ndim == 4:
        print("\nThis is a one-hot encoded mask with shape", mask.shape)
        print("Slice shape:", slice_data.shape)
        
        n_classes = mask.shape[-1]
        print(f"Number of classes: {n_classes}")
        
        # Print unique values
        print("Unique values in one-hot encoded mask:", np.unique(mask))
        
        # Print class distribution in this slice
        print("\nClass distribution in this slice:")
        for i in range(n_classes):
            channel = slice_data[..., i]
            non_zeros = np.count_nonzero(channel)
            total = channel.size
            percentage = (non_zeros / total) * 100
            print(f"  Class {i}: {non_zeros} pixels ({percentage:.2f}%)")
        
        # Print decoded values
        print("\nDecoded (integer) values in this slice:")
        unique_vals, counts = np.unique(decoded_slice, return_counts=True)
        for val, count in zip(unique_vals, counts):
            percentage = (count / decoded_slice.size) * 100
            print(f"  Value {val}: {count} pixels ({percentage:.2f}%)")
        
        # Visualize the data
        colors = ['black', 'red', 'green', 'blue']
        cmap = ListedColormap(colors[:n_classes])
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Plot each channel in the one-hot encoded slice
        for i in range(n_classes):
            ax = fig.add_subplot(2, n_classes, i+1)
            ax.imshow(slice_data[..., i], cmap='gray')
            class_names = ['Background', 'Non-enhancing Tumor', 'Edema', 'Enhancing Tumor']
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            ax.set_title(f"{class_name} (Channel {i})")
            ax.axis('off')
        
        # Plot the decoded integer values with color mapping
        ax_decoded = fig.add_subplot(2, 1, 2)
        im = ax_decoded.imshow(decoded_slice, cmap=cmap, vmin=0, vmax=n_classes-1)
        ax_decoded.set_title(f"Decoded Class Labels - {axis_name.capitalize()} Slice {slice_idx}")
        ax_decoded.axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax_decoded, orientation='horizontal', shrink=0.8)
        cbar.set_ticks(range(n_classes))
        cbar.set_ticklabels([f"{i}: {class_names[i] if i < len(class_names) else f'Class {i}'}" 
                            for i in range(n_classes)])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{filename}_{axis_name}_slice{slice_idx}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        # Print actual values in a section of the slice
        sample_size = 10
        print(f"\nActual values in a {sample_size}x{sample_size} section of the {axis_name} slice {slice_idx}:")
        
        center_i = decoded_slice.shape[0] // 2 - sample_size // 2
        center_j = decoded_slice.shape[1] // 2 - sample_size // 2
        
        sample_section = decoded_slice[center_i:center_i + sample_size, 
                                      center_j:center_j + sample_size]
        
        # Create a visual representation of the values
        print("\nDecoded integer values (class labels):")
        for i in range(sample_section.shape[0]):
            row = " ".join(f"{val}" for val in sample_section[i])
            print(row)
        
        # Save raw values to a text file for reference
        txt_output = os.path.join(output_dir, f"{filename}_{axis_name}_slice{slice_idx}_values.txt")
        with open(txt_output, 'w') as f:
            f.write(f"Mask: {mask_path}\n")
            f.write(f"{axis_name.capitalize()} slice {slice_idx}\n\n")
            f.write("Decoded integer values (class labels):\n")
            for i in range(decoded_slice.shape[0]):
                row = " ".join(f"{val}" for val in decoded_slice[i])
                f.write(row + "\n")
        
        print(f"\nSaved full slice values to: {txt_output}")
        print(f"Saved visualization to: {output_path}")
        
    else:
        # For non one-hot encoded masks
        print("\nThis is a standard mask with shape", mask.shape)
        print("Slice shape:", decoded_slice.shape)
        
        # Print unique values
        unique_vals, counts = np.unique(decoded_slice, return_counts=True)
        print("Unique values in this slice:", unique_vals)
        
        print("\nValue distribution in this slice:")
        for val, count in zip(unique_vals, counts):
            percentage = (count / decoded_slice.size) * 100
            print(f"  Value {val}: {count} pixels ({percentage:.2f}%)")
        
        # Visualize the data
        plt.figure(figsize=(12, 10))
        plt.imshow(decoded_slice)
        plt.title(f"Mask Values - {axis_name.capitalize()} Slice {slice_idx}")
        plt.colorbar(label='Class')
        plt.axis('off')
        
        output_path = os.path.join(output_dir, f"{filename}_{axis_name}_slice{slice_idx}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        # Print actual values in a section of the slice
        sample_size = 10
        print(f"\nActual values in a {sample_size}x{sample_size} section of the {axis_name} slice {slice_idx}:")
        
        center_i = decoded_slice.shape[0] // 2 - sample_size // 2
        center_j = decoded_slice.shape[1] // 2 - sample_size // 2
        
        sample_section = decoded_slice[center_i:center_i + sample_size, 
                                      center_j:center_j + sample_size]
        
        for i in range(sample_section.shape[0]):
            row = " ".join(f"{val}" for val in sample_section[i])
            print(row)
        
        # Save raw values to a text file for reference
        txt_output = os.path.join(output_dir, f"{filename}_{axis_name}_slice{slice_idx}_values.txt")
        with open(txt_output, 'w') as f:
            f.write(f"Mask: {mask_path}\n")
            f.write(f"{axis_name.capitalize()} slice {slice_idx}\n\n")
            for i in range(decoded_slice.shape[0]):
                row = " ".join(f"{val}" for val in decoded_slice[i])
                f.write(row + "\n")
        
        print(f"\nSaved full slice values to: {txt_output}")
        print(f"Saved visualization to: {output_path}")


def main():
    """Main function to handle script execution."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python view_mask_slice.py <mask_file> [slice_index] [axis]")
        print("  mask_file: Path to mask .npy file")
        print("  slice_index: (Optional) Index of slice to view (default: middle slice)")
        print("  axis: (Optional) Axis to slice along (0=z/axial, 1=y/coronal, 2=x/sagittal, default: 0)")
        
        # Try to find mask files in common directories
        mask_files = []
        for search_dir in ['../../MET-data/t1c_only/masks/', '../../MET-data/input_data/masks/']:
            mask_files.extend(glob(os.path.join(search_dir, '*.npy')))
        
        if mask_files:
            print(f"\nFound {len(mask_files)} mask files. Example usage:")
            example_file = os.path.basename(mask_files[0])
            print(f"  python view_mask_slice.py {example_file} 60 0")
            sys.exit(1)
        else:
            print("\nNo mask files found in common directories.")
            sys.exit(1)
    
    # Get the mask file path
    mask_path = sys.argv[1]
    
    # Check if the path is relative or absolute
    if not os.path.isabs(mask_path):
        # Try to find the file in common directories
        found = False
        for search_dir in ['../../MET-data/t1c_only/masks/', '../../MET-data/input_data/masks/']:
            full_path = os.path.join(search_dir, mask_path)
            if os.path.exists(full_path):
                mask_path = full_path
                found = True
                break
        
        if not found:
            print(f"Could not find mask file: {mask_path}")
            sys.exit(1)
    
    # Get the slice index if provided
    slice_idx = None
    if len(sys.argv) > 2:
        try:
            slice_idx = int(sys.argv[2])
        except ValueError:
            print("Slice index must be an integer.")
            sys.exit(1)
    
    # Get the axis if provided
    axis = 0
    if len(sys.argv) > 3:
        try:
            axis = int(sys.argv[3])
            if axis not in [0, 1, 2]:
                print("Axis must be 0 (z/axial), 1 (y/coronal), or 2 (x/sagittal).")
                axis = 0
        except ValueError:
            print("Axis must be an integer (0, 1, or 2).")
            sys.exit(1)
    
    # Call the view function
    view_mask_slice(mask_path, slice_idx, axis)


if __name__ == "__main__":
    main()