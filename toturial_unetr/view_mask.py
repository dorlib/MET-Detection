#!/usr/bin/env python3
"""
Script to load and print detailed information about a mask numpy array.
This helps understand the structure and content of the mask arrays created
during preprocessing. Shows both one-hot encoded and decoded integer labels.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
from matplotlib.colors import ListedColormap

def print_mask_info(mask_path):
    """Print detailed information about a mask numpy array."""
    print(f"Loading mask from: {mask_path}")
    
    # Load the mask
    mask = np.load(mask_path)
    
    # Basic information
    print("\nBasic Information:")
    print(f"  Shape: {mask.shape}")
    print(f"  Data type: {mask.dtype}")
    print(f"  Min value: {mask.min()}")
    print(f"  Max value: {mask.max()}")
    print(f"  Unique values: {np.unique(mask)}")
    
    # Check if one-hot encoded
    is_one_hot = (mask.ndim == 4 and mask.shape[-1] >= 2 and 
                 np.array_equal(np.unique(mask), np.array([0, 1])))
    print(f"  One-hot encoded: {is_one_hot}")
    
    if is_one_hot:
        # Convert from one-hot encoding to integer labels
        print("\nConverting from one-hot encoding to integer labels (argmax)...")
        decoded_mask = np.argmax(mask, axis=-1)
        print(f"  Decoded shape: {decoded_mask.shape}")
        print(f"  Decoded unique values: {np.unique(decoded_mask)}")
    
    # Distribution of values in each class/channel
    if is_one_hot and mask.shape[-1] == 4:  # One-hot encoded with 4 classes
        print("\nClass distribution (one-hot encoded):")
        for i in range(mask.shape[-1]):
            non_zeros = np.count_nonzero(mask[..., i])
            percentage = (non_zeros / (mask.shape[0] * mask.shape[1] * mask.shape[2])) * 100
            print(f"  Class {i}: {non_zeros} voxels ({percentage:.2f}%)")
    else:
        print("\nValue distribution:")
        unique_vals, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique_vals, counts):
            percentage = (count / mask.size) * 100
            print(f"  Value {val}: {count} voxels ({percentage:.2f}%)")
    
    # Print sample values from both original and decoded mask
    mid_slice = mask.shape[0] // 2  # Middle slice along z-axis
    mid_row_start = mask.shape[1] // 2 - 2
    mid_col_start = mask.shape[2] // 2 - 2
    
    print(f"\nSample 5x5 patch from middle slice (z={mid_slice}):")
    if is_one_hot:
        # Print a 5x5 section from the middle of the slice for each channel
        print("  One-hot encoded values:")
        for class_idx in range(mask.shape[-1]):
            print(f"  Class {class_idx} channel:")
            patch = mask[mid_slice, mid_row_start:mid_row_start+5, mid_col_start:mid_col_start+5, class_idx]
            for row in patch:
                print("    ", row)
        
        # Show the decoded (integer) values
        print("\n  Decoded integer class labels:")
        decoded_patch = decoded_mask[mid_slice, mid_row_start:mid_row_start+5, mid_col_start:mid_col_start+5]
        for row in decoded_patch:
            print("    ", row)
    else:
        print("  Raw values:")
        patch = mask[mid_slice, mid_row_start:mid_row_start+5, mid_col_start:mid_col_start+5]
        for row in patch:
            print("    ", row)

def visualize_mask_slices(mask_path, output_dir="./mask_visualizations"):
    """Create visualizations of different slices of the mask."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the mask
    mask = np.load(mask_path)
    
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(mask_path))[0]
    
    # Create a colormap for the segmentation classes
    colors = ['black', 'red', 'green', 'blue'] 
    cmap = ListedColormap(colors[:mask.shape[-1]] if mask.ndim == 4 else colors)
    
    # Prepare figure
    if mask.ndim == 4 and mask.shape[-1] > 1:  # One-hot encoded
        # Get class indices (convert from one-hot to class indices)
        mask_indices = np.argmax(mask, axis=-1)
        
        # Create visualizations for different views
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Axial view (middle slice)
        mid_slice_z = mask_indices.shape[0] // 2
        im1 = axes[0].imshow(mask_indices[mid_slice_z], cmap=cmap, vmin=0, vmax=3)
        axes[0].set_title(f'Axial Slice (z={mid_slice_z})')
        
        # Coronal view
        mid_slice_y = mask_indices.shape[1] // 2
        im2 = axes[1].imshow(mask_indices[:, mid_slice_y, :], cmap=cmap, vmin=0, vmax=3)
        axes[1].set_title(f'Coronal Slice (y={mid_slice_y})')
        
        # Sagittal view
        mid_slice_x = mask_indices.shape[2] // 2
        im3 = axes[2].imshow(mask_indices[:, :, mid_slice_x], cmap=cmap, vmin=0, vmax=3)
        axes[2].set_title(f'Sagittal Slice (x={mid_slice_x})')
        
        # Add colorbar
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.7)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['Background', 'Non-enhancing Tumor', 'Edema', 'Enhancing Tumor'])
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_views_decoded.png"))
        plt.close()
        
        # Also show the raw one-hot encoded channels
        for class_idx in range(mask.shape[-1]):
            class_names = ['Background', 'Non-enhancing Tumor', 'Edema', 'Enhancing Tumor']
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Axial view
            axes[0].imshow(mask[mid_slice_z, :, :, class_idx], cmap='hot')
            axes[0].set_title(f'{class_name} - Axial')
            
            # Coronal view
            axes[1].imshow(mask[:, mid_slice_y, :, class_idx], cmap='hot')
            axes[1].set_title(f'{class_name} - Coronal')
            
            # Sagittal view
            axes[2].imshow(mask[:, :, mid_slice_x, class_idx], cmap='hot')
            axes[2].set_title(f'{class_name} - Sagittal')
            
            for ax in axes:
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}_class{class_idx}_{class_name.replace(' ', '_')}.png"))
            plt.close()
    else:
        # Non one-hot encoded mask
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        mid_slice_z = mask.shape[0] // 2
        mid_slice_y = mask.shape[1] // 2
        mid_slice_x = mask.shape[2] // 2
        
        im1 = axes[0].imshow(mask[mid_slice_z], cmap=cmap)
        axes[0].set_title(f'Axial Slice (z={mid_slice_z})')
        
        im2 = axes[1].imshow(mask[:, mid_slice_y, :], cmap=cmap)
        axes[1].set_title(f'Coronal Slice (y={mid_slice_y})')
        
        im3 = axes[2].imshow(mask[:, :, mid_slice_x], cmap=cmap)
        axes[2].set_title(f'Sagittal Slice (x={mid_slice_x})')
        
        # Add colorbar
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.7)
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_views.png"))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def create_side_by_side_comparison(mask_path, output_dir="./mask_visualizations"):
    """Create a side-by-side comparison of one-hot encoded vs decoded integer mask"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the mask
    mask = np.load(mask_path)
    
    # Only proceed if this is a one-hot encoded mask
    if not (mask.ndim == 4 and mask.shape[-1] >= 2):
        print("Not a one-hot encoded mask. Skipping comparison.")
        return
        
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(mask_path))[0]
    
    # Convert one-hot to integer labels
    decoded_mask = np.argmax(mask, axis=-1)
    
    # Get a middle slice
    mid_slice = mask.shape[0] // 2
    
    # Create a figure showing both representations side by side
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Plot each channel from one-hot encoding
    for i in range(4):
        axes[i].imshow(mask[mid_slice, :, :, i], cmap='gray')
        class_names = ['Background', 'Non-enhancing Tumor', 'Edema', 'Enhancing Tumor']
        axes[i].set_title(f'Channel {i}\n({class_names[i]})')
        axes[i].axis('off')
    
    # Plot the decoded mask with color mapping
    colors = ['black', 'red', 'green', 'blue']
    cmap = ListedColormap(colors)
    im = axes[4].imshow(decoded_mask[mid_slice], cmap=cmap, vmin=0, vmax=3)
    axes[4].set_title('Decoded Integer Labels\n(argmax of one-hot)')
    axes[4].axis('off')
    
    # Add colorbar to the decoded mask
    cbar = fig.colorbar(im, ax=axes[4], orientation='vertical', shrink=0.8)
    cbar.set_ticks([0.4, 1.2, 2.0, 2.8])
    cbar.set_ticklabels(['0: Background', '1: Non-enhancing', '2: Edema', '3: Enhancing'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_one_hot_vs_decoded.png"), dpi=150)
    plt.close()
    
    print(f"Comparison visualization saved to {output_dir}/{filename}_one_hot_vs_decoded.png")

def main():
    """Main function to handle script execution."""
    # Define the directory to search for mask files
    mask_dir = '../../MET-data/input_data/masks/'
    
    # Find all mask files
    mask_files = sorted(glob(os.path.join(mask_dir, 'mask_*.npy')))
    
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        # Try alternate location
        mask_dir = '../../MET-data/t1c_only/masks/'
        mask_files = sorted(glob(os.path.join(mask_dir, '*.npy')))
        if not mask_files:
            print(f"No mask files found in alternate location {mask_dir}")
            sys.exit(1)
    
    # Get the file to analyze
    if len(mask_files) > 0:
        # Use the first mask file or a specific one if specified by argument
        if len(sys.argv) > 1 and sys.argv[1].isdigit():
            idx = int(sys.argv[1])
            if idx < len(mask_files):
                mask_path = mask_files[idx]
            else:
                print(f"Index {idx} out of range. Using first mask file.")
                mask_path = mask_files[0]
        else:
            mask_path = mask_files[0]
            
        print(f"Found {len(mask_files)} mask files.")
        print_mask_info(mask_path)
        visualize_mask_slices(mask_path)
        create_side_by_side_comparison(mask_path)
    else:
        print("No mask files found!")

if __name__ == "__main__":
    main()