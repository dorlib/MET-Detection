#!/usr/bin/env python3
"""
Script to generate a synthetic test mask with all tissue types (including edema)
for visualization and testing purposes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_synthetic_mask(output_dir="./synthetic_masks", save_visualizations=True):
    """
    Generate a synthetic mask with all tissue types.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the synthetic mask
    save_visualizations : bool
        Whether to save visualizations of the mask
    
    Returns:
    --------
    str
        Path to the generated mask file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define mask dimensions
    shape = (128, 128, 128, 4)  # 128x128x128 volume with 4 channels (one-hot encoding)
    
    # Initialize mask with zeros
    mask = np.zeros(shape, dtype=np.float32)
    
    # Create background (class 0)
    # Set all voxels initially to background
    mask[:, :, :, 0] = 1.0
    
    # Create a non-enhancing tumor region (class 1)
    # A spherical region at center-right
    center = (64, 80, 64)
    radius = 10
    x, y, z = np.ogrid[:128, :128, :128]
    tumor_region = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
    mask[tumor_region, 0] = 0  # Remove background label
    mask[tumor_region, 1] = 1  # Set non-enhancing tumor
    
    # Create an edema region (class 2)
    # A larger spherical region surrounding the tumor
    center = (64, 80, 64)
    inner_radius = radius  # Same as tumor radius
    outer_radius = radius + 8  # Edema extends beyond the tumor
    x, y, z = np.ogrid[:128, :128, :128]
    edema_region = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= outer_radius**2) & \
                  ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 > inner_radius**2)
    mask[edema_region, 0] = 0  # Remove background label
    mask[edema_region, 2] = 1  # Set edema
    
    # Create an enhancing tumor region (class 3)
    # A small region within the non-enhancing tumor
    center = (64, 75, 64)  # Slightly offset from the tumor center
    radius = 5
    x, y, z = np.ogrid[:128, :128, :128]
    enhancing_region = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
    mask[enhancing_region, 0] = 0  # Remove background label
    mask[enhancing_region, 1] = 0  # Remove non-enhancing tumor label
    mask[enhancing_region, 3] = 1  # Set enhancing tumor
    
    # Save the synthetic mask
    mask_path = os.path.join(output_dir, "synthetic_all_classes.npy")
    np.save(mask_path, mask)
    print(f"Synthetic mask saved to {mask_path}")
    
    # Generate visualization if requested
    if save_visualizations:
        # Create a decoded version (integer labels)
        decoded_mask = np.argmax(mask, axis=-1)
        
        # Get a middle slice
        mid_slice = decoded_mask.shape[0] // 2
        
        # Create a visualization
        colors = ['black', 'red', 'green', 'blue']
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(decoded_mask[mid_slice], cmap=cmap, vmin=0, vmax=3)
        plt.title(f"Synthetic Mask - Axial Slice {mid_slice}")
        plt.colorbar(ticks=[0, 1, 2, 3], 
                    label='Tissue Class', 
                    orientation='vertical')
        plt.axis('off')
        
        vis_path = os.path.join(output_dir, "synthetic_mask_visualization.png")
        plt.savefig(vis_path, dpi=150)
        plt.close()
        print(f"Visualization saved to {vis_path}")
        
        # Print class distribution
        print("\nClass distribution in synthetic mask:")
        for i in range(4):
            non_zeros = np.count_nonzero(mask[..., i])
            percentage = (non_zeros / (mask.shape[0] * mask.shape[1] * mask.shape[2])) * 100
            class_names = ['Background', 'Non-enhancing Tumor', 'Edema', 'Enhancing Tumor']
            print(f"  Class {i} ({class_names[i]}): {non_zeros} voxels ({percentage:.2f}%)")
    
    return mask_path

if __name__ == "__main__":
    # Generate the synthetic mask
    mask_path = generate_synthetic_mask()
    
    # Now use the view_mask_slice script to view it
    try:
        print("\nLaunching view_mask_slice.py to show the synthetic mask...\n")
        import subprocess
        subprocess.run(["python", "view_mask_slice.py", mask_path])
    except Exception as e:
        print(f"Failed to launch view_mask_slice.py: {e}")
        print("You can view the mask manually by running:")
        print(f"python view_mask_slice.py {mask_path}")