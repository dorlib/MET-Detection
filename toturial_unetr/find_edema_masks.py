#!/usr/bin/env python3
"""
Script to find mask files containing edema (class 2) or enhancing tumor (class 3)
in the mask directory and display statistics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from glob import glob

def find_masks_with_classes(mask_dir, target_classes=[2, 3], top_n=5):
    """
    Find mask files containing specific classes and analyze their distribution.
    
    Parameters:
    -----------
    mask_dir : str
        Directory containing mask .npy files
    target_classes : list
        List of class indices to search for (default: [2, 3] for edema and enhancing tumor)
    top_n : int
        Number of top masks to return for each class
        
    Returns:
    --------
    dict
        Dictionary with statistics and top masks for each target class
    """
    print(f"Searching for masks in: {mask_dir}")
    
    # Find all mask files
    mask_files = sorted(glob(os.path.join(mask_dir, '*.npy')))
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return None
        
    print(f"Found {len(mask_files)} mask files. Analyzing...")
    
    # Initialize statistics
    result = {c: {'masks': [], 'percentages': []} for c in target_classes}
    
    # Analyze each mask
    for i, mask_path in enumerate(mask_files):
        if i % 10 == 0:  # Print progress every 10 files
            print(f"Processing file {i+1}/{len(mask_files)}: {os.path.basename(mask_path)}")
        
        try:
            # Load mask
            mask = np.load(mask_path)
            
            # Check if one-hot encoded
            if mask.ndim == 4 and mask.shape[-1] >= max(target_classes) + 1:
                # Calculate percentage of each target class
                for class_idx in target_classes:
                    non_zeros = np.count_nonzero(mask[..., class_idx])
                    total = mask.shape[0] * mask.shape[1] * mask.shape[2]
                    percentage = (non_zeros / total) * 100
                    
                    if percentage > 0:  # Only track masks with non-zero presence
                        result[class_idx]['masks'].append(mask_path)
                        result[class_idx]['percentages'].append(percentage)
        except Exception as e:
            print(f"Error processing {mask_path}: {str(e)}")
    
    # Sort and find top masks for each class
    for class_idx in target_classes:
        if result[class_idx]['masks']:
            # Sort by percentage (descending)
            sorted_indices = np.argsort(result[class_idx]['percentages'])[::-1]
            
            # Get top N masks
            top_masks = [result[class_idx]['masks'][i] for i in sorted_indices[:top_n]]
            top_percentages = [result[class_idx]['percentages'][i] for i in sorted_indices[:top_n]]
            
            class_names = {0: 'Background', 1: 'Non-enhancing Tumor', 
                         2: 'Edema', 3: 'Enhancing Tumor'}
            class_name = class_names.get(class_idx, f'Class {class_idx}')
            
            print(f"\nFound {len(result[class_idx]['masks'])} masks containing {class_name} (class {class_idx})")
            print(f"Top {min(top_n, len(top_masks))} masks with highest {class_name} percentage:")
            
            for j, (mask_path, percentage) in enumerate(zip(top_masks, top_percentages)):
                print(f"  {j+1}. {os.path.basename(mask_path)}: {percentage:.2f}%")
                
            result[class_idx]['top_masks'] = top_masks
            result[class_idx]['top_percentages'] = top_percentages
        else:
            print(f"\nNo masks found containing class {class_idx}")
    
    return result

def visualize_mask_with_class(mask_path, class_idx, output_dir="./mask_class_visualizations"):
    """
    Visualize a mask focusing on a specific class.
    
    Parameters:
    -----------
    mask_path : str
        Path to the mask file
    class_idx : int
        Index of the class to focus on
    output_dir : str
        Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mask
    mask = np.load(mask_path)
    filename = os.path.splitext(os.path.basename(mask_path))[0]
    
    # Check if one-hot encoded
    if not (mask.ndim == 4 and mask.shape[-1] > class_idx):
        print(f"Mask is not in expected format or doesn't contain class {class_idx}")
        return
        
    # Convert to class indices
    decoded_mask = np.argmax(mask, axis=-1)
    
    # Create highlight mask (binary mask for the target class)
    highlight_mask = (decoded_mask == class_idx)
    
    # Get middle slices for each axis
    mid_slices = [mask.shape[i] // 2 for i in range(3)]
    
    # Find a good slice with the target class
    # Count the presence of class in each slice along each axis
    counts = []
    for axis in range(3):
        axis_counts = []
        for i in range(mask.shape[axis]):
            if axis == 0:
                slice_count = np.sum(decoded_mask[i, :, :] == class_idx)
            elif axis == 1:
                slice_count = np.sum(decoded_mask[:, i, :] == class_idx)
            else:
                slice_count = np.sum(decoded_mask[:, :, i] == class_idx)
            axis_counts.append(slice_count)
        counts.append(axis_counts)
    
    # Get the slice with maximum class presence for each axis
    best_slices = [np.argmax(counts[i]) for i in range(3)]
    
    # Create figure with 6 subplots: 3 full segmentation, 3 class highlight
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define class colors and names
    colors = ['black', 'red', 'green', 'blue']
    cmap = ListedColormap(colors[:mask.shape[-1]])
    class_names = {0: 'Background', 1: 'Non-enhancing Tumor', 
                 2: 'Edema', 3: 'Enhancing Tumor'}
    class_name = class_names.get(class_idx, f'Class {class_idx}')
    
    # Get the best slice for each view
    views = ['axial', 'coronal', 'sagittal']
    
    for i, (axis, view) in enumerate(zip(range(3), views)):
        # Get the best slice for this axis
        slice_idx = best_slices[axis]
        
        # Full segmentation
        if axis == 0:  # axial
            full_slice = decoded_mask[slice_idx, :, :]
            class_slice = mask[slice_idx, :, :, class_idx]
        elif axis == 1:  # coronal
            full_slice = decoded_mask[:, slice_idx, :]
            class_slice = mask[:, slice_idx, :, class_idx]
        else:  # sagittal
            full_slice = decoded_mask[:, :, slice_idx]
            class_slice = mask[:, :, slice_idx, class_idx]
        
        # Plot full segmentation
        im1 = axes[0, i].imshow(full_slice, cmap=cmap, vmin=0, vmax=len(colors)-1)
        axes[0, i].set_title(f'{view.capitalize()} Slice {slice_idx} - All Classes')
        axes[0, i].axis('off')
        
        # Plot class highlight
        axes[1, i].imshow(class_slice, cmap='hot')
        axes[1, i].set_title(f'{view.capitalize()} Slice {slice_idx} - {class_name} Only')
        axes[1, i].axis('off')
    
    # Add colorbar for the full segmentation
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_ticks(range(len(colors)))
    cbar.set_ticklabels([class_names.get(i, f'Class {i}') for i in range(len(colors))])
    
    # Add overall title
    class_percentage = (np.sum(mask[..., class_idx]) / (mask.shape[0] * mask.shape[1] * mask.shape[2])) * 100
    fig.suptitle(f'Mask: {filename}\n{class_name} Coverage: {class_percentage:.2f}%', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.9)
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{filename}_class{class_idx}_{class_name.replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization to: {output_path}")
    
    # Get class statistics
    for i, cls in enumerate(range(mask.shape[-1])):
        non_zeros = np.count_nonzero(mask[..., cls])
        total = mask.shape[0] * mask.shape[1] * mask.shape[2]
        percentage = (non_zeros / total) * 100
        print(f"  Class {cls} ({class_names.get(cls, f'Class {cls}')}): {non_zeros} voxels ({percentage:.2f}%)")

def main():
    # Define mask directory - using the path from user input
    mask_dir = '../../MET-data/input_data/masks/'
    
    print(f"Looking for masks in relative path: {mask_dir}")
    # Also try an absolute path
    import os
    abs_mask_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), mask_dir))
    print(f"Absolute path: {abs_mask_dir}")
    
    # Check if directories exist
    if not os.path.isdir(mask_dir):
        print(f"Directory not found: {mask_dir}")
    else:
        print(f"Directory exists: {mask_dir}")
        # List files in directory
        files = os.listdir(mask_dir)
        print(f"Found {len(files)} files in directory")
        print("First 5 files:", files[:5] if len(files) > 5 else files)
    
    # Try the absolute path if relative path doesn't work
    if not os.path.isdir(mask_dir):
        if os.path.isdir(abs_mask_dir):
            print(f"Using absolute path instead: {abs_mask_dir}")
            mask_dir = abs_mask_dir
        else:
            print(f"Absolute path also not found: {abs_mask_dir}")
            # Ask the user where the masks might be
            print("\nCouldn't find mask directory. Let's check for npy files in other locations:")
            
            # Try to find mask files in other potential locations
            potential_dirs = [
                '../../MET-data/t1c_only/masks/',
                '../MET-data/input_data/masks/',
                '../MET-data/masks/',
                './masks/',
                '../masks/'
            ]
            
            for search_dir in potential_dirs:
                abs_search_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), search_dir))
                if os.path.isdir(abs_search_dir):
                    print(f"Found potential mask directory: {search_dir} ({abs_search_dir})")
                    mask_files = [f for f in os.listdir(abs_search_dir) if f.endswith('.npy')]
                    if mask_files:
                        print(f"  Contains {len(mask_files)} .npy files")
                        print(f"  First few: {mask_files[:3] if len(mask_files) > 3 else mask_files}")
                        mask_dir = abs_search_dir
                        break
            
    # Try to sample one mask and see what's in it
    try:
        import glob
        import numpy as np
        
        # Find any npy files
        mask_files = glob.glob(os.path.join(mask_dir, '*.npy'))
        
        if not mask_files:
            print(f"No .npy files found in {mask_dir}")
            # Try to find any npy files in the workspace
            search_path = os.path.join(os.path.dirname(__file__), '../..')
            all_npy_files = []
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.npy'):
                        all_npy_files.append(os.path.join(root, file))
                        if len(all_npy_files) >= 10:
                            break
            
            if all_npy_files:
                print(f"Found {len(all_npy_files)} .npy files elsewhere in the workspace.")
                print("First few:")
                for npy_file in all_npy_files[:5]:
                    print(f"  {npy_file}")
                    
                # Try to load one of these files to check format
                sample_file = all_npy_files[0]
                print(f"\nTrying to load sample file: {sample_file}")
                sample_mask = np.load(sample_file)
                print(f"Shape: {sample_mask.shape}")
                
                if sample_mask.ndim == 4:
                    print("This looks like a one-hot encoded mask.")
                    print(f"Channels (classes): {sample_mask.shape[-1]}")
                    print("Class distribution:")
                    for i in range(sample_mask.shape[-1]):
                        non_zeros = np.count_nonzero(sample_mask[..., i])
                        percentage = (non_zeros / (sample_mask.shape[0] * sample_mask.shape[1] * sample_mask.shape[2])) * 100
                        print(f"  Class {i}: {non_zeros} voxels ({percentage:.2f}%)")
                else:
                    print(f"This doesn't look like the expected one-hot encoded mask format.")
                    print(f"Unique values: {np.unique(sample_mask)}")
                
                # Set mask_dir to the directory of the first npy file
                mask_dir = os.path.dirname(sample_file)
                print(f"\nUsing this directory as mask_dir: {mask_dir}")
                
                # Try to run the search again with this new path
                print("\nRetrying search with new path...")
                result = find_masks_with_classes(mask_dir, target_classes=[2, 3], top_n=3)
                if result:
                    for class_idx in result:
                        if 'top_masks' in result[class_idx] and result[class_idx]['top_masks']:
                            print(f"\nVisualizing top masks for class {class_idx}:")
                            for i, mask_path in enumerate(result[class_idx]['top_masks']):
                                print(f"  {i+1}. {os.path.basename(mask_path)}")
                                visualize_mask_with_class(mask_path, class_idx)
                return
            else:
                print("No .npy files found anywhere in the workspace.")
                sys.exit(1)
        
        # If we have mask files, sample one to check
        sample_file = mask_files[0]
        print(f"\nAnalyzing sample mask file: {sample_file}")
        
        sample_mask = np.load(sample_file)
        print(f"Shape: {sample_mask.shape}")
        
        if sample_mask.ndim == 4:
            print(f"This is a one-hot encoded mask with {sample_mask.shape[-1]} classes.")
            print("Class distribution:")
            for i in range(sample_mask.shape[-1]):
                non_zeros = np.count_nonzero(sample_mask[..., i])
                percentage = (non_zeros / (sample_mask.shape[0] * sample_mask.shape[1] * sample_mask.shape[2])) * 100
                print(f"  Class {i}: {non_zeros} voxels ({percentage:.2f}%)")
                
            # Check if there's a problem with specific classes
            for i in range(sample_mask.shape[-1]):
                if np.count_nonzero(sample_mask[..., i]) == 0:
                    print(f"Class {i} has zero voxels. This might be a problem.")
        else:
            print("This doesn't look like a one-hot encoded mask.")
            print(f"Unique values: {np.unique(sample_mask)}")
            
    except Exception as e:
        print(f"Error analyzing mask file: {str(e)}")
    
    # Find masks containing edema (class 2) and enhancing tumor (class 3)
    result = find_masks_with_classes(mask_dir, target_classes=[2, 3], top_n=3)
    
    if not result:
        print("No suitable masks found.")
        print("\nThis could be because:")
        print("1. There genuinely are no masks with edema or enhancing tumor")
        print("2. The mask files might be in a different directory")
        print("3. The mask format might be different than expected")
        print("4. The class indices might be different in your dataset")
        
        print("\nHere's what to check:")
        print("1. Verify the path to your mask files")
        print("2. Check how the segmentation was done - were classes 2 and 3 included?")
        print("3. If your files have edema labeled differently (maybe as class 1?), update the search.")
        return
    
    # If we found masks, visualize them
    for class_idx in result:
        if 'top_masks' in result[class_idx] and result[class_idx]['top_masks']:
            print(f"\nVisualizing top masks for class {class_idx}:")
            
            for i, mask_path in enumerate(result[class_idx]['top_masks']):
                print(f"  {i+1}. {os.path.basename(mask_path)}")
                visualize_mask_with_class(mask_path, class_idx)

if __name__ == "__main__":
    main()