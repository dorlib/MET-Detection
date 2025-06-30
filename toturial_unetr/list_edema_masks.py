#!/usr/bin/env python3
"""
Script to list masks that have non-zero voxels in class 2 (edema) and class 3 (enhancing tumor)
and provide their indices for use with view_mask.py
"""

import os
import numpy as np
import glob

# Define the mask directory
MASK_DIR = '../../MET-data/input_data/masks/'

def list_masks_with_classes():
    """List masks with non-zero voxels in classes 2 and 3"""
    print(f"Searching for masks in: {MASK_DIR}")
    
    # Find all mask files
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, '*.npy')))
    if not mask_files:
        print(f"No mask files found in {MASK_DIR}")
        return
    
    print(f"Found {len(mask_files)} mask files. Analyzing...")
    
    # Initialize result lists
    class2_masks = []  # For edema
    class3_masks = []  # For enhancing tumor
    
    # Process each mask
    for i, mask_path in enumerate(mask_files):
        try:
            # Load mask
            mask = np.load(mask_path)
            
            # Check if one-hot encoded with at least 4 channels
            if mask.ndim == 4 and mask.shape[-1] >= 4:
                # Calculate percentage for class 2 (edema)
                class2_voxels = np.count_nonzero(mask[..., 2])
                total_voxels = mask.shape[0] * mask.shape[1] * mask.shape[2]
                class2_percentage = (class2_voxels / total_voxels) * 100
                
                # Calculate percentage for class 3 (enhancing tumor)
                class3_voxels = np.count_nonzero(mask[..., 3])
                class3_percentage = (class3_voxels / total_voxels) * 100
                
                # Store masks with non-zero voxels
                if class2_voxels > 0:
                    class2_masks.append({
                        'index': i,
                        'filename': os.path.basename(mask_path),
                        'path': mask_path,
                        'percentage': class2_percentage
                    })
                
                if class3_voxels > 0:
                    class3_masks.append({
                        'index': i,
                        'filename': os.path.basename(mask_path),
                        'path': mask_path,
                        'percentage': class3_percentage
                    })
        except Exception as e:
            print(f"Error processing {mask_path}: {str(e)}")
    
    # Sort by percentage (descending)
    class2_masks.sort(key=lambda x: x['percentage'], reverse=True)
    class3_masks.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Print results for Class 2 (Edema)
    print("\n=== Class 2 (Edema) ===")
    print(f"Found {len(class2_masks)} masks containing edema.")
    if class2_masks:
        print("\nTop 10 masks with highest edema percentage:")
        for i, mask in enumerate(class2_masks[:10]):
            print(f"  {i+1}. Index: {mask['index']} - {mask['filename']} - {mask['percentage']:.2f}%")
            
        print("\nTo view these masks with view_mask_slice.py, use:")
        for i, mask in enumerate(class2_masks[:5]):
            print(f"  python view_mask_slice.py {mask['filename']}")
            
        print("\nTo use these specific indices with view_mask.py:")
        for i, mask in enumerate(class2_masks[:5]):
            print(f"  python view_mask.py {mask['index']}")
    
    # Print results for Class 3 (Enhancing Tumor)
    print("\n=== Class 3 (Enhancing Tumor) ===")
    print(f"Found {len(class3_masks)} masks containing enhancing tumor.")
    if class3_masks:
        print("\nTop 10 masks with highest enhancing tumor percentage:")
        for i, mask in enumerate(class3_masks[:10]):
            print(f"  {i+1}. Index: {mask['index']} - {mask['filename']} - {mask['percentage']:.2f}%")
            
        print("\nTo view these masks with view_mask_slice.py, use:")
        for i, mask in enumerate(class3_masks[:5]):
            print(f"  python view_mask_slice.py {mask['filename']}")
            
        print("\nTo use these specific indices with view_mask.py:")
        for i, mask in enumerate(class3_masks[:5]):
            print(f"  python view_mask.py {mask['index']}")
    
    # Find masks that have both classes
    both_classes = []
    filenames_with_class2 = {mask['filename']: mask for mask in class2_masks}
    for mask in class3_masks:
        if mask['filename'] in filenames_with_class2:
            both_classes.append({
                'filename': mask['filename'],
                'index': mask['index'],
                'class2_percentage': filenames_with_class2[mask['filename']]['percentage'],
                'class3_percentage': mask['percentage'],
                'total_percentage': filenames_with_class2[mask['filename']]['percentage'] + mask['percentage']
            })
    
    # Sort by total percentage
    both_classes.sort(key=lambda x: x['total_percentage'], reverse=True)
    
    print("\n=== Masks with both Edema and Enhancing Tumor ===")
    print(f"Found {len(both_classes)} masks containing both classes.")
    if both_classes:
        print("\nTop 10 masks with highest combined percentage:")
        for i, mask in enumerate(both_classes[:10]):
            print(f"  {i+1}. Index: {mask['index']} - {mask['filename']} - "
                  f"Edema: {mask['class2_percentage']:.2f}%, "
                  f"Enhancing: {mask['class3_percentage']:.2f}%, "
                  f"Total: {mask['total_percentage']:.2f}%")
            
        print("\nTo view these masks with view_mask_slice.py, use:")
        for i, mask in enumerate(both_classes[:5]):
            print(f"  python view_mask_slice.py {mask['filename']}")
            
        print("\nTo use these specific indices with view_mask.py:")
        for i, mask in enumerate(both_classes[:5]):
            print(f"  python view_mask.py {mask['index']}")

if __name__ == "__main__":
    list_masks_with_classes()