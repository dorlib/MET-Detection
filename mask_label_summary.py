# mask_label_summary.py
# Iterate through .npy mask volumes, report label counts, and generate color-coded mask images

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Directories
MASK_DIR    = "MET_samples/masks"
OUTPUT_CSV  = "MET_samples/label_summary.csv"
OUTPUT_IMG_DIR  = "MET_samples/binary_images"

# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Define colormap: 0 -> blue, 1 -> red
cmap = ListedColormap(['blue', 'red'])

# Collect summary rows
df_rows = []

# Process each mask file
for fname in sorted(os.listdir(MASK_DIR)):
    if not fname.endswith('.npy'):
        continue
    filepath = os.path.join(MASK_DIR, fname)
    mask_vol = np.load(filepath)

    # Count labels and record
    labels, counts = np.unique(mask_vol, return_counts=True)
    for lbl, cnt in zip(labels, counts):
        df_rows.append({'filename': fname, 'label': int(lbl), 'count': int(cnt)})

    # Generate a 2D color-coded image from the central slice
    depth = mask_vol.shape[0]
    mid_slice = mask_vol[depth // 2, :, :]
    # Ensure only 0 and 1 values
    binary_slice = np.where(mid_slice > 0, 1, 0).astype(np.uint8)

    # Save color-coded mask image
    img_name = fname.replace('.npy', '_slice.png')
    img_path = os.path.join(OUTPUT_IMG_DIR, img_name)
    plt.imsave(img_path, binary_slice, cmap=cmap)
    print(f"Saved color-coded slice image: {img_path}")

# Build DataFrame and save CSV
df = pd.DataFrame(df_rows, columns=['filename', 'label', 'count'])
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved label summary CSV: {OUTPUT_CSV}")

# Print summaries
print("\nPer-file label counts:")
for fname, group in df.groupby('filename'):
    print(f"File: {fname}")
    for _, row in group.sort_values('count', ascending=False).iterrows():
        print(f"  Label {row['label']}: {row['count']} voxels")
    print()

print("Overall label frequencies:")
for lbl, total in df.groupby('label')['count'].sum().sort_values(ascending=False).items():
    print(f"Label {lbl}: {total} voxels total")

