import numpy as np
import os

image_dir = "../../MET-data/input_data/images"
mask_dir = "../../MET-data/input_data/masks"

for file in sorted(os.listdir(image_dir)):
    if file.endswith(".npy"):
        try:
            data = np.load(os.path.join(image_dir, file))
            print(f"Loaded {file}: shape={data.shape}, dtype={data.dtype}")
        except Exception as e:
            print(f"Error loading {file}: {e}")


for file in sorted(os.listdir(mask_dir)):
    if file.endswith(".npy"):
        try:
            data = np.load(os.path.join(mask_dir, file))
            print(f"Loaded {file}: shape={data.shape}, dtype={data.dtype}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
