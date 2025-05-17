#!/usr/bin/env python3
# unetr.py - Train UNETR on 3D .npy Volumes (Training One Layer Only)

import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to auto-close

# ────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────
BASE_DIR = "../Data/"
TRAIN_IMG_DIR  = os.path.join(BASE_DIR, "input_data_128/train/images/")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "input_data_128/train/masks/")
OUTPUT_DIR     = os.path.join(BASE_DIR, "generated_comparisons/")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LAYER_NAMES  = ["FLAIR", "T1CE", "T2"]

# ────────────────────────────────────────────────────
# Multi-Axis Alignment Check
# ────────────────────────────────────────────────────
def check_alignment_all_axes(image_path, mask_path):
    print("Image path:", image_path)
    print("Mask path :", mask_path)

    img  = np.load(image_path)  # (D, H, W, 3)
    mask = np.load(mask_path)   # (D, H, W) or (D, H, W, C)

    if mask.ndim == 4:
        mask = np.argmax(mask, axis=-1)

    flair = img[..., 0]
    t1ce = img[..., 1]
    t2   = img[..., 2]

    d, h, w = t1ce.shape
    d_mid, h_mid, w_mid = d // 2, h // 2, w // 2

    fig, axs = plt.subplots(4, 3, figsize=(24, 16), dpi=200)

    axs[0][0].imshow(flair[d_mid])
    axs[0][0].set_title("FLAIR Axial (Z)")
    axs[0][1].imshow(flair[:, h_mid, :])
    axs[0][1].set_title("FLAIR Coronal (Y)")
    axs[0][2].imshow(flair[:, :, w_mid])
    axs[0][2].set_title("FLAIR Sagittal (X)")

    axs[1][0].imshow(t1ce[d_mid])
    axs[1][0].set_title("T1CE Axial (Z)")
    axs[1][1].imshow(t1ce[:, h_mid, :])
    axs[1][1].set_title("T1CE Coronal (Y)")
    axs[1][2].imshow(t1ce[:, :, w_mid])
    axs[1][2].set_title("T1CE Sagittal (X)")

    axs[2][0].imshow(t2[d_mid])
    axs[2][0].set_title("T2 Axial (Z)")
    axs[2][1].imshow(t2[:, h_mid, :])
    axs[2][1].set_title("T2 Coronal (Y)")
    axs[2][2].imshow(t2[:, :, w_mid])
    axs[2][2].set_title("T2 Sagittal (X)")

    axs[3][0].imshow(mask[d_mid], cmap="viridis")
    axs[3][0].set_title("Mask Axial (Z)")
    axs[3][1].imshow(mask[:, h_mid, :], cmap="viridis")
    axs[3][1].set_title("Mask Coronal (Y)")
    axs[3][2].imshow(mask[:, :, w_mid], cmap="viridis")
    axs[3][2].set_title("Mask Sagittal (X)")

    for row in axs:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    filename = os.path.basename(image_path).replace(".npy", "_axes_check_all_scans.png")
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    time.sleep(1)

# ────────────────────────────────────────────────────
# Run Comparison Loop
# ────────────────────────────────────────────────────
if __name__ == "__main__":
    image_list = sorted(os.listdir(TRAIN_IMG_DIR))
    mask_list  = sorted(os.listdir(TRAIN_MASK_DIR))

    for image_file, mask_file in zip(image_list, mask_list):
        image_path = os.path.join(TRAIN_IMG_DIR, image_file)
        mask_path  = os.path.join(TRAIN_MASK_DIR, mask_file)
        check_alignment_all_axes(image_path, mask_path)

