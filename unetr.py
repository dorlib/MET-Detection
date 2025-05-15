#!/usr/bin/env python3
# unetr.py
# ----------------------------------
# 3-D UNETR training on .npy volumes
# ----------------------------------

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_3D as sm
from self_attention_cv import UNETR
import matplotlib.pyplot as plt  # noqa: F401  (kept for future plots)

# ──────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────
BASE_DIR = "../Data/"
TRAIN_IMG_DIR  = os.path.join(BASE_DIR, "input_data_128/train/images/")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "input_data_128/train/masks/")
VAL_IMG_DIR    = os.path.join(BASE_DIR, "input_data_128/val/images/")
VAL_MASK_DIR   = os.path.join(BASE_DIR, "input_data_128/val/masks/")

DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")	
print("DEVICE", DEVICE)
BATCH_SIZE   = 4
NUM_CLASSES  = 4
NUM_EPOCHS   = 100
LR           = 1e-4

# ──────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────
class NpyImageDataset(Dataset):
    """
    Expects .npy files.
      • Images  : shape (D, H, W, C)
      • Masks   : either
          a) class indices (D, H, W)                int
          b) one-hot      (D, H, W, C)              int/float
    Returns Tensors:
      img  → (C, D, H, W) float32
      mask → (D, H, W)     int64      (class indices)
    """
    def __init__(self, img_dir, img_list, mask_dir, mask_list):
        self.img_dir   = img_dir
        self.img_list  = img_list
        self.mask_dir  = mask_dir
        self.mask_list = mask_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img  = np.load(os.path.join(self.img_dir,  self.img_list[idx])).astype(np.float32)
        mask = np.load(os.path.join(self.mask_dir, self.mask_list[idx]))

        # collapse one-hot mask to class indices, if needed
        if mask.ndim == 4 and mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)           # (D,H,W)

        img  = torch.tensor(img).permute(3, 0, 1, 2)  # (C,D,H,W)
        mask = torch.tensor(mask, dtype=torch.long)   # (D,H,W)
        return img, mask


def sanity_prints():
    train_imgs  = sorted(os.listdir(TRAIN_IMG_DIR))
    train_masks = sorted(os.listdir(TRAIN_MASK_DIR))
    val_imgs    = sorted(os.listdir(VAL_IMG_DIR))
    val_masks   = sorted(os.listdir(VAL_MASK_DIR))

    print(f"Training images : {len(train_imgs)}")
    print(f"Training masks  : {len(train_masks)}")
    print(f"Validation imgs : {len(val_imgs)}")
    print(f"Validation masks: {len(val_masks)}")

    ds = NpyImageDataset(TRAIN_IMG_DIR, train_imgs, TRAIN_MASK_DIR, train_masks)
    x, y = ds[0]
    print(f"Sample img shape  {x.shape}, dtype {x.dtype}")
    print(f"Sample mask shape {y.shape}, dtype {y.dtype}, classes {torch.unique(y)}")


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────
def build_unetr():
    model = UNETR(
        img_shape=(128, 128, 128),
        input_dim=3,
        output_dim=NUM_CLASSES,
        embed_dim=128,
        patch_size=16,
        num_heads=4,
        ext_layers=[3, 6, 9, 12],
        norm='instance',
        base_filters=16,
        dim_linear_block=1024
    ).to(DEVICE)

    params_m = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"UNETR has {params_m:.2f} M parameters")
    return model


# ──────────────────────────────────────────────────────────────
# Class-balance loss
# ──────────────────────────────────────────────────────────────
def weighted_ce_loss():
    cols = [str(i) for i in range(NUM_CLASSES)]
    df   = pd.DataFrame(columns=cols)

    for p in sorted(glob.glob(os.path.join(TRAIN_MASK_DIR, "*.npy"))):
        m = np.load(p)
        if m.ndim == 4:                       # one-hot on disk
            m = np.argmax(m, axis=-1)
        vals, cnt = np.unique(m, return_counts=True)
        row = {c: 0 for c in cols}
        row.update(dict(zip(vals.astype(str), cnt)))
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    label_sum     = df.sum()
    total         = label_sum.sum()
    weights       = [total / (NUM_CLASSES * label_sum[str(i)]) for i in range(NUM_CLASSES)]
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    print(f"Class weights: {weights}")
    return nn.CrossEntropyLoss(weight=weights_tensor)


# ──────────────────────────────────────────────────────────────
# Training & validation loops
# ──────────────────────────────────────────────────────────────
def standardise_masks(t):
    """
    Bring `t` to shape (B, D, H, W) int64, class indices.
    """
    if t.ndim == 5 and t.shape[-1] == NUM_CLASSES:           # one-hot last
        t = torch.argmax(t, dim=-1)
    elif t.ndim == 5 and t.shape[1] == 1:                    # (B,1,D,H,W)
        t = t.squeeze(1)
    return t.long()


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = standardise_masks(y).to(DEVICE)

        logits = model(x)
        loss   = criterion(logits, y)
        running += loss.item()
    return running / len(loader)


def train(model, train_loader, val_loader, criterion):
    opt = optim.Adam(model.parameters(), lr=LR)
    print("Starting training…", flush=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        run_loss = 0.0

        for bidx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = standardise_masks(y).to(DEVICE)

            # ─ Diagnostics: first epoch, first two batches ──
            if epoch == 0 and bidx < 2:
                print(f"\nEpoch 0 Batch {bidx}")
                print(f"inputs : {x.shape} {x.dtype}")
                print(f"targets: {y.shape} {y.dtype}  uniq={torch.unique(y)[:10]}")

            opt.zero_grad()
            logits = model(x)                 # (B,C,D,H,W)
            loss   = criterion(logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item()

        train_loss = run_loss / len(train_loader)
        val_loss   = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS}  "
              f"train loss {train_loss:.4f}  val loss {val_loss:.4f}")

    torch.save(model.state_dict(),
               os.path.join(BASE_DIR, "saved_models/brats_3d.pth"))


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    # file lists
    train_imgs  = sorted(os.listdir(TRAIN_IMG_DIR))
    train_masks = sorted(os.listdir(TRAIN_MASK_DIR))
    val_imgs    = sorted(os.listdir(VAL_IMG_DIR))
    val_masks   = sorted(os.listdir(VAL_MASK_DIR))

    sanity_prints()

    # datasets / loaders
    train_ds = NpyImageDataset(TRAIN_IMG_DIR, train_imgs, TRAIN_MASK_DIR, train_masks)
    val_ds   = NpyImageDataset(VAL_IMG_DIR,   val_imgs,   VAL_MASK_DIR,   val_masks)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # model, loss, train
    model     = build_unetr()
    criterion = weighted_ce_loss()
    train(model, train_ld, val_ld, criterion)


if __name__ == "__main__":
    main()

