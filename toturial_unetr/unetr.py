#!/usr/bin/env python3
# unetr.py - 3D UNETR training, prediction, and test plotting

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
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
BASE_DIR            = "../Data/"
TRAIN_IMG_DIR       = os.path.join(BASE_DIR, "input_data_128/train/images/")
TRAIN_MASK_DIR      = os.path.join(BASE_DIR, "input_data_128/train/masks/")
VAL_IMG_DIR         = os.path.join(BASE_DIR, "input_data_128/val/images/")
VAL_MASK_DIR        = os.path.join(BASE_DIR, "input_data_128/val/masks/")
TEST_IMG_DIR        = os.path.join(BASE_DIR, "input_data_128/test/images/")
TEST_MASK_DIR       = os.path.join(BASE_DIR, "input_data_128/test/masks/")
SAVED_MODEL_PATH    = os.path.join(BASE_DIR, "saved_models/brats_3d.pth")

DEVICE              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE          = 4
NUM_CLASSES         = 4
NUM_EPOCHS          = 100
LR                  = 1e-4

# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class NpyImageDataset(Dataset):
    def __init__(self, img_dir, img_list, mask_dir, mask_list):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir, self.img_list[idx])).astype(np.float32)
        mask = np.load(os.path.join(self.mask_dir, self.mask_list[idx]))
        if mask.ndim == 4 and mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)
        img = torch.tensor(img).permute(3, 0, 1, 2)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask

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
# Loss
# ──────────────────────────────────────────────────────────────
def weighted_ce_loss():
    cols = [str(i) for i in range(NUM_CLASSES)]
    df = pd.DataFrame(columns=cols)
    for p in sorted(glob.glob(os.path.join(TRAIN_MASK_DIR, "*.npy"))):
        m = np.load(p)
        if m.ndim == 4:
            m = np.argmax(m, axis=-1)
        vals, cnt = np.unique(m, return_counts=True)
        row = {c: 0 for c in cols}
        row.update(dict(zip(vals.astype(str), cnt)))
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    label_sum = df.sum()
    total = label_sum.sum()
    weights = [total / (NUM_CLASSES * label_sum[str(i)]) for i in range(NUM_CLASSES)]
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))

# ──────────────────────────────────────────────────────────────
# Training and Evaluation
# ──────────────────────────────────────────────────────────────

def standardise_masks(t):
    if t.ndim == 5 and t.shape[-1] == NUM_CLASSES:
        t = torch.argmax(t, dim=-1)
    elif t.ndim == 5 and t.shape[1] == 1:
        t = t.squeeze(1)
    return t.long()


def compute_metrics(preds, targets, num_classes):
    preds = preds.flatten()
    targets = targets.flatten()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    acc = correct / total
    recall = 0
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        act = (targets == cls).sum().item()
        recall += (tp / act) if act != 0 else 0
    recall /= num_classes
    return acc, recall


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = total_acc = total_recall = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = standardise_masks(y).to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        acc, recall = compute_metrics(preds.cpu(), y.cpu(), NUM_CLASSES)
        total_acc += acc
        total_recall += recall
    size = len(loader)
    return running_loss/size, total_acc/size, total_recall/size


def train(model, train_loader, val_loader, criterion):
    opt = optim.Adam(model.parameters(), lr=LR)
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_recalls = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = standardise_masks(y).to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        train_loss = run_loss / len(train_loader)
        val_loss, val_acc, val_recall = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_recalls.append(val_recall)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  acc {val_acc:.4f}  recall {val_recall:.4f}", flush=True)
    print("finished training, saving model...")
    os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
    print("Finished training and saved model.")
    plot_metrics(val_losses, val_recalls, val_accuracies)

# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────
def plot_metrics(losses, recalls, accuracies, title="Training Metrics"):
    epochs = range(1, len(losses)+1)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.plot(epochs, losses, 'b-', label='Loss')
    ax2.plot(epochs, recalls, 'r-', label='Recall')
    ax3.plot(epochs, accuracies, 'g-', label='Accuracy')
    for ax, lbl in zip((ax1,ax2,ax3), ('Loss','Recall','Accuracy')):
        ax.set_xlabel('Epoch')
        ax.set_title(f'{lbl}')
        ax.grid(True)
        ax.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# ──────────────────────────────────────────────────────────────
# Testing and Plotting
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def test_and_plot(model, loader, device, output_dir="test_results"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = standardise_masks(y).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        masks = y.cpu().numpy()
        images = x.cpu().numpy()  # shape [B, C, D, H, W]
        depth = preds.shape[1]
        mid = depth // 2

        for i in range(preds.shape[0]):
            # Extract mid-plane slices
            orig_vol = images[i]  # (C, D, H, W)
            orig_slice = np.transpose(orig_vol[:, mid, :, :], (1, 2, 0))
            pred_slice = preds[i, mid]
            mask_slice = masks[i, mid]

            fname = loader.dataset.img_list[batch_idx * loader.batch_size + i]

            fig, axs = plt.subplots(1, 3, figsize=(18, 8), dpi=300)
            # Original image
            axs[0].imshow(orig_slice, interpolation='nearest')
            axs[0].set_title('Original', fontsize=12)
            # Prediction
            axs[1].imshow(pred_slice, interpolation='nearest')
            axs[1].set_title('Prediction', fontsize=12)
            # Ground truth mask
            axs[2].imshow(mask_slice, interpolation='nearest')
            axs[2].set_title('Ground Truth', fontsize=12)

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_slice.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)

    print(f"Saved test results in {output_dir}")

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    # Prepare datasets
    train_imgs = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.npy')])
    train_masks = sorted([f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.npy')])
    val_imgs   = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.npy')])
    val_masks  = sorted([f for f in os.listdir(VAL_MASK_DIR) if f.endswith('.npy')])
    train_ds = NpyImageDataset(TRAIN_IMG_DIR, train_imgs, TRAIN_MASK_DIR, train_masks)
    val_ds   = NpyImageDataset(VAL_IMG_DIR, val_imgs, VAL_MASK_DIR, val_masks)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Build and (optionally) train model
    model = build_unetr()
    criterion = weighted_ce_loss()
    if os.path.exists(SAVED_MODEL_PATH):
        model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
        print("Loaded existing model, skipping training.")
    else:
        train(model, train_ld, val_ld, criterion)

    # Run predictions and plotting on test set
    test_imgs  = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.npy')])
    test_masks = sorted([f for f in os.listdir(TEST_MASK_DIR) if f.endswith('.npy')])
    test_ds = NpyImageDataset(TEST_IMG_DIR, test_imgs, TEST_MASK_DIR, test_masks)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_and_plot(model, test_ld, DEVICE)

if __name__ == "__main__":
    main()

