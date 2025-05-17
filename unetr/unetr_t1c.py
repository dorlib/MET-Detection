#!/usr/bin/env python3
# unetr_t1ce_training.py - 3D UNETR training on T1CE-only .npy volumes

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
BASE_DIR           = "../Data/"
TRAIN_IMG_DIR      = os.path.join(BASE_DIR, "input_data_128/train/images/")
TRAIN_MASK_DIR     = os.path.join(BASE_DIR, "input_data_128/train/masks/")
VAL_IMG_DIR        = os.path.join(BASE_DIR, "input_data_128/val/images/")
VAL_MASK_DIR       = os.path.join(BASE_DIR, "input_data_128/val/masks/")
TEST_IMG_DIR       = os.path.join(BASE_DIR, "input_data_128/test/images/")
TEST_MASK_DIR      = os.path.join(BASE_DIR, "input_data_128/test/masks/")
SAVED_MODEL_PATH   = os.path.join(BASE_DIR, "saved_models/brats_t1ce_1.pth")
BATCH_VIS_DIR      = os.path.join(BASE_DIR, "batch_visualizations")
GETITEM_VIS_DIR  = os.path.join(BASE_DIR, "getitem_visualizations")


DEVICE             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE         = 4
NUM_CLASSES        = 4
NUM_EPOCHS         = 100
LR                 = 1e-4


# ──────────────────────────────────────────────────────────────
# Utility: numeric sort key for consistent pairing
# ──────────────────────────────────────────────────────────────
def numeric_sort_key(fname: str) -> int:
    base = os.path.basename(fname)
    parts = base.split('_')
    if len(parts) >= 2:
        num = parts[1].split('.')[0]
        try:
            return int(num)
        except ValueError:
            return 0
    return 0
    
    
# ──────────────────────────────────────────────────────────────
# Dataset (T1CE-only) with filename tracking
# ──────────────────────────────────────────────────────────────
class NpyT1cImageDataset(Dataset):
    def __init__(self, img_dir, img_list, mask_dir, mask_list):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)
        
        
        # Assert matching numeric indices
        assert numeric_sort_key(img_name) == numeric_sort_key(mask_name), \
            f"Filename mismatch: {img_name} vs {mask_name}"

        img = np.load(img_path)
        if img.ndim == 4 and img.shape[-1] > 1:
            img = img[..., 1]  # select T1CE channel
        img = img[..., np.newaxis].astype(np.float32)

        mask = np.load(mask_path)
        if mask.ndim == 4 and mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)

        img_tensor = torch.tensor(img).permute(3, 2, 0, 1)  # (C, D, H, W)
        mask_tensor = torch.tensor(mask, dtype=torch.long)   # (D, H, W)
        

        # Visualize mid-slice and save
        os.makedirs(GETITEM_VIS_DIR, exist_ok=True)
        depth = img_tensor.shape[1]
        mid = depth // 2
        img_slice = img_tensor[0, mid].cpu().numpy()
        mask_slice = mask_tensor[mid].cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=150)
        axs[0].imshow(img_slice, interpolation='nearest')
        axs[0].set_title(f"Image: {img_name}"); axs[0].axis('off')
        axs[1].imshow(mask_slice, interpolation='nearest')
        axs[1].set_title(f"Mask: {mask_name}"); axs[1].axis('off')
        plt.tight_layout()
        save_name = f"{numeric_sort_key(img_name)}_{img_name.replace('.npy','')}_{mask_name}".replace('/', '_')
        fig.savefig(os.path.join(GETITEM_VIS_DIR, save_name + '.png'))
        plt.close(fig)
        
        return img_tensor, mask_tensor, img_name, mask_name

# ──────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────

def build_unetr():
    model = UNETR(
        img_shape=(128, 128, 128),
        input_dim=1,
        output_dim=NUM_CLASSES,
        embed_dim=128,
        patch_size=16,
        num_heads=4,
        ext_layers=[3, 6, 9, 12],
        norm='instance',
        base_filters=16,
        dim_linear_block=1024
    ).to(DEVICE)
    print(f"UNETR has {sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    return model

# ──────────────────────────────────────────────────────────────
# Loss computation
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
# Batch visualization
# ──────────────────────────────────────────────────────────────

def save_batch_visualization(x_batch, y_batch, epoch, batch_idx, img_names, mask_names):
    batch_dir = os.path.join(BATCH_VIS_DIR, f"epoch_{epoch}", f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)
    x_np = x_batch.cpu().numpy()
    y_np = y_batch.cpu().numpy()
    depth = x_np.shape[2]
    mid = depth // 2
    for i in range(x_np.shape[0]):
        img_slice = x_np[i, 0, mid, :, :]
        mask_slice = y_np[i, mid, :, :]
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
        axs[0].imshow(img_slice, cmap='gray'); axs[0].set_title(f"T1CE {img_names[i]}"); axs[0].axis('off')
        axs[1].imshow(mask_slice, cmap='gray'); axs[1].set_title(f"Mask {mask_names[i]}"); axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, f"{img_names[i].split('.')[0]}_{mask_names[i].split('.')[0]}.png"))
        plt.close(fig)

# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    loss_sum = acc_sum = 0.0
    for x, y, _, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        preds = torch.argmax(logits, dim=1)
        acc_sum += (preds == y).float().mean().item()
    N = len(loader)
    return loss_sum/N, acc_sum/N

# ──────────────────────────────────────────────────────────────
# Plot training curves
# ──────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, val_accs, output_path='training_curves.png'):
    epochs = range(1, len(train_losses)+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.plot(epochs, train_losses, '-', label='Train Loss')
    ax1.plot(epochs, val_losses, '-', label='Val Loss')
    ax1.set(title='Loss', xlabel='Epoch', ylabel='Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, val_accs, '-', label='Val Acc')
    ax2.set(title='Accuracy', xlabel='Epoch', ylabel='Acc')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, criterion):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    os.makedirs(BATCH_VIS_DIR, exist_ok=True)
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(NUM_EPOCHS):
        model.train(); epoch_loss = 0.0
        for batch_idx, (x, y, img_names, mask_names) in enumerate(train_loader):
            # log filenames
            print(f"Epoch {epoch} Batch {batch_idx} - Images: {img_names} Masks: {mask_names}")
            # visualize first epoch
            if epoch == 0:
                save_batch_visualization(x, y, epoch, batch_idx, img_names, mask_names)
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        t_loss = epoch_loss/len(train_loader)
        v_loss, v_acc = validate(model, val_loader, criterion)
        train_losses.append(t_loss); val_losses.append(v_loss); val_accs.append(v_acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train: {t_loss:.4f} Val: {v_loss:.4f} Acc: {v_acc:.4f}")
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
    plot_training_curves(train_losses, val_losses, val_accs)

# ──────────────────────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def test_and_plot(model, loader, device, output_dir="test_results_t1ce"):
    model.eval(); os.makedirs(output_dir, exist_ok=True)
    for b, (x,y,_,_) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        preds = torch.argmax(model(x), dim=1).cpu().numpy()
        vols = x.cpu().numpy(); masks=y.cpu().numpy()
        mid = preds.shape[1]//2
        for i in range(preds.shape[0]):
            oslice = np.transpose(vols[i,:,mid],(1,2,0))
            pslice = preds[i,mid]; mslice = masks[i,mid]
            fig,axs=plt.subplots(1,3,figsize=(18,8));
            axs[0].imshow(oslice,cmap='gray'); axs[0].set_title('Orig'); axs[0].axis('off')
            axs[1].imshow(pslice,cmap='gray'); axs[1].set_title('Pred'); axs[1].axis('off')
            axs[2].imshow(mslice,cmap='gray'); axs[2].set_title('GT'); axs[2].axis('off')
            plt.tight_layout(); plt.savefig(os.path.join(output_dir,f"{b}_{i}.png")); plt.close()
    print(f"Saved test results in {output_dir}")

# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Prepare data loaders
    train_imgs = sorted(f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.npy'))
    train_masks = sorted(f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.npy'))
    val_imgs = sorted(f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.npy'))
    val_masks = sorted(f for f in os.listdir(VAL_MASK_DIR) if f.endswith('.npy'))
    train_ds = NpyT1cImageDataset(TRAIN_IMG_DIR,train_imgs,TRAIN_MASK_DIR,train_masks)
    val_ds   = NpyT1cImageDataset(VAL_IMG_DIR,val_imgs,VAL_MASK_DIR,val_masks)
    train_ld = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    val_ld   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)

    model = build_unetr(); criterion = weighted_ce_loss()
    if os.path.exists(SAVED_MODEL_PATH):
        model.load_state_dict(torch.load(SAVED_MODEL_PATH,map_location=DEVICE))
        print("Loaded existing model, skipping training.")
    else:
        train(model,train_ld,val_ld,criterion)

    # Always run testing visualization
    test_imgs = sorted(f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.npy'))
    test_masks = sorted(f for f in os.listdir(TEST_MASK_DIR) if f.endswith('.npy'))
    test_ds = NpyT1cImageDataset(TEST_IMG_DIR,test_imgs,TEST_MASK_DIR,test_masks)
    test_ld = DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
    test_and_plot(model,test_ld,DEVICE)

