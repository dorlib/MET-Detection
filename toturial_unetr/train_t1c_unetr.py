#!/usr/bin/env python3
# train_t1c_unetr.py - Train UNETR on t1c-only data using PyTorch

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import local modules
from unter_model import unetr_model

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
BASE_DIR = "../../MET-data/t1c_only/"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images/")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "masks/")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test/images/")
TEST_MASK_DIR = os.path.join(BASE_DIR, "test/masks/")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "saved_models/unetr_t1c_model.pth")
LOG_DIR = "./logs/unetr_t1c/"
TEST_RESULTS_DIR = "./test_results_t1c/"

# Training parameters
BATCH_SIZE = 2  # Smaller batch size due to model complexity
NUM_CLASSES = 4
NUM_EPOCHS = 100
LR = 1e-4
PATIENCE = 10  # For early stopping
VAL_SPLIT = 0.2  # 20% of data for validation
RANDOM_SEED = 42
IMG_SIZE = (128, 128, 128)  # H, W, D

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class T1cDataset(Dataset):
    """Dataset for loading T1c images and masks"""
    def __init__(self, img_dir, img_list, mask_dir, mask_list):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        
        # Load numpy arrays
        img = np.load(img_path).astype(np.float32)  # Shape: H, W, D, C=1
        mask = np.load(mask_path)                   # Shape: H, W, D, C=4 (one-hot encoded)
        
        # Convert to PyTorch tensors with appropriate dimensions
        # For 3D medical imaging, PyTorch expects: [C, D, H, W]
        img = torch.tensor(img).permute(3, 2, 0, 1)  # Reshape to [C, D, H, W]
        
        if mask.ndim == 4 and mask.shape[-1] > 1:  # One-hot encoded
            mask = torch.tensor(mask).permute(3, 2, 0, 1)  # [C, D, H, W]
        else:
            mask = torch.argmax(torch.tensor(mask), dim=-1)  # Convert one-hot to class indices
            mask = mask.unsqueeze(0)  # Add channel dimension [1, D, H, W]
            
        return img, mask

# ──────────────────────────────────────────────────────────────
# PyTorch Lightning Module
# ──────────────────────────────────────────────────────────────
class UNETRModule(pl.LightningModule):
    def __init__(self, img_size=IMG_SIZE, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = unetr_model(img_size[0], img_size[1], img_size[2], in_channels, num_classes)
        
        # Modified loss function with class weights to emphasize tumor over edema
        # Higher weights for tumor classes (1 and 3)
        class_weights = torch.tensor([0.1, 2.0, 0.5, 2.0])  # Lower weight for background (0) and edema (2)
        self.loss_function = DiceCELoss(
            to_onehot_y=True, 
            softmax=True,
            ce_weight=class_weights.to(DEVICE) if torch.cuda.is_available() else class_weights
        )
        
        # Separate metric for tumor-only evaluation (combining classes 1 and 3)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.tumor_dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        self.val_imgs = None
        self.val_masks = None
        self.val_preds = None
        self.training_metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_tumor_dice": []  # Track tumor-specific Dice score
        }
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        loss = self.loss_function(outputs, masks)
        
        # Log training metrics
        self.log("train_loss", loss, prog_bar=True)
        self.training_metrics["train_loss"].append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        # Use sliding window inference for better predictions on validation
        outputs = sliding_window_inference(
            images, roi_size=IMG_SIZE, sw_batch_size=1, predictor=self.model
        )
        
        loss = self.loss_function(outputs, masks)
        
        # Calculate Dice score for all classes
        self.dice_metric(y_pred=outputs, y=masks)
        
        # Calculate tumor-only Dice score (classes 1 and 3)
        # Create a binary tumor mask (combine classes 1 and 3)
        if masks.shape[1] > 1:  # If one-hot encoded
            tumor_mask = torch.zeros_like(masks)
            tumor_mask[:, 1] = masks[:, 1]  # Non-enhancing tumor (class 1)
            tumor_mask[:, 3] = masks[:, 3]  # Enhancing tumor (class 3)
            
            # Create tumor-only predictions
            tumor_pred = torch.zeros_like(outputs)
            tumor_pred[:, 1] = outputs[:, 1]  # Non-enhancing tumor (class 1)
            tumor_pred[:, 3] = outputs[:, 3]  # Enhancing tumor (class 3)
            
            # Calculate tumor-only Dice score
            self.tumor_dice_metric(y_pred=tumor_pred, y=tumor_mask)
        
        # Store some validation samples for visualization
        if batch_idx == 0:
            self.val_imgs = images.detach().clone()
            self.val_masks = masks.detach().clone()
            self.val_preds = outputs.detach().clone()
        
        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_dice", mean_dice, prog_bar=True)
        
        # Log tumor-specific Dice score
        mean_tumor_dice = self.tumor_dice_metric.aggregate().item()
        self.tumor_dice_metric.reset()
        self.log("val_tumor_dice", mean_tumor_dice, prog_bar=True)
        
        # Store metrics for plotting
        self.training_metrics["val_loss"].append(torch.stack([x["val_loss"] for x in outputs]).mean().item())
        self.training_metrics["val_dice"].append(mean_dice)
        self.training_metrics["val_tumor_dice"].append(mean_tumor_dice)
        
        # For visualization, get the last batch
        if self.current_epoch % 10 == 0:  # Visualize every 10 epochs
            self._log_images()
    
    def _log_images(self):
        """Log example predictions for visualization"""
        # This would typically be done with tensorboard or another logger
        # For simplicity, we'll just save to disk
        if not hasattr(self, 'val_imgs') or self.val_imgs is None:
            return
            
        os.makedirs("predictions", exist_ok=True)
        
        # Process multiple samples in the batch
        for i in range(min(2, self.val_imgs.shape[0])):  # Process up to 2 samples
            image = self.val_imgs[i, 0].cpu().numpy()  # First image, first channel
            true_mask = torch.argmax(self.val_masks[i], dim=0).cpu().numpy()  # Convert one-hot to class indices
            pred_mask = torch.softmax(self.val_preds[i], dim=0).argmax(dim=0).cpu().numpy()
            
            # Get tumor-only masks (classes 1 and 3)
            true_tumor = np.zeros_like(true_mask)
            true_tumor[(true_mask == 1) | (true_mask == 3)] = 1  # Combine classes 1 and 3
            
            pred_tumor = np.zeros_like(pred_mask)
            pred_tumor[(pred_mask == 1) | (pred_mask == 3)] = 1  # Combine classes 1 and 3
            
            # Save a middle slice
            mid_slice = image.shape[0] // 2
            
            # Create standard visualization
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.title("T1c Image")
            plt.imshow(image[mid_slice], cmap="gray")
            plt.subplot(132)
            plt.title("True Mask")
            plt.imshow(true_mask[mid_slice])
            plt.subplot(133)
            plt.title("Predicted Mask")
            plt.imshow(pred_mask[mid_slice])
            plt.tight_layout()
            plt.savefig(f"predictions/epoch_{self.current_epoch}_sample_{i}.png")
            plt.close()
            
            # Create tumor-specific visualization
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.title("T1c Image")
            plt.imshow(image[mid_slice], cmap="gray")
            plt.subplot(132)
            plt.title("True Tumor Regions")
            plt.imshow(true_tumor[mid_slice], cmap="hot")
            plt.subplot(133)
            plt.title("Predicted Tumor Regions")
            plt.imshow(pred_tumor[mid_slice], cmap="hot")
            plt.tight_layout()
            plt.savefig(f"predictions/epoch_{self.current_epoch}_sample_{i}_tumor_only.png")
            plt.close()

# ──────────────────────────────────────────────────────────────
# Plotting Functions
# ──────────────────────────────────────────────────────────────
def plot_metrics(losses, recalls, accuracies, tumor_accuracies=None, title="Training Metrics"):
    """Plot training metrics over time"""
    epochs = range(1, len(losses)+1)
    
    if tumor_accuracies is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, losses, 'b-', label='Loss')
    ax2.plot(epochs, accuracies, 'g-', label='Dice Score')
    
    if tumor_accuracies is not None:
        ax3.plot(epochs, tumor_accuracies, 'r-', label='Tumor-Only Dice')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Tumor-Only Dice')
        ax3.set_title('Tumor-Only Dice Score over Epochs')
        ax3.grid(True)
        ax3.legend()
    
    for ax, lbl in zip((ax1, ax2), ('Loss', 'Dice Score')):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(lbl)
        ax.set_title(f'{lbl} over Epochs')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def test_and_plot(model, loader, device, output_dir=TEST_RESULTS_DIR):
    """Test the model and plot the results"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            
            # Use sliding window inference for better predictions
            logits = sliding_window_inference(
                x, roi_size=IMG_SIZE, sw_batch_size=1, predictor=model
            )
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            if y.shape[1] > 1:  # If one-hot encoded
                masks = torch.argmax(y, dim=1).cpu().numpy()
            else:
                masks = y[:, 0].cpu().numpy()  # Take first channel if not one-hot
                
            images = x.cpu().numpy()  # shape [B, C, D, H, W]
            
            for i in range(preds.shape[0]):
                # Extract multiple slices: mid-depth, mid-height, and mid-width
                img_vol = images[i, 0]  # (D, H, W), take first channel
                pred_vol = preds[i]
                mask_vol = masks[i]
                
                mid_d = img_vol.shape[0] // 2
                mid_h = img_vol.shape[1] // 2
                mid_w = img_vol.shape[2] // 2
                
                # Create a figure with 3 rows (one for each view) and 3 columns (image, pred, gt)
                fig, axs = plt.subplots(3, 3, figsize=(18, 15), dpi=150)
                
                # Axial view (mid-depth slice)
                axs[0, 0].imshow(img_vol[mid_d], cmap='gray')
                axs[0, 0].set_title('Original (Axial)')
                axs[0, 1].imshow(pred_vol[mid_d])
                axs[0, 1].set_title('Prediction (Axial)')
                axs[0, 2].imshow(mask_vol[mid_d])
                axs[0, 2].set_title('Ground Truth (Axial)')
                
                # Coronal view (mid-height slice)
                axs[1, 0].imshow(img_vol[:, mid_h, :], cmap='gray')
                axs[1, 0].set_title('Original (Coronal)')
                axs[1, 1].imshow(pred_vol[:, mid_h, :])
                axs[1, 1].set_title('Prediction (Coronal)')
                axs[1, 2].imshow(mask_vol[:, mid_h, :])
                axs[1, 2].set_title('Ground Truth (Coronal)')
                
                # Sagittal view (mid-width slice)
                axs[2, 0].imshow(img_vol[:, :, mid_w], cmap='gray')
                axs[2, 0].set_title('Original (Sagittal)')
                axs[2, 1].imshow(pred_vol[:, :, mid_w])
                axs[2, 1].set_title('Prediction (Sagittal)')
                axs[2, 2].imshow(mask_vol[:, :, mid_w])
                axs[2, 2].set_title('Ground Truth (Sagittal)')
                
                for row in axs:
                    for ax in row:
                        ax.axis('off')
                
                plt.tight_layout()
                
                # Use the filename from dataset or generate a unique one
                if hasattr(loader.dataset, 'img_list'):
                    fname = os.path.splitext(loader.dataset.img_list[batch_idx * loader.batch_size + i])[0]
                else:
                    fname = f"test_case_{batch_idx}_{i}"
                    
                out_path = os.path.join(output_dir, f"{fname}_slices.png")
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
    
    print(f"Saved test results in {output_dir}")

# ──────────────────────────────────────────────────────────────
# Training and Evaluation
# ──────────────────────────────────────────────────────────────
def train_model():
    """Train the UNETR model"""
    # Create directories
    os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_MASK_DIR, exist_ok=True)
    
    # Load file lists
    img_list = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.npy')])
    mask_list = sorted([f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.npy')])
    
    if len(img_list) != len(mask_list):
        raise ValueError(f"Number of images ({len(img_list)}) does not match number of masks ({len(mask_list)})")
    
    # Create train/val split
    indices = np.arange(len(img_list))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - VAL_SPLIT))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_img_list = [img_list[i] for i in train_indices]
    train_mask_list = [mask_list[i] for i in train_indices]
    val_img_list = [img_list[i] for i in val_indices]
    val_mask_list = [mask_list[i] for i in val_indices]
    
    # Create test set from validation set (or use separate test set if available)
    test_img_list = val_img_list[:5]  # Use first 5 validation samples as test
    test_mask_list = val_mask_list[:5]
    
    print(f"Training on {len(train_img_list)} samples, validating on {len(val_img_list)} samples")
    print(f"Testing on {len(test_img_list)} samples")
    
    # Create datasets and data loaders
    train_ds = T1cDataset(TRAIN_IMG_DIR, train_img_list, TRAIN_MASK_DIR, train_mask_list)
    val_ds = T1cDataset(TRAIN_IMG_DIR, val_img_list, TRAIN_MASK_DIR, val_mask_list)
    test_ds = T1cDataset(TRAIN_IMG_DIR, test_img_list, TRAIN_MASK_DIR, test_mask_list)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Use batch size 1 for testing to generate better visualizations
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create callbacks for training
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.dirname(SAVED_MODEL_PATH),
        filename="unetr_t1c_{epoch:02d}_{val_dice:.3f}",
        save_top_k=3,
        monitor="val_dice",
        mode="max",
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_dice",
        min_delta=0.001,
        patience=PATIENCE,
        verbose=True,
        mode="max"
    )
    
    # Create the logger
    logger = TensorBoardLogger(LOG_DIR, name="unetr_t1c")
    
    # Create the model
    model = UNETRModule()
    
    # Check if model exists and load it if it does
    if os.path.exists(SAVED_MODEL_PATH):
        model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded existing model from {SAVED_MODEL_PATH}, skipping training.")
        # Test the pre-loaded model
        test_and_plot(model, test_loader, DEVICE)
        return model, None
    
    # Train the model
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision for faster training
        log_every_n_steps=10,
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
    print(f"Model saved to {SAVED_MODEL_PATH}")
    
    # Plot training metrics
    plot_metrics(
        model.training_metrics["val_loss"], 
        model.training_metrics.get("val_recall", [0] * len(model.training_metrics["val_loss"])),
        model.training_metrics["val_dice"],
        model.training_metrics.get("val_tumor_dice", None),
        "UNETR T1c Training Metrics"
    )
    
    # Test the model
    test_and_plot(model, test_loader, DEVICE)
    
    return model, trainer

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train the model
    model, trainer = train_model()
    
    print("Training complete!")