import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from monai.transforms import Compose, RandFlip, RandRotate, RandZoom, ResizeWithPadOrCrop, ResizeWithPadOrCropd, \
    RandFlipd, RandRotated, RandZoomd
from monai.losses import DiceLoss


# ---------------------------
# Helper Function: Load NIfTI
# ---------------------------
def load_nifti(filepath):
    """
    Load a NIfTI image and normalize its values.
    """
    nifti_image = nib.load(filepath)
    image_data = nifti_image.get_fdata()
    # Normalize between 0 and 1
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    return image_data


# ------------------------------
# Dataset Class for NIfTI Images
# ------------------------------
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset class for 3D MRI NIfTI images and masks.
        Args:
            root_dir: Root directory containing subdirectories of cases.
            transform: Optional transformation pipeline.
        """
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform

        # Scan all directories for t1c and seg files
        for case_folder in os.listdir(root_dir):
            case_path = os.path.join(root_dir, case_folder)
            if not os.path.isdir(case_path):
                continue
            t1c_path = os.path.join(case_path, f"{case_folder}-t1c.nii.gz")
            seg_path = os.path.join(case_path, f"{case_folder}-seg.nii.gz")
            if os.path.exists(t1c_path) and os.path.exists(seg_path):
                self.image_paths.append(t1c_path)
                self.mask_paths.append(seg_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_nifti(self.image_paths[idx])
        mask = load_nifti(self.mask_paths[idx])

        # Add channel dimension (C, D, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            transformed = self.transform({"image": image, "mask": mask})
            image = transformed["image"]
            mask = transformed["mask"]

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# ------------------------------
# Transformer Model for 3D Data
# ------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, input_shape=(128, 128, 128), patch_size=16, num_classes=1):
        super(VisionTransformer, self).__init__()
        self.config = ViTConfig(
            image_size=input_shape[1:],  # H, W
            patch_size=patch_size,
            num_channels=1,  # Single-channel input
            num_labels=num_classes
        )
        self.vit = ViTModel(self.config)
        self.channel_mapper = nn.Linear(self.config.hidden_size, 128)
        self.head = nn.Conv3d(128, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError(f"Expected 1 channel in the input, but got {C}.")

        # Reshape for ViT processing
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)  # (B*D, C, H, W)

        # Pass through ViT
        features = self.vit(x).last_hidden_state  # (B*D, num_patches, hidden_size)
        features = features.mean(dim=1)  # Global average pooling: (B*D, hidden_size)

        # Map features and reshape for 3D conv
        features = self.channel_mapper(features)  # (B*D, 128)
        features = features.view(B, D, 128, 1, 1).permute(0, 2, 1, 3, 4)  # (B, 128, D, 1, 1)

        # Final 3D convolution
        output = self.head(features)  # (B, num_classes, D, 1, 1)
        return output.expand(-1, -1, -1, H, W)  # Broadcast spatial dimensions


# ----------------------------
# Data Preparation
# ----------------------------
def prepare_data(root_dir):
    """
    Prepare datasets and dataloaders.
    """
    transform = Compose([
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=(128, 128, 128)),  # Apply to both
        RandFlipd(keys=["image", "mask"], spatial_axis=[0, 1, 2], prob=0.5),        # Random flip
        RandRotated(keys=["image", "mask"], range_x=10, range_y=10, range_z=10, prob=0.5),  # Random rotation
        RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.5),    # Random zoom
    ])

    dataset = NiftiDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=2, shuffle=True)


# ----------------------------
# Training Function
# ----------------------------
def train(model, dataloader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")


# ----------------------------
# Main Script
if __name__ == "__main__":
    # Root directory for the dataset
    root_dir = "../MET - data/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData"

    # Prepare data
    dataloader = prepare_data(root_dir)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(input_shape=(128, 128, 128)).to(device)
    criterion = DiceLoss(sigmoid=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train(model, dataloader, optimizer, criterion, num_epochs=10)

