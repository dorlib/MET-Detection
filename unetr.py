# === IMPORT LIBRARIES ===
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader
from monai.config import print_config
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Compose, Spacingd, Orientationd,
    RandSpatialCropd, NormalizeIntensityd, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, ToTensord, CenterSpatialCropd, Activations, AsDiscrete, SpatialPadd
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from torchsummary import summary

# === SET DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_config()

# === SET SEED FOR REPRODUCIBILITY ===
set_determinism(seed=0)

# === DEFINE DATASET PATH ===
data_dir = "../MET-data"
train_dir = os.path.join(data_dir, "trainingData_unetr")
val_dir = os.path.join(data_dir, "ValidationData_unetr")

# === IMAGE MODALITIES AND SEGMENTATION LABEL ===
image_patterns = ["*t1c.nii", "*t2f.nii", "*t2w.nii"]
label_pattern = "*seg.nii"  # Mask file


# === FUNCTION TO SCAN PATIENT FOLDERS AND FIND FILES ===
def get_patient_data_paths(data_path):
    patient_dirs = glob.glob(os.path.join(data_path, "*"))  # List all patient folders
    data_list = []
    for patient_dir in patient_dirs:
        image_paths = []
        for pattern in image_patterns:
            matched_files = glob.glob(os.path.join(patient_dir, pattern))  # Find matching files
            if matched_files:
                image_paths.append(matched_files[0])  # Take the first match

        # Find the segmentation mask
        label_files = glob.glob(os.path.join(patient_dir, label_pattern))
        if not image_paths or not label_files:
            print(f"⚠️ Warning: Missing data in {patient_dir}")
            continue  # Skip this patient if images or labels are missing

        data_list.append({"image": image_paths, "label": label_files[0]})

    return data_list


# === GET TRAINING AND VALIDATION DATA ===
train_data = get_patient_data_paths(train_dir)
val_data = get_patient_data_paths(val_dir)

# === PRINT SAMPLE DATA ===
print("Train samples:", train_data[:2])  # Print first 2 training samples
print("Validation samples:", val_data[:2])  # Print first 2 validation samples

# === DEFINE TRANSFORMS ===
roi_size = [128, 128, 64]  # Crop size
pixdim = (1.5, 1.5, 2.0)  # Voxel spacing

# TRAINING TRANSFORMS
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(160, 160, 80)),  # Explicitly pad to match model
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

# VALIDATION TRANSFORMS
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

# === CREATE DATASETS & DATALOADERS ===
train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# === IMPORT UNETR MODEL ===
from self_attention_cv import UNETR

# === SETUP MODEL PARAMETERS ===
num_heads = 10  # 12 normally
embed_dim = 320  # Match feature map size to avoid mismatch

# === INITIALIZE MODEL ===
model = UNETR(
    img_shape=(160, 160, 80),
    input_dim=3,
    output_dim=3,
    embed_dim=320,  # Adjust to match input dimensions
    patch_size=16,
    num_heads=10,
    # ext_layers=[3, 6, 9, 12],
    # norm='instance',
    # base_filters=16,
    # dim_linear_block=2048
).to(device)

# === PRINT MODEL PARAMETER COUNT ===
pytorch_total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Parameters in millions: {pytorch_total_params}')

summary(model, (3, 160, 160, 80))

# === DEFINE LOSS FUNCTION & OPTIMIZER ===
loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# === TRAINING LOOP ===
max_epochs = 180
val_interval = 5
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

torch.cuda.empty_cache()

for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}/{max_epochs}")

    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        print(f"Before Model Forward Pass: Input Shape = {inputs.shape}")

        try:
            outputs = model(inputs)
            print(f"✅ Model Forward Pass Successful: Output Shape = {outputs.shape}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            raise e  # Stop execution to analyze error

        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}")

# === SAVE FINAL MODEL ===
torch.save(model.state_dict(), os.path.join(data_dir, "last.pth"))
print(f"Training completed. Best Metric: {best_metric:.4f} at epoch {best_metric_epoch}")

# === PLOT LOSS ===
plt.figure("train", (12, 6))
plt.title("Epoch Average Loss")
plt.xlabel("Epoch")
plt.plot(range(len(epoch_loss_values)), epoch_loss_values, color="red")
plt.show()
