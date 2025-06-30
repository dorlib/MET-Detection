from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
)
from monai.data import CacheDataset, DataLoader

def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.1),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ])
    return train_transforms, val_transforms

def get_data_loaders(train_files, val_files, train_transforms, val_transforms, batch_size=1):
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader
