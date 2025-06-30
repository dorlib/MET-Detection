from monai.transforms import (
    Compose,
    LambdaD,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
)
from custom_transforms import LoadNpy

def get_transforms():
    train_transforms = Compose([
        LoadNpy(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # Apply before adding channel dimension
        LambdaD(keys=["image", "label"], func=lambda x: x[None, ...]),  # Add channel dimension
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.1),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])
    return train_transforms

def get_val_transforms():
    val_transforms = Compose([
        LoadNpy(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),  # Apply before adding channel dimension
        LambdaD(keys=["image"], func=lambda x: x[None, ...]),  # Add channel dimension
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    ])
    return val_transforms
