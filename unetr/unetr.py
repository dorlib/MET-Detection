import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from monai.losses import DiceCELoss
from monai.transforms import (AsDiscrete, Compose, RandFlipd, RandRotate90d, 
                              RandCropByPosNegLabeld, RandShiftIntensityd)
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NumpyDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".npy")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".npy")]) if mask_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]).astype(np.float32)
        image = torch.from_numpy(image).permute(3, 0, 1, 2)

        if self.mask_paths:
            mask = np.load(self.mask_paths[idx]).astype(np.float32)
            mask = torch.from_numpy(mask).permute(3, 0, 1, 2)

            if self.transform:
                augmented = self.transform({"image": image, "label": mask})
                print(f"Augmented output: {augmented}")
                if not isinstance(augmented, dict) or "image" not in augmented or "label" not in augmented:
                    raise TypeError("Expected a dictionary with keys 'image' and 'label' from transforms.")
                image, mask = augmented["image"], augmented["label"]

            return {"image": image, "label": mask}
        else:
            return {"image": image}


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNETR(
            in_channels=3,  # Assuming 3 input channels (flair, t1ce, t2)
            out_channels=4,  # Assuming 4 classes (background, label1, label2, label3)
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0,
        )

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.post_label = AsDiscrete(to_onehot=4)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.epoch_loss_values = []
        self.metric_values = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        pass  # Data is preprocessed and saved as Numpy arrays

    def setup(self, stage):
        train_transforms = Compose([
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(128, 128, 128), pos=1,
                                   neg=1, num_samples=1),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.1),
            RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ])

        self.train_ds = NumpyDataset("../../MET-data/input_data/images", "../../MET-data/input_data/masks", transform=train_transforms)
        self.val_ds = NumpyDataset("../../MET-data/input_data/validation_images")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4, weight_decay=1e-5)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Visualize a training sample
        if batch_idx == 0:
            img = images[0].cpu().numpy().transpose(1, 2, 3, 0)[:, :, :, 0]  # First channel only
            lbl = labels[0].cpu().numpy().transpose(1, 2, 3, 0).argmax(axis=-1)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Training Image")
            plt.imshow(img[:, :, img.shape[2] // 2], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("Training Mask")
            plt.imshow(lbl[:, :, lbl.shape[2] // 2], cmap="gray")
            plt.show()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        outputs = self.forward(images)
        predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        # Visualize and save the predictions
        for i, pred in enumerate(predictions):
            plt.figure(figsize=(8, 8))
            plt.title(f"Validation Prediction - Batch {batch_idx}, Image {i}")
            plt.imshow(pred[64], cmap="gray")  # Show a middle slice of the prediction
            plt.show()

        self.validation_step_outputs.append({"predictions": predictions})

    def on_validation_epoch_end(self):
        dice_scores = []
        for output in self.validation_step_outputs:
            predictions = output["predictions"]
            for pred in predictions:
                # Assuming ground truth masks exist for validation
                # Compute Dice score here if you have labels
                pass  # Add dice score computation here if needed

        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        self.log("val_dice", avg_dice, prog_bar=True, logger=True)
        print(f"Average validation Dice score: {avg_dice:.4f}")
        self.validation_step_outputs.clear()  # Clear saved outputs to free memory


# Training setup
net = Net()

# Callbacks
checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints", filename="best_metric_model", save_top_k=1, monitor="val_loss", mode="min")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_logger = TensorBoardLogger(save_dir="logs", name="unetr_logs")

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stopping],
    logger=tensorboard_logger,
    devices=4,  # Specify number of CPU devices (cores), e.g., 1
    accelerator="cpu"
)
trainer.fit(net)
