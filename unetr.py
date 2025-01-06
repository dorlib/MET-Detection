#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: Train UNETR on .npy data that has labels (for training) and 
run inference on unlabeled .npy data (for final predictions).
Also show how to visualize losses and predictions.
"""

import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from monai.config import print_config
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    AsDiscrete,
    decollate_batch,
)
from monai.data import CacheDataset, DataLoader
from monai.transforms.transform import MapTransform


# -------------------------------------------------------------------------
# 1) Custom transform to load .npy
# -------------------------------------------------------------------------
class LoadNumpyd(MapTransform):
    """
    Custom transform to load arrays stored as .npy files.
    Expecting keys like 'image' or 'label' that map to file paths.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:  # e.g. ["image", "label"]
            npy_path = d[key]
            d[key] = np.load(npy_path)  # shape might be [H, W, D], or [H, W, D, C], etc.
        return d


# -------------------------------------------------------------------------
# 2) Datasets
#    A) "train_files" -> images + labels  (for training)
#    B) "val_files"   -> images + labels  (optional small validation set)
#    C) "test_files"  -> images only      (unlabeled; for final predictions)
# -------------------------------------------------------------------------
train_files = [
    {"image": "/path/to/train_img0.npy", "label": "/path/to/train_label0.npy"},
    {"image": "/path/to/train_img1.npy", "label": "/path/to/train_label1.npy"},
    # ...
]
val_files = [
    {"image": "/path/to/val_img0.npy", "label": "/path/to/val_label0.npy"},
    {"image": "/path/to/val_img1.npy", "label": "/path/to/val_label1.npy"},
    # ...
]
test_files = [
    {"image": "/path/to/test_img0.npy"},  # no label here
    {"image": "/path/to/test_img1.npy"},
    # ...
]


# -------------------------------------------------------------------------
# 3) Define transforms
#    NOTE:
#    - Use RandCropByPosNegLabeld only for labeled data (training).
#    - For unlabeled data, skip any transform requiring "label".
# -------------------------------------------------------------------------
train_transforms = Compose([
    LoadNumpyd(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        num_samples=2,
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
    RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
])

val_transforms = Compose([
    LoadNumpyd(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
])

# For test (unlabeled) data, we only load the image:
test_transforms = Compose([
    LoadNumpyd(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
])


# -------------------------------------------------------------------------
# 4) LightningModule: define model, train/val step, etc.
# -------------------------------------------------------------------------
class Net(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        out_channels=4,
        max_epochs=100,
        check_val=10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # -- Model
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(96, 96, 96),
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

        # -- Loss
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

        # -- Post-processing & metrics
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=out_channels)
        self.post_label = AsDiscrete(to_onehot=True, n_classes=out_channels)
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )

        # -- For tracking
        self.best_val_dice = 0.0
        self.best_val_epoch = 0
        self.epoch_loss_values = []
        self.metric_values = []

        # store steps outputs
        self.validation_step_outputs = []

        self.max_epochs = max_epochs
        self.check_val = check_val

    def forward(self, x):
        return self.model(x)

    # ---------------------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------------------
    def prepare_data(self):
        """
        Prepare CacheDatasets for training, validation, and testing.
        If you have no labeled val data, skip creating self.val_ds.
        """
        # Training set (labeled)
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        # Validation set (labeled) -> can be empty if you have no labels
        if len(val_files) > 0 and "label" in val_files[0]:
            self.val_ds = CacheDataset(
                data=val_files,
                transform=val_transforms,
                cache_rate=1.0,
                num_workers=2,
            )
        else:
            self.val_ds = None

        # Test set (unlabeled)
        self.test_ds = CacheDataset(
            data=test_files,
            transform=test_transforms,
            cache_rate=1.0,
            num_workers=2,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        # Only return if we have a labeled validation dataset
        if self.val_ds is not None:
            return DataLoader(
                self.val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        return None

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # ---------------------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        images, labels = batch["image"].cuda(), batch["label"].cuda()
        preds = self.forward(images)
        loss = self.loss_function(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        # average over each mini-batch in the epoch
        avg_loss = torch.stack(outputs).mean()
        self.epoch_loss_values.append(avg_loss.item())

    # ---------------------------------------------------------------------
    # Validation (only if we have labels)
    # ---------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"].cuda(), batch["label"].cuda()

        # sliding window if needed
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        preds = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)

        val_loss = self.loss_function(preds, labels)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        preds_list = decollate_batch(preds)
        labels_list = decollate_batch(labels)
        preds_converted = [self.post_pred(p) for p in preds_list]
        labels_converted = [self.post_label(l) for l in labels_list]
        self.dice_metric(y_pred=preds_converted, y=labels_converted)

        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        if self.val_ds is None:
            return  # no labeled val data

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.metric_values.append(mean_val_dice)

        # track best
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        self.log("val_dice", mean_val_dice, on_epoch=True, prog_bar=True)
        print(
            f"validation dice: {mean_val_dice:.4f} | "
            f"best: {self.best_val_dice:.4f} at epoch {self.best_val_epoch}"
        )

    # ---------------------------------------------------------------------
    # Test / Predict
    # ---------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        """
        Here we do inference on unlabeled data. We only have 'image' in the batch dict.
        We'll run the model forward pass and return the predicted segmentation.

        We'll store the predictions in self.test_outputs for later usage if desired.
        """
        images = batch["image"].cuda()

        # do sliding window if large volume
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        preds = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        # argmax to get final segmentation
        preds_argmax = torch.argmax(preds, dim=1, keepdim=True)  # shape [B,1,H,W,D]

        # We can store or return it
        return preds_argmax.cpu().numpy()  # returning as .numpy()

    def test_epoch_end(self, outputs):
        """
        outputs is a list of test_step return values
        We can do further processing or saving here.
        """
        # For example, save the predictions to disk as .npy
        # or just keep them in memory
        self.test_outputs = outputs  # store for inspection
        print(f"Test / inference done. Got {len(outputs)} volumes.")


# -------------------------------------------------------------------------
# 5) Main function to run everything
# -------------------------------------------------------------------------
def main():
    print_config()  # optional: prints MONAI's version, etc.

    # Create model
    net = Net(
        in_channels=1,
        out_channels=4,  # background + 3 classes, example
        max_epochs=50,
        check_val=5,
    )

    # Save top checkpoint by "val_dice" if we do have some labeled val data
    checkpoint_callback = ModelCheckpoint(
        dirpath="./runs/",
        filename="best_metric_model",
        save_top_k=1,
        monitor="val_dice",  # only works if we have labeled val
        mode="max",
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else None,
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_callback],
        default_root_dir="./runs/",
    )

    # TRAIN
    trainer.fit(net)

    # VISUALIZE LOSS (and val dice if it exists)
    # ------------------------------------------
    # net.epoch_loss_values = training losses
    # net.metric_values     = validation dice (only if we had labels)

    plt.figure("loss_and_dice", (12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Training Loss")
    plt.plot(range(len(net.epoch_loss_values)), net.epoch_loss_values, label="Train")
    plt.xlabel("Epoch")
    plt.legend()

    if len(net.metric_values) > 0:
        plt.subplot(1, 2, 2)
        plt.title("Val Dice")
        plt.plot(range(len(net.metric_values)), net.metric_values, label="Val")
        plt.xlabel("Validation Check")
        plt.legend()

    plt.tight_layout()
    plt.show()

    # TEST / PREDICT on unlabeled data
    # ------------------------------------------
    # This will call net.test_step(...) and net.test_epoch_end(...)
    trainer.test(net)

    # If you want to see predictions from net.test_outputs:
    # Each item in net.test_outputs is an array of shape [B,1,H,W,D] (argmax)
    # Here we do a small example of visualizing a single slice from the first volume
    test_preds = net.test_outputs  # list of length len(test_ds)
    # test_preds[i].shape -> (batch_size=1, 1, H, W, D)

    if len(test_preds) > 0:
        volume0 = test_preds[0][0, 0]  # [H,W,D]
        print("Predicted volume shape:", volume0.shape)

        # For example, visualize a middle slice
        slice_idx = volume0.shape[-1] // 2
        plt.figure("test_prediction", (12, 5))
        plt.imshow(volume0[..., slice_idx], cmap="gray")
        plt.title("Predicted segmentation (test volume #0, middle slice)")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
