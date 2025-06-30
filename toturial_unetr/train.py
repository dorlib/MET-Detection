import os
import torch
import pytorch_lightning as pl
from monai.losses import DiceCELoss
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from pytorch_lightning.callbacks import ModelCheckpoint
from data_transforms import get_transforms, get_val_transforms
from unter_model import unetr_model


class UNETRNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = unetr_model(128, 128, 128, 3, 4)
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]

        outputs = sliding_window_inference(images, (128, 128, 128), 4, self.model)
        outputs = torch.argmax(outputs, dim=1)  # Get predicted mask as class indices

        # Save predicted mask
        for i, output in enumerate(outputs):
            np.save(f"predictions/val_pred_{batch_idx}_{i}.npy", output.cpu().numpy())

        # Log example prediction
        if batch_idx == 0:
            self.log_example(images[0].cpu().numpy(), output.cpu().numpy())

    def log_example(self, image, prediction):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Validation Image (Sample Slice)")
        plt.imshow(image[0, :, :, 64], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask (Sample Slice)")
        plt.imshow(prediction[:, :, 64])
        plt.savefig("predictions/example_prediction.png")
        plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


if __name__ == "__main__":
    # Prepare directories
    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    # Load data
    train_transforms = get_transforms()
    val_transforms = get_val_transforms()

    train_files = [
        {"image": os.path.join("../../MET-data/input_data/images", img),
         "label": os.path.join("../../MET-data/input_data/masks", msk)}
        for img, msk in zip(
            sorted(f for f in os.listdir("../../MET-data/input_data/images") if f.endswith(".npy")),
            sorted(f for f in os.listdir("../../MET-data/input_data/masks") if f.endswith(".npy"))
        )
    ]

    val_files = [
        {"image": os.path.join("../../MET-data/input_data/validation_images", img)}
        for img in sorted(f for f in os.listdir("../../MET-data/input_data/validation_images") if f.endswith(".npy"))
    ]

    train_loader = DataLoader(CacheDataset(train_files, train_transforms, cache_rate=1.0), batch_size=2, shuffle=True)
    val_loader = DataLoader(CacheDataset(val_files, val_transforms, cache_rate=1.0), batch_size=1)

    # Callbacks and training
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100, callbacks=[checkpoint_callback])
    model = UNETRNet()
    trainer.fit(model, train_loader, val_loader)

    # Load a saved prediction
    pred = np.load("predictions/val_pred_0_0.npy")

    # Visualize a slice
    slice_idx = 64  # Choose a slice to visualize
    plt.imshow(pred[:, :, slice_idx], cmap="viridis")
    plt.title(f"Predicted Mask Slice {slice_idx}")
    plt.colorbar()
    plt.show()
