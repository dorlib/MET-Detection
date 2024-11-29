# backend/app/services/training_service.py

import torch
from torch.utils.data import DataLoader
from backend.models.vit_segmentation import VisionTransformerSegmentation
from backend.data.datasets import MRIImageDataset
from backend.db.db import get_db
from sqlalchemy.orm import Session
import os
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm


def get_transform():
    """Define data augmentations and preprocessing steps"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TrainingService:
    def __init__(self, db: Session, epochs: int, batch_size: int, learning_rate: float):
        self.db = db
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Prepare the dataset and DataLoader
        self.train_dataset = MRIImageDataset(root_dir='backend/data/raw', transform=get_transform())
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Define the Vision Transformer model for segmentation
        self.model = VisionTransformerSegmentation(num_classes=1)

        # Define the optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()  # For binary segmentation, adjust accordingly

    def start_training(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Start training loop
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Loop through the dataset
            for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                images = images.to(device)
                masks = masks.to(device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                loss = self.criterion(outputs, masks)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(self.train_loader)}")

            # Optionally, save checkpoints after every epoch
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        checkpoint_dir = 'backend/models/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
