"""Script to initiate training."""
import torch
import torch.optim as optim
import torch.nn as nn
from backend.models.vit_segmentation import VisionTransformer
from backend.data.datasets import train_loader, val_loader

# Initialize the model
model = VisionTransformer(num_classes=1)  # For binary segmentation, or adjust for multiple classes

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary segmentation, or change for multi-class
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss todo: should be more precise for segmentation i.e. nice (look for loss function for segmentation of MRI brain)
        loss = criterion(outputs, masks)
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    # Log the loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), './trained_vit_model.pth')
