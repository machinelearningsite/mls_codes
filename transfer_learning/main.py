import torch
import torchvision.models as models
import torch.nn as nn

# Load the pre-trained ResNet18 model
base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze the base model layers
for param in base_model.parameters():
    param.requires_grad = False

# Modify the final layer for binary classification
num_features = base_model.fc.in_features
base_model.fc = nn.Linear(num_features, 1)

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(root='dataset/train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(base_model.fc.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = base_model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")