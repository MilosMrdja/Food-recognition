import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from copy import deepcopy

NUM_CLASSES = 82
NUM_EPOCHS = 80

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
extract_path = '/content/dataset1/split_dataset'

TRAIN_CNN2 = os.path.join(extract_path, 'training')
VALID_CNN2 = os.path.join(extract_path, 'validation')
TEST_CNN2 = os.path.join(extract_path, 'testing')
MODELS_DIR = '/content'

# ----- Data transforms -----
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(TRAIN_CNN2, transform=train_transform)
valid_dataset = datasets.ImageFolder(VALID_CNN2, transform=valid_transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ----- Model -----
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# sgd optimizer

# ----- Training loop -----
best_acc = 0.0
best_model_wts = None

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save best
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = deepcopy(model.state_dict())
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(best_model_wts, os.path.join(MODELS_DIR, "best_model_cnn2.pth"))
        print(f"✅ Best model updated! Acc={best_acc:.2f}%")
