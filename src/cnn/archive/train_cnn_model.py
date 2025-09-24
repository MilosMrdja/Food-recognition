import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from config import TRAIN_CNN, VALID_CNN, MODELS_DIR
from src.cnn.archive.cnn_class import Net, get_loss, get_optimizer

classes = (
    "AW cola",
    "Beijing Beef",
    "Chow Mein",
    "Fried Rice",
    "Hashbrown",
    "Honey Walnut Shrimp",
    "Kung Pao Chicken",
    "String Bean Chicken Breast",
    "Super Greens",
    "The Original Orange Chicken",
    "White Steamed Rice",
    "black pepper rice bowl",
    "burger",
    "carrot_eggs",
    "cheese burger",
    "chicken waffle",
    "chicken_nuggets",
    "chinese_cabbage",
    "chinese_sausage",
    "crispy corn",
    "curry",
    "french fries",
    "fried chicken",
    "fried_chicken",
    "fried_dumplings",
    "fried_eggs",
    "mango chicken pocket",
    "mozza burger",
    "mung_bean_sprouts",
    "nugget",
    "perkedel",
    "rice",
    "sprite",
    "tostitos cheese dip sauce",
    "triangle_hash_brown",
    "water_spinach",
)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main(net, criterion, optimizer, num_epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainloader = DataLoader(
        datasets.ImageFolder(TRAIN_CNN, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=2
    )
    validloader = DataLoader(
        datasets.ImageFolder(VALID_CNN, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=2
    )

    best_acc = 0.0
    for epoch in range(num_epochs):
        # ---- Train ----
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)

        # ---- Validation ----
        net.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(validloader)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = f"{MODELS_DIR}/cnn_ep{epoch+1}_acc{val_acc:.2f}.pth"
            torch.save(net.state_dict(), model_path)
            print(f"✅ Best model saved: {model_path}")


if __name__ == '__main__':
    net = Net(num_classes=36)
    criterion = get_loss()
    optimizer = get_optimizer(net)
    main(net, criterion, optimizer)