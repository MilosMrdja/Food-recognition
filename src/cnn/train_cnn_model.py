import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from config import TRAIN_CNN, VALID_CNN, MODELS_DIR
from cnn_class import Net, get_loss, get_optimizer

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

def main(net: Net, criterion, optimizer, num_epochs=2, batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = datasets.ImageFolder(TRAIN_CNN, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    validset = datasets.ImageFolder(VALID_CNN, transform=transform)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_acc = 0.0

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/200:.3f}")
                running_loss = 0.0

        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validloader:
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch + 1} finished. Test Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(net.state_dict(), f"{MODELS_DIR}/cnn_model.pth")
            print(f"✅ Best model saved with Acc: {best_acc:.2f}%")

    print("Finished Training")

if __name__ == '__main__':
    net = Net(num_classes=36)
    criterion = get_loss()
    optimizer = get_optimizer(net)
    main(net, criterion, optimizer)