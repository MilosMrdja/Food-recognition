import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from config import TRAIN_CNN, TEST_CNN
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

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = datasets.ImageFolder(TRAIN_CNN, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = datasets.ImageFolder(TEST_CNN, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))

    classes = trainset.classes
    print(' '.join(f'{classes[labels[j]]}' for j in range(len(labels))))

if __name__ == '__main__':
    main()