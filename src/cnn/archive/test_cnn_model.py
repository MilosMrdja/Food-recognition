from src.cnn.archive.cnn_class import Net
import torch
from config import MODELS_DIR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import TEST_CNN

def test(batch_size=16):
    # 1. laod model
    net = Net(num_classes=36)
    net.load_state_dict(torch.load(f"{MODELS_DIR}/cnn_model.pth"))
    net.eval()

    # 2. test loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = datasets.ImageFolder(TEST_CNN, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 3. testing
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)  # predikcija
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test set accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test()