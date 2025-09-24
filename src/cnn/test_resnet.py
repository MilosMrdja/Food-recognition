import torch
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from config import MODELS_DIR, TEST_CNN2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_data():
    test_dataset = datasets.ImageFolder(TEST_CNN2, transform=transform)
    test__loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    return test__loader

def load_model(num_classes=82):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(f"{MODELS_DIR}/final/best_model_cnn2.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def evaluate(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            #print(predicted, labels)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"Test Accuracy: {val_acc:.2f}%")


def predict_image(image_path, model, transform, device=DEVICE):
    img = Image.open(image_path).convert("RGB")

    img_t = transform(img)

    img_t = img_t.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_t)
        _, pred_class = torch.max(output, 1)

    return pred_class.item()

def main():
    print(f"Using device: {DEVICE}")
    valid_loader = load_data()
    model = load_model()
    evaluate(model, valid_loader)

if __name__ == "__main__":
     main()
    # model = load_model()
    # class_idx = predict_image("input_images/rice.jpg", model, transform)
    # print(f"Predikovana klasa: {class_idx}")
