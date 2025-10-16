import cv2
import numpy as np
import torch
from PIL import Image
import torch
from torchvision import models, transforms
from ultralytics.models import YOLO
from PIL import Image
import cv2
import numpy as np
import torch

from config import YOLO_MODEL, CNN_MODEL, IMAGES_DIR, FOOD_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(num_classes=82):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(CNN_MODEL, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def print_yolo_predict(results):
    img_with_boxes = results[0].plot()
    cv2.imshow("Detections", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_crops(results, img_path):
    img = Image.open(img_path).convert("RGB")

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = img.crop((x1, y1, x2, y2))

        crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Crop {i}", crop_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def classify_crops(results, img_path, cnn_model, device, pad_px=0):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        x1_new = max(0, x1 - pad_px)
        y1_new = max(0, y1 - pad_px)
        x2_new = min(W, x2 + pad_px)
        y2_new = min(H, y2 + pad_px)

        crop = img.crop((x1_new, y1_new, x2_new, y2_new))

        # crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        # cv2.imshow(f"Crop {i}", crop_cv)

        input_tensor = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = cnn_model(input_tensor)
            pred_idx = out.argmax(1).item()

        print(f"Crop {i}: predicted class = {FOOD_CLASSES[pred_idx]} | Box: {(x1_new, y1_new, x2_new, y2_new)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def classify_images(results, img_path, cnn_model, device):
    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = cnn_model(input_tensor)
        pred_idx = out.argmax(1).item()

    print(f"predicted class = {FOOD_CLASSES[pred_idx]}")

def main():
    yolo = YOLO(YOLO_MODEL)
    cnn = load_model()
    #image_path = "data/cnn/split_dataset/testing/33/829005.jpg" #cookie yolo ok, cnn radi
    image_path = "data/cnn/split_dataset/testing/1/732481.jpg" # apple radi
    #image_path = "data/cnn/split_dataset/training/43/832103.jpg"
    #image_path = IMAGES_DIR / "test.jpg" # yolo radi, cnn ne
    results = yolo(str(image_path))
    classify_images(results, image_path, cnn, DEVICE)
    print_yolo_predict(results)
    #show_crops(results, image_path)
    classify_crops(results, image_path, cnn, DEVICE)

if __name__ == '__main__':
    main()