
from ultralytics import YOLO


def main():
    # 1️⃣ Trening
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    # Start training
    model.train(
        data="data.yml",
        epochs=100,
        imgsz=512,
        batch=16,
        device=0,
        augment=True,
        degrees=5,
        translate=0.05,
        scale=0.1
    )


if __name__ == "__main__":
    main()
