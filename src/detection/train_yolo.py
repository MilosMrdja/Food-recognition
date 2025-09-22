
import cv2
import ultralytics
from ultralytics import YOLO


def main():
    # 1️⃣ Trening
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    # Start training
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=512,
        batch=16,
        device=0,
        augment=True,
        degrees=5,
        translate=0.05,
        scale=0.1
    )

    # 2️⃣ Evaluacija
    model = YOLO("models/best.pt")
    results = model.val()
    print("Evaluation results:", results)

    # 3️⃣ Inferencija / detekcija na novoj slici
    img = cv2.imread("input_images/test.jpg")


    results = model.predict(
        source=img,
        conf=0.25,
        save=False
    )

    # 3️⃣ Vizuelizacija i crtanje bbox-ova
    for r in results:
        img_with_boxes = r.plot()
        cv2.imshow("Food recognition", img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
