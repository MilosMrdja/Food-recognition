# 2️⃣ Evaluacija
import cv2
from ultralytics import YOLO

model = YOLO("models/yolo/v2/best.pt")


# 3️⃣ Inferencija / detekcija na novoj slici
img = cv2.imread("input_images/1.jpg")

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