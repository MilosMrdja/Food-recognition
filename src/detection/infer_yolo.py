# 2️⃣ Evaluacija
import cv2
from ultralytics.models import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # učitavanje najboljeg modela
results = model.val()
print("Evaluation results:", results)

# 3️⃣ Inferencija / detekcija na novoj slici
img = cv2.imread("input_images/test.jpg")

results = model.predict(
    source=img,  # zameni sa svojom slikom
    conf=0.25,  # minimalna confidence vrednost
    save=False  # ne čuvamo automatski, samo prikazujemo
)

# 3️⃣ Vizuelizacija i crtanje bbox-ova
for r in results:
    img_with_boxes = r.plot()  # vraća sliku sa iscrtanim bbox-ovima
    cv2.imshow("Food recognition", img_with_boxes)
    cv2.waitKey(0)  # pritisni bilo koji taster da zatvoriš prozor