from ultralytics import YOLO

model = YOLO("models/yolo/v2/best.pt")

results = model.predict(
    source="data/processedV2/images/test",
    conf=0.25,
    save=True,
    save_txt=True
)

# images= [0, 50, 100]
img_with_boxes = results[100].plot()
import cv2
cv2.imshow("Detections", img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
