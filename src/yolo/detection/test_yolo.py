from ultralytics import YOLO

from config import YOLO_MODEL_V2, TEST_IMAGES_DIR, YOLO_MODEL_V1

model = YOLO(YOLO_MODEL_V2) # model_v1 vs. model_v2

results = model.predict(
    source=TEST_IMAGES_DIR,
    conf=0.25,
    save=True,
    save_txt=True
)

images= [0, 50, 100]
for i in images:
    img_with_boxes = results[i].plot()
    import cv2
    cv2.imshow("Detections", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
