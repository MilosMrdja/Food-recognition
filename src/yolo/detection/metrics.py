from ultralytics import YOLO

model = YOLO("models/yolo/v2/best.pt")
metrics = model.val(data="data.yml")
print(metrics)