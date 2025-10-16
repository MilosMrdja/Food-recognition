from ultralytics import YOLO
import matplotlib.pyplot as plt
from config import YOLO_MODEL_V2

model = YOLO(YOLO_MODEL_V2)
metrics = model.val(data="data.yml", split="test")

results = metrics.results_dict

precision = results.get("metrics/precision(B)", 0)
recall    = results.get("metrics/recall(B)", 0)
map50     = results.get("metrics/mAP50(B)", 0)
map50_95  = results.get("metrics/mAP50-95(B)", 0)

labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
values = [precision, recall, map50, map50_95]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.ylim(0, 1.1)
plt.title("YOLO Validation Metrics")
plt.ylabel("Score")

for bar, val in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()
