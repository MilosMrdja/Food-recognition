from pathlib import Path

ROOT = Path("data/yolo/archive/Food-Image-Segmentation-using-YOLOv5-1/Food-Image-Segmentation-using-YOLOv5-1")


OUTPUT_DIR = Path("data/yolo/processedV2")

DIR_LABELS = OUTPUT_DIR / "labels"

TRAIN_FOLDER = ROOT / "train"
TEST_FOLDER   = ROOT / "test"
VALID_FOLDER  = ROOT / "valid"

SPLITS = ["train", "valid", "test"]