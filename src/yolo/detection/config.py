from pathlib import Path

YOLO_MODEL_V2 = Path("models/yolo/v2/best.pt")
YOLO_MODEL_V1 = Path("models/yolo/v1/best.pt")

ROOT = Path("data/yolo/processedV2")

IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"

TEST_IMAGES_DIR = IMAGES_DIR / "test"
TEST_LABELS_DIR = LABELS_DIR / "test"

TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
TRAIN_LABELS_DIR = LABELS_DIR / "train"

VALID_IMAGES_DIR = IMAGES_DIR / "valid"
VALID_LABELS_DIR = LABELS_DIR / "valid"

