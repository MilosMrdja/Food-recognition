from pathlib import Path

ROOT = Path("data/archive/Food-Image-Segmentation-using-YOLOv5-1/Food-Image-Segmentation-using-YOLOv5-1")

TEST_DIR = ROOT / "test"
TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "valid"

TEST_IMAGES_DIR = TEST_DIR / "images"
TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
VAL_IMAGES_DIR = VAL_DIR / "images"

TEST_LABELS_DIR = TEST_DIR / "labels"
TRAIN_LABELS_DIR = TRAIN_DIR / "labels"
VAL_LABELS_DIR = VAL_DIR / "labels"

OUTPUT_DIR = Path("data/cnn_dataset")

ROOT_CNN_DATASET= Path("data/cnn_dataset")

TRAIN_CNN = ROOT_CNN_DATASET / "train"
TEST_CNN = ROOT_CNN_DATASET / "test"
VALID_CNN = ROOT_CNN_DATASET / "valid"

MODELS_DIR = Path("models/cnn")
