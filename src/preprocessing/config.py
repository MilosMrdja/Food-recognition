from pathlib import Path

ROOT = Path("data/vfn_1_0")

IMAGES_DIR = ROOT / "Images"
IMAGES_512_DIR = ROOT / "Images_512"
META_DIR   = ROOT / "Meta"

OUTPUT_DIR = Path("data/processed")

TRAIN_FILE = META_DIR / "training.txt"
VAL_FILE   = META_DIR / "validation.txt"
TEST_FILE  = META_DIR / "testing.txt"
ANNOT_FILE = META_DIR / "annotations.txt"
