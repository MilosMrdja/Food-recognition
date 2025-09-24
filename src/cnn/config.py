from pathlib import Path


#------ FINAL for vfn dataset--------
ROOT_CNN2_DATASET= Path("data/cnn/split_dataset")

TRAIN_CNN2 = ROOT_CNN2_DATASET / "training"
VALID_CNN2 = ROOT_CNN2_DATASET / "validation"
TEST_CNN2 = ROOT_CNN2_DATASET / "testing"

MODELS_DIR = Path("models/cnn")
