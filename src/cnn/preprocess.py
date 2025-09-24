import os
import shutil
from pathlib import Path

DATASET_DIR = Path("data/vfn_1_0")
IMAGES_DIR = DATASET_DIR / "Images"
META_DIR = DATASET_DIR / "Meta"
OUTPUT_DIR = Path("data/split_dataset")

splits = ["training", "validation", "testing"]

def read_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def create_split_folders():

    for split in splits:
        for class_id in range(82):
            (OUTPUT_DIR / split / str(class_id)).mkdir(parents=True, exist_ok=True)

def build_filename_to_class():

    mapping = {}
    for class_id in range(82):
        class_path = IMAGES_DIR / str(class_id)
        for file in class_path.iterdir():
            if file.is_file():
                mapping[file.name] = str(class_id)
    return mapping

def copy_images(file_list, split, name_to_class):
    for fname in file_list:
        if fname not in name_to_class:
            print(f"[WARN] {fname} not found in images/*")
            continue
        cls = name_to_class[fname]
        src = IMAGES_DIR / cls / fname
        dst = OUTPUT_DIR / split / cls / fname
        shutil.copy2(src, dst)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    create_split_folders()

    name_to_class = build_filename_to_class()

    for split in splits:
        txt_file = META_DIR / f"{split}.txt"
        files = read_list(txt_file)
        print(f"Processing {split}: {len(files)} images")
        copy_images(files, split, name_to_class)

    print("Done! Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
