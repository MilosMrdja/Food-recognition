
import shutil
from pathlib import Path

from config import IMAGES_DIR, META_DIR, ROOT_CNN2_DATASET

splits = ["training", "validation", "testing"]

def read_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def create_split_folders():

    for split in splits:
        for class_id in range(82):
            (ROOT_CNN2_DATASET / split / str(class_id)).mkdir(parents=True, exist_ok=True)

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
        dst = ROOT_CNN2_DATASET / split / cls / fname
        shutil.copy2(src, dst)

def main():
    ROOT_CNN2_DATASET.mkdir(exist_ok=True)
    create_split_folders()

    name_to_class = build_filename_to_class()

    for split in splits:
        txt_file = META_DIR / f"{split}.txt"
        files = read_list(txt_file)
        print(f"Processing {split}: {len(files)} images")
        copy_images(files, split, name_to_class)

    print("Done! Saved to:", ROOT_CNN2_DATASET)

if __name__ == "__main__":
    main()
