import shutil
from pathlib import Path
from config import (
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR,
    VAL_IMAGES_DIR,   VAL_LABELS_DIR,
    TEST_IMAGES_DIR,  TEST_LABELS_DIR,
    OUTPUT_DIR
)
id_to_name = {
    0:  "AW-cola",
    1:  "Beijing-Beef",
    2:  "Chow-Mein",
    3:  "Fried-Rice",
    4:  "Hashbrown",
    5:  "Honey-Walnut-Shrimp",
    6:  "Kung-Pao-Chicken",
    7:  "String-Bean-Chicken-Breast",
    8:  "Super-Greens",
    9:  "The-Original-Orange-Chicken",
    10: "White-Steamed-Rice",
    11: "black-pepper-rice-bowl",
    12: "burger",
    13: "carrot_eggs",
    14: "cheese-burger",
    15: "chicken-waffle",
    16: "chicken_nuggets",
    17: "chinese-cabbage",
    18: "chinese-sausage",
    19: "crispy-corn",
    20: "curry",
    21: "french-fries",
    22: "fried-chicken",
    23: "fried_chicken",
    24: "fried-dumplings",
    25: "fried-eggs",
    26: "mango-chicken-pocket",
    27: "mozza-burger",
    28: "mung-bean_sprouts",
    29: "nugget",
    30: "perkedel",
    31: "rice",
    32: "sprite",
    33: "tostitos-cheese-dip-sauce",
    34: "triangle_hash_brown",
    35: "water_spinach",
}


def convert_split(images_dir: Path, labels_dir: Path, split_name: str):
    out_dir = OUTPUT_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for label_file in labels_dir.glob("*.txt"):
        img_name = label_file.stem
        img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            continue

        with open(label_file) as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            class_id = int(first_line.split()[0])   # broj 0–35

        class_name = id_to_name[class_id]
        class_folder = out_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, class_folder / img_path.name)


convert_split(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, "train")
convert_split(VAL_IMAGES_DIR,   VAL_LABELS_DIR,   "valid")
convert_split(TEST_IMAGES_DIR,  TEST_LABELS_DIR,  "test")
