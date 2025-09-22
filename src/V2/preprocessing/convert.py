import shutil
from config import ROOT, SPLITS, OUTPUT_DIR

for split in SPLITS:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

count = 0
for split in SPLITS:
    old_images_dir = ROOT / split / "images"
    old_labels_dir = ROOT / split / "labels"

    new_images_dir = OUTPUT_DIR / "images" / split
    new_labels_dir = OUTPUT_DIR / "labels" / split

    for img_file in old_images_dir.glob("*.jpg"):
        shutil.copy(img_file, new_images_dir / img_file.name)

    for txt_file in old_labels_dir.glob("*.txt"):
        new_lines = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                parts[0] = "0"
                new_lines.append(" ".join(parts))


        # if len(new_lines) > 6:
        #     new_lines = new_lines[:6]

        with open(new_labels_dir / txt_file.name, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        count += 1
        if count % 100 == 0:
            print(f"Preprocessed {count} images")

print(f"✅ Dataset is ready for training: '{OUTPUT_DIR}'")
print(f"📝 Total images processed: {count}")
# 3496
