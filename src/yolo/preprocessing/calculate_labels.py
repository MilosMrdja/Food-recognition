
from config import SPLITS, DIR_LABELS
count = 0
for split in SPLITS:
    split_dir = DIR_LABELS / split
    label_files = list(split_dir.glob("*.txt"))

    total_labels = 0
    print(f"\n📂 Split: {split}")
    print(f"📄 Total label files: {len(label_files)}")

    for txt_file in label_files:

        with open(txt_file, "r") as f:
            lines = [line for line in f if line.strip()]
            num_labels = len(lines)
            count += len(lines)
            total_labels += num_labels
            print(f"{txt_file.name}: {num_labels} label(s)")

    print(f"📝 Total labels in {split}: {total_labels}")

print(f"📝Total labels: {count}")
# max6 = max = 10000+
# max4 = 9755
# max2 = 5975
