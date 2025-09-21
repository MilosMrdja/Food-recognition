import shutil
from PIL import Image

from config import OUTPUT_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE, ANNOT_FILE, IMAGES_512_DIR
from utils import load_list, voc_to_yolo, build_image_map

def main():
    # 1. load lsits
    splits = {
        "train": load_list(TRAIN_FILE),
        "val":   load_list(VAL_FILE),
        "test":  load_list(TEST_FILE)
    }

    # 2. Map
    image_map = build_image_map(IMAGES_512_DIR)

    # 3. load config
    annots = {}
    with open(ANNOT_FILE) as f:
        for line in f:
            fname, xmin, ymin, xmax, ymax, cls = line.split()
            annots.setdefault(fname, []).append(
                (float(xmin), float(ymin), float(xmax), float(ymax), int(cls))
            )

    # 4. create dirs
    multi_bbox_count = 0
    for s in ["train","val","test"]:
        (OUTPUT_DIR / f"images/{s}").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"labels/{s}").mkdir(parents=True, exist_ok=True)

    # 5. create labels
    for split, name_set in splits.items():
        print(f"📝Split: {split}, images counter: {len(name_set)}")
        for fname in name_set:
            src_img = image_map[fname]

            img = Image.open(src_img)
            w, h = img.size

            shutil.copy(src_img, OUTPUT_DIR / f"images/{split}/{fname}")

            label_path = OUTPUT_DIR / f"labels/{split}/{fname.replace('.jpg', '.txt')}"
            bboxes = annots.get(fname, [])
            if len(bboxes) > 1:
                multi_bbox_count += 1
            with open(label_path, "w") as out:
                for xmin, ymin, xmax, ymax, cls in bboxes:
                    x_c, y_c, w_n, h_n = voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
                    out.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    print(f"📊 Number of images with multi bounding boxes: {multi_bbox_count}")

if __name__ == "__main__":
    main()

