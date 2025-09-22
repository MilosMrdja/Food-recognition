import shutil
from PIL import Image
import cv2
import numpy as np

from config import OUTPUT_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE, ANNOT_FILE, IMAGES_DIR
from utils import load_list, voc_to_yolo, build_image_map

def main():
    # 1. load lists
    splits = {
        "train": load_list(TRAIN_FILE),
        "val":   load_list(VAL_FILE),
        "test":  load_list(TEST_FILE)
    }

    # 2. Map
    image_map = build_image_map(IMAGES_DIR)

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

    # 5. create labels + vizuelna provera
    for split, name_set in splits.items():
        print(f"📝Split: {split}, images counter: {len(name_set)}")
        for fname in name_set:
            src_img = image_map[fname]

            img_pil = Image.open(src_img)
            w, h = img_pil.size
            img_np = np.array(img_pil)

            shutil.copy(src_img, OUTPUT_DIR / f"images/{split}/{fname}")

            label_path = OUTPUT_DIR / f"labels/{split}/{fname.replace('.jpg', '.txt')}"
            bboxes = annots.get(fname, [])
            if len(bboxes) > 1:
                multi_bbox_count += 1
            with open(label_path, "w") as out:
                for xmin, ymin, xmax, ymax, cls in bboxes:
                    x_c, y_c, w_n, h_n = voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
                    out.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

                    # Vizuelizacija bbox-a na slici
                    x1 = int((x_c - w_n/2) * w)
                    y1 = int((y_c - h_n/2) * h)
                    x2 = int((x_c + w_n/2) * w)
                    y2 = int((y_c + h_n/2) * h)
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_np, "food", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Prikaz slike sa box-evima
            cv2.imshow(f"{split}: {fname}", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                return

    print(f"📊 Number of images with multi bounding boxes: {multi_bbox_count}")

if __name__ == "__main__":
    main()
