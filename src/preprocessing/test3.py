import cv2
import random
from pathlib import Path
from utils import voc_to_yolo

from config import IMAGES_DIR, TRAIN_FILE, ANNOT_FILE, OUTPUT_DIR

# --- KONFIGURACIJA ---
SAMPLE_COUNT = 5

# --- UČITAVANJE ANOTACIJA ---
annots = {}
with open(ANNOT_FILE) as f:
    for line in f:
        fname, xmin, ymin, xmax, ymax, cls = line.strip().split()
        annots.setdefault(fname, []).append(
            (float(xmin), float(ymin), float(xmax), float(ymax), int(cls))
        )

# --- UČITAVANJE TRENING SKUPA ---
with open(TRAIN_FILE) as f:
    train_imgs = [line.strip() for line in f]

# --- NASUMIČAN IZBOR ---
sample_imgs = random.sample(train_imgs, min(SAMPLE_COUNT, len(train_imgs)))

# --- PROLAZ KROZ NASUMIČNE SLIKE ---
for fname in sample_imgs:
    matches = list(IMAGES_DIR.rglob(fname))
    if not matches:
        print(f"Slika {fname} nije pronađena!")
        continue

    img_path = matches[0]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Ne može da učita sliku {img_path}")
        continue

    h, w = img.shape[:2]

    # --- 1. PRIKAZ ORIGINALNIH VOC BOXOVA ---
    img_voc = img.copy()
    for xmin, ymin, xmax, ymax, cls in annots.get(fname, []):
        cv2.rectangle(img_voc, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        cv2.putText(img_voc, str(cls), (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow(f"{fname} - VOC boxes", img_voc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 2. YOLO LABEL ---
    label_dir = OUTPUT_DIR / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / fname.replace(".jpg", ".txt")
    img_yolo = img.copy()
    with open(label_path, "w") as out:
        for xmin, ymin, xmax, ymax, cls in annots.get(fname, []):
            # Konverzija VOC -> YOLO
            x_c, y_c, w_n, h_n = voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
            out.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

            # --- 3. Iscrtavanje YOLO bbox-a ---
            x1 = int((x_c - w_n/2) * w)
            y1 = int((y_c - h_n/2) * h)
            x2 = int((x_c + w_n/2) * w)
            y2 = int((y_c + h_n/2) * h)
            cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_yolo, "food", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # --- PRIKAZ YOLO BOXOVA ---
    cv2.imshow(f"{fname} - YOLO boxes", img_yolo)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:  # ESC za izlaz
        break
