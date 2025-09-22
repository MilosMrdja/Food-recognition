import os

import cv2


images_dir = "data/processed/images/train"
labels_dir = "data/processed/labels/train"
# PRINT LABELS
for img_name in os.listdir(images_dir):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        for line in f:
            cls, x_c, y_c, bw, bh = map(float, line.strip().split())
            # Pretvori iz YOLO formata u piksele
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Check labels", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()