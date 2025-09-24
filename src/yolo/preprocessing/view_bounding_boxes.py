
from PIL import Image, ImageDraw

from config import OUTPUT_DIR, SPLITS

# View bounding boxes on the images
for split in SPLITS:
    images_dir = OUTPUT_DIR / "images" / split
    labels_dir = OUTPUT_DIR / "labels" / split

    print(f"Processing {split} images...")

    for img_file in images_dir.glob("*.jpg"):
        label_file = labels_dir / img_file.name.replace(".jpg", ".txt")

        if not label_file.exists():
            print(f"⚠️ Label file missing for {img_file.name}")
            continue

        img = Image.open(img_file)
        draw = ImageDraw.Draw(img)
        w, h = img.size

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_c, y_c, w_box, h_box = map(float, parts)
                x0 = (x_c - w_box / 2) * w
                y0 = (y_c - h_box / 2) * h
                x1 = (x_c + w_box / 2) * w
                y1 = (y_c + h_box / 2) * h

                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)


        img.show()
