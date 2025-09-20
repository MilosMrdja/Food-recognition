import os
import pandas as pd

image_dir = "data/processed/images"
label_dir = "data/processed/labels"
os.makedirs(label_dir, exist_ok=True)

annotations = pd.read_csv("data/vfn_1_0/meta/annotations.txt", sep=" ", header=None)
annotations.columns = ["img_name","x_min","y_min","x_max","y_max","class_id"]

count_labels = 0

for idx, row in annotations.iterrows():
    img_name = row["img_name"]
    class_id = str(int(row["class_id"]))

    txt_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_dir, txt_name)

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    w, h = 512, 512
    x_center = ((row["x_min"] + row["x_max"]) / 2) / w
    y_center = ((row["y_min"] + row["y_max"]) / 2) / h
    box_width = (row["x_max"] - row["x_min"]) / w
    box_height = (row["y_max"] - row["y_min"]) / h

    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    count_labels += 1
    if count_labels % 100 == 0:
        print(f"📄 Processed {count_labels} labels..")

print(f"✅ Done! Processed {count_labels} labels")
