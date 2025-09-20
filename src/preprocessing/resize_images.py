from PIL import Image
import os

input_dir = "data/vfn_1_0/Images"
output_dir = "data/processed/images"
os.makedirs(output_dir, exist_ok=True)

count = 0

for root, dirs, files in os.walk(input_dir):
    rel_path = os.path.relpath(root, input_dir)
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)

    for img_name in files:
        if img_name.lower().endswith((".jpg", ".jpeg")):
            output_name = os.path.splitext(img_name)[0] + ".png"
            output_path = os.path.join(output_subdir, output_name)

            if os.path.exists(output_path):
                print(f"Skipping {img_name}")
                continue

            img_path = os.path.join(root, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((512, 512))
                img.save(output_path)
                count += 1
                if count % 100 == 0:
                    print(f"Preprocessed {count} images")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

print(f"✅DONE: Processed {count} images")
# 17829
