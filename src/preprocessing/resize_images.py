from PIL import Image
import os

from config import IMAGES_DIR, IMAGES_512_DIR

input_dir = IMAGES_DIR
output_dir = IMAGES_512_DIR
os.makedirs(output_dir, exist_ok=True)

count = 0
skipped_files = 0
error_files = 0
for root, dirs, files in os.walk(input_dir):
    rel_path = os.path.relpath(root, input_dir)
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)

    for img_name in files:
        if img_name.lower().endswith((".jpg", ".jpeg")):
            output_name = img_name
            output_path = os.path.join(output_subdir, output_name)

            if os.path.exists(output_path):
                skipped_files += 1
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
                error_files += 1
                print(f"Error processing {img_name}: {e}")

print(f"✅DONE: Processed {count} images")
print(f"📝Skipped files: {skipped_files} images")
print(f"❌ Error files: {error_files} images")
# 17829
