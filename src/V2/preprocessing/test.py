import hashlib

from config import SPLITS, OUTPUT_DIR

hashes = {}
duplicates = []

for split in SPLITS:
    images_dir = OUTPUT_DIR / split / "images"
    for img_file in images_dir.glob("*.jpg"):
        with open(img_file, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash in hashes:
            duplicates.append((img_file.name, hashes[file_hash]))
        else:
            hashes[file_hash] = img_file.name

print(f"Duplicates: {duplicates}")
