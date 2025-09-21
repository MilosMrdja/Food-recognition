from pathlib import Path


def load_list(path):
    return set(open(path).read().split())

def voc_to_yolo(xmin, ymin, xmax, ymax, w_img, h_img):
    x_center = ((xmin + xmax) / 2) / w_img
    y_center = ((ymin + ymax) / 2) / h_img
    width  = (xmax - xmin) / w_img
    height = (ymax - ymin) / h_img
    return x_center, y_center, width, height

def build_image_map(images_dir: Path):

    image_map = {}
    for cat_folder in images_dir.iterdir():
        for img in cat_folder.glob("*.jpg"):
            image_map[img.name] = img
    return image_map
