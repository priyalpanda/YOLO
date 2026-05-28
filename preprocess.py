import os
import shutil
from PIL import Image, ImageStat, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def gather_images(input_path):
    paths = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in EXTS:
                    paths.append(os.path.join(root, f))
    elif os.path.isfile(input_path):
        if os.path.splitext(input_path)[1].lower() in EXTS:
            paths = [input_path]
    return sorted(paths)


def is_corrupt(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return False
    except Exception:
        return True


def is_blank(path, thresh=5.0):
    try:
        with Image.open(path) as im:
            gray = im.convert('L')
            arr = np.array(gray)
            return float(arr.std()) < thresh
    except Exception:
        return True


def is_blurry_cv(path, lap_thresh=100.0):
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        fm = cv2.Laplacian(img, cv2.CV_64F).var()
        return fm < lap_thresh
    except Exception:
        return True


def sharpen_pil(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def preprocess_images(input_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_paths = gather_images(input_path)
    valid = []
    # filter corrupt/blank
    for p in tqdm(image_paths, desc='Scanning images'):
        if is_corrupt(p):
            continue
        if is_blank(p):
            continue
        valid.append(p)

    if not valid:
        return []

    processed = []
    for p in tqdm(valid, desc='Preprocessing'):
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                # detect blur and sharpen if needed
                if is_blurry_cv(p):
                    im = sharpen_pil(im)
                out_path = os.path.join(out_dir, os.path.basename(p))
                im.save(out_path)
                processed.append(out_path)
        except Exception:
            continue

    return processed
