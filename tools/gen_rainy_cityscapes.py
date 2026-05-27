"""Offline RainMix generator: produces a parallel auxiliary domain from source images.

Port of DA-Detect/efficientderain-master/generate_rainy_cityscape.py adapted to
the YOLO-G project layout.

Usage:
    python tools/gen_rainy_cityscapes.py \
        --source-images /path/to/cityscapes/images/train \
        --rain-masks    /path/to/rain_masks/ \
        --output        /path/to/cityscapes/images/train_rainy

Outputs PNGs with the same filenames as the source images.
"""
import argparse
import os
import random
import sys
from glob import glob
from pathlib import Path

# Ensure the project root (parent of tools/) is on sys.path so
# `from tools import ...` works when the script is run as
# `python tools/gen_rainy_cityscapes.py` from the project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

# Local imports (the AugMix port)
from tools import augment_and_mix


_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def _image_files(directory: str):
    """Return sorted list of image filenames in *directory* (case-insensitive ext filter)."""
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(_IMAGE_EXTS)
    )


def random_rain_mask(rain_path: str) -> np.ndarray:
    files = _image_files(rain_path)
    idx = random.randint(0, len(files) - 1)
    return cv2.imread(os.path.join(rain_path, files[idx]))


def make_rain_layer(rain_mask: np.ndarray, size=(2048, 1024)) -> np.ndarray:
    layer = cv2.resize(rain_mask, size)
    layer = layer.astype(np.float32) / 255.0
    return cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)


def rain_aug(img_rgb: np.ndarray, rain_mask: np.ndarray, size=(2048, 1024)) -> np.ndarray:
    """Screen-blend a RainMix-augmented rain layer onto img_rgb."""
    img = img_rgb.astype(np.float32) / 255.0
    rain_layer = make_rain_layer(rain_mask, size=size)
    rain_aug_layer = augment_and_mix.augment_and_mix(rain_layer, severity=3, width=3, depth=-1) * 1.0
    out = img + rain_aug_layer - img * rain_aug_layer   # screen blend
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-images', required=True, help='dir of source PNG/JPG images')
    ap.add_argument('--rain-masks', required=True, help='dir of rain-streak mask images')
    ap.add_argument('--output', required=True, help='dir to write rainy-augmented images')
    ap.add_argument('--size', nargs=2, type=int, default=[2048, 1024], help='resize target (W H)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.rain_masks):
        sys.exit(f'ERROR: --rain-masks path does not exist or is not a directory: {args.rain_masks}')
    if not _image_files(args.rain_masks):
        sys.exit(f'ERROR: --rain-masks dir contains no image files (*.png/jpg/jpeg/bmp): {args.rain_masks}')

    src_paths = sorted(glob(os.path.join(args.source_images, '*')))
    Path(args.output).mkdir(parents=True, exist_ok=True)

    for src in tqdm(src_paths, desc='RainMix'):
        img_bgr = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, tuple(args.size))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = random_rain_mask(args.rain_masks)
        out_rgb = rain_aug(img_rgb, mask, size=tuple(args.size))
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output, Path(src).name), out_bgr)


if __name__ == '__main__':
    main()
