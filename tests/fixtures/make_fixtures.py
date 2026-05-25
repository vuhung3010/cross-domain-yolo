"""Generate deterministic toy images + YOLO-format labels for tests/fixtures/.

Usage: python tests/fixtures/make_fixtures.py
Produces 4 source images (with labels), 4 target images (no labels),
4 aux images (no labels), all 64x64 RGB PNGs.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
SEED = 1234
IMG_SIZE = 64


def make_image(rng, kind):
    """kind in {'source', 'target', 'aux'} — each has a distinct color bias."""
    bias = {'source': [200, 100, 100], 'target': [100, 200, 100], 'aux': [100, 100, 200]}[kind]
    img = rng.integers(0, 80, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = (img + np.array(bias, dtype=np.uint8)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    rng = np.random.default_rng(SEED)
    for kind in ('source', 'target', 'aux'):
        img_dir = ROOT / kind / 'images'
        lbl_dir = ROOT / kind / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = make_image(rng, kind)
            img.save(img_dir / f'{kind}_{i:02d}.png')
            # YOLO label format: class cx cy w h (normalized). Source has 1 box; others empty.
            if kind == 'source':
                (lbl_dir / f'{kind}_{i:02d}.txt').write_text('0 0.5 0.5 0.3 0.3\n')
            else:
                (lbl_dir / f'{kind}_{i:02d}.txt').write_text('')
    # Manifests for YOLO dataloader
    for kind in ('source', 'target', 'aux'):
        manifest = ROOT / f'{kind}.txt'
        imgs = sorted((ROOT / kind / 'images').glob('*.png'))
        manifest.write_text('\n'.join(str(p) for p in imgs) + '\n')
    print(f'Fixtures written to {ROOT}')


if __name__ == '__main__':
    main()
