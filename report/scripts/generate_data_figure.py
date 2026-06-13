#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path('/home/kacchan/Downloads/cityscapes/yolov5_format/images')
OUT = Path(__file__).resolve().parents[1] / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ('Cityscapes source', ROOT / 'train' / 'aachen_000000_000019.jpg'),
    ('Foggy Cityscapes target', ROOT / 'train_foggy' / 'aachen_000000_000019.jpg'),
    ('RainMix auxiliary', ROOT / 'train_rainy' / 'aachen_000000_000019.jpg'),
]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, (title, path) in zip(axes, SAMPLES):
    if not path.exists():
        raise FileNotFoundError(path)
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

fig.suptitle('Representative data domains used in the ablation study', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'fig_data_domains.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'fig_data_domains.pdf', bbox_inches='tight')
plt.close(fig)
print(OUT / 'fig_data_domains.png')
print(OUT / 'fig_data_domains.pdf')
