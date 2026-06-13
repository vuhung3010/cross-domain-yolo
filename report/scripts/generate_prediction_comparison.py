#!/usr/bin/env python3
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

SOURCE = Path('/home/kacchan/Downloads/cityscapes/yolov5_format/images/train_foggy/aachen_000000_000019.jpg')
BASELINE = Path('/home/kacchan/VuHung/project/nhandang_project_2/runs/train/sanity_pr1_baseline/weights/best.pt')
IMAGE_LEVEL = Path('/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful/weights/best.pt')
BEST = Path('/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful_neck_all_alpha075_50ep/weights/best.pt')
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def predict(weights, title, conf=0.25):
    device = select_device('cpu')
    model = DetectMultiBackend(str(weights), device=device, data=str(ROOT / 'domain' / 'city_foggycity.yaml'))
    model.model.float().eval()
    stride, names = model.stride, model.names

    im0 = cv2.imread(str(SOURCE))
    if im0 is None:
        raise FileNotFoundError(SOURCE)
    im = letterbox(im0, 640, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = torch.from_numpy(im.copy()).to(device).float() / 255.0
    im = im[None]

    pred = model(im)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    pred = non_max_suppression(pred, conf, 0.45, None, False, max_det=1000)[0]

    annotator = Annotator(im0.copy(), line_width=3, example=str(names))
    count = 0
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()
        count = len(pred)
        for *xyxy, score, cls in reversed(pred):
            c = int(cls)
            label = f'{names[c]} {float(score):.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
    out = cv2.cvtColor(annotator.result(), cv2.COLOR_BGR2RGB)
    return out, f'{title}\n{count} detections at confidence ≥ {conf}'


panels = [
    predict(BASELINE, 'Source-only YOLO-G'),
    predict(IMAGE_LEVEL, 'YOLO-G image-level DA'),
    predict(BEST, 'Best multi-scale neck AdvGRL'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
fig.suptitle('Qualitative prediction comparison on the same Foggy Cityscapes image', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'fig_prediction_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'fig_prediction_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print(OUT / 'fig_prediction_comparison.png')
print(OUT / 'fig_prediction_comparison.pdf')
