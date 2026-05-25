"""Tests for the dumb model and DAImgHead."""
import torch
from models.da_classifier import DAImgHead


def test_da_img_head_output_shape():
    head = DAImgHead(in_channels=1024)
    x = torch.randn(2, 1024, 4, 4)
    y = head(x)
    assert y.shape == (2, 1, 4, 4)


def test_da_img_head_grad_flows():
    head = DAImgHead(in_channels=1024)
    x = torch.randn(2, 1024, 4, 4, requires_grad=True)
    head(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.yolo_GRL import Model


def test_forward_once_returns_tuple_with_backbone_feat():
    """Model._forward_once must return (det_pred, backbone_feat).

    backbone_feat is the SPPF output (layer 9) with channels=1024 for YOLOv5-L.
    Spatial dims = input // 32. We use 64x64 -> 2x2.
    """
    cfg = str(ROOT / 'configs' / 'domain' / 'yolov5l_GRL.yaml')
    model = Model(cfg=cfg, ch=3, nc=8)
    model.train()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 2, f'expected 2-tuple, got {type(out)} len={len(out) if hasattr(out, "__len__") else "n/a"}'
    det_pred, backbone_feat = out
    # det_pred in training is a LIST of per-scale tensors (P3, P4, P5)
    assert isinstance(det_pred, list) and len(det_pred) == 3
    # backbone_feat is the SPPF output: [B, 1024, H/32, W/32]
    assert backbone_feat.shape == (2, 1024, 2, 2), f'got {backbone_feat.shape}'
