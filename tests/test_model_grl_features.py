"""Tests for DA feature capture in models/yolo_GRL.py."""
from pathlib import Path

import torch

from models.yolo_GRL import Model

ROOT = Path(__file__).parent.parent


@torch.no_grad()
def test_yolo_grl_returns_da_feature_dict():
    model = Model(str(ROOT / 'configs/domain/yolov5l_GRL.yaml'), ch=3, nc=1).eval()
    det_pred, features = model(torch.zeros(1, 3, 64, 64))

    assert det_pred is not None
    assert set(features) >= {'sppf', 'neck_p3', 'neck_p4', 'neck_p5'}
    assert features['sppf'].shape[1] == 1024
    assert features['neck_p3'].shape[1] == 256
    assert features['neck_p4'].shape[1] == 512
    assert features['neck_p5'].shape[1] == 1024
    assert features['neck_p3'].shape[-1] > features['neck_p4'].shape[-1]
    assert features['neck_p4'].shape[-1] > features['neck_p5'].shape[-1]
