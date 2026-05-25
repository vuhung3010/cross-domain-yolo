"""Tests for utils/domain_loss.py"""
import math
import torch
from utils.domain_loss import da_img_loss, triplet_img_loss


def test_da_img_loss_shape_and_value():
    # 2 source + 2 target, [B=4, 1, H=2, W=2]. With zero logits everywhere,
    # BCE-with-logits(0, label) = log(2) for both label values.
    logits = torch.zeros(4, 1, 2, 2)
    loss = da_img_loss(logits, source_count=2)
    assert loss.shape == ()
    assert math.isclose(loss.item(), math.log(2.0), rel_tol=1e-5)


def test_da_img_loss_perfect_prediction():
    # Source: large positive logits (predict 1); target: large negative (predict 0).
    logits = torch.cat([
        torch.full((2, 1, 2, 2), 10.0),    # source
        torch.full((2, 1, 2, 2), -10.0),   # target
    ], dim=0)
    loss = da_img_loss(logits, source_count=2)
    assert loss.item() < 0.001


def test_triplet_loss_zero_when_neg_far():
    # anchor and positive identical; negative is far.
    a = torch.zeros(1, 8)
    p = torch.zeros(1, 8)
    n = torch.full((1, 8), 10.0)
    loss = triplet_img_loss(a, p, n, margin=1.0)
    # d(a,p)=0, d(a,n)≈28.3, margin=1 -> max(-27.3, 0) = 0
    assert loss.item() == 0.0


def test_triplet_loss_positive_when_pos_far():
    a = torch.zeros(1, 8)
    p = torch.full((1, 8), 5.0)   # far positive
    n = torch.full((1, 8), 0.1)   # close negative
    loss = triplet_img_loss(a, p, n, margin=1.0)
    assert loss.item() > 0.0
