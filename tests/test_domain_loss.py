"""Tests for utils/domain_loss.py"""
import math
import torch
import torch.nn.functional as F
from utils.domain_loss import da_img_faithful_loss_pair, da_img_loss, triplet_img_loss


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


def test_da_img_faithful_loss_pair_uses_source_zero_target_one_labels():
    source_logits = torch.tensor([[[[0.25, -0.25]]]], dtype=torch.float32)
    target_logits = torch.tensor([[[[-0.5, 0.5]]]], dtype=torch.float32)

    loss = da_img_faithful_loss_pair(source_logits, target_logits)

    expected_source = F.binary_cross_entropy_with_logits(source_logits, torch.zeros_like(source_logits))
    expected_target = F.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits))
    expected = 0.5 * expected_source + 0.5 * expected_target
    assert torch.allclose(loss, expected)


def test_da_img_faithful_loss_pair_matches_original_yolog_flattening():
    source_logits = torch.randn(2, 1, 3, 4)
    target_logits = torch.randn(2, 1, 3, 4)

    loss = da_img_faithful_loss_pair(source_logits, target_logits)

    source_feature = source_logits.permute(0, 2, 3, 1).reshape(source_logits.shape[0], -1)
    target_feature = target_logits.permute(0, 2, 3, 1).reshape(target_logits.shape[0], -1)
    expected_source = F.binary_cross_entropy_with_logits(source_feature, torch.zeros_like(source_feature))
    expected_target = F.binary_cross_entropy_with_logits(target_feature, torch.ones_like(target_feature))
    expected = 0.5 * expected_source + 0.5 * expected_target
    assert torch.allclose(loss, expected)


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
