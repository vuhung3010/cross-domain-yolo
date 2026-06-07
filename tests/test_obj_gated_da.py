import torch
import torch.nn.functional as F

from train_GRL import _build_objectness_gates, _weighted_da_bce_loss, da_img_faithful_gated_loss_multi


def _detect_tensor(obj_logits):
    # YOLO Detect train output layout: [B, A, H, W, no], objectness at index 4.
    b, a, h, w = obj_logits.shape
    pred = torch.zeros(b, a, h, w, 6, dtype=obj_logits.dtype, device=obj_logits.device)
    pred[..., 4] = obj_logits
    return pred


def test_build_objectness_gates_sigmoid_max_floor_detach():
    p3_obj = torch.tensor([[[[-10.0, 0.0], [2.0, -2.0]], [[1.0, -1.0], [0.5, 3.0]]]], requires_grad=True)
    p4_obj = torch.full((1, 2, 1, 1), -10.0, requires_grad=True)
    p5_obj = torch.full((1, 2, 1, 1), 0.0, requires_grad=True)

    gates = _build_objectness_gates(
        [_detect_tensor(p3_obj), _detect_tensor(p4_obj), _detect_tensor(p5_obj)],
        ['neck_p3', 'neck_p4', 'neck_p5'],
        floor=0.05,
    )

    expected_p3 = torch.sigmoid(p3_obj).amax(dim=1, keepdim=True).clamp_min(0.05)
    assert list(gates) == ['neck_p3', 'neck_p4', 'neck_p5']
    assert gates['neck_p3'].shape == (1, 1, 2, 2)
    assert torch.allclose(gates['neck_p3'], expected_p3)
    assert torch.all(gates['neck_p4'] == torch.full((1, 1, 1, 1), 0.05))
    assert torch.allclose(gates['neck_p5'], torch.full((1, 1, 1, 1), 0.5))
    assert gates['neck_p3'].requires_grad is False
    assert gates['neck_p4'].requires_grad is False
    assert gates['neck_p5'].requires_grad is False


def test_build_objectness_gates_rejects_non_neck_all_order():
    pred = [_detect_tensor(torch.zeros(1, 2, 1, 1)) for _ in range(3)]

    try:
        _build_objectness_gates(pred, ['sppf'], floor=0.05)
    except ValueError as exc:
        assert 'neck-all' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_weighted_da_bce_loss_normalizes_by_gate_sum_clamp():
    logits = torch.tensor([[[[0.0, 1.0], [-1.0, 2.0]]]])
    gate = torch.tensor([[[[1.0, 0.5], [0.25, 0.25]]]])
    raw = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
    expected = (raw * gate).sum() / gate.sum().clamp_min(1.0)

    actual = _weighted_da_bce_loss(logits, 1.0, gate)

    assert torch.allclose(actual, expected)


def test_weighted_da_bce_loss_uses_clamp_when_gate_sum_below_one():
    logits = torch.tensor([[[[0.0, 0.0]]]])
    gate = torch.tensor([[[[0.1, 0.2]]]])
    raw = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    expected = (raw * gate).sum() / torch.tensor(1.0)

    actual = _weighted_da_bce_loss(logits, 0.0, gate)

    assert torch.allclose(actual, expected)


def test_da_img_faithful_gated_loss_multi_averages_scales():
    source_logits = {
        'neck_p3': torch.zeros(1, 1, 1, 1),
        'neck_p4': torch.ones(1, 1, 1, 1),
        'neck_p5': torch.full((1, 1, 1, 1), -1.0),
    }
    target_logits = {
        'neck_p3': torch.ones(1, 1, 1, 1),
        'neck_p4': torch.zeros(1, 1, 1, 1),
        'neck_p5': torch.full((1, 1, 1, 1), 2.0),
    }
    source_gates = {name: torch.ones(1, 1, 1, 1) for name in source_logits}
    target_gates = {name: torch.ones(1, 1, 1, 1) for name in target_logits}

    per_scale = []
    for name in source_logits:
        s = F.binary_cross_entropy_with_logits(source_logits[name], torch.zeros_like(source_logits[name]))
        t = F.binary_cross_entropy_with_logits(target_logits[name], torch.ones_like(target_logits[name]))
        per_scale.append(0.5 * s + 0.5 * t)
    expected = torch.stack(per_scale).mean()

    actual = da_img_faithful_gated_loss_multi(source_logits, target_logits, source_gates, target_gates)

    assert torch.allclose(actual, expected)
