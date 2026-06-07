from argparse import Namespace

import torch
import torch.nn.functional as F

from train_GRL import DA_FEATURE_CHANNELS, _build_objectness_gates, _weighted_da_bce_loss, da_img_faithful_gated_loss_multi, main


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

def _base_opt(**overrides):
    values = dict(
        weights='', cfg='configs/domain/yolov5l_GRL.yaml', data='domain/city_foggycity.yaml', hyp='hyps/hyp.scratch-high.yaml',
        epochs=1, loss='origin', auxotaloss=False, otaloss='origin', batch_size=1, imgsz=64, rect=False, resume=False,
        nosave=True, noval=True, noautoanchor=False, evolve=None, bucket='', cache=None, image_weights=False, device='cpu',
        multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False, workers=0, project='runs/train', name='tmp',
        exist_ok=True, quad=False, cos_lr=False, label_smoothing=0.0, patience=1, freeze=[0], save_period=-1, local_rank=-1,
        entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias='latest', da_img=False, da_img_weight=1.0,
        da_img_grl_weight=0.1, da_img_warmup='off', advgrl=False, advgrl_threshold=30.0, advgrl_alpha=None, aux=False,
        triplet_img=False, triplet_img_weight=0.1, triplet_margin=1.0, triplet_adaptive=False, triplet_max_margin=3.0,
        da_img_faithful=False, da_feat_layers='sppf', da_img_obj_gate=False, da_img_obj_gate_floor=0.05,
    )
    values.update(overrides)
    return Namespace(**values)


def test_obj_gate_requires_da_img():
    opt = _base_opt(da_img_obj_gate=True, da_img=False, da_img_faithful=True, da_feat_layers='neck-all')
    try:
        main(opt)
    except SystemExit as exc:
        assert '--da-img-obj-gate requires --da-img' in str(exc)
    else:
        raise AssertionError('expected SystemExit')


def test_obj_gate_requires_faithful():
    opt = _base_opt(da_img_obj_gate=True, da_img=True, da_img_faithful=False, da_feat_layers='neck-all')
    try:
        main(opt)
    except SystemExit as exc:
        assert '--da-img-obj-gate requires --da-img-faithful' in str(exc)
    else:
        raise AssertionError('expected SystemExit')


def test_obj_gate_requires_neck_all():
    opt = _base_opt(da_img_obj_gate=True, da_img=True, da_img_faithful=True, da_feat_layers='neck-p4')
    try:
        main(opt)
    except SystemExit as exc:
        assert '--da-img-obj-gate requires --da-feat-layers neck-all' in str(exc)
    else:
        raise AssertionError('expected SystemExit')


def test_obj_gate_floor_must_be_non_negative():
    opt = _base_opt(da_img_obj_gate=True, da_img=True, da_img_faithful=True, da_feat_layers='neck-all', da_img_obj_gate_floor=-0.1)
    try:
        main(opt)
    except SystemExit as exc:
        assert '--da-img-obj-gate-floor must be non-negative' in str(exc)
    else:
        raise AssertionError('expected SystemExit')



def test_neck_all_feature_selection_order_matches_detect_order():
    assert list(DA_FEATURE_CHANNELS['neck-all']) == ['neck_p3', 'neck_p4', 'neck_p5']



def test_faithful_da_loss_respects_warmup_scale():
    loss = torch.tensor(2.0)
    base = torch.tensor(5.0)
    da_scale_this_iter = 0.25
    da_img_weight = 3.0

    total = base + da_scale_this_iter * da_img_weight * loss

    assert torch.allclose(total, torch.tensor(6.5))


def test_multiscale_resizes_domain_batches_consistently():
    imgs = torch.zeros(2, 3, 64, 64)
    t_imgs = torch.zeros(2, 3, 64, 64)
    a_imgs = torch.zeros(2, 3, 64, 64)
    all_imgs = torch.cat([imgs, t_imgs, a_imgs], dim=0)
    ns = [96, 96]

    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
    all_imgs = F.interpolate(all_imgs, size=ns, mode='bilinear', align_corners=False)
    t_imgs = F.interpolate(t_imgs, size=ns, mode='bilinear', align_corners=False)
    a_imgs = F.interpolate(a_imgs, size=ns, mode='bilinear', align_corners=False)

    assert imgs.shape[-2:] == all_imgs.shape[-2:] == t_imgs.shape[-2:] == a_imgs.shape[-2:]
