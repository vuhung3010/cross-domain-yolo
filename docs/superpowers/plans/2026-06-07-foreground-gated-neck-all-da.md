# Foreground/Objectness-Gated Neck-All DA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exact report semantics for `--da-img-obj-gate`: faithful neck-all DA weighted by detached Detect objectness gates.

**Architecture:** Keep DA logic outside `models/yolo_GRL.py`. The model already returns Detect predictions plus feature dict taps (`sppf`, `neck_p3`, `neck_p4`, `neck_p5`). Add focused helpers in `train_GRL.py` for objectness gate construction and gated multi-scale DA loss, then wire them into the existing faithful DA branch.

**Tech Stack:** Python, PyTorch, YOLOv5/YOLO-G, pytest.

---

## File Structure

- Modify `train_GRL.py`
  - Add `_build_objectness_gates(det_pred, feature_names, floor)`.
  - Add `_weighted_da_bce_loss(logits, target_value, gate)`.
  - Add `da_img_faithful_gated_loss_multi(source_logits, target_logits, source_gates, target_gates)`.
  - Add CLI flags `--da-img-obj-gate`, `--da-img-obj-gate-floor`.
  - Add validation rules.
  - Wire gated loss into faithful DA branch, including detached `L_c` path.
- Modify `tests/test_advgrl.py` or create `tests/test_obj_gated_da.py`
  - Prefer new file for focused unit tests.
- Modify `tests/test_train_smoke.py`
  - Add smoke args for objectness-gated faithful neck-all DA if smoke matrix is centralized.
- No change to `models/yolo_GRL.py` unless tests reveal current feature dict behavior regressed.
- No change to `utils/da_logger.py`; exact report semantics add no gate diagnostics.

---

## Task 1: Add Objectness Gate Unit Tests

**Files:**
- Create: `tests/test_obj_gated_da.py`
- Modify: none

- [ ] **Step 1: Write failing tests for objectness gates**

Create `tests/test_obj_gated_da.py`:

```python
import torch

from train_GRL import _build_objectness_gates


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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_obj_gated_da.py -v
```

Expected: FAIL with import error or missing `_build_objectness_gates`.

- [ ] **Step 3: Commit failing tests if your workflow commits red tests**

Usually skip committing red tests unless using strict TDD checkpoints. If committing:

```bash
git add tests/test_obj_gated_da.py
git commit -m "test: cover objectness gate construction"
```

---

## Task 2: Implement Objectness Gate Helper

**Files:**
- Modify: `train_GRL.py:75-100`
- Test: `tests/test_obj_gated_da.py`

- [ ] **Step 1: Add helper implementation**

In `train_GRL.py`, after `_feature()` and before `_gap_mean()`, add:

```python
def _build_objectness_gates(det_pred, feature_names, floor=0.05):
    """Build detached [B,1,H,W] objectness gates for faithful neck-all DA."""
    expected = ['neck_p3', 'neck_p4', 'neck_p5']
    if list(feature_names) != expected:
        raise ValueError('--da-img-obj-gate is only supported with --da-feat-layers neck-all')
    if floor < 0:
        raise ValueError('--da-img-obj-gate-floor must be non-negative')
    if not isinstance(det_pred, (list, tuple)) or len(det_pred) < 3:
        raise ValueError('Detect predictions must be a P3/P4/P5 list for objectness gates')

    gates = {}
    for name, pred in zip(expected, det_pred[:3]):
        # Training Detect output layout is [B, anchors, H, W, outputs].
        obj = pred[..., 4].sigmoid().amax(dim=1, keepdim=True).clamp_min(floor)
        gates[name] = obj.detach()
    return gates
```

- [ ] **Step 2: Run gate tests**

Run:

```bash
pytest tests/test_obj_gated_da.py::test_build_objectness_gates_sigmoid_max_floor_detach tests/test_obj_gated_da.py::test_build_objectness_gates_rejects_non_neck_all_order -v
```

Expected: PASS.

- [ ] **Step 3: Commit helper**

```bash
git add train_GRL.py tests/test_obj_gated_da.py
git commit -m "feat: build detached objectness gates"
```

---

## Task 3: Add Gated DA Loss Unit Tests

**Files:**
- Modify: `tests/test_obj_gated_da.py`
- Test: `tests/test_obj_gated_da.py`

- [ ] **Step 1: Add failing loss tests**

Append to `tests/test_obj_gated_da.py`:

```python
import torch.nn.functional as F

from train_GRL import _weighted_da_bce_loss, da_img_faithful_gated_loss_multi


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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_obj_gated_da.py -v
```

Expected: FAIL with missing `_weighted_da_bce_loss` or `da_img_faithful_gated_loss_multi`.
---

## Task 4: Implement Gated DA Loss Helpers

**Files:**
- Modify: `train_GRL.py:75-110`
- Test: `tests/test_obj_gated_da.py`

- [ ] **Step 1: Add `F` import if missing**

At imports in `train_GRL.py`, ensure this exists:

```python
import torch.nn.functional as F
```

If `F` is already imported, do not duplicate it.

- [ ] **Step 2: Add loss helper implementations**

In `train_GRL.py`, after `_build_objectness_gates()`, add:

```python
def _weighted_da_bce_loss(logits, target_value, gate):
    """Spatial BCE weighted by detached objectness gate, normalized by gate mass."""
    target = torch.full_like(logits, float(target_value))
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    if gate.shape != raw.shape:
        raise ValueError(f'gate shape {tuple(gate.shape)} must match logits shape {tuple(raw.shape)}')
    return (raw * gate).sum() / gate.sum().clamp_min(1.0)


def da_img_faithful_gated_loss_multi(source_logits, target_logits, source_gates, target_gates):
    """Faithful source=0/target=1 DA loss averaged over gated neck-all scales."""
    losses = []
    for name in source_logits:
        source_loss = _weighted_da_bce_loss(source_logits[name], 0.0, source_gates[name])
        target_loss = _weighted_da_bce_loss(target_logits[name], 1.0, target_gates[name])
        losses.append(0.5 * source_loss + 0.5 * target_loss)
    return torch.stack(losses).mean()
```

- [ ] **Step 3: Run all object-gated unit tests**

Run:

```bash
pytest tests/test_obj_gated_da.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit gated loss helpers**

```bash
git add train_GRL.py tests/test_obj_gated_da.py
git commit -m "feat: add gated faithful da loss"
```

---

## Task 5: Add CLI Flags And Validation Tests

**Files:**
- Modify: `tests/test_obj_gated_da.py`
- Modify: `train_GRL.py` later in Task 6

- [ ] **Step 1: Inspect existing validation function**

Search `train_GRL.py` for parser and validation:

```bash
grep -n "da-feat-layers\|da_img_faithful\|def parse_opt\|def main" train_GRL.py
```

Expected: parser includes `--da-feat-layers`; `main()` contains DA flag dependency validation.

- [ ] **Step 2: Add failing validation tests**

Append to `tests/test_obj_gated_da.py`:

```python
from argparse import Namespace

from train_GRL import main


def _base_opt(**overrides):
    values = dict(
        weights='', cfg='configs/domain/yolov5l_GRL.yaml', data='domain/city_foggycity.yaml', hyp='hyps/hyp.scratch-high.yaml',
        epochs=1, batch_size=1, imgsz=64, rect=False, resume=False, nosave=True, noval=True, noautoanchor=False,
        noplots=True, evolve=None, bucket='', cache=None, image_weights=False, device='cpu', multi_scale=False,
        single_cls=False, optimizer='SGD', sync_bn=False, workers=0, project='runs/train', name='tmp', exist_ok=True,
        quad=False, cos_lr=False, label_smoothing=0.0, patience=1, freeze=[0], save_period=-1, seed=0,
        local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias='latest',
        da_img=False, da_img_weight=1.0, da_img_grl_weight=0.1, advgrl=False, advgrl_alpha=None,
        advgrl_beta=30.0, advgrl_eps=1e-7, advgrl_threshold=30.0, da_img_warmup='off', aux=False,
        triplet_img=False, triplet_img_weight=1.0, triplet_margin=1.0, triplet_adaptive=False,
        triplet_max_margin=2.0, da_img_faithful=False, da_feat_layers='sppf',
        da_img_obj_gate=False, da_img_obj_gate_floor=0.05,
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
```

If `_base_opt` duplicates an existing test helper in this repo, reuse the existing helper instead and add only missing attrs.

- [ ] **Step 3: Run validation tests to verify failure**

Run:

```bash
pytest tests/test_obj_gated_da.py::test_obj_gate_requires_da_img tests/test_obj_gated_da.py::test_obj_gate_requires_faithful tests/test_obj_gated_da.py::test_obj_gate_requires_neck_all tests/test_obj_gated_da.py::test_obj_gate_floor_must_be_non_negative -v
```

Expected: FAIL because parser/validation fields do not exist yet or validation is missing.

---

## Task 6: Implement CLI Flags And Validation

**Files:**
- Modify: `train_GRL.py` parser and `main()` validation block
- Test: `tests/test_obj_gated_da.py`

- [ ] **Step 1: Add parser flags**

In `parse_opt()` near existing DA flags, add:

```python
parser.add_argument('--da-img-obj-gate', action='store_true', help='weight faithful neck-all image DA by detached Detect objectness gates')
parser.add_argument('--da-img-obj-gate-floor', type=float, default=0.05, help='minimum objectness gate value for --da-img-obj-gate')
```

- [ ] **Step 2: Add validation rules**

In `main(opt)`, after existing DA dependency validation, add exactly:

```python
    if opt.da_img_obj_gate and not opt.da_img:
        raise SystemExit('--da-img-obj-gate requires --da-img')
    if opt.da_img_obj_gate and not opt.da_img_faithful:
        raise SystemExit('--da-img-obj-gate requires --da-img-faithful')
    if opt.da_img_obj_gate and opt.da_feat_layers != 'neck-all':
        raise SystemExit('--da-img-obj-gate requires --da-feat-layers neck-all')
    if opt.da_img_obj_gate_floor < 0:
        raise SystemExit('--da-img-obj-gate-floor must be non-negative')
```

- [ ] **Step 3: Run validation tests**

Run:

```bash
pytest tests/test_obj_gated_da.py::test_obj_gate_requires_da_img tests/test_obj_gated_da.py::test_obj_gate_requires_faithful tests/test_obj_gated_da.py::test_obj_gate_requires_neck_all tests/test_obj_gated_da.py::test_obj_gate_floor_must_be_non_negative -v
```

Expected: PASS.

- [ ] **Step 4: Run all object-gated tests**

Run:

```bash
pytest tests/test_obj_gated_da.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit CLI and validation**

```bash
git add train_GRL.py tests/test_obj_gated_da.py
git commit -m "feat: add objectness gate cli validation"
```
---

## Task 7: Wire Gated Loss Into Faithful DA Branch

**Files:**
- Modify: `train_GRL.py:529-566`
- Test: `tests/test_obj_gated_da.py`

- [ ] **Step 1: Add integration test for matching gated objective**

Append to `tests/test_obj_gated_da.py`:

```python
from train_GRL import DA_FEATURE_CHANNELS


def test_neck_all_feature_selection_order_matches_detect_order():
    assert list(DA_FEATURE_CHANNELS['neck-all']) == ['neck_p3', 'neck_p4', 'neck_p5']
```

- [ ] **Step 2: Modify faithful DA branch**

In `train_GRL.py`, replace the faithful DA block from:

```python
                        source_logits_detached = {
                            name: classifier_heads[name](_feature(backbone_features, name).detach())
                            for name in da_feature_channels
                        }
                        target_logits_detached = {
                            name: classifier_heads[name](_feature(target_backbone_features, name).detach())
                            for name in da_feature_channels
                        }
                        L_c_tensor = da_img_faithful_loss_multi(source_logits_detached, target_logits_detached)
```

through:

```python
                        loss_da_image = da_img_faithful_loss_multi(source_logits, target_logits)
```

with:

```python
                        source_gates = None
                        target_gates = None
                        if opt.da_img_obj_gate:
                            feature_names = list(da_feature_channels)
                            source_gates = _build_objectness_gates(
                                det_pred_src, feature_names, floor=opt.da_img_obj_gate_floor)
                            target_gates = _build_objectness_gates(
                                target_det_pred, feature_names, floor=opt.da_img_obj_gate_floor)

                        source_logits_detached = {
                            name: classifier_heads[name](_feature(backbone_features, name).detach())
                            for name in da_feature_channels
                        }
                        target_logits_detached = {
                            name: classifier_heads[name](_feature(target_backbone_features, name).detach())
                            for name in da_feature_channels
                        }
                        if opt.da_img_obj_gate:
                            L_c_tensor = da_img_faithful_gated_loss_multi(
                                source_logits_detached, target_logits_detached, source_gates, target_gates)
                        else:
                            L_c_tensor = da_img_faithful_loss_multi(source_logits_detached, target_logits_detached)
                        L_c_this_iter = float(L_c_tensor.item())
                        if opt.advgrl:
                            alpha = opt.advgrl_alpha if opt.advgrl_alpha is not None else default_alpha()
                            lambda_adv_this_iter = compute_lambda_adv(
                                L_c_this_iter,
                                lambda_0=opt.da_img_grl_weight,
                                alpha=alpha,
                                beta=opt.advgrl_threshold,
                            )
                        else:
                            lambda_adv_this_iter = opt.da_img_grl_weight
                        source_logits = {
                            name: classifier_heads[name](
                                gradient_scalar(_feature(backbone_features, name), -lambda_adv_this_iter)
                            )
                            for name in da_feature_channels
                        }
                        target_logits = {
                            name: classifier_heads[name](
                                gradient_scalar(_feature(target_backbone_features, name), -lambda_adv_this_iter)
                            )
                            for name in da_feature_channels
                        }
                        if opt.da_img_obj_gate:
                            loss_da_image = da_img_faithful_gated_loss_multi(
                                source_logits, target_logits, source_gates, target_gates)
                        else:
                            loss_da_image = da_img_faithful_loss_multi(source_logits, target_logits)
```

Important: the snippet references `target_det_pred`. If current code discards target Detect predictions, update the target forward in the faithful separate-forward path from:

```python
_, target_backbone_features = model(t_imgs)
```

to:

```python
target_det_pred, target_backbone_features = model(t_imgs)
```

Keep existing variable names if the file uses `target_imgs` rather than `t_imgs` in this branch.

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest tests/test_obj_gated_da.py -v
```

Expected: PASS.

- [ ] **Step 4: Run existing DA loss tests**

Run:

```bash
pytest tests/test_domain_loss.py tests/test_advgrl.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit wiring**

```bash
git add train_GRL.py tests/test_obj_gated_da.py
git commit -m "feat: gate faithful neck-all da loss"
```

---

## Task 8: Add Smoke Coverage

**Files:**
- Modify: `tests/test_train_smoke.py`
- Test: `tests/test_train_smoke.py`

- [ ] **Step 1: Inspect smoke parametrization**

Run:

```bash
grep -n "da-feat-layers\|advgrl\|da-img-faithful\|parametrize" tests/test_train_smoke.py
```

Expected: shows existing smoke flag combos.

- [ ] **Step 2: Add objectness-gated combo**

If `tests/test_train_smoke.py` has a list of CLI combos, add this combo:

```python
[
    '--da-img',
    '--da-img-faithful',
    '--advgrl',
    '--advgrl-alpha', '0.75',
    '--da-feat-layers', 'neck-all',
    '--da-img-obj-gate',
]
```

If the file uses pytest params with ids, use:

```python
pytest.param(
    ['--da-img', '--da-img-faithful', '--advgrl', '--advgrl-alpha', '0.75', '--da-feat-layers', 'neck-all', '--da-img-obj-gate'],
    id='faithful-advgrl-neck-all-objgate',
)
```

- [ ] **Step 3: Run smoke test for new combo only**

Run the exact node id after inspecting the file. Example:

```bash
pytest tests/test_train_smoke.py -v -s -k objgate
```

Expected: PASS. If CUDA is unavailable and the test skips by design, expected: SKIPPED with existing CUDA skip reason.

- [ ] **Step 4: Commit smoke coverage**

```bash
git add tests/test_train_smoke.py
git commit -m "test: smoke objectness-gated neck-all da"
```

---

## Task 9: Regression Tests And Static Checks

**Files:**
- No code changes expected
- Test: full relevant suite

- [ ] **Step 1: Run focused DA unit tests**

```bash
pytest tests/test_obj_gated_da.py tests/test_domain_loss.py tests/test_advgrl.py tests/test_da_warmup.py -v
```

Expected: PASS.

- [ ] **Step 2: Run train smoke tests**

```bash
pytest tests/test_train_smoke.py -v -s
```

Expected: PASS or CUDA-dependent skips consistent with existing behavior.

- [ ] **Step 3: Run parser help smoke**

```bash
python train_GRL.py --help | grep -E "da-img-obj-gate|da-img-obj-gate-floor"
```

Expected output includes both flags:

```text
--da-img-obj-gate
--da-img-obj-gate-floor
```

- [ ] **Step 4: Check git diff**

```bash
git diff -- train_GRL.py tests/test_obj_gated_da.py tests/test_train_smoke.py
```

Expected: only objectness-gated DA helpers, CLI/validation, faithful DA branch wiring, tests.

---

## Task 10: Manual Command Dry Run

**Files:**
- No code changes expected unless dry run reveals parser/validation bug

- [ ] **Step 1: Run invalid combo checks**

```bash
python train_GRL.py --da-img-obj-gate --cfg configs/domain/yolov5l_GRL.yaml --data domain/city_foggycity.yaml --hyp hyps/hyp.scratch-high.yaml --epochs 1 --batch-size 1 --img 64 --weights '' --nosave --noval
```

Expected: exits with:

```text
--da-img-obj-gate requires --da-img
```

Run:

```bash
python train_GRL.py --da-img --da-img-obj-gate --cfg configs/domain/yolov5l_GRL.yaml --data domain/city_foggycity.yaml --hyp hyps/hyp.scratch-high.yaml --epochs 1 --batch-size 1 --img 64 --weights '' --nosave --noval
```

Expected: exits with:

```text
--da-img-obj-gate requires --da-img-faithful
```

- [ ] **Step 2: Run exact report command parse check**

Do not start a 50-epoch training run. Use `--epochs 0` if supported by current train script; otherwise skip this step.

```bash
python train_GRL.py \
  --weights '' \
  --cfg configs/domain/yolov5l_GRL.yaml \
  --data domain/city_foggycity.yaml \
  --hyp hyps/hyp.scratch-high.yaml \
  --epochs 0 \
  --batch-size 1 \
  --img 64 \
  --name objgate_parse_check \
  --project /tmp/yolog_objgate_parse_check \
  --da-img \
  --da-img-faithful \
  --advgrl \
  --advgrl-alpha 0.75 \
  --da-feat-layers neck-all \
  --da-img-obj-gate \
  --nosave \
  --noval
```

Expected: parser/validation accepts flags. If dataset/model loading proceeds and fails due local data/weights availability, record exact failure; do not treat data absence as object-gate failure.

- [ ] **Step 3: Commit any dry-run fixes**

Only if Step 1 or Step 2 required code changes:

```bash
git add train_GRL.py tests/test_obj_gated_da.py tests/test_train_smoke.py
git commit -m "fix: align objectness gate cli behavior"
```
---

## Task 11: Final Review And Documentation Check

**Files:**
- Modify only if review finds mismatch
- Relevant docs:
  - `docs/superpowers/specs/2026-06-03-foreground-gated-da-design.md`
  - `yolog_city_foggy_da_full_report.md`

- [ ] **Step 1: Confirm exact report semantics**

Check implementation against this checklist:

```text
[x] --da-img-obj-gate exists
[x] --da-img-obj-gate-floor default is 0.05
[x] flag requires --da-img
[x] flag requires --da-img-faithful
[x] flag requires --da-feat-layers neck-all
[x] source label is 0
[x] target label is 1
[x] objectness logits use sigmoid
[x] objectness gate maxes over anchors
[x] gate shape is [B,1,H,W]
[x] gate is clamped by floor
[x] gate is detached
[x] weighted BCE divides by clamp(sum(gate), min=1)
[x] neck-all loss averages P3/P4/P5
[x] AdvGRL detached L_c uses same gated objective
[x] no gate diagnostics added to da_losses.csv
[x] no default behavior changed
```

- [ ] **Step 2: Request code review**

Use the code-review skill or subagent review per execution mode. Ask reviewer to focus on:

```text
Review objectness-gated faithful neck-all DA implementation. Check exact report semantics, tensor shapes, detach behavior, AdvGRL L_c parity, CLI validation, and whether non-gated behavior is unchanged.
```

- [ ] **Step 3: Apply review fixes if needed**

For each accepted finding:

```bash
git add train_GRL.py tests/test_obj_gated_da.py tests/test_train_smoke.py
git commit -m "fix: address objectness gate review finding"
```

Use a more specific commit message if only one concrete issue is fixed.

- [ ] **Step 4: Final status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing untracked files may remain, such as:

```text
?? 2210.15176v1.pdf
?? yolog_city_foggy_da_full_report.md
```

If implementation files are modified but uncommitted, commit or report why not.

---

## Self-Review

Spec coverage:
- CLI flags and validation: Tasks 5-6.
- Dumb-model architecture: File Structure, Task 7 uses existing feature dict; no DA logic added to model.
- Feature selection neck-all: Task 7 integration test and existing `DA_FEATURE_CHANNELS`.
- Objectness gates: Tasks 1-2.
- Gated loss formula: Tasks 3-4.
- AdvGRL L_c parity: Task 7 wiring and Task 11 checklist.
- No diagnostics: File Structure and Task 11 checklist.
- Tests/smoke: Tasks 1, 3, 5, 8, 9.

Placeholder scan:
- No TBD/TODO placeholders.
- All code steps include concrete snippets or commands.
- One conditional area remains: smoke test insertion depends on existing parametrization shape; plan gives exact `pytest.param` content and inspection command.

Type/name consistency:
- Helpers: `_build_objectness_gates`, `_weighted_da_bce_loss`, `da_img_faithful_gated_loss_multi` used consistently.
- Feature names: `neck_p3`, `neck_p4`, `neck_p5` match `DA_FEATURE_CHANNELS` and model feature dict.
- CLI names: `da_img_obj_gate`, `da_img_obj_gate_floor` match parser flag conversion.
