# YOLO-G DA-Img Faithful Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--da-img-faithful` as an independently-toggleable mode that reproduces original YOLO-G image-level DA behavior: two separate source/target forwards, GRL weight `-0.1`, source label `0`, target label `1`, and `0.5 * source_BCE + 0.5 * target_BCE`.

**Architecture:** Keep the current dumb-model design: `models/yolo_GRL.py` still returns `(det_pred, backbone_feat)`, and `train_GRL.py` owns the DA branch. Faithful mode is a branch inside `train_GRL.py` that runs a second model forward on `t_imgs` and applies a new pure helper `da_img_faithful_loss_pair(...)` to separate source/target logits. This preserves existing `--da-img` behavior for PR2+ comparisons while adding a clean original-YOLO-G baseline.

**Tech Stack:** Python 3.10+, PyTorch, pytest, YOLOv5-L/YOLO-G training loop. No new dependencies.

---

## File Structure

**Modify:**
- `utils/domain_loss.py` — add `da_img_faithful_loss_pair(source_logits, target_logits)` for original YOLO-G source=0 / target=1 convention and 0.5/0.5 weighting.
- `utils/advgrl.py` — no behavior change; keep current PR2+ `advgrl_step` path untouched.
- `train_GRL.py` — add `--da-img-faithful`; validate dependency/conflicts; branch the training forward pass so faithful mode uses separate target forward and separate BCE losses.
- `tests/test_domain_loss.py` — add unit tests for faithful label convention and 0.5/0.5 weighting. Create this file if it does not exist.
- `tests/test_train_smoke.py` — add `--da-img --da-img-faithful` smoke case.
- `tools/run_ablations.sh` — add an optional seventh ablation for faithful YOLO-G baseline.

**Do not modify:**
- `models/yolo_GRL.py` — model stays dumb.
- `models/da_classifier.py` — keep DA head shape/initialization unchanged for fair comparison to current PR2+ path.
- `utils/domain_grl.py` — existing `gradient_scalar(feat, -weight)` already matches original baked `GradientScalarLayer(-0.1)` semantics.

**Behavior matrix:**

| Flags | Forward pattern | DA label convention | Loss weighting | Allowed with PR2+ extensions? |
|---|---|---|---|---|
| `--da-img` | concatenated `[source,target]` forward | source=1, target=0 | `opt.da_img_weight * BCE(all)` | yes |
| `--da-img --advgrl` | concatenated `[source,target]` forward + AdvGRL detached pass | source=1, target=0 | `opt.da_img_weight * BCE(all)` | yes |
| `--da-img --da-img-faithful` | separate source forward + separate target forward | source=0, target=1 | `opt.da_img_weight * (0.5*src_BCE + 0.5*tgt_BCE)` | no |

**CLI defaults:**
- `--da-img-faithful`: default `False`.
- `--da-img-weight`: keep global default `1.0` for backwards compatibility.
- In faithful mode, the implementation must multiply `0.5*src + 0.5*tgt` by `opt.da_img_weight`. Users can set `--da-img-weight 1.0` to match original exactly because the 0.5/0.5 factors are inside the faithful helper.
- `--da-img-grl-weight`: keep global default `0.1`; faithful mode applies it as `gradient_scalar(feat, -opt.da_img_grl_weight)`, matching original `GradientScalarLayer(-0.1)`.

**Flag validation:**
- `--da-img-faithful` requires `--da-img`.
- `--da-img-faithful` conflicts with `--advgrl`.
- `--da-img-faithful` conflicts with `--aux`.
- `--da-img-faithful` conflicts with `--triplet-img`.
- `--da-img-faithful` conflicts with `--da-img-warmup gate` and `--da-img-warmup ramp`; only `--da-img-warmup off` is allowed.

**Why conflicts:** Faithful mode is original YOLO-G PR1-equivalent DA only. `--advgrl`, `--aux`, `--triplet-img`, and `--da-img-warmup` are later extensions; allowing them would make the baseline no longer faithful.

---

## Task 1: Add faithful DA loss helper + unit tests

**Files:**
- Modify: `utils/domain_loss.py`
- Create or modify: `tests/test_domain_loss.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_domain_loss.py` if missing. If it already exists, append these tests without deleting existing tests.

```python
import torch
import torch.nn.functional as F

from utils.domain_loss import da_img_faithful_loss_pair


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_domain_loss.py -v
```

Expected: FAIL with `ImportError: cannot import name 'da_img_faithful_loss_pair'` or `AttributeError` if import style differs.

- [ ] **Step 3: Implement the helper**

Modify `utils/domain_loss.py` to import no new dependencies and add this function after `da_img_loss`:

```python
def da_img_faithful_loss_pair(source_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Original YOLO-G image-level DA loss: source=0, target=1, 0.5/0.5."""
    source_feature = source_logits.permute(0, 2, 3, 1).reshape(source_logits.shape[0], -1)
    target_feature = target_logits.permute(0, 2, 3, 1).reshape(target_logits.shape[0], -1)
    source_loss = F.binary_cross_entropy_with_logits(source_feature, torch.zeros_like(source_feature))
    target_loss = F.binary_cross_entropy_with_logits(target_feature, torch.ones_like(target_feature))
    return 0.5 * source_loss + 0.5 * target_loss
```

Keep existing `da_img_loss(...)` unchanged. Its source=1/target=0 convention remains current PR2+ behavior.

- [ ] **Step 4: Run tests to verify helper passes**

Run:

```bash
pytest tests/test_domain_loss.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/domain_loss.py tests/test_domain_loss.py
git commit -m "add faithful da image loss"
```

---

## Task 2: Add `--da-img-faithful` CLI validation

**Files:**
- Modify: `train_GRL.py`

- [ ] **Step 1: Add parser flag**

In `train_GRL.py`, locate the DA flag block near:

```python
    parser.add_argument('--da-img', action='store_true', help='enable image-level DANN (S vs T)')
    parser.add_argument('--da-img-weight', type=float, default=1.0, help='multiplier on loss_da_image')
    parser.add_argument('--da-img-grl-weight', type=float, default=0.1, help='lambda_0 - GRL weight on the DA branch')
```

Add immediately after `--da-img`:

```python
    parser.add_argument('--da-img-faithful', action='store_true',
                        help='use original YOLO-G image-level DA: separate source/target forwards, source=0, target=1')
```

- [ ] **Step 2: Add dependency/conflict validation**

In `main(opt, callbacks=Callbacks())`, locate the flag dependency validation block. Add these checks after existing `--advgrl requires --da-img` check:

```python
    if opt.da_img_faithful and not opt.da_img:
        raise SystemExit('--da-img-faithful requires --da-img')
    if opt.da_img_faithful and opt.advgrl:
        raise SystemExit('--da-img-faithful conflicts with --advgrl (faithful mode reproduces original YOLO-G only)')
    if opt.da_img_faithful and opt.aux:
        raise SystemExit('--da-img-faithful conflicts with --aux (faithful mode uses source+target only)')
    if opt.da_img_faithful and opt.triplet_img:
        raise SystemExit('--da-img-faithful conflicts with --triplet-img (faithful mode has no aux negative)')
    if opt.da_img_faithful and opt.da_img_warmup != 'off':
        raise SystemExit('--da-img-faithful requires --da-img-warmup off')
```

The order matters: keep existing `--triplet-img requires --aux` and `--triplet-img requires --da-img` checks below this block. That way `--da-img --da-img-faithful --triplet-img` reports the faithful conflict, not only missing aux.

- [ ] **Step 3: Add parser/validation tests if a CLI test file exists**

If there is an existing CLI parser test file, add these tests there. If no such file exists, skip this step and rely on the smoke test in Task 4.

```python
import pytest

from train_GRL import main, parse_opt


def test_da_img_faithful_requires_da_img():
    opt = parse_opt(known=True)
    opt.da_img = False
    opt.da_img_faithful = True
    with pytest.raises(SystemExit, match='--da-img-faithful requires --da-img'):
        main(opt)
```

Do not create a new parser test harness if none exists; `main(opt)` performs environment checks after validation and is not ideal for isolated unit tests in this repo.

- [ ] **Step 4: Syntax-check parser changes**

Run:

```bash
python -m py_compile train_GRL.py
```

Expected: command exits 0.

- [ ] **Step 5: Commit**

```bash
git add train_GRL.py
git commit -m "add da-img faithful flag validation"
```

---

## Task 3: Wire faithful two-forward training branch

**Files:**
- Modify: `train_GRL.py`

- [ ] **Step 1: Import helper and GRL function**

At the top of `train_GRL.py`, locate:

```python
from utils.domain_loss import da_img_loss, triplet_img_loss
```

Replace with:

```python
from utils.domain_loss import da_img_loss, da_img_faithful_loss_pair, triplet_img_loss
```

Locate imports near `advgrl_step`. Add:

```python
from utils.domain_grl import gradient_scalar
```

If `gradient_scalar` is already imported in `train_GRL.py`, do not add a duplicate import.

- [ ] **Step 2: Keep combined forward for non-faithful mode**

Locate the current forward block inside training loop:

```python
            # Forward
            with amp.autocast(enabled=cuda):
                det_pred, backbone_feat = model(all_imgs)
                # Batch sizes for each domain slice in backbone_feat.
                # Order: [source | target | aux]. Defined once here so DA + triplet share them.
                B_t = t_imgs.size(0) if t_imgs is not None else 0
                B_a = a_imgs.size(0) if a_imgs is not None else 0
                # Slice predictions to source rows only for the detection loss.
                det_pred_src = [p[:B_s] for p in det_pred]
```

Replace that setup with:

```python
            # Forward
            with amp.autocast(enabled=cuda):
                if opt.da_img_faithful:
                    det_pred, backbone_feat = model(imgs)
                    _, target_backbone_feat = model(t_imgs)
                    B_t = t_imgs.size(0)
                    B_a = 0
                    det_pred_src = det_pred
                else:
                    det_pred, backbone_feat = model(all_imgs)
                    # Batch sizes for each domain slice in backbone_feat.
                    # Order: [source | target | aux]. Defined once here so DA + triplet share them.
                    B_t = t_imgs.size(0) if t_imgs is not None else 0
                    B_a = a_imgs.size(0) if a_imgs is not None else 0
                    # Slice predictions to source rows only for the detection loss.
                    det_pred_src = [p[:B_s] for p in det_pred]
```

This exactly matches original YOLO-G's two model forwards for faithful mode:

```python
pred, source_pred_label = model(imgs)
_, target_pred_label = model(t_imgs)
```

- [ ] **Step 3: Branch the DA loss block**

Locate current DA block:

```python
                if classifier_head is not None:
                    da_scale_this_iter = compute_da_warmup_scale(ni=ni, nw=nw, mode=opt.da_img_warmup)
                    if da_scale_this_iter > 0.0:
                        st_feat = backbone_feat[:B_s + B_t]
                        alpha = opt.advgrl_alpha if opt.advgrl_alpha is not None else default_alpha()
                        loss_da_image, lambda_adv_this_iter, L_c_this_iter = advgrl_step(
                            st_feat, source_count=B_s, classifier=classifier_head,
                            use_advgrl=opt.advgrl, lambda_0=opt.da_img_grl_weight,
                            alpha=alpha, beta=opt.advgrl_threshold,
                        )
                        loss_da_image_this_iter = loss_da_image
                        loss = loss + da_scale_this_iter * opt.da_img_weight * loss_da_image
```

Replace with:

```python
                if classifier_head is not None:
                    da_scale_this_iter = compute_da_warmup_scale(ni=ni, nw=nw, mode=opt.da_img_warmup)
                    if opt.da_img_faithful:
                        source_feat_grl = gradient_scalar(backbone_feat, -opt.da_img_grl_weight)
                        target_feat_grl = gradient_scalar(target_backbone_feat, -opt.da_img_grl_weight)
                        source_logits = classifier_head(source_feat_grl)
                        target_logits = classifier_head(target_feat_grl)
                        loss_da_image = da_img_faithful_loss_pair(source_logits, target_logits)
                        lambda_adv_this_iter = opt.da_img_grl_weight
                        L_c_this_iter = float(loss_da_image.detach().item())
                        loss_da_image_this_iter = loss_da_image
                        loss = loss + opt.da_img_weight * loss_da_image
                    elif da_scale_this_iter > 0.0:
                        st_feat = backbone_feat[:B_s + B_t]
                        alpha = opt.advgrl_alpha if opt.advgrl_alpha is not None else default_alpha()
                        loss_da_image, lambda_adv_this_iter, L_c_this_iter = advgrl_step(
                            st_feat, source_count=B_s, classifier=classifier_head,
                            use_advgrl=opt.advgrl, lambda_0=opt.da_img_grl_weight,
                            alpha=alpha, beta=opt.advgrl_threshold,
                        )
                        loss_da_image_this_iter = loss_da_image
                        loss = loss + da_scale_this_iter * opt.da_img_weight * loss_da_image
```

Notes:
- `da_scale_this_iter` remains `1.0` in faithful mode because validation only permits `--da-img-warmup off`.
- `lambda_adv_this_iter` logs `0.1` by default, matching original baked GRL weight magnitude.
- `L_c_this_iter` logs the faithful loss value for diagnostics only. Original YOLO-G did not log this; logging it does not affect training.

- [ ] **Step 4: Ensure triplet block cannot run in faithful mode**

No code change needed if Task 2 validation is present. `--da-img-faithful` conflicts with `--triplet-img`, so this block stays unreachable:

```python
                if opt.triplet_img:
                    F_S = _gap_mean(backbone_feat[:B_s])
                    F_T = _gap_mean(backbone_feat[B_s:B_s + B_t])
                    F_A = _gap_mean(backbone_feat[B_s + B_t:B_s + B_t + B_a])
```

- [ ] **Step 5: Syntax-check training code**

Run:

```bash
python -m py_compile train_GRL.py
```

Expected: command exits 0.

- [ ] **Step 6: Commit**

```bash
git add train_GRL.py
git commit -m "wire da-img faithful training path"
```

---

## Task 4: Add smoke test + ablation entry

**Files:**
- Modify: `tests/test_train_smoke.py`
- Modify: `tools/run_ablations.sh`

- [ ] **Step 1: Add smoke test parameter**

In `tests/test_train_smoke.py`, locate:

```python
@pytest.mark.parametrize('flags,name', [
    ([],                                                                           'baseline'),
    (['--da-img'],                                                                 'daimg'),
    (['--da-img', '--advgrl'],                                                     'advgrl'),
```

Insert faithful mode after `daimg`:

```python
    (['--da-img', '--da-img-faithful'],                                             'daimg_faithful'),
```

The final list should include baseline, current daimg, faithful daimg, advgrl, aux, triplet, full.

- [ ] **Step 2: Strengthen smoke artifact assertion for faithful mode**

In the existing DA CSV assertion block:

```python
    if '--da-img' in flags:
        da_csv = run_dir / 'da_losses.csv'
        assert da_csv.exists(), f'[{name}] da_losses.csv missing: {da_csv}'
        # Header + at least one data row.
        lines = da_csv.read_text().splitlines()
        assert len(lines) > 1, f'[{name}] da_losses.csv has no data rows (got {len(lines)} lines)'
```

Append:

```python
        if '--da-img-faithful' in flags:
            header = lines[0].split(',')
            first_row = lines[1].split(',')
            row = dict(zip(header, first_row))
            assert row['lambda_adv'] == '0.1'
            assert row['da_scale'] == '1.0'
```

This verifies the faithful branch logs fixed GRL weight and no warmup scaling.

- [ ] **Step 3: Add ablation 07 faithful baseline**

In `tools/run_ablations.sh`, locate:

```bash
# 2. --da-img only (original YOLO-G)
python train_GRL.py $BASE_ARGS --name ablation_02_daimg --da-img
```

Replace the comment and add faithful run:

```bash
# 2. --da-img only (current PR2+ dumb-model implementation)
python train_GRL.py $BASE_ARGS --name ablation_02_daimg --da-img

# 7. Faithful original YOLO-G image-level DA baseline
python train_GRL.py $BASE_ARGS --name ablation_07_daimg_faithful --da-img --da-img-faithful
```

Keep numbering `07` so existing ablation result folders remain stable.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_domain_loss.py -v
```

Expected: PASS.

If CUDA is available, run:

```bash
pytest tests/test_train_smoke.py -v -s
```

Expected: PASS, including `smoke_daimg_faithful` with `weights/last.pt` and non-empty `da_losses.csv`.

If CUDA is not available, expected: smoke tests SKIPPED with reason `needs CUDA`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_train_smoke.py tools/run_ablations.sh
git commit -m "test da-img faithful smoke path"
```

---

## Task 5: End-to-end faithful baseline sanity run

**Files:**
- No code files. Produces run artifacts under `runs/train/sanity_daimg_faithful/`.

- [ ] **Step 1: Run a short faithful sanity check**

Run from `yolo-G/`:

```bash
python train_GRL.py \
  --weights yolov5l.pt \
  --cfg configs/domain/yolov5l_GRL.yaml \
  --data domain/city_foggycity.yaml \
  --hyp data/hyps/hyp.scratch-high.yaml \
  --epochs 10 \
  --batch-size 8 \
  --img 640 \
  --name sanity_daimg_faithful \
  --da-img \
  --da-img-faithful
```

Expected:
- Training completes with no NaN and no crash.
- `runs/train/sanity_daimg_faithful/weights/last.pt` exists.
- `runs/train/sanity_daimg_faithful/da_losses.csv` exists.
- `da_losses.csv` has `lambda_adv` fixed at `0.1` and `da_scale` fixed at `1.0`.

- [ ] **Step 2: Inspect DA diagnostics**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
p = Path('runs/train/sanity_daimg_faithful/da_losses.csv')
df = pd.read_csv(p)
print('rows', len(df))
print('loss_da_image min/max', df['loss_da_image'].min(), df['loss_da_image'].max())
print('lambda_adv unique head', sorted(df['lambda_adv'].dropna().unique())[:5])
print('da_scale unique head', sorted(df['da_scale'].dropna().unique())[:5])
assert len(df) > 0
assert (df['lambda_adv'].dropna() == 0.1).all()
assert (df['da_scale'].dropna() == 1.0).all()
assert df['loss_da_image'].notna().all()
PY
```

Expected:
- Script exits 0.
- `lambda_adv unique head [0.1]`.
- `da_scale unique head [1.0]`.

- [ ] **Step 3: Commit only code/test changes, not run artifacts**

No commit if Tasks 1-4 already committed. `runs/` is gitignored and must not be staged.

Run:

```bash
git status --short
```

Expected:
- No tracked code changes if all previous commits already happened.
- Run artifacts under `runs/` do not appear because `.gitignore` excludes them.

---

## Acceptance Criteria

- `python -m py_compile train_GRL.py` exits 0.
- `pytest tests/test_domain_loss.py -v` passes.
- `pytest tests/test_train_smoke.py -v -s` passes on CUDA, or skips cleanly when CUDA is unavailable.
- `python train_GRL.py ... --da-img --da-img-faithful` completes a sanity run with no NaN and no crash.
- `--da-img-faithful` without `--da-img` exits with a clear validation error.
- `--da-img-faithful` with `--advgrl`, `--aux`, `--triplet-img`, or non-off `--da-img-warmup` exits with a clear validation error.
- Existing `--da-img`, `--da-img --advgrl`, `--da-img --aux`, `--da-img --aux --triplet-img`, and full method behavior remains unchanged.

## Self-Review

**Spec coverage:**
- New `--da-img-faithful` flag: Task 2.
- Separate source/target forward passes: Task 3.
- Fixed GRL weight `-0.1` by default: Task 3 via `gradient_scalar(..., -opt.da_img_grl_weight)` and existing default `0.1`.
- Source label `0`, target label `1`: Task 1 helper.
- `0.5 * source_loss + 0.5 * target_loss`: Task 1 helper.
- Conflicts with PR2+ extensions: Task 2 validation.
- Smoke/no-NaN gate: Task 4 and Task 5.

**Placeholder scan:** No `TBD`, `TODO`, or unspecified implementation steps. Every code change has exact file and code snippet.

**Type consistency:** `da_img_faithful_loss_pair(source_logits, target_logits)` is imported and called consistently. `target_backbone_feat` is defined only inside `opt.da_img_faithful` branch and used only inside that same branch.

## Execution Handoff

Recommended execution: use `superpowers:subagent-driven-development` to implement Task 1 through Task 5 sequentially, with spec-compliance review then code-quality review after each task.
