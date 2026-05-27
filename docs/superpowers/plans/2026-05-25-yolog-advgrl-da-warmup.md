# YOLO-G AdvGRL DA-Warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently-toggleable `--da-img-warmup` flag (`off` / `gate` / `ramp`) that suppresses or ramps the DA loss during the LR warmup period, to stabilize PR 2 (`--da-img`) and PR 3 (`--advgrl`) which both catastrophically collapse at the warmup peak LR on Foggy Cityscapes.

**Architecture:** A small pure helper `utils/da_warmup.py::compute_da_warmup_scale(ni, nw, mode) -> float` computes the per-iteration DA-loss scale. `train_GRL.py` multiplies the existing `loss_da_image` by this scalar before adding it to the total loss. `gate` mode also short-circuits the (otherwise expensive) `advgrl_step` two-pass forward while the scale is zero. `da_logger.py` is extended with a `da_scale` column so we can plot the warmup behavior alongside `lambda_adv` and `L_c`.

**Tech Stack:** PyTorch, NumPy, Python 3.10+, pytest. No new dependencies.

**Default behavior preserved:** `--da-img-warmup off` (the default) reproduces existing PR 2 / PR 3 behavior byte-for-byte — the existing sanity-gate results remain valid as faithful-original baselines.

**Phasing:** This is one logical PR (call it PR 3.5 — Stabilization). It slots between PR 3 and PR 4 in the existing plan at `docs/superpowers/plans/2026-05-25-yolog-advgrl.md`. After this lands, PR 4 (`--aux` RainMix) and PR 5 (`--triplet-img`) can use `--da-img-warmup=ramp` in their sanity gates to clear the gate.

---

## File Structure

**Create:**
- `utils/da_warmup.py` — single pure function `compute_da_warmup_scale(ni: int, nw: int, mode: str) -> float`.
- `tests/test_da_warmup.py` — unit tests for `off` / `gate` / `ramp` modes covering boundary conditions and invalid inputs.

**Modify:**
- `train_GRL.py` — add `--da-img-warmup` CLI flag; multiply `loss_da_image` by `compute_da_warmup_scale(...)` before adding to total loss; short-circuit `advgrl_step` when scale is exactly 0; log `da_scale` per-iter.
- `utils/da_logger.py` — append `da_scale` to `self._fields`.

**Touch order:** helper + tests first (no cross-dependencies), then logger field (one-line change), then wire-in (depends on both).

---

## Task 3.5.1: `compute_da_warmup_scale` helper + unit tests

**Files:**
- Create: `utils/da_warmup.py`
- Test: `tests/test_da_warmup.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/test_da_warmup.py
"""Tests for utils.da_warmup.compute_da_warmup_scale."""
from __future__ import annotations

import math
import pytest

from utils.da_warmup import compute_da_warmup_scale


class TestOffMode:
    def test_off_returns_one_at_iter_zero(self):
        assert compute_da_warmup_scale(ni=0, nw=1000, mode='off') == 1.0

    def test_off_returns_one_mid_warmup(self):
        assert compute_da_warmup_scale(ni=500, nw=1000, mode='off') == 1.0

    def test_off_returns_one_post_warmup(self):
        assert compute_da_warmup_scale(ni=5000, nw=1000, mode='off') == 1.0


class TestGateMode:
    def test_gate_returns_zero_at_iter_zero(self):
        assert compute_da_warmup_scale(ni=0, nw=1000, mode='gate') == 0.0

    def test_gate_returns_zero_mid_warmup(self):
        assert compute_da_warmup_scale(ni=999, nw=1000, mode='gate') == 0.0

    def test_gate_returns_zero_at_boundary(self):
        # ni == nw is still inside the warmup window (matches train_GRL.py: `if ni <= nw`).
        assert compute_da_warmup_scale(ni=1000, nw=1000, mode='gate') == 0.0

    def test_gate_returns_one_just_past_boundary(self):
        assert compute_da_warmup_scale(ni=1001, nw=1000, mode='gate') == 1.0

    def test_gate_returns_one_post_warmup(self):
        assert compute_da_warmup_scale(ni=5000, nw=1000, mode='gate') == 1.0


class TestRampMode:
    def test_ramp_returns_zero_at_iter_zero(self):
        assert compute_da_warmup_scale(ni=0, nw=1000, mode='ramp') == 0.0

    def test_ramp_returns_half_at_midpoint(self):
        assert math.isclose(compute_da_warmup_scale(ni=500, nw=1000, mode='ramp'), 0.5)

    def test_ramp_returns_one_at_boundary(self):
        assert compute_da_warmup_scale(ni=1000, nw=1000, mode='ramp') == 1.0

    def test_ramp_clamps_to_one_post_warmup(self):
        assert compute_da_warmup_scale(ni=5000, nw=1000, mode='ramp') == 1.0

    def test_ramp_is_monotonic(self):
        scales = [compute_da_warmup_scale(ni=i, nw=1000, mode='ramp') for i in range(0, 1100, 100)]
        for a, b in zip(scales, scales[1:]):
            assert a <= b


class TestInvalid:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match='mode'):
            compute_da_warmup_scale(ni=0, nw=1000, mode='banana')

    def test_zero_nw_with_ramp_returns_one(self):
        # Edge case: nw=0 means "no warmup". Ramp should immediately be at 1.0
        # rather than dividing by zero. Same return as 'off'.
        assert compute_da_warmup_scale(ni=0, nw=0, mode='ramp') == 1.0
        assert compute_da_warmup_scale(ni=5, nw=0, mode='ramp') == 1.0

    def test_zero_nw_with_gate_returns_one(self):
        # nw=0 means no warmup window — gate is immediately open.
        assert compute_da_warmup_scale(ni=0, nw=0, mode='gate') == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_da_warmup.py -v`
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'utils.da_warmup'`.

- [ ] **Step 3: Implement the helper**

```python
# utils/da_warmup.py
"""DA-loss warmup scaling — pure helper used by train_GRL.py.

The scalar returned by `compute_da_warmup_scale` multiplies `loss_da_image`
before it is added to the total loss. Modes:
    off:  always 1.0 (no warmup — DA loss applied from iter 0).
    gate: 0.0 while ni <= nw, then 1.0. Hard cutoff at the LR-warmup boundary.
    ramp: linear from 0.0 at ni=0 to 1.0 at ni=nw, clamped at 1.0 after.

`nw == 0` short-circuits to 1.0 (no warmup window) for both gate and ramp.
"""
from __future__ import annotations

_VALID_MODES = ('off', 'gate', 'ramp')


def compute_da_warmup_scale(ni: int, nw: int, mode: str) -> float:
    """Return the scalar to multiply `loss_da_image` by at iteration `ni`.

    Args:
        ni: current global iteration index (matches `ni` in train_GRL.py).
        nw: number of LR-warmup iterations (matches `nw` in train_GRL.py).
        mode: one of 'off', 'gate', 'ramp'.

    Returns:
        Python float in [0.0, 1.0].
    """
    if mode == 'off':
        return 1.0
    if nw <= 0:
        return 1.0
    if mode == 'gate':
        return 0.0 if ni <= nw else 1.0
    if mode == 'ramp':
        if ni >= nw:
            return 1.0
        return ni / nw
    raise ValueError(f'Unknown mode {mode!r}; expected one of {_VALID_MODES}')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_da_warmup.py -v`
Expected: 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/da_warmup.py tests/test_da_warmup.py
git commit -m "feat: compute_da_warmup_scale helper + unit tests"
```

---

## Task 3.5.2: Wire `--da-img-warmup` flag into `train_GRL.py` + extend `DALogger`

**Files:**
- Modify: `utils/da_logger.py` — add `da_scale` to `self._fields`
- Modify: `train_GRL.py` — add CLI flag, compute scale, scale DA loss, short-circuit advgrl_step when scale==0, log da_scale per-iter

- [ ] **Step 1: Extend `DALogger` fields**

Modify `utils/da_logger.py` line 13 — append `'da_scale'` to the fields list:

```python
self._fields = ['epoch', 'iter', 'loss_det', 'loss_da_image', 'loss_triplet_img', 'lambda_adv', 'L_c', 'da_scale']
```

No other changes — `setdefault('', ...)` at line 17 already handles missing keys.

- [ ] **Step 2: Add the CLI flag**

In `train_GRL.py`, locate the DA flag block at lines 648–651 (look for the comment `# Domain adaptation flags (all default off; --da-img reproduces YOLO-G original)`).

Add immediately after line 651 (after `--da-img-grl-weight`):

```python
    parser.add_argument('--da-img-warmup', type=str, default='off', choices=['off', 'gate', 'ramp'],
                        help='DA-loss warmup: off=apply from iter 0 (faithful original), '
                             'gate=zero during LR warmup, ramp=linear 0→1 over LR warmup')
```

- [ ] **Step 3: Import helper at top of train_GRL.py**

In `train_GRL.py`, locate the imports for `advgrl_step` (line 66: `from utils.advgrl import advgrl_step, default_alpha`).

Add immediately below:

```python
from utils.da_warmup import compute_da_warmup_scale
```

- [ ] **Step 4: Wire the scale into the DA loss block**

In `train_GRL.py`, locate the DA block at lines 441–454 inside the training loop forward pass:

```python
                lambda_adv_this_iter = None
                L_c_this_iter = None
                loss_da_image_this_iter = None
                if classifier_head is not None:
                    B_t = t_imgs.size(0)
                    st_feat = backbone_feat[:B_s + B_t]
                    alpha = opt.advgrl_alpha if opt.advgrl_alpha is not None else default_alpha()
                    loss_da_image, lambda_adv_this_iter, L_c_this_iter = advgrl_step(
                        st_feat, source_count=B_s, classifier=classifier_head,
                        use_advgrl=opt.advgrl, lambda_0=opt.da_img_grl_weight,
                        alpha=alpha, beta=opt.advgrl_threshold,
                    )
                    loss_da_image_this_iter = loss_da_image
                    loss = loss + opt.da_img_weight * loss_da_image
```

Replace with:

```python
                lambda_adv_this_iter = None
                L_c_this_iter = None
                loss_da_image_this_iter = None
                da_scale_this_iter = 1.0
                if classifier_head is not None:
                    da_scale_this_iter = compute_da_warmup_scale(ni=ni, nw=nw, mode=opt.da_img_warmup)
                    if da_scale_this_iter > 0.0:
                        B_t = t_imgs.size(0)
                        st_feat = backbone_feat[:B_s + B_t]
                        alpha = opt.advgrl_alpha if opt.advgrl_alpha is not None else default_alpha()
                        loss_da_image, lambda_adv_this_iter, L_c_this_iter = advgrl_step(
                            st_feat, source_count=B_s, classifier=classifier_head,
                            use_advgrl=opt.advgrl, lambda_0=opt.da_img_grl_weight,
                            alpha=alpha, beta=opt.advgrl_threshold,
                        )
                        loss_da_image_this_iter = loss_da_image
                        loss = loss + da_scale_this_iter * opt.da_img_weight * loss_da_image
                    # else: da_scale == 0.0 — skip advgrl_step entirely (no forward, no grads).
                    # loss_da_image_this_iter / lambda_adv_this_iter / L_c_this_iter stay None.
```

**Rationale:** When `gate` mode is active and `ni <= nw`, `da_scale == 0.0` and we skip the entire two-pass `advgrl_step`. This saves compute and avoids backpropagating zero-weighted gradients. When `ramp` mode is active, `da_scale > 0` from `ni=1` onward, so we always run `advgrl_step` and L_c is logged throughout.

- [ ] **Step 5: Add `da_scale` to the per-iter `da_logger.log` call**

In `train_GRL.py`, locate the DA logging block at lines 467–476:

```python
            # DA per-iter logging
            if da_logger is not None:
                da_logger.log(
                    epoch=epoch,
                    iter=ni,
                    loss_det=float(loss_det_unscaled.item()),
                    loss_da_image=float(loss_da_image_this_iter.item()) if loss_da_image_this_iter is not None else '',
                    lambda_adv=lambda_adv_this_iter if lambda_adv_this_iter is not None else '',
                    L_c=L_c_this_iter if L_c_this_iter is not None else '',
                )
```

Modify to add `da_scale`:

```python
            # DA per-iter logging
            if da_logger is not None:
                da_logger.log(
                    epoch=epoch,
                    iter=ni,
                    loss_det=float(loss_det_unscaled.item()),
                    loss_da_image=float(loss_da_image_this_iter.item()) if loss_da_image_this_iter is not None else '',
                    lambda_adv=lambda_adv_this_iter if lambda_adv_this_iter is not None else '',
                    L_c=L_c_this_iter if L_c_this_iter is not None else '',
                    da_scale=da_scale_this_iter if classifier_head is not None else '',
                )
```

- [ ] **Step 6: Smoke-test with default mode (regression check)**

Run a 1-epoch training with `--da-img-warmup off` (default) to confirm faithful-original behavior is unchanged:

```bash
python train_GRL.py --da-img --epochs 1 --batch-size 8 --img 640 --weights yolov5l.pt \
    --data <path-to-cityscapes-to-foggy.yaml> --name smoke_warmup_off_regression
```

Expected: no crashes; `runs/train/smoke_warmup_off_regression/da_losses.csv` has `da_scale=1.0` for every iter (the new column).

- [ ] **Step 7: Smoke-test gate mode**

```bash
python train_GRL.py --da-img --da-img-warmup gate --epochs 1 --batch-size 8 --img 640 \
    --weights yolov5l.pt --data <path> --name smoke_warmup_gate
```

Expected: no crashes; `da_scale=0.0` for `ni <= nw` and `loss_da_image` column is empty (because we short-circuited `advgrl_step`); `da_scale=1.0` and `loss_da_image` present after `ni > nw`.

- [ ] **Step 8: Smoke-test ramp mode**

```bash
python train_GRL.py --da-img --da-img-warmup ramp --epochs 1 --batch-size 8 --img 640 \
    --weights yolov5l.pt --data <path> --name smoke_warmup_ramp
```

Expected: no crashes; `da_scale` rises linearly from ~0 to 1.0 over the first `nw` iters; `loss_da_image` and `L_c` populated throughout.

- [ ] **Step 9: Run full unit-test suite to confirm nothing else broke**

Run: `pytest tests/ -v`
Expected: all existing tests still pass (24 from PR 1–3) + 14 new from Task 3.5.1 = 38 PASS.

- [ ] **Step 10: Commit**

```bash
git add train_GRL.py utils/da_logger.py
git commit -m "feat: --da-img-warmup={off,gate,ramp} flag with da_scale logging"
```

---

## Task 3.5.3: Sanity gate — confirm DA warmup fixes the collapse

**Files:** none modified — this is an empirical validation step.

**Reference numbers** (from `memory/yolog_pr1_sanity_baseline.md`):
- PR 1 source-only baseline mAP@0.5 = **0.379** at epoch 9.
- ±1.0 slack: passing range is **0.369–0.389**.

**Prior failed runs** (for comparison):
- PR 2 (`--da-img`) without warmup: mAP@0.5 = 0.031 at epoch 9 — **fails by 12×**.
- PR 3 (`--advgrl`) without warmup: mAP@0.5 = 0.030 at epoch 9 — **fails by 12×** (AdvGRL doesn't help because L_c stays ≈ ln(2) > α).

- [ ] **Step 1: Run sanity gate with `--da-img --da-img-warmup=gate`**

```bash
python train_GRL.py --da-img --da-img-warmup gate --epochs 10 --batch-size 8 --img 640 \
    --weights yolov5l.pt --data <path-to-cityscapes-to-foggy.yaml> \
    --hyp hyps/hyp.scratch-low.yaml --name sanity_pr35_daimg_gate
```

- [ ] **Step 2: Check `runs/train/sanity_pr35_daimg_gate/results.csv` epoch 9 row**

Expected: `metrics/mAP_0.5` in **0.369–0.389** (within ±1.0 of baseline 0.379). The clean detection-only warmup should produce a trajectory very close to PR 1 baseline for epochs 0–2 (because DA loss is zero), then add stabilized DA pressure from epoch 3+.

- [ ] **Step 3: Run sanity gate with `--da-img --da-img-warmup=ramp`**

```bash
python train_GRL.py --da-img --da-img-warmup ramp --epochs 10 --batch-size 8 --img 640 \
    --weights yolov5l.pt --data <path> --hyp hyps/hyp.scratch-low.yaml \
    --name sanity_pr35_daimg_ramp
```

- [ ] **Step 4: Check `runs/train/sanity_pr35_daimg_ramp/results.csv` epoch 9 row**

Expected: `metrics/mAP_0.5` in **0.369–0.389**. Ramp should be smoother than gate (no discontinuity) and may exceed baseline slightly if DA adaptation is genuinely helping on the foggy target.

- [ ] **Step 5: Run sanity gate with `--da-img --advgrl --da-img-warmup=ramp`**

```bash
python train_GRL.py --da-img --advgrl --da-img-warmup ramp --epochs 10 --batch-size 8 \
    --img 640 --weights yolov5l.pt --data <path> --hyp hyps/hyp.scratch-low.yaml \
    --name sanity_pr35_advgrl_ramp
```

- [ ] **Step 6: Check `runs/train/sanity_pr35_advgrl_ramp/results.csv` epoch 9 row**

Expected: `metrics/mAP_0.5` in **0.369–0.389**. With warmup applied, the backbone should remain stable long enough for the classifier to actually learn → L_c should drop below α at some point → AdvGRL's dynamic weighting should finally engage. Check `da_losses.csv` for an epoch where `L_c < 0.629` (default α) and `lambda_adv < 0.1` — that's the evidence the AdvGRL gating activates.

- [ ] **Step 7: Update memory if the gate passes**

If all three sanity runs land in the passing range, update `/home/kacchan/.claude/projects/-home-kacchan-VuHung-project-nhandang-project-2/memory/MEMORY.md` and create a new memory file:

```markdown
---
name: yolog-da-warmup-finding
description: DA warmup (gate or ramp) stabilizes --da-img and --advgrl on Foggy Cityscapes; without it, both collapse at LR-peak.
metadata:
  type: project
---

PR 3.5 finding: `--da-img-warmup={gate,ramp}` is load-bearing for PR 2/PR 3 stability on Foggy Cityscapes (10ep, bs=8, img=640).

- Without warmup: PR 2 mAP@0.5 = 0.031, PR 3 mAP@0.5 = 0.030 (12× regression from PR 1's 0.379)
- With ramp: <fill in actual number>
- With gate: <fill in actual number>

**Why**: The original YOLO-G code did NOT gate DA loss during LR warmup; neither does Paper 2's AdvGRL spec. Without warmup, the GRL signal from a randomly-initialized DA classifier at LR-peak (~ep 2) destabilizes the backbone before detection has stabilized. AdvGRL alone doesn't help because L_c stays at ~ln(2) > default α ≈ 0.629 (hard regime, λ_adv = λ_0).

**How to apply**: PR 4 (`--aux` RainMix) and PR 5 (`--triplet-img`) sanity gates should be run WITH `--da-img-warmup=ramp` to clear the gate. The faithful-original runs (warmup=off) remain documented as the motivating baselines in the ablation table.
```

- [ ] **Step 8: Commit the sanity gate evidence**

Optionally commit the `results.csv` artifacts under `runs/train/sanity_pr35_*/` if the project tracks them (check `.gitignore` first — they may be excluded).

```bash
# Only if runs/ is tracked:
git add runs/train/sanity_pr35_daimg_gate/results.csv runs/train/sanity_pr35_daimg_ramp/results.csv runs/train/sanity_pr35_advgrl_ramp/results.csv
git commit -m "chore: PR 3.5 sanity gate results (da-img-warmup stabilizes PR 2/PR 3)"
```

---

## Self-review notes

**Spec coverage:**
- ✅ User chose `gate + ramp` → both implemented in Task 3.5.1 helper and exercised in Task 3.5.3 sanity.
- ✅ User chose `reuse LR warmup nw` → helper takes `nw` as a parameter; train_GRL.py passes the existing `nw` value (line 315) directly. No new CLI flag for duration.
- ✅ User chose `off as default` → CLI default is `'off'`, which short-circuits to scale=1.0, preserving byte-for-byte faithful-original behavior.

**Type consistency:**
- Helper returns `float`. `loss = loss + da_scale * opt.da_img_weight * loss_da_image` — PyTorch auto-broadcasts the Python float across the tensor. Tested via Task 3.5.1.

**Placeholder scan:**
- `<path-to-cityscapes-to-foggy.yaml>` and `<path>` in Task 3.5.3 are user-environment-specific (the user has the same path already used in PR 1/2/3 sanity runs — they can reuse it).
- `<fill in actual number>` in Task 3.5.3 Step 7 memory template — these are placeholders to be filled with the actual sanity numbers when running.
- No other placeholders.

**Risks / edge cases:**
- If sanity gates fail (e.g., warmup helps but doesn't fully close the gap), the failure mode is informative: either lower `--da-img-grl-weight` from 0.1 → 0.05, or extend warmup beyond `nw`. Document in commit message and consider as PR 3.6 follow-up.
- If `da_scale=0.0` but `classifier_head is not None`, `loss_da_image_this_iter` stays None, so `da_logger.log` gets `loss_da_image=''` — already handled by `setdefault('', ...)` at `da_logger.py:17`.
- DDP / AMP: scaling happens before the `if RANK != -1: loss *= WORLD_SIZE` line, so DDP scaling still works correctly on the combined loss. AMP autocast wraps the forward; scaling a tensor by a Python float is autocast-safe.
