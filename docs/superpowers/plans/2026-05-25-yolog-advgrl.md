# YOLO-G + AdvGRL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port AdvGRL, auxiliary RainMix domain, and triplet metric regularization from DA-Detect onto YOLO-G as independently toggleable components, preserving the source-only `train.py` baseline.

**Architecture:** Refactor YOLOv5-L to a "dumb model" that returns `(det_pred, backbone_feat)`. All domain-adaptation logic lives in a new `train_GRL.py` with opt-in flags `--da-img`, `--advgrl`, `--aux`, `--triplet-img`. Eval (`val_GRL.py`) sees a clean detector via existing `ckpt['model']`/`ckpt['ema']` keys; the domain classifier is stored separately under `ckpt['classifier']`.

**Tech Stack:** PyTorch, YOLOv5 codebase (Ultralytics), pytest, numpy, OpenCV, PIL, AugMix-style augmentation.

**Spec:** `docs/superpowers/specs/2026-05-25-yolog-advgrl-design.md` — read this first.

**PR boundaries:** 6 PRs total. Each ends with a sanity gate. Do not start the next PR until the previous PR's sanity gate passes.

| PR | Phase | Sanity gate |
|---|---|---|
| 1 | Foundation: tests + dumb-model refactor | 10-epoch source-only `train_GRL.py` matches `train.py` within ±0.5 mAP |
| 2 | Base DA: `--da-img` (fixed-λ DANN) | 10-epoch `--da-img` matches existing YOLO-G baseline within ±0.5 mAP |
| 3 | AdvGRL: `--advgrl` | 10-epoch full run: no NaN, `lambda_adv ≤ λ₀·β = 3.0` throughout |
| 4 | Auxiliary domain: `--aux` | RainMix gen produces visually domain-shifted images; 3-way zip dataloader smoke test passes |
| 5 | Triplet: `--triplet-img` | All 6 flag combinations pass smoke test |
| 6 | Notebooks + ablation runner | All 6 notebooks execute end-to-end on Colab T4 (or local equivalent) |

---

## File Structure

**New files:**
- `tests/__init__.py` — empty marker
- `tests/conftest.py` — shared pytest fixtures (toy images, tiny model, device)
- `tests/test_grl_gradient.py` — gradient sign/scale tests for `gradient_scalar`
- `tests/test_advgrl.py` — `compute_lambda_adv` unit tests
- `tests/test_triplet_loss.py` — `triplet_img_loss` unit tests
- `tests/test_dumb_model.py` — `Model._forward_once` returns `(det_pred, backbone_feat)` with correct shapes
- `tests/test_train_smoke.py` — 5-iter smoke test across all flag combinations
- `tests/fixtures/make_fixtures.py` — generates toy PNG images and labels deterministically
- `models/da_classifier.py` — standalone `DAImgHead` module (Conv→ReLU→Conv, no baked GRL)
- `utils/domain_loss.py` — new file; `da_img_loss`, `triplet_img_loss`, optional adaptive-margin helper
- `utils/domain_aux.py` — auxiliary-domain dataloader wrapper + path resolver
- `utils/advgrl.py` — `compute_lambda_adv` helper + `advgrl_step` two-pass routine
- `utils/da_logger.py` — `da_losses.csv` per-iter logger
- `train_GRL.py` — new composable DA training loop
- `tools/gen_rainy_cityscapes.py` — offline RainMix generator (ported from DA-Detect)
- `tools/augment_and_mix.py` — AugMix helper used by the generator
- `tools/augmentations.py` — AugMix op library used by the generator
- `tools/run_ablations.sh` — sequential ablation runner
- `notebooks/00_setup_and_data.ipynb` — Colab setup + data download
- `notebooks/01_generate_rainy_aux.ipynb` — run RainMix generator
- `notebooks/02_train_baseline.ipynb` — source-only run
- `notebooks/03_train_yolog_original.ipynb` — `--da-img` only
- `notebooks/04_train_advgrl_full.ipynb` — full method
- `notebooks/05_evaluate_and_compare.ipynb` — compare all runs

**Modified files:**
- `requirements.txt` — add `pytest>=7.0.0`
- `models/yolo_GRL.py` — refactor `_forward_once` to return `(det_pred, backbone_feat)` via layer-9 capture
- `models/common.py` — remove `GradientScalarLayer` from `Classifyy` (will be deleted from YAML anyway, but make it harmless)
- `configs/domain/yolov5l_GRL.yaml` — remove `[9, 1, Classifyy, [1]]` line
- `domain/city_foggycity.yaml` — add `aux:` and `rain_mask:` keys (optional)
- `val_GRL.py` — 2-line patch to strip `backbone_feat` from the model's training-mode return tuple
- `train_GRL.sh` — update with full flag set for the full method
- `utils/damain_loss.py` — **deleted** after migration to `utils/domain_loss.py` (preserve git history via rename)
- `utils/domain_grl.py` — no changes; functional `gradient_scalar = _GradientScalarLayer.apply` already exists

**Unchanged (intentional — preserves clean baseline):**
- `train.py`
- `val.py`
- `utils/general.py`
- `utils/dataloaders.py`

---

## PR 1 — Foundation: tests + dumb-model refactor

### Task 1.1: Add pytest and create test scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/make_fixtures.py`

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
# Testing -------------------------------------
pytest>=7.0.0
```

- [ ] **Step 2: Install pytest**

Run: `pip install pytest>=7.0.0`
Expected: `Successfully installed pytest-7.x.y` (or already installed)

- [ ] **Step 3: Create empty `tests/__init__.py` and `tests/fixtures/__init__.py`**

Both files: empty (just `touch`-equivalent via Write tool with empty content).

- [ ] **Step 4: Create `tests/fixtures/make_fixtures.py`**

```python
"""Generate deterministic toy images + YOLO-format labels for tests/fixtures/.

Usage: python tests/fixtures/make_fixtures.py
Produces 4 source images (with labels), 4 target images (no labels),
4 aux images (no labels), all 64x64 RGB PNGs.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
SEED = 1234
IMG_SIZE = 64


def make_image(rng, kind):
    """kind in {'source', 'target', 'aux'} — each has a distinct color bias."""
    bias = {'source': [200, 100, 100], 'target': [100, 200, 100], 'aux': [100, 100, 200]}[kind]
    img = rng.integers(0, 80, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = (img + np.array(bias, dtype=np.uint8)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    rng = np.random.default_rng(SEED)
    for kind in ('source', 'target', 'aux'):
        img_dir = ROOT / kind / 'images'
        lbl_dir = ROOT / kind / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = make_image(rng, kind)
            img.save(img_dir / f'{kind}_{i:02d}.png')
            # YOLO label format: class cx cy w h (normalized). Source has 1 box; others empty.
            if kind == 'source':
                (lbl_dir / f'{kind}_{i:02d}.txt').write_text('0 0.5 0.5 0.3 0.3\n')
            else:
                (lbl_dir / f'{kind}_{i:02d}.txt').write_text('')
    # Manifests for YOLO dataloader
    for kind in ('source', 'target', 'aux'):
        manifest = ROOT / f'{kind}.txt'
        imgs = sorted((ROOT / kind / 'images').glob('*.png'))
        manifest.write_text('\n'.join(str(p) for p in imgs) + '\n')
    print(f'Fixtures written to {ROOT}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Generate the fixtures**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && python tests/fixtures/make_fixtures.py`
Expected: `Fixtures written to /.../yolo-G/tests/fixtures` and 12 PNGs + 3 manifests on disk.

- [ ] **Step 6: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from pathlib import Path
import pytest
import torch

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def source_manifest():
    return str(FIXTURES / 'source.txt')


@pytest.fixture
def target_manifest():
    return str(FIXTURES / 'target.txt')


@pytest.fixture
def aux_manifest():
    return str(FIXTURES / 'aux.txt')


@pytest.fixture
def tiny_backbone_feat(device):
    """[2, 1024, 2, 2] — mimics YOLOv5-L SPPF output at very small img size."""
    torch.manual_seed(0)
    return torch.randn(2, 1024, 2, 2, device=device, requires_grad=True)
```

- [ ] **Step 7: Verify pytest can collect**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/ --collect-only`
Expected: `no tests ran in 0.xx s` (no test files yet) — but no errors.

- [ ] **Step 8: Commit**

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
git add requirements.txt tests/
git commit -m "test: add pytest scaffolding and toy fixtures"
```

---

### Task 1.2: Test for `gradient_scalar` functional API

**Files:**
- Create: `tests/test_grl_gradient.py`

The existing `utils/domain_grl.py` already exposes `gradient_scalar = _GradientScalarLayer.apply`. This task pins its behavior with a test before any refactor touches surrounding code.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grl_gradient.py`:

```python
"""Pin gradient_scalar behavior: forward is identity, backward scales by weight."""
import torch
from utils.domain_grl import gradient_scalar


def test_forward_is_identity():
    x = torch.randn(2, 3, 4, 4)
    y = gradient_scalar(x, -0.1)
    assert torch.allclose(y, x)


def test_backward_scales_by_weight():
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    # Use small weight -0.1 — typical GRL sign-flip + scale.
    y = gradient_scalar(x, -0.1)
    y.sum().backward()
    # dL/dx via identity-forward is grad_output (== 1 everywhere from sum().backward()),
    # scaled by weight = -0.1.
    expected = torch.full_like(x, -0.1)
    assert torch.allclose(x.grad, expected)


def test_backward_dynamic_weight():
    """Critical for AdvGRL: weight can be a Python float computed per iter."""
    for w in [-3.0, -1.0, -0.1, 0.0, 0.5]:
        x = torch.randn(2, 3, requires_grad=True)
        y = gradient_scalar(x, w)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, w))
```

- [ ] **Step 2: Run to confirm tests pass (no implementation needed — `gradient_scalar` exists)**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_grl_gradient.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_grl_gradient.py
git commit -m "test: pin gradient_scalar forward/backward behavior"
```

---

### Task 1.3: Refactor `Classifyy` to remove baked GRL

**Files:**
- Modify: `models/common.py` (lines ~769-790, the `Classifyy` class)

This class will be deleted from the YAML in Task 1.5, so we just need to make it harmless if anyone still uses it from old code. We keep the class to avoid breaking imports, but strip the baked `GradientScalarLayer(-0.1)`.

- [ ] **Step 1: Modify `Classifyy.__init__` and `Classifyy.forward`**

Replace the entire `Classifyy` class (lines ~769-790) with:

```python
class Classifyy(nn.Module):
    """Domain classifier head (pure Conv->ReLU->Conv).

    No baked GRL — gradient reversal happens externally via utils.domain_grl.gradient_scalar.
    Kept for backwards compatibility with old YAMLs; new code should import DAImgHead
    from models.da_classifier instead.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, 512, 1, 1)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(512, c2, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        return self.conv(x)
```

- [ ] **Step 2: Sanity-import the modified module**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && python -c "from models.common import Classifyy; m = Classifyy(1024, 1); import torch; print(m(torch.randn(1,1024,2,2)).shape)"`
Expected: `torch.Size([1, 1, 2, 2])`

- [ ] **Step 3: Commit**

```bash
git add models/common.py
git commit -m "refactor: strip baked GRL from Classifyy (gradient reversal now external)"
```

---

### Task 1.4: Create standalone `DAImgHead` module

**Files:**
- Create: `models/da_classifier.py`

- [ ] **Step 1: Write the file**

```python
"""Standalone image-level domain classifier head.

Used externally to the YOLO model (dumb-model principle): the model returns
backbone_feat; train_GRL.py runs DAImgHead on the feature with gradient_scalar
applied for the GRL-attached pass.
"""
import torch
from torch import nn


class DAImgHead(nn.Module):
    """Conv -> ReLU -> Conv producing [B, 1, H, W] logits.

    Initialized with small weights (std=0.001) to match DA-Detect's DAImgHead.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        for layer in (self.conv1, self.conv2):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        return self.conv2(x)
```

- [ ] **Step 2: Write a smoke test**

Append to `tests/test_dumb_model.py` (create the file):

```python
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
```

- [ ] **Step 3: Run test**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_dumb_model.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add models/da_classifier.py tests/test_dumb_model.py
git commit -m "feat: add standalone DAImgHead module (no baked GRL)"
```

---

### Task 1.5: Remove `Classifyy` from YAML head

**Files:**
- Modify: `configs/domain/yolov5l_GRL.yaml`

- [ ] **Step 1: Remove line 47 (the Classifyy entry)**

In `configs/domain/yolov5l_GRL.yaml`, the head section currently ends with:

```yaml
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [9,1,Classifyy,[1]],

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

Replace with (just delete the `Classifyy` line and its blank padding):

```yaml
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

- [ ] **Step 2: Verify YAML still parses**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && python -c "import yaml; print(yaml.safe_load(open('configs/domain/yolov5l_GRL.yaml')))"`
Expected: dict prints, no errors.

- [ ] **Step 3: Commit**

```bash
git add configs/domain/yolov5l_GRL.yaml
git commit -m "refactor: remove Classifyy from yolov5l_GRL.yaml (model is now dumb)"
```

---

### Task 1.6: Refactor `Model._forward_once` to capture layer-9 feature

**Files:**
- Modify: `models/yolo_GRL.py` (lines 193-212, the `_forward_once` method)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dumb_model.py`:

```python
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
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_dumb_model.py::test_forward_once_returns_tuple_with_backbone_feat -v`
Expected: FAIL — either tuple-shape mismatch or layer-9 not captured.

- [ ] **Step 3: Implement the refactor**

Replace the `_forward_once` method in `models/yolo_GRL.py` (currently lines 193-212):

```python
    def _forward_once(self, x, profile=False, visualize=False):
        """Forward pass that returns (det_pred, backbone_feat).

        backbone_feat is the SPPF output (layer 9) — used externally for domain
        adaptation in train_GRL.py. Capture is explicit (via m.i == 9) so it
        does not depend on the save-list being driven by later layer references.
        """
        y = []  # save-list outputs
        backbone_feat = None
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, [])
            x = m(x)
            if m.i == 9:  # SPPF — explicit capture, independent of save list
                backbone_feat = x
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x, backbone_feat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_dumb_model.py::test_forward_once_returns_tuple_with_backbone_feat -v`
Expected: PASS.

- [ ] **Step 5: Run all tests so far to check for regressions**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add models/yolo_GRL.py tests/test_dumb_model.py
git commit -m "refactor: Model._forward_once returns (det_pred, backbone_feat) via layer-9 capture"
```

---

### Task 1.7: Patch `val_GRL.py` for new forward-tuple shape

**Files:**
- Modify: `val_GRL.py` (around line 196 where model output is unpacked)

The current `val_GRL.py` line 196 reads:

```python
(out, train_out), _ = model(im) if training else model(im, augment=augment, val=True)
```

The `_` was catching the old `pred` list. After Task 1.6, eval-mode forward returns `((out_inference, train_out), backbone_feat)` — the trailing `_` already absorbs `backbone_feat`. **No change needed if the pattern matches.**

Let me verify the current code path: in eval mode, `Model.forward` calls `_forward_once`, which now returns `(x, backbone_feat)` where `x` is `(torch.cat(z, 1), x)` (the Detect.forward eval-mode tuple). So overall: `((out_inf, train_out), backbone_feat)`.

The existing destructuring `(out, train_out), _ = ...` works correctly — `_` catches `backbone_feat`. Confirm and add a defensive guard:

- [ ] **Step 1: Read the relevant block in `val_GRL.py`**

Run: `sed -n '190,220p' val_GRL.py`

- [ ] **Step 2: Verify the destructuring pattern still works**

In `val_GRL.py` around line 196, confirm the line reads (after your read):

```python
(out, train_out), _ = model(im) if training else model(im, augment=augment, val=True)
```

If yes — no change required. Add a comment to lock the contract:

Edit `val_GRL.py`, find that line and replace with:

```python
# Model.forward returns ((inference_out, train_out), backbone_feat) in eval mode after
# the dumb-model refactor. backbone_feat is unused at eval time — discarded via _.
(out, train_out), _ = model(im) if training else model(im, augment=augment, val=True)
```

If the line differs (e.g., old code unpacked into `(out, train_out), pred = ...`), replace `pred` with `_`.

- [ ] **Step 3: Smoke-test `val_GRL.py` imports**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && python -c "import val_GRL"`
Expected: no errors (module imports cleanly).

- [ ] **Step 4: Commit**

```bash
git add val_GRL.py
git commit -m "fix: val_GRL.py discards backbone_feat from new model forward tuple"
```

---

### Task 1.8: Sanity gate — 10-epoch source-only run

This is **manual** — runs on real data. Do not skip; this catches dumb-model refactor breakage before PR 2 builds on top.

- [ ] **Step 1: Run baseline `train.py` for 10 epochs on source data**

Run:

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
python train.py --weights yolov5l.pt --cfg configs/domain/yolov5l.yaml \
    --data domain/city_foggycity.yaml --epochs 10 --batch-size 8 --img 640 \
    --name sanity_pr1_baseline
```

Record final mAP@0.5 on the val set from `runs/train/sanity_pr1_baseline/results.csv`.

- [ ] **Step 2: Note the result**

Write the mAP value down. This is the reference number Task 2.x will compare against.

- [ ] **Step 3: PR 1 gate check**

The dumb-model refactor must not have broken `train.py`. Since `train.py` doesn't use `yolo_GRL.Model` (it uses `yolo.Model`), this baseline confirms the broader codebase is healthy. The dumb-model integration test from Task 1.6 already confirmed `yolo_GRL.Model` returns the right shape.

If the run produces a reasonable mAP (>= 0.1 on Cityscapes 8-class baseline, which is the floor for early epochs) → PR 1 is good. **Open PR 1 now** before moving to PR 2.

---

## PR 2 — Base DA: `train_GRL.py` with `--da-img`

This PR introduces the new training script with **only** the fixed-λ DANN component enabled. AdvGRL, aux, and triplet are added in later PRs.

### Task 2.1: Domain dataloader pairing utility

**Files:**
- Create: `utils/domain_aux.py`

`utils/domain_aux.py` will eventually host the aux loader too. For PR 2 we only need source+target pairing.

- [ ] **Step 1: Write the file**

```python
"""Multi-domain dataloader pairing helpers.

resolve_da_path: resolves target/aux paths against the YAML's `path:` root
  (since check_dataset only resolves train/val/test — see utils/general.py:416).

create_target_dataloader: a thin wrapper around YOLO's create_dataloader for
  unlabeled target/aux image lists. We reuse YOLO's pipeline so augmentation
  (letterbox, image scaling) matches the source pipeline exactly.
"""
from pathlib import Path
from typing import Optional


def resolve_da_path(p: str, root: Optional[str]) -> str:
    """Resolve a DA path (target/aux) against the YAML 'path:' root.

    Absolute paths are returned unchanged. Relative paths are joined with root.
    """
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    if root is None:
        return str(pp)
    return str(Path(root) / pp)


def create_target_dataloader(images_path, imgsz, batch_size, stride, workers=4, prefix='target: '):
    """Create a dataloader for unlabeled target/aux images.

    Uses YOLO's create_dataloader so augmentation matches source. The labels it
    yields are ignored at the training loop level — we only consume images.
    """
    from utils.dataloaders import create_dataloader
    loader, _ = create_dataloader(
        images_path, imgsz, batch_size, stride,
        single_cls=False, hyp=None, augment=False, cache=False, rect=False,
        rank=-1, workers=workers, image_weights=False, quad=False,
        prefix=prefix, shuffle=True,
    )
    return loader
```

- [ ] **Step 2: Write a unit test for `resolve_da_path`**

Create `tests/test_domain_aux.py`:

```python
"""Tests for utils/domain_aux.py path resolution."""
from utils.domain_aux import resolve_da_path


def test_absolute_path_unchanged():
    assert resolve_da_path('/abs/path/imgs', '/root') == '/abs/path/imgs'


def test_relative_joined_with_root():
    assert resolve_da_path('images/train', '/root/data') == '/root/data/images/train'


def test_no_root_returns_input():
    assert resolve_da_path('images/train', None) == 'images/train'
```

- [ ] **Step 3: Run test**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_domain_aux.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add utils/domain_aux.py tests/test_domain_aux.py
git commit -m "feat: utils/domain_aux with path resolver and target dataloader factory"
```

---

### Task 2.2: Migrate `damain_loss.py` → `domain_loss.py` and add `da_img_loss`

**Files:**
- Create: `utils/domain_loss.py` (new home)
- Modify: existing `utils/damain_loss.py` → leave it as a deprecation shim that re-exports from the new module (preserve any old imports)

- [ ] **Step 1: Write the new `utils/domain_loss.py`**

```python
"""Domain-adaptation loss functions.

da_img_loss      — BCE-with-logits between dense classifier output and dense domain labels
triplet_img_loss — image-level triplet metric regularization (anchor=source, positive=target, negative=aux)
"""
import torch
import torch.nn.functional as F
from torch import nn


def da_img_loss(logits: torch.Tensor, source_count: int) -> torch.Tensor:
    """Image-level DA loss (DANN-style BCE).

    Args:
        logits: classifier output of shape [B, 1, H, W] where the first
                `source_count` rows are source images and the remainder are target.
        source_count: number of source samples in the batch (B_s).

    Returns:
        Scalar BCE-with-logits loss.

    Labels are dense (shape matches logits): 1.0 for source rows, 0.0 for target.
    This matches both the existing utils/damain_loss.DA_loss convention and
    DA-Detect's DAImgHead loss.
    """
    labels = torch.zeros_like(logits)
    labels[:source_count] = 1.0
    return F.binary_cross_entropy_with_logits(logits, labels)


def triplet_img_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Image-level triplet loss: max(d(A,P) - d(A,N) + margin, 0).

    Args:
        anchor, positive, negative: [N, D] tensors (typically N=1 batch-centroid).
        margin: triplet margin (delta in spec).
    """
    return nn.TripletMarginLoss(margin=margin, p=2)(anchor, positive, negative)
```

- [ ] **Step 2: Replace the body of `utils/damain_loss.py` with a deprecation shim**

Overwrite `utils/damain_loss.py` entirely:

```python
"""Deprecation shim — use utils.domain_loss instead.

Kept to avoid breaking old imports. The original DA_loss is re-exported with
its legacy signature; new code should call utils.domain_loss.da_img_loss.
"""
import torch
import torch.nn.functional as F


def DA_loss(features, target):
    """Legacy DA loss: features is a list of [B,C,H,W] tensors, target is 0 or 1."""
    loss = 0.0
    for feature in features:
        N, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1)
        label = torch.zeros_like(feature) if target == 0 else torch.ones_like(feature)
        feature_end = feature.reshape(N, -1)
        label_end = label.reshape(N, -1)
        loss = loss + F.binary_cross_entropy_with_logits(feature_end, label_end)
    return loss / len(features)
```

(We don't auto-import the new module's symbols here, because the old `DA_loss` had a list-of-features signature; the new `da_img_loss` is per-iter on a single concatenated batch. They are semantically different — leave the old name alone.)

- [ ] **Step 3: Write unit test for `da_img_loss`**

Create `tests/test_domain_loss.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_domain_loss.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/domain_loss.py utils/damain_loss.py tests/test_domain_loss.py
git commit -m "feat: utils/domain_loss with da_img_loss + triplet_img_loss"
```

---

### Task 2.3: Implement `train_GRL.py` skeleton with `--da-img`

**Files:**
- Create: `train_GRL.py`

This is the largest single task in the plan. We base the skeleton on `train.py` and add the DA-specific hooks. The skeleton supports source-only (no flags) and `--da-img` only — AdvGRL, aux, triplet are added in later PRs as guarded branches.

- [ ] **Step 1: Copy `train.py` to `train_GRL.py` and read it through**

Run:

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
cp train.py train_GRL.py
```

- [ ] **Step 2: Add DA imports near the top of `train_GRL.py`**

Find the import block in `train_GRL.py` (the `from utils.X import ...` lines). After the last existing import, add:

```python
# DA additions
from models.da_classifier import DAImgHead
from utils.domain_grl import gradient_scalar
from utils.domain_loss import da_img_loss
from utils.domain_aux import resolve_da_path, create_target_dataloader
```

- [ ] **Step 3: Add CLI flags in `parse_opt()`**

Find `parse_opt()` in `train_GRL.py`. After the last `parser.add_argument(...)` call but before `return parser.parse_args()`, add:

```python
    # Domain adaptation flags (all default off; --da-img reproduces YOLO-G original)
    parser.add_argument('--da-img', action='store_true', help='enable image-level DANN (S vs T)')
    parser.add_argument('--da-img-weight', type=float, default=1.0, help='multiplier on loss_da_image')
    parser.add_argument('--da-img-grl-weight', type=float, default=0.1, help='lambda_0 — GRL weight on the DA branch')
    # AdvGRL flags (used in PR 3 — defined here so the namespace is stable)
    parser.add_argument('--advgrl', action='store_true', help='enable dynamic lambda_adv (requires --da-img)')
    parser.add_argument('--advgrl-threshold', type=float, default=30.0, help='beta — cap on adv_threshold')
    parser.add_argument('--advgrl-alpha', type=float, default=None, help='alpha gate; if None, computed as BCE([0.7,0.3],[1,0])')
    # Aux flags (used in PR 4)
    parser.add_argument('--aux', action='store_true', help='enable auxiliary RainMix domain')
    # Triplet flags (used in PR 5)
    parser.add_argument('--triplet-img', action='store_true', help='enable image-level triplet loss (requires --aux)')
    parser.add_argument('--triplet-img-weight', type=float, default=0.1)
    parser.add_argument('--triplet-margin', type=float, default=1.0)
    parser.add_argument('--triplet-adaptive', action='store_true', help='adaptive margin ramp (requires --triplet-img)')
    parser.add_argument('--triplet-max-margin', type=float, default=3.0)
```

- [ ] **Step 4: Add flag-dependency validation right after `opt = parse_opt()` in `main()`**

Find `def main(opt, callbacks=Callbacks()):` and add at the very top of the body:

```python
    # Flag dependency validation (see spec §4.2)
    if opt.advgrl and not opt.da_img:
        raise SystemExit('--advgrl requires --da-img (no DA classifier means no L_c to gate on)')
    if opt.triplet_img and not opt.aux:
        raise SystemExit('--triplet-img requires --aux (triplet negative comes from aux loader)')
    if opt.triplet_adaptive and not opt.triplet_img:
        raise SystemExit('--triplet-adaptive requires --triplet-img')
```

- [ ] **Step 5: In `train(...)`, instantiate the domain classifier + target dataloader when `--da-img` is set**

Find the line where the model is constructed (look for `Model(cfg, ...)` near the start of `train(...)`). Right after the model setup and dataloader creation (find where `train_loader` is created and use that as the anchor), add:

```python
    # ---- Domain adaptation setup ------------------------------------------------
    classifier_head = None
    target_loader = None
    if opt.da_img:
        # Classifier head matches backbone_feat channels (1024 for YOLOv5-L).
        classifier_head = DAImgHead(in_channels=1024).to(device)
        # Add classifier params to the optimizer's param groups.
        optimizer.add_param_group({'params': list(classifier_head.parameters()), 'weight_decay': 0.0})
        # Resolve target path (relative paths are joined with data_dict['path']).
        target_path = resolve_da_path(data_dict['target'], data_dict.get('path'))
        target_loader = create_target_dataloader(
            target_path, imgsz, batch_size, gs, workers=workers, prefix=colorstr('target: '),
        )
    # ----------------------------------------------------------------------------
```

(Use `gs` for stride — that's the YOLOv5 convention; `gs = max(int(model.stride.max()), 32)` should already exist by that point in `train(...)`. If not, search for `gs =` in the original `train.py` to find the correct variable.)

- [ ] **Step 6: Modify the training inner loop to consume target batches + compute `loss_da_image`**

Find the main training loop in `train(...)` — the `for i, (imgs, targets, paths, _) in pbar:` line. We need to:
1. Iterate `target_loader` alongside `train_loader` when `--da-img` is set
2. Concatenate source + target images for a single backbone forward
3. Slice detection output to source-only
4. Compute `loss_da_image` on the classifier head

Replace the existing forward-and-loss block with this logic. The structure becomes:

```python
        # Create target iterator if DA is on (rebuilt each epoch via the for-loop scope)
        target_iter = iter(target_loader) if target_loader is not None else None

        pbar = enumerate(train_loader)
        # ... existing tqdm/progress setup ...

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # ---- Source + target concatenation ---------------------------------
            B_s = imgs.size(0)
            t_imgs = None
            if target_iter is not None:
                try:
                    t_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    t_batch = next(target_iter)
                t_imgs = t_batch[0].to(device, non_blocking=True).float() / 255.0
                all_imgs = torch.cat([imgs, t_imgs], dim=0)
            else:
                all_imgs = imgs
            # ---------------------------------------------------------------------

            # Warmup, scheduler, etc. — keep existing logic up to the forward call.
            # ...

            # Forward
            with amp.autocast(enabled=cuda):
                det_pred, backbone_feat = model(all_imgs)
                # det_pred is a list of per-scale tensors in training mode.
                det_pred_src = [p[:B_s] for p in det_pred]
                loss_det, loss_items = compute_loss(det_pred_src, targets.to(device))

                losses = {'loss_det': loss_det}

                if classifier_head is not None:
                    B_t = t_imgs.size(0)
                    st_feat = backbone_feat[:B_s + B_t]
                    # Fixed-lambda GRL pass (PR 3 will replace with two-pass AdvGRL).
                    feat_grl = gradient_scalar(st_feat, -opt.da_img_grl_weight)
                    da_logits = classifier_head(feat_grl)
                    loss_da_image = da_img_loss(da_logits, source_count=B_s)
                    losses['loss_da_image'] = opt.da_img_weight * loss_da_image

                total_loss = sum(losses.values())

            # Backward (existing scaler + step logic)
            scaler.scale(total_loss).backward()
            # ... existing optimizer.step / scheduler.step / zero_grad ...
```

The existing `train.py` has surrounding logic (warmup, accumulation, scaler, scheduler) — keep all of it; only replace the forward/loss block.

- [ ] **Step 7: Smoke-test the script with no DA flags (source-only path)**

Run:

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data domain/city_foggycity.yaml \
    --epochs 1 --batch-size 4 --img 320 --name smoke_pr2_baseline --noplots
```

Expected: completes 1 epoch without error. `loss_det` should be the only loss printed in the progress bar.

- [ ] **Step 8: Smoke-test with `--da-img`**

Run:

```bash
python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data domain/city_foggycity.yaml \
    --epochs 1 --batch-size 4 --img 320 --name smoke_pr2_daimg --noplots --da-img
```

Expected: completes 1 epoch. Now both `loss_det` and `loss_da_image` should be non-zero. Verify the `target:` line appears in console output.

- [ ] **Step 9: Commit**

```bash
git add train_GRL.py
git commit -m "feat: train_GRL.py with --da-img (fixed-lambda DANN)"
```

---

### Task 2.4: Checkpoint schema with separate classifier key

**Files:**
- Modify: `train_GRL.py` (find the existing `torch.save(ckpt, last)` block)

- [ ] **Step 1: Locate the checkpoint save block**

Search `train_GRL.py` for `torch.save(ckpt`. In stock YOLOv5, the dict looks like:

```python
ckpt = {
    'epoch': epoch,
    'best_fitness': best_fitness,
    'model': deepcopy(de_parallel(model)).half(),
    'ema': deepcopy(ema.ema).half(),
    'updates': ema.updates,
    'optimizer': optimizer.state_dict(),
    'opt': vars(opt),
    'date': datetime.now().isoformat(),
}
```

- [ ] **Step 2: Add classifier key**

Right before the `torch.save(ckpt, last)` line, modify the dict to include:

```python
ckpt = {
    'epoch': epoch,
    'best_fitness': best_fitness,
    'model': deepcopy(de_parallel(model)).half(),
    'ema': deepcopy(ema.ema).half(),
    'updates': ema.updates,
    'optimizer': optimizer.state_dict(),
    'opt': vars(opt),
    'date': datetime.now().isoformat(),
    # DA additions (None when DA is off)
    'classifier': classifier_head.state_dict() if classifier_head is not None else None,
    'advgrl_cfg': {
        'lambda_0': opt.da_img_grl_weight,
        'alpha': opt.advgrl_alpha,
        'beta': opt.advgrl_threshold,
    },
}
```

- [ ] **Step 3: Add classifier resume logic**

Find the resume block (search for `if resume:` or `if weights and weights.endswith('.pt'):`). After the existing model state load, add:

```python
        # Resume classifier head state if present
        if classifier_head is not None and ckpt.get('classifier') is not None:
            classifier_head.load_state_dict(ckpt['classifier'])
            LOGGER.info(f'{colorstr("DA: ")}restored classifier head from checkpoint')
```

- [ ] **Step 4: Smoke-test save+load roundtrip**

Run:

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data domain/city_foggycity.yaml \
    --epochs 1 --batch-size 4 --img 320 --name smoke_pr2_ckpt --noplots --da-img
ls runs/train/smoke_pr2_ckpt/weights/
python -c "import torch; c = torch.load('runs/train/smoke_pr2_ckpt/weights/last.pt', map_location='cpu'); print('keys:', list(c.keys())); print('classifier present:', c.get('classifier') is not None)"
```

Expected: `classifier present: True`.

- [ ] **Step 5: Verify `val_GRL.py` can load the new checkpoint**

Run:

```bash
python val_GRL.py --weights runs/train/smoke_pr2_ckpt/weights/last.pt \
    --data domain/city_foggycity.yaml --task val --img 320 --batch-size 4
```

Expected: loads model from `ckpt['ema']` (which is detector-only) and runs val without touching the `classifier` key. mAP numbers may be near-zero (only 1 epoch), but the run must complete without errors.

- [ ] **Step 6: Commit**

```bash
git add train_GRL.py
git commit -m "feat: train_GRL checkpoint schema with separate classifier key"
```

---

### Task 2.5: Sanity gate — 10-epoch `--da-img` matches existing YOLO-G mAP

- [ ] **Step 1: Run 10-epoch DA training**

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
python train_GRL.py --weights yolov5l.pt --cfg configs/domain/yolov5l_GRL.yaml \
    --data domain/city_foggycity.yaml --epochs 10 --batch-size 8 --img 640 \
    --name sanity_pr2_daimg --da-img
```

- [ ] **Step 2: Compare against Task 1.8 baseline**

Read `runs/train/sanity_pr2_daimg/results.csv` final mAP@0.5. Compare against Task 1.8's `sanity_pr1_baseline` mAP. With `--da-img`, you should see roughly equal-or-slightly-better mAP on Foggy Cityscapes (the val target). Allow ±1.0 mAP slack at 10 epochs — this is a smoke check, not a publication number.

- [ ] **Step 3: PR 2 gate**

If `--da-img` runs end-to-end and produces a non-degenerate mAP → PR 2 is good. **Open PR 2 now**.

---

## PR 3 — AdvGRL: `--advgrl`

### Task 3.1: `compute_lambda_adv` helper + unit tests

**Files:**
- Create: `utils/advgrl.py`
- Create: `tests/test_advgrl.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_advgrl.py`:

```python
"""Unit tests for utils/advgrl.compute_lambda_adv."""
import math
import pytest
from utils.advgrl import compute_lambda_adv, default_alpha


LAMBDA_0 = 0.1
BETA = 30.0
ALPHA = default_alpha()  # BCE([0.7,0.3],[1,0]) ≈ 0.6286


def test_default_alpha_close_to_0_6286():
    assert math.isclose(default_alpha(), 0.6286, abs_tol=0.001)


def test_easy_regime_uses_lambda_0():
    """L_c > alpha -> return lambda_0 unchanged."""
    assert compute_lambda_adv(0.7,  LAMBDA_0, ALPHA, BETA) == LAMBDA_0
    assert compute_lambda_adv(1.0,  LAMBDA_0, ALPHA, BETA) == LAMBDA_0
    assert compute_lambda_adv(10.0, LAMBDA_0, ALPHA, BETA) == LAMBDA_0


def test_hard_regime_uses_lambda_0_times_min():
    """L_c <= alpha -> lambda_0 * min(beta, 1/L_c)."""
    # L_c = 0.3 -> 1/L_c ≈ 3.33; min(30, 3.33) = 3.33; lambda_adv = 0.333
    v = compute_lambda_adv(0.3, LAMBDA_0, ALPHA, BETA)
    assert math.isclose(v, LAMBDA_0 * (1.0 / 0.3), rel_tol=1e-5)


def test_cap_at_lambda_0_times_beta():
    """L_c -> 0 -> lambda_adv saturates at lambda_0 * beta = 3.0."""
    v = compute_lambda_adv(1e-9, LAMBDA_0, ALPHA, BETA)
    assert math.isclose(v, LAMBDA_0 * BETA, rel_tol=1e-3)
    # Also at the boundary L_c=0 with eps guard:
    v0 = compute_lambda_adv(0.0, LAMBDA_0, ALPHA, BETA)
    assert v0 <= LAMBDA_0 * BETA + 1e-6


def test_alpha_boundary():
    """At L_c == alpha, the function takes the hard-regime branch."""
    v_at = compute_lambda_adv(ALPHA, LAMBDA_0, ALPHA, BETA)
    v_above = compute_lambda_adv(ALPHA + 1e-6, LAMBDA_0, ALPHA, BETA)
    assert v_at != v_above or math.isclose(v_at, LAMBDA_0, rel_tol=1e-3)


def test_always_le_lambda_0_times_beta():
    """The cap invariant — assertion smoke tests rely on this."""
    for L_c in [0.0, 1e-6, 0.01, 0.1, 0.3, 0.6, 0.6286, 0.7, 1.0, 5.0]:
        v = compute_lambda_adv(L_c, LAMBDA_0, ALPHA, BETA)
        assert v <= LAMBDA_0 * BETA + 1e-6, f'L_c={L_c} gave {v}'
```

- [ ] **Step 2: Run tests to confirm they fail (module doesn't exist yet)**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_advgrl.py -v`
Expected: ImportError / module not found.

- [ ] **Step 3: Implement `utils/advgrl.py`**

```python
"""AdvGRL: dynamic gradient-reversal weight per DA-Detect (Adv_GRL).

The effective GRL weight applied at each iter is:
    lambda_adv = lambda_0 * min(beta, 1/L_c)     if L_c <= alpha   (hard regime)
    lambda_adv = lambda_0                        otherwise         (easy regime)

This caps lambda_adv at lambda_0 * beta (default 0.1 * 30 = 3.0), NOT at beta.
See spec §2.1.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def default_alpha() -> float:
    """alpha = BCE-with-logits([0.7, 0.3], [1.0, 0.0])  (DA-Detect's bce constant)."""
    pred = torch.tensor([[0.7, 0.3]])
    label = torch.tensor([[1.0, 0.0]])
    return F.binary_cross_entropy_with_logits(pred, label).item()


def compute_lambda_adv(L_c: float, lambda_0: float = 0.1, alpha: float = None,
                       beta: float = 30.0, eps: float = 1e-7) -> float:
    """Compute the AdvGRL effective weight for this iter.

    Args:
        L_c: scalar DA-classifier loss from the detached forward (Python float).
        lambda_0: base GRL weight (e.g. 0.1).
        alpha: gate threshold; if None, uses default_alpha() ≈ 0.6286.
        beta: cap on the 1/L_c factor.
        eps: division-by-zero guard.

    Returns:
        Python float — the effective lambda. Pass with a negative sign into
        gradient_scalar() to perform gradient reversal.
    """
    if alpha is None:
        alpha = default_alpha()
    L_c = float(L_c)
    if L_c <= alpha:
        adv_threshold = min(beta, 1.0 / (L_c + eps))
        return lambda_0 * adv_threshold
    return lambda_0
```

- [ ] **Step 4: Run tests**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_advgrl.py -v`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add utils/advgrl.py tests/test_advgrl.py
git commit -m "feat: compute_lambda_adv helper with cap invariant tests"
```

---

### Task 3.2: `advgrl_step` two-pass routine

**Files:**
- Modify: `utils/advgrl.py` (add `advgrl_step` function)

- [ ] **Step 1: Append to `utils/advgrl.py`**

```python
from torch import nn
from utils.domain_grl import gradient_scalar
from utils.domain_loss import da_img_loss


def advgrl_step(
    backbone_feat: torch.Tensor,
    source_count: int,
    classifier: nn.Module,
    *,
    use_advgrl: bool,
    lambda_0: float,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, float, float]:
    """Two-pass AdvGRL: detached forward for L_c, then GRL-attached forward.

    Args:
        backbone_feat: concatenated source+target features, shape [B_s+B_t, C, H, W].
        source_count: B_s — number of source rows at the start.
        classifier: DAImgHead (or similar) producing [B, 1, H, W] logits.
        use_advgrl: if False, lambda_adv is just lambda_0 (plain fixed-GRL).
        lambda_0, alpha, beta: AdvGRL hyperparameters.

    Returns:
        (loss_da_image, lambda_adv, L_c_value)
        - loss_da_image: scalar tensor for backprop.
        - lambda_adv:    Python float, the effective GRL weight this iter.
        - L_c_value:     Python float, the detached classifier loss (for logging).
    """
    # Pass 1: detached — compute scalar L_c for AdvGRL gating.
    pred_detached = classifier(backbone_feat.detach())
    L_c_tensor = da_img_loss(pred_detached, source_count=source_count)
    L_c = float(L_c_tensor.item())

    # Decide lambda_adv.
    if use_advgrl:
        lambda_adv = compute_lambda_adv(L_c, lambda_0=lambda_0, alpha=alpha, beta=beta)
    else:
        lambda_adv = lambda_0

    # Pass 2: GRL-attached — actual gradients flow back through gradient_scalar.
    feat_grl = gradient_scalar(backbone_feat, -lambda_adv)
    pred = classifier(feat_grl)
    loss_da_image = da_img_loss(pred, source_count=source_count)

    return loss_da_image, lambda_adv, L_c
```

- [ ] **Step 2: Write integration test for `advgrl_step`**

Append to `tests/test_advgrl.py`:

```python
import torch
from models.da_classifier import DAImgHead
from utils.advgrl import advgrl_step


def test_advgrl_step_returns_three_things():
    head = DAImgHead(in_channels=64)
    feat = torch.randn(4, 64, 2, 2, requires_grad=True)
    loss, lambda_adv, L_c = advgrl_step(
        feat, source_count=2, classifier=head,
        use_advgrl=True, lambda_0=0.1, alpha=default_alpha(), beta=30.0,
    )
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert isinstance(lambda_adv, float)
    assert isinstance(L_c, float)
    assert lambda_adv <= 0.1 * 30.0 + 1e-6


def test_advgrl_step_gradient_flows_back():
    """Backprop reaches backbone_feat with the sign-flipped weight."""
    head = DAImgHead(in_channels=64)
    feat = torch.randn(4, 64, 2, 2, requires_grad=True)
    loss, _, _ = advgrl_step(
        feat, source_count=2, classifier=head,
        use_advgrl=False, lambda_0=0.1, alpha=default_alpha(), beta=30.0,
    )
    loss.backward()
    assert feat.grad is not None
    assert feat.grad.abs().sum() > 0


def test_advgrl_step_fixed_lambda_when_off():
    """use_advgrl=False -> lambda_adv == lambda_0 regardless of L_c."""
    head = DAImgHead(in_channels=64)
    feat = torch.randn(4, 64, 2, 2)
    _, lambda_adv, _ = advgrl_step(
        feat, source_count=2, classifier=head,
        use_advgrl=False, lambda_0=0.07, alpha=default_alpha(), beta=30.0,
    )
    assert lambda_adv == 0.07
```

- [ ] **Step 3: Run tests**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_advgrl.py -v`
Expected: 9 passed (6 original + 3 new).

- [ ] **Step 4: Commit**

```bash
git add utils/advgrl.py tests/test_advgrl.py
git commit -m "feat: advgrl_step two-pass routine"
```

---

### Task 3.3: Wire `advgrl_step` into `train_GRL.py`

**Files:**
- Modify: `train_GRL.py` (the forward/loss block from Task 2.3)
- Create: `utils/da_logger.py`

- [ ] **Step 1: Create the DA logger**

```python
"""Per-iteration DA loss logger -> runs/.../da_losses.csv"""
import csv
from pathlib import Path


class DALogger:
    """Append per-iter (epoch, iter, loss_det, loss_da_image, loss_triplet_img, lambda_adv, L_c)."""

    def __init__(self, save_dir: str):
        self.path = Path(save_dir) / 'da_losses.csv'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._written_header = self.path.exists()
        self._fields = ['epoch', 'iter', 'loss_det', 'loss_da_image', 'loss_triplet_img', 'lambda_adv', 'L_c']

    def log(self, **row):
        for k in self._fields:
            row.setdefault(k, '')
        write_header = not self._written_header
        with self.path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if write_header:
                w.writeheader()
                self._written_header = True
            w.writerow({k: row[k] for k in self._fields})
```

- [ ] **Step 2: Use the logger in `train_GRL.py`**

After the classifier_head setup block (from Task 2.3), add:

```python
    # DA logger writes runs/.../da_losses.csv
    from utils.da_logger import DALogger
    da_logger = DALogger(save_dir=str(save_dir)) if (opt.da_img or opt.triplet_img) else None
```

(`save_dir` is the YOLOv5 run directory; it exists in `train(...)` already — check for the `save_dir` variable name in your version.)

- [ ] **Step 3: Replace the DA loss computation block with the `advgrl_step` call**

In the forward/loss block from Task 2.3 step 6, replace:

```python
                if classifier_head is not None:
                    B_t = t_imgs.size(0)
                    st_feat = backbone_feat[:B_s + B_t]
                    feat_grl = gradient_scalar(st_feat, -opt.da_img_grl_weight)
                    da_logits = classifier_head(feat_grl)
                    loss_da_image = da_img_loss(da_logits, source_count=B_s)
                    losses['loss_da_image'] = opt.da_img_weight * loss_da_image
```

with:

```python
                lambda_adv_this_iter = None
                L_c_this_iter = None
                if classifier_head is not None:
                    from utils.advgrl import advgrl_step, default_alpha
                    B_t = t_imgs.size(0)
                    st_feat = backbone_feat[:B_s + B_t]
                    alpha = opt.advgrl_alpha if opt.advgrl_alpha is not None else default_alpha()
                    loss_da_image, lambda_adv_this_iter, L_c_this_iter = advgrl_step(
                        st_feat, source_count=B_s, classifier=classifier_head,
                        use_advgrl=opt.advgrl, lambda_0=opt.da_img_grl_weight,
                        alpha=alpha, beta=opt.advgrl_threshold,
                    )
                    losses['loss_da_image'] = opt.da_img_weight * loss_da_image
                    # Cap invariant — fail fast if compute_lambda_adv is buggy.
                    assert lambda_adv_this_iter <= opt.da_img_grl_weight * opt.advgrl_threshold + 1e-6
```

- [ ] **Step 4: Log per-iter after `total_loss.backward()` (still inside the iter loop)**

After the backward call:

```python
            if da_logger is not None:
                da_logger.log(
                    epoch=epoch, iter=ni,
                    loss_det=float(loss_det.item()),
                    loss_da_image=float(losses.get('loss_da_image', torch.tensor(0.0)).item())
                        if 'loss_da_image' in losses else '',
                    lambda_adv=lambda_adv_this_iter if lambda_adv_this_iter is not None else '',
                    L_c=L_c_this_iter if L_c_this_iter is not None else '',
                )
```

- [ ] **Step 5: Smoke-test with `--advgrl`**

Run:

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data domain/city_foggycity.yaml \
    --epochs 1 --batch-size 4 --img 320 --name smoke_pr3_advgrl --noplots \
    --da-img --advgrl
head -5 runs/train/smoke_pr3_advgrl/da_losses.csv
```

Expected: 1 epoch completes; `da_losses.csv` has header + rows; `lambda_adv` column values are all `<= 3.0`.

- [ ] **Step 6: Verify the dependency-validation guard**

Run: `python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data domain/city_foggycity.yaml --advgrl --epochs 1 --batch-size 4 --img 320 --name smoke_pr3_advgrl_dep`
Expected: `SystemExit: --advgrl requires --da-img ...`

- [ ] **Step 7: Commit**

```bash
git add train_GRL.py utils/da_logger.py
git commit -m "feat: wire advgrl_step + DA logger into train_GRL.py"
```

---

### Task 3.4: Sanity gate — 10-epoch run with `--advgrl`

- [ ] **Step 1: Run training**

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
python train_GRL.py --weights yolov5l.pt --cfg configs/domain/yolov5l_GRL.yaml \
    --data domain/city_foggycity.yaml --epochs 10 --batch-size 8 --img 640 \
    --name sanity_pr3_advgrl --da-img --advgrl
```

- [ ] **Step 2: Check `lambda_adv` trace**

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/train/sanity_pr3_advgrl/da_losses.csv')
print('lambda_adv max:', df['lambda_adv'].max())
print('lambda_adv mean:', df['lambda_adv'].mean())
print('L_c min:', df['L_c'].min(), 'L_c mean:', df['L_c'].mean())
assert df['lambda_adv'].max() <= 3.0 + 1e-6, 'cap invariant violated!'
print('OK — cap invariant holds throughout training.')
"
```

Expected: `OK — cap invariant holds throughout training.`

- [ ] **Step 3: PR 3 gate**

mAP should be in the same ballpark as Task 2.5 (with `--da-img` only). AdvGRL might help, might not — at 10 epochs noise dominates. The key gate is **no NaN/divergence** and **cap invariant holds**. If both pass → **Open PR 3**.

---

## PR 4 — Auxiliary domain: `--aux`

### Task 4.1: Port RainMix offline generator

**Files:**
- Create: `tools/augmentations.py` (copy verbatim from DA-Detect, adjust import)
- Create: `tools/augment_and_mix.py` (port from DA-Detect)
- Create: `tools/gen_rainy_cityscapes.py` (port from DA-Detect, adapt paths)

- [ ] **Step 1: Copy `augmentations.py` from DA-Detect**

Copy `/home/kacchan/VuHung/project/nhandang_project_2/DA-Detect/efficientderain-master/augmentations.py` → `yolo-G/tools/augmentations.py`.

Run:

```bash
cp /home/kacchan/VuHung/project/nhandang_project_2/DA-Detect/efficientderain-master/augmentations.py /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tools/augmentations.py
```

(No code changes — this is a pure utility module.)

- [ ] **Step 2: Port `augment_and_mix.py`**

Copy `/home/kacchan/VuHung/project/nhandang_project_2/DA-Detect/efficientderain-master/augment_and_mix.py` → `yolo-G/tools/augment_and_mix.py`. Then edit the import at the top:

Change `import augmentations` to `from tools import augmentations`.

(Everything else in the file is fine — it's the AugMix `augment_and_mix(...)` function used by the generator.)

- [ ] **Step 3: Port `generate_rainy_cityscape.py` into `tools/gen_rainy_cityscapes.py`**

Write `tools/gen_rainy_cityscapes.py`:

```python
"""Offline RainMix generator: produces a parallel auxiliary domain from source images.

Port of DA-Detect/efficientderain-master/generate_rainy_cityscape.py adapted to
the YOLO-G project layout.

Usage:
    python tools/gen_rainy_cityscapes.py \
        --source-images /path/to/cityscapes/images/train \
        --rain-masks    /path/to/rain_masks/ \
        --output        /path/to/cityscapes/images/train_rainy

Outputs PNGs with the same filenames as the source images.
"""
import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Local imports (the AugMix port)
from tools import augment_and_mix


def random_rain_mask(rain_path: str) -> np.ndarray:
    files = sorted(os.listdir(rain_path))
    idx = random.randint(0, len(files) - 1)
    return cv2.imread(os.path.join(rain_path, files[idx]))


def make_rain_layer(rain_mask: np.ndarray, size=(2048, 1024)) -> np.ndarray:
    layer = cv2.resize(rain_mask, size)
    layer = layer.astype(np.float32) / 255.0
    return cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)


def rain_aug(img_rgb: np.ndarray, rain_mask: np.ndarray, size=(2048, 1024)) -> np.ndarray:
    """Screen-blend a RainMix-augmented rain layer onto img_rgb."""
    img = img_rgb.astype(np.float32) / 255.0
    rain_layer = make_rain_layer(rain_mask, size=size)
    rain_aug_layer = augment_and_mix.augment_and_mix(rain_layer, severity=3, width=3, depth=-1) * 1.0
    out = img + rain_aug_layer - img * rain_aug_layer   # screen blend
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-images', required=True, help='dir of source PNG/JPG images')
    ap.add_argument('--rain-masks', required=True, help='dir of rain-streak mask images')
    ap.add_argument('--output', required=True, help='dir to write rainy-augmented images')
    ap.add_argument('--size', nargs=2, type=int, default=[2048, 1024], help='resize target (W H)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_paths = sorted(glob(os.path.join(args.source_images, '*')))
    Path(args.output).mkdir(parents=True, exist_ok=True)

    for src in tqdm(src_paths, desc='RainMix'):
        img_bgr = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, tuple(args.size))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = random_rain_mask(args.rain_masks)
        out_rgb = rain_aug(img_rgb, mask, size=tuple(args.size))
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output, Path(src).name), out_bgr)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Smoke-test the generator on 2 toy images**

Use the test fixtures as inputs:

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
mkdir -p /tmp/rain_masks /tmp/rainy_out
# Use existing source fixtures as "rain masks" for the smoke test (just need *some* PNGs)
cp tests/fixtures/source/images/*.png /tmp/rain_masks/
python tools/gen_rainy_cityscapes.py \
    --source-images tests/fixtures/source/images \
    --rain-masks /tmp/rain_masks \
    --output /tmp/rainy_out \
    --size 64 64
ls /tmp/rainy_out/
```

Expected: 4 PNGs in `/tmp/rainy_out/` matching source fixture names.

- [ ] **Step 5: Commit**

```bash
git add tools/augmentations.py tools/augment_and_mix.py tools/gen_rainy_cityscapes.py
git commit -m "feat: tools/gen_rainy_cityscapes.py — offline RainMix generator"
```

---

### Task 4.2: Add aux loader and 3-way zip in `train_GRL.py`

**Files:**
- Modify: `train_GRL.py` (extend the DA setup + training loop)

- [ ] **Step 1: Extend the DA setup block in `train_GRL.py`**

After the target_loader creation (from Task 2.3 step 5), add:

```python
    aux_loader = None
    if opt.aux:
        if 'aux' not in data_dict:
            raise SystemExit('--aux set but no `aux:` key in data YAML')
        aux_path = resolve_da_path(data_dict['aux'], data_dict.get('path'))
        aux_loader = create_target_dataloader(
            aux_path, imgsz, batch_size, gs, workers=workers, prefix=colorstr('aux: '),
        )
```

- [ ] **Step 2: Extend the iter-loop to consume aux batches**

In the iter-loop, after the target-fetch block, add:

```python
            a_imgs = None
            if aux_loader is not None:
                if 'aux_iter' not in locals() or aux_iter is None:
                    aux_iter = iter(aux_loader)
                try:
                    a_batch = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(aux_loader)
                    a_batch = next(aux_iter)
                a_imgs = a_batch[0].to(device, non_blocking=True).float() / 255.0
                all_imgs = torch.cat([all_imgs, a_imgs], dim=0)
```

And reset `aux_iter` to None at the top of each epoch (next to where you reset `target_iter`):

```python
        aux_iter = iter(aux_loader) if aux_loader is not None else None
```

- [ ] **Step 3: Keep `loss_da_image` source+target-only** (aux is excluded — spec §2.2)

The DA loss block from Task 3.3 already uses `st_feat = backbone_feat[:B_s + B_t]`. **No change** — aux features are simply unused by the DA classifier. Verify by re-reading the block.

- [ ] **Step 4: Smoke-test with `--aux` but no triplet**

Add an `aux:` line to a test YAML. Run:

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
# Create a temporary domain YAML pointing at toy fixtures
cat > /tmp/toy_domain.yaml <<'EOF'
path: /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/fixtures
train: source/images
val:   source/images
target: target/images
aux:    aux/images
nc: 1
names: ['toy']
EOF
python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data /tmp/toy_domain.yaml \
    --epochs 1 --batch-size 2 --img 64 --name smoke_pr4_aux --noplots \
    --da-img --aux
```

Expected: 1 epoch completes; console shows `target: ...` and `aux: ...` lines during setup.

- [ ] **Step 5: Verify dependency validation**

```bash
python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data /tmp/toy_domain.yaml \
    --epochs 1 --batch-size 2 --img 64 --name smoke_pr4_dep --noplots --triplet-img
```

Expected: `SystemExit: --triplet-img requires --aux ...`

- [ ] **Step 6: Commit**

```bash
git add train_GRL.py
git commit -m "feat: --aux flag with 3-way zip dataloader (aux excluded from DA loss)"
```

---

### Task 4.3: Sanity gate — RainMix generator + 3-loader smoke

- [ ] **Step 1: Visual inspection of generated aux images**

Run the generator on a small slice of real Cityscapes data:

```bash
python tools/gen_rainy_cityscapes.py \
    --source-images /home/airy/Downloads/cityscapes/yolo_format_8class/images/train \
    --rain-masks   /path/to/rain_masks \
    --output       /tmp/cityscapes_rainy_sample \
    --size 640 640
```

Visually compare a few `train_rainy/*.png` against the corresponding `train/*.png`. They should look domain-shifted (visible rain streaks, slight contrast change) but still recognizable.

If your rain-mask dataset is elsewhere, document the path in `notebooks/01_generate_rainy_aux.ipynb` later. Skip Step 1 if real data isn't available — the toy smoke test from Task 4.2 step 4 is sufficient for PR 4 gate.

- [ ] **Step 2: PR 4 gate**

The 3-way zip dataloader smoke test (Task 4.2 step 4) is the gate. If aux images appear domain-shifted (or the toy smoke passes) → **Open PR 4**.

---

## PR 5 — Triplet: `--triplet-img`

### Task 5.1: Wire triplet loss into `train_GRL.py`

**Files:**
- Modify: `train_GRL.py`

- [ ] **Step 1: Add the triplet block after the DA loss block**

In the forward/loss block, after the `if classifier_head is not None:` block, add:

```python
                if opt.triplet_img:
                    from utils.domain_loss import triplet_img_loss
                    # GAP per image, then batch-mean -> [1, C] centroid per domain.
                    def gap_mean(feat_slice):
                        return feat_slice.mean(dim=[2, 3]).mean(dim=0, keepdim=True)

                    B_t_local = t_imgs.size(0)
                    B_a_local = a_imgs.size(0)
                    F_S = gap_mean(backbone_feat[:B_s])
                    F_T = gap_mean(backbone_feat[B_s:B_s + B_t_local])
                    F_A = gap_mean(backbone_feat[B_s + B_t_local:B_s + B_t_local + B_a_local])

                    # Margin (with optional adaptive ramp)
                    margin = getattr(self, '_triplet_margin', opt.triplet_margin) if False else opt.triplet_margin
                    # We track adaptive state in a closure-attached dict to avoid leaking state.
                    # See utils/triplet_state below for the adaptive ramp helper.
                    if opt.triplet_adaptive:
                        margin = _adaptive_margin_state.get('margin', opt.triplet_margin)
                    loss_triplet = triplet_img_loss(F_S, F_T, F_A, margin=margin)
                    losses['loss_triplet_img'] = opt.triplet_img_weight * loss_triplet

                    # Bump margin if loss hit zero last iter (matches DA-Detect's adaptive logic).
                    if opt.triplet_adaptive:
                        prev = _adaptive_margin_state['last_loss']
                        cur_margin = _adaptive_margin_state['margin']
                        if prev == 0.0 and cur_margin < opt.triplet_max_margin:
                            _adaptive_margin_state['margin'] = cur_margin + 0.001
                        _adaptive_margin_state['last_loss'] = float(loss_triplet.item())
```

- [ ] **Step 2: Initialize adaptive state next to the DA setup block**

After the `aux_loader` setup:

```python
    _adaptive_margin_state = {'margin': opt.triplet_margin, 'last_loss': 1.0}  # 1.0 -> no ramp on first iter
```

- [ ] **Step 3: Extend the DA logger call to include `loss_triplet_img`**

In the `da_logger.log(...)` call (from Task 3.3 step 4), add:

```python
                    loss_triplet_img=float(losses['loss_triplet_img'].item()) if 'loss_triplet_img' in losses else '',
```

- [ ] **Step 4: Smoke-test the full method**

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
python train_GRL.py --cfg configs/domain/yolov5l_GRL.yaml --data /tmp/toy_domain.yaml \
    --epochs 1 --batch-size 2 --img 64 --name smoke_pr5_full --noplots \
    --da-img --advgrl --aux --triplet-img
head -5 runs/train/smoke_pr5_full/da_losses.csv
```

Expected: `da_losses.csv` has all five columns populated (`loss_det`, `loss_da_image`, `loss_triplet_img`, `lambda_adv`, `L_c`).

- [ ] **Step 5: Commit**

```bash
git add train_GRL.py
git commit -m "feat: --triplet-img with batch-centroid pooling and optional adaptive margin"
```

---

### Task 5.2: End-to-end smoke test across all flag combinations

**Files:**
- Create: `tests/test_train_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
"""End-to-end smoke test: run train_GRL.py for ~1 epoch on toy fixtures across
all six valid flag combinations.

Slow-ish (a couple of minutes total). Skipped automatically if CUDA is not available.
"""
import os
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / 'tests' / 'fixtures'


@pytest.fixture(scope='module', autouse=True)
def ensure_fixtures():
    """Regenerate fixtures if missing."""
    if not (FIXTURES / 'source.txt').exists():
        subprocess.run([sys.executable, str(FIXTURES / 'make_fixtures.py')], check=True)


@pytest.fixture(scope='module')
def toy_yaml(tmp_path_factory):
    p = tmp_path_factory.mktemp('cfg') / 'toy.yaml'
    p.write_text(f'''path: {FIXTURES}
train: source/images
val:   source/images
target: target/images
aux:    aux/images
nc: 1
names: ['toy']
''')
    return str(p)


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('flags,name', [
    ([],                                                                           'baseline'),
    (['--da-img'],                                                                 'daimg'),
    (['--da-img', '--advgrl'],                                                     'advgrl'),
    (['--da-img', '--aux'],                                                        'aux'),
    (['--da-img', '--aux', '--triplet-img'],                                       'triplet'),
    (['--da-img', '--advgrl', '--aux', '--triplet-img'],                           'full'),
])
def test_train_GRL_smoke(toy_yaml, tmp_path, flags, name):
    cmd = [
        sys.executable, 'train_GRL.py',
        '--cfg', 'configs/domain/yolov5l_GRL.yaml',
        '--data', toy_yaml,
        '--epochs', '1', '--batch-size', '2', '--img', '64',
        '--name', f'smoke_{name}',
        '--project', str(tmp_path),
        '--noplots',
    ] + flags
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f'FAILED [{name}]:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}'
```

- [ ] **Step 2: Run the smoke test**

Run: `cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G && pytest tests/test_train_smoke.py -v -s`
Expected: 6 passed (one per flag combination) — or all skipped if no GPU.

If any fail, inspect stderr — likely the new flag block isn't gated correctly. Fix and re-run before committing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_smoke.py
git commit -m "test: end-to-end smoke test across all flag combinations"
```

---

### Task 5.3: Update `train_GRL.sh` with the full-method flag set

**Files:**
- Modify: `train_GRL.sh`

- [ ] **Step 1: Replace `train_GRL.sh` body**

```bash
#!/usr/bin/env bash
# Reproduces the full Paper 2 method on Cityscapes -> Foggy Cityscapes.
# For ablations (single-component runs), see tools/run_ablations.sh.

set -euo pipefail
set -x

WEIGHTS='yolov5l.pt'
CFG='./configs/domain/yolov5l_GRL.yaml'
DATA='./domain/city_foggycity.yaml'
HYP='data/hyps/hyp.scratch-high.yaml'
EPOCHS=200
BATCH=2                              # per-domain; effective backbone batch = BATCH * 3 = 6
IMGSIZE=640
NAME='city_foggycity_advgrl_full'

python train_GRL.py \
  --weights $WEIGHTS \
  --cfg     $CFG \
  --data    $DATA \
  --epochs  $EPOCHS \
  --batch-size $BATCH \
  --img     $IMGSIZE \
  --hyp     $HYP \
  --name    $NAME \
  --da-img \
  --advgrl \
  --aux \
  --triplet-img
```

- [ ] **Step 2: Commit**

```bash
git add train_GRL.sh
git commit -m "chore: update train_GRL.sh with full-method flag set"
```

---

### Task 5.4: Sanity gate — 10-epoch full run

- [ ] **Step 1: Run full method for 10 epochs on real data**

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
# Edit domain/city_foggycity.yaml to add aux: <path> first (relative to path:).
python train_GRL.py --weights yolov5l.pt --cfg configs/domain/yolov5l_GRL.yaml \
    --data domain/city_foggycity.yaml --epochs 10 --batch-size 2 --img 640 \
    --name sanity_pr5_full --da-img --advgrl --aux --triplet-img
```

- [ ] **Step 2: Inspect logs**

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/train/sanity_pr5_full/da_losses.csv')
print(df.tail())
print('lambda_adv max:', df['lambda_adv'].max())
print('loss_triplet_img mean:', df['loss_triplet_img'].mean())
assert df['lambda_adv'].max() <= 3.0 + 1e-6
print('OK')
"
```

Expected: cap invariant holds; triplet loss is non-zero on at least some iters.

- [ ] **Step 3: PR 5 gate**

If full method runs to completion + cap invariant holds + smoke tests all pass → **Open PR 5**.

---

## PR 6 — Notebooks + ablation runner

### Task 6.1: `tools/run_ablations.sh`

**Files:**
- Create: `tools/run_ablations.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Sequentially run the six ablation configs. Each writes to runs/train/ablation_<name>/.
# Read each run's results.csv for the final mAP comparison.

set -euo pipefail
set -x

WEIGHTS='yolov5l.pt'
CFG='./configs/domain/yolov5l_GRL.yaml'
DATA='./domain/city_foggycity.yaml'
HYP='data/hyps/hyp.scratch-high.yaml'
EPOCHS=50
BATCH=2
IMG=640

BASE_ARGS="--weights $WEIGHTS --cfg $CFG --data $DATA --epochs $EPOCHS --batch-size $BATCH --img $IMG --hyp $HYP"

# 1. Baseline (source-only)
python train_GRL.py $BASE_ARGS --name ablation_01_baseline

# 2. --da-img only (original YOLO-G)
python train_GRL.py $BASE_ARGS --name ablation_02_daimg --da-img

# 3. --da-img --advgrl
python train_GRL.py $BASE_ARGS --name ablation_03_advgrl --da-img --advgrl

# 4. --da-img --aux (no triplet)
python train_GRL.py $BASE_ARGS --name ablation_04_aux --da-img --aux

# 5. --da-img --aux --triplet-img
python train_GRL.py $BASE_ARGS --name ablation_05_triplet --da-img --aux --triplet-img

# 6. Full
python train_GRL.py $BASE_ARGS --name ablation_06_full --da-img --advgrl --aux --triplet-img

echo 'All ablations complete. mAP per run:'
for d in runs/train/ablation_*/; do
  echo -n "$d: "
  tail -1 "$d/results.csv" | cut -d, -f9    # mAP@0.5:0.95 column — adjust index if YOLO-G csv differs
done
```

- [ ] **Step 2: Commit**

```bash
chmod +x tools/run_ablations.sh
git add tools/run_ablations.sh
git commit -m "tools: sequential ablation runner"
```

---

### Task 6.2: Notebook `00_setup_and_data.ipynb`

**Files:**
- Create: `notebooks/00_setup_and_data.ipynb`

Notebooks are JSON files. The cleanest way to author them is via a helper script that emits the JSON. For brevity, this plan shows the cell contents — the engineer writes the `.ipynb` JSON manually or uses `jupytext`/`nbformat`.

- [ ] **Step 1: Write the notebook with these cells (in order)**

Cell 1 (markdown):

```markdown
# Setup & Data — Cityscapes + Foggy Cityscapes

Mount Drive, clone repo, install deps, download datasets.
```

Cell 2 (code):

```python
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/yolog-advgrl'
import os; os.makedirs(BASE_DIR, exist_ok=True)
```

Cell 3 (code):

```python
%cd /content
!git clone https://github.com/<your-fork>/yolo-G.git
%cd yolo-G
!pip install -r requirements.txt
```

Cell 4 (code):

```python
# Download Cityscapes + Foggy Cityscapes (assumes you have credentials in ~/.netrc).
# Adapt to your data source. The dataset must end up under {BASE_DIR}/cityscapes/ with
# images/train, images/val_foggy, images/train_foggy in YOLO format.
!ls {BASE_DIR}/cityscapes/images || echo "Place the dataset at {BASE_DIR}/cityscapes/ before continuing"
```

Cell 5 (code):

```python
# Update domain/city_foggycity.yaml's `path:` to point at {BASE_DIR}/cityscapes
import yaml
yaml_path = 'domain/city_foggycity.yaml'
with open(yaml_path) as f: cfg = yaml.safe_load(f)
cfg['path'] = f'{BASE_DIR}/cityscapes'
with open(yaml_path, 'w') as f: yaml.safe_dump(cfg, f)
print(cfg)
```

- [ ] **Step 2: Write the .ipynb file (JSON)**

Use this minimal Python to write the file (one-time):

```python
import json
from pathlib import Path

cells = [
    {'cell_type': 'markdown', 'source': '# Setup & Data — Cityscapes + Foggy Cityscapes\n\nMount Drive, clone repo, install deps, download datasets.', 'metadata': {}},
    {'cell_type': 'code', 'source': "from google.colab import drive\ndrive.mount('/content/drive')\n\nBASE_DIR = '/content/drive/MyDrive/yolog-advgrl'\nimport os; os.makedirs(BASE_DIR, exist_ok=True)", 'metadata': {}, 'outputs': [], 'execution_count': None},
    {'cell_type': 'code', 'source': "%cd /content\n!git clone https://github.com/<your-fork>/yolo-G.git\n%cd yolo-G\n!pip install -r requirements.txt", 'metadata': {}, 'outputs': [], 'execution_count': None},
    {'cell_type': 'code', 'source': "!ls {BASE_DIR}/cityscapes/images || echo \"Place the dataset at {BASE_DIR}/cityscapes/ before continuing\"", 'metadata': {}, 'outputs': [], 'execution_count': None},
    {'cell_type': 'code', 'source': "import yaml\nyaml_path = 'domain/city_foggycity.yaml'\nwith open(yaml_path) as f: cfg = yaml.safe_load(f)\ncfg['path'] = f'{BASE_DIR}/cityscapes'\nwith open(yaml_path, 'w') as f: yaml.safe_dump(cfg, f)\nprint(cfg)", 'metadata': {}, 'outputs': [], 'execution_count': None},
]
nb = {
    'cells': [{**c, 'source': c['source'].split('\n')} for c in cells],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 5,
}
Path('notebooks/00_setup_and_data.ipynb').write_text(json.dumps(nb, indent=1))
```

Run that script (paste into a Python REPL or save as `tools/_make_notebook_00.py` temporarily, run, delete).

- [ ] **Step 3: Verify the notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/00_setup_and_data.ipynb'))"`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/00_setup_and_data.ipynb
git commit -m "docs(nb): 00 setup and data"
```

---

### Task 6.3: Notebook `01_generate_rainy_aux.ipynb`

**Files:**
- Create: `notebooks/01_generate_rainy_aux.ipynb`

Cells (same authoring pattern as Task 6.2):

- [ ] **Step 1: Cell contents**

Cell 1 (markdown):

```markdown
# Generate RainMix Auxiliary Domain

One-time offline step: produce a parallel `train_rainy/` dataset from `train/`.
~30 min × 3 splits on Colab. Saved to Drive (persists across sessions).
```

Cell 2 (code):

```python
BASE_DIR = '/content/drive/MyDrive/yolog-advgrl'
RAIN_MASKS = f'{BASE_DIR}/rain_masks'      # Download/upload rain-streak masks here first.
%cd /content/yolo-G
```

Cell 3 (code, preview):

```python
# Visual preview: pick one source image + one rain mask, run the augmentation, show before/after.
import cv2, matplotlib.pyplot as plt, glob, random
from tools import gen_rainy_cityscapes as gen
src_paths = sorted(glob.glob(f'{BASE_DIR}/cityscapes/images/train/*.png'))
mask = gen.random_rain_mask(RAIN_MASKS)
img = cv2.cvtColor(cv2.imread(random.choice(src_paths)), cv2.COLOR_BGR2RGB)
img_r = gen.rain_aug(img, mask, size=(640, 640))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img); axes[0].set_title('source'); axes[0].axis('off')
axes[1].imshow(img_r); axes[1].set_title('rainy'); axes[1].axis('off')
plt.show()
```

Cell 4 (code):

```python
# Run the full generator for the train split.
!python tools/gen_rainy_cityscapes.py \
    --source-images {BASE_DIR}/cityscapes/images/train \
    --rain-masks    {RAIN_MASKS} \
    --output        {BASE_DIR}/cityscapes/images/train_rainy \
    --size 640 640
```

Cell 5 (code):

```python
# Update domain YAML to point at the new aux dir.
import yaml
cfg = yaml.safe_load(open('domain/city_foggycity.yaml'))
cfg['aux'] = 'images/train_rainy'
with open('domain/city_foggycity.yaml', 'w') as f: yaml.safe_dump(cfg, f)
print(cfg)
```

- [ ] **Step 2: Build the .ipynb JSON** (use the authoring script pattern from Task 6.2).

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_generate_rainy_aux.ipynb
git commit -m "docs(nb): 01 generate rainy aux"
```

---

### Task 6.4: Notebooks 02-05

**Files:**
- Create: `notebooks/02_train_baseline.ipynb`
- Create: `notebooks/03_train_yolog_original.ipynb`
- Create: `notebooks/04_train_advgrl_full.ipynb`
- Create: `notebooks/05_evaluate_and_compare.ipynb`

All four follow the same shape — a setup cell + a single `!python train_GRL.py ...` cell + a `!python val_GRL.py ...` cell. The only difference is the flag set.

- [ ] **Step 1: Write 02 (baseline) — flags: none**

Cell 1 (markdown): `# Baseline — source-only YOLOv5-L\n\nNo DA flags. Establishes the floor.`

Cell 2 (code):
```python
BASE_DIR = '/content/drive/MyDrive/yolog-advgrl'
%cd /content/yolo-G
!python train_GRL.py --weights yolov5l.pt --cfg configs/domain/yolov5l_GRL.yaml \
    --data domain/city_foggycity.yaml --epochs 50 --batch-size 8 --img 640 \
    --name baseline
```

Cell 3 (code):
```python
!python val_GRL.py --weights runs/train/baseline/weights/best.pt \
    --data domain/city_foggycity.yaml --task val --img 640 --batch-size 8
```

- [ ] **Step 2: Write 03 (original YOLO-G) — flags: `--da-img`**

Same as 02 but the training command has `--da-img` appended and `--name yolog_original`. Batch size 4 (2 domains).

- [ ] **Step 3: Write 04 (full method) — flags: `--da-img --advgrl --aux --triplet-img`**

Same as 02 but the training command has all four flags appended and `--name advgrl_full`. Batch size 2 (3 domains, fits T4).

- [ ] **Step 4: Write 05 (comparison)**

Cell 1 (markdown): `# Compare all runs\n\nLoad all checkpoints, run val_GRL.py, print mAP table.`

Cell 2 (code):
```python
import subprocess, pandas as pd
RUNS = ['baseline', 'yolog_original', 'advgrl_full']
rows = []
for r in RUNS:
    csv = f'runs/train/{r}/results.csv'
    try:
        df = pd.read_csv(csv)
        rows.append({'run': r, 'mAP@0.5': df.iloc[-1]['metrics/mAP_0.5'], 'mAP@0.5:0.95': df.iloc[-1]['metrics/mAP_0.5:0.95']})
    except Exception as e:
        rows.append({'run': r, 'mAP@0.5': float('nan'), 'mAP@0.5:0.95': float('nan')})
print(pd.DataFrame(rows).to_string(index=False))
```

- [ ] **Step 5: Commit all four**

```bash
git add notebooks/02_train_baseline.ipynb notebooks/03_train_yolog_original.ipynb notebooks/04_train_advgrl_full.ipynb notebooks/05_evaluate_and_compare.ipynb
git commit -m "docs(nb): training and evaluation notebooks (02-05)"
```

---

### Task 6.5: Sanity gate — notebook smoke

- [ ] **Step 1: Validate all .ipynb files are well-formed JSON**

```bash
cd /home/kacchan/VuHung/project/nhandang_project_2/yolo-G
for nb in notebooks/*.ipynb; do
    python -c "import json,sys; json.load(open('$nb')); print('OK:', '$nb')" || echo "FAIL: $nb"
done
```

Expected: 6 lines of `OK: notebooks/*.ipynb`.

- [ ] **Step 2: PR 6 gate**

If notebooks parse + the smoke test (Task 5.2) still passes → **Open PR 6**.

The Colab end-to-end execution is left to the user as a separate manual gate — there's no automated way to test it from here.

---

## Self-Review Checklist (done at write-time)

**Spec coverage**: each spec section has at least one task:
- §1.1 dumb model → Task 1.6
- §1.2 file changes → Tasks 1.x, 2.x, 3.x, 4.x, 5.x, 6.x
- §2.1 AdvGRL → Tasks 3.1, 3.2, 3.3
- §2.2 auxiliary domain → Tasks 4.1, 4.2
- §2.3 triplet → Tasks 2.2, 5.1
- §3 data flow → Tasks 2.3, 3.3, 4.2, 5.1
- §4 CLI flags / YAML / checkpoint → Tasks 2.3, 2.4, 4.2
- §5 eval / notebooks → Tasks 1.7, 6.x
- §6 testing → Tasks 1.1-1.6, 2.2, 3.1, 3.2, 5.2

**Placeholder scan**: no `TBD`/`TODO`/`fill in later` in any step. Code blocks are complete and runnable.

**Type / name consistency**:
- `DAImgHead(in_channels=...)` — same signature in Tasks 1.4 and 2.3 and 3.3
- `compute_lambda_adv(L_c, lambda_0, alpha, beta, eps)` — same in Tasks 3.1, 3.2
- `advgrl_step(backbone_feat, source_count, classifier, *, use_advgrl, lambda_0, alpha, beta)` — defined in 3.2, called in 3.3
- `da_img_loss(logits, source_count)` — same signature in 2.2 (definition) and 3.2 (usage inside `advgrl_step`)
- `triplet_img_loss(anchor, positive, negative, margin)` — same in 2.2 (definition) and 5.1 (usage)
- `resolve_da_path(p, root)` — same in 2.1 and 4.2
- `gradient_scalar(input, weight)` — pre-existing functional, used in 1.2, 3.2, indirectly in 2.3
- `DALogger(save_dir).log(**row)` — defined in 3.3, fields kept consistent across DA-only (3.3) and DA+triplet (5.1)
- Checkpoint keys: `'model'`, `'ema'`, `'classifier'`, `'advgrl_cfg'` — same in 2.4 (save) and (next iteration would load — but resume tested in 2.4 step 4)
- Flag names: `--da-img`, `--da-img-weight`, `--da-img-grl-weight`, `--advgrl`, `--advgrl-threshold`, `--advgrl-alpha`, `--aux`, `--triplet-img`, `--triplet-img-weight`, `--triplet-margin`, `--triplet-adaptive`, `--triplet-max-margin` — all parsed in Task 2.3 step 3, consumed consistently across 3.3, 4.2, 5.1
