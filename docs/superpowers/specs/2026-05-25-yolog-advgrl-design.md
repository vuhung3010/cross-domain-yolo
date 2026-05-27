# YOLO-G + AdvGRL Design Spec

**Date:** 2026-05-25
**Author:** assistant + user collaboration
**Status:** design — pending user sign-off

## Goal

Port the three improvements from Paper 2 (DA-Detect, *Adversarial Gradient Reversal Layer for Domain-Adaptive Object Detection*, 2210.15176) onto the YOLO-G implementation (Paper 1, YOLOv5-L cross-domain detector) so that **each improvement is independently toggleable** for ablation studies.

The three improvements:
1. **AdvGRL** — dynamic gradient-reversal weight λ_adv that emphasizes hard examples
2. **Auxiliary domain (RainMix)** — synthesized third domain between source and target
3. **Triplet metric regularization** — `L^R_img` pulls source/target features together while pushing source/auxiliary apart

Source-only `train.py` is preserved untouched as the clean baseline. All new code lives behind opt-in flags in a new `train_GRL.py`.

## Non-goals

- Instance-level DA / triplet on ROIs (YOLO is single-stage, no ROI pooling — only `L^R_img` ports)
- Multi-GPU / DDP training (single-GPU only in v1)
- Online RainMix at training time (offline generator only)
- Replacing `train.py` (kept as clean source-only baseline)
- Other auxiliary domains (snow, night, blur) — architecture supports swapping via `aux:` YAML key, but only RainMix generator ships
- Hyperparameter sweeps or new eval metrics

## Success criteria

1. `train_GRL.py` with no DA flags reproduces `train.py` source-only mAP within ±0.5 mAP
2. `train_GRL.py --da-img` reproduces existing YOLO-G mAP within ±0.5 mAP
3. Full method `--da-img --advgrl --aux --triplet-img` runs to 50 epochs without NaN/divergence
4. All unit + integration tests pass
5. All 6 Colab notebooks execute end-to-end on T4

We are **not** promising the full method beats published YOLO-G mAP — that's the user's empirical question to answer.

---

## 1. Architecture & file layout

### 1.1 "Dumb model" principle

The YOLOv5-L model class becomes domain-agnostic. `Model._forward_once(x)` returns `(det_pred, backbone_feat)` where `backbone_feat` is the SPPF output at layer 9 (shape `[B, 1024, H/32, W/32]` for YOLOv5-L). All domain-adaptation logic (GRL, classifier head, AdvGRL two-pass, triplet) lives **outside** the model in `train_GRL.py`. The model itself has no GRL, no classifier head baked in, no awareness of source/target/aux distinction.

**Layer-9 capture mechanism**: `parse_model()` only adds a layer to the save list when a later layer references it via `from:` (see `models/yolo_GRL.py:511`). Removing the `Classifyy` reference from the YAML head means layer 9 would no longer be saved. The fix is **not** to rely on the save list — instead, `_forward_once` is modified to explicitly capture `x` into a local `backbone_feat` variable when `m.i == 9`:

```python
# inside Model._forward_once
backbone_feat = None
for m in self.model:
    if m.f != -1:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
    x = m(x)
    if m.i == 9:                       # SPPF output
        backbone_feat = x
    y.append(x if m.i in self.save else None)
return x, backbone_feat                # x is det_pred (list[Tensor] in training)
```

This is robust to YAML edits (does not depend on later layers referencing layer 9) and keeps the save list driven solely by detection-graph needs.

Rationale for the whole refactor: AdvGRL needs a detached forward of the classifier head to compute the scalar `L_c` *before* deciding `λ_adv`, then a second GRL-attached forward for backprop. Baking the classifier into the YAML graph (current `Classifyy` layer) would require a redundant backbone forward in the two-pass scheme. Decoupling fixes this and also makes the model reusable for eval (no domain head loaded at inference).

### 1.2 File changes

| file | change |
|---|---|
| `models/yolo_GRL.py` | refactor `_forward_once` to return `(det_pred, backbone_feat)`; remove implicit `Classifyy` call |
| `models/common.py` | remove `GradientScalarLayer(-0.1)` from `Classifyy`; make it a pure Conv→ReLU→Conv head (~lines 730-790) |
| `configs/domain/yolov5l_GRL.yaml` | remove `[9, 1, Classifyy, [1]]` from head (no save-list reliance — layer 9 captured explicitly in `_forward_once`) |
| `utils/domain_grl.py` | extend with `compute_lambda_adv(L_c, lambda_0, alpha, beta)` helper |
| `utils/damain_loss.py` | rename to `utils/domain_loss.py`; add `triplet_img_loss(anchor, pos, neg, margin)` |
| `utils/domain_aux.py` | **new** — auxiliary-domain dataloader wrapper |
| `domain/city_foggycity.yaml` | add optional `aux:` and `rain_mask:` keys |
| `train_GRL.py` | **new** — composable DA training loop |
| `val_GRL.py` | 2-line patch to strip `backbone_feat` from forward tuple before NMS |
| `train_GRL.sh` | update with new flag set |
| `tools/gen_rainy_cityscapes.py` | **new** — offline RainMix generator (port of DA-Detect's `generate_rainy_cityscape.py`) |
| `tools/run_ablations.sh` | **new** — sequential ablation runner |
| `notebooks/` | **new** dir, 6 Colab notebooks |
| `tests/` | **new** dir, unit + integration tests |
| `docs/superpowers/specs/` | this design + future planning docs |
| `train.py` | **untouched** — preserved source-only baseline |

---

## 2. Component design

### 2.1 AdvGRL (two-pass)

Mirrors DA-Detect's official implementation (`maskrcnn_benchmark/modeling/da_heads/da_heads.py:DomainAdaptationModule_triplet.DA_Img_component` + `Adv_GRL`).

**Classifier output and label shape**: the existing `Classifyy` head (`models/common.py:769`) outputs dense `[B, 1, H, W]` logits (Conv→ReLU→Conv, no flatten). Labels must match this shape — the existing `utils/damain_loss.py:DA_loss` already uses `zeros_like(feature)` / `ones_like(feature)` for exactly this reason. Domain labels in our two-pass code are therefore dense tensors of the same shape as the classifier output, not 1-D `[B]` vectors.

**Functional GRL**: the existing `utils/domain_grl.py` already exposes `gradient_scalar = _GradientScalarLayer.apply` — a functional form that accepts `(input, weight)` at forward time. We use it directly for dynamic `lambda_adv` instead of constructing a fresh `GradientScalarLayer` Module each iter.

**Pseudocode per training iteration:**

```python
from utils.domain_grl import gradient_scalar  # functional, takes (input, weight) at call time

# Pass 1: detached forward to compute scalar L_c for AdvGRL gating
pred_detached = classifier(backbone_feat.detach())              # [B_st, 1, H, W]
labels = torch.zeros_like(pred_detached)                        # dense, matches pred shape
labels[:B_s] = 1.0                                              # source = 1, target = 0
L_c = F.binary_cross_entropy_with_logits(pred_detached, labels)

# Decide lambda_adv (matches DA-Detect's Adv_GRL exactly):
#   adv_threshold = min(beta, 1 / L_c)
#   lambda_adv    = lambda_0 * adv_threshold
# so the effective cap is lambda_0 * beta = 0.1 * 30 = 3.0.
if use_advgrl:
    if L_c <= alpha:                     # hard example regime
        adv_threshold = min(beta, 1.0 / (L_c.item() + eps))
        lambda_adv = lambda_0 * adv_threshold
    else:                                # easy example regime
        lambda_adv = lambda_0
else:
    lambda_adv = lambda_0                # plain fixed-lambda GRL

# Pass 2: GRL-attached forward for actual backprop
feat_grl = gradient_scalar(backbone_feat, -lambda_adv)          # functional GRL
pred = classifier(feat_grl)
loss_da_image = F.binary_cross_entropy_with_logits(pred, labels)
```

**`compute_lambda_adv` helper** (in `utils/domain_grl.py`):
```python
def compute_lambda_adv(L_c, lambda_0=0.1, alpha=0.6286, beta=30.0, eps=1e-7):
    if L_c <= alpha:
        return lambda_0 * min(beta, 1.0 / (L_c + eps))
    return lambda_0
```

**Constants** (defaults match DA-Detect exactly — see `DA-Detect/maskrcnn_benchmark/config/defaults.py`):
- `lambda_0 = 0.1` (`DA_IMG_advGRL_WEIGHT`)
- `alpha = BCE([0.7, 0.3], [1, 0]) ≈ 0.6286` (computed at startup)
- `beta = 30.0` (`DA_ADV_GRL_THRESHOLD`)
- effective cap on `lambda_adv` = `lambda_0 * beta = 3.0`
- `eps = 1e-7` (L_c=0 guard)

### 2.2 Auxiliary domain (RainMix)

**Offline generation** (`tools/gen_rainy_cityscapes.py`): port of DA-Detect's `efficientderain-master/generate_rainy_cityscape.py`. AugMix-style augmentation on rain-streak masks, then screen-blend onto Cityscapes images:

```python
img_rainy = img + rain_layer - img * rain_layer   # screen blend, [0,1] domain
```

Output: a parallel image dir matching Cityscapes layout, listed in a `train.txt`/`val.txt` pointed to by the YAML's `aux:` key.

**Wrapper** (`utils/domain_aux.py`): a third `DataLoader` paired with source + target via `zip(source_loader, target_loader, aux_loader)`. Drops the shortest cycle.

**Important**: aux features are used **only** for the triplet loss and as filler in the concatenated backbone forward. They are **NOT** fed into the DA image classifier. If aux were labeled as "target-side" for the DA classifier, the DA loss would try to make source and aux indistinguishable while the triplet loss tries to push them apart — direct objective conflict. This matches DA-Detect, which uses only source + positive_target for the DA image loss and reserves negative_target for triplet only (`DA-Detect/maskrcnn_benchmark/modeling/detector/generalized_rcnn.py:90-99`).

### 2.3 Triplet metric regularization

`L^R_img = max(d(F_S, F_T) − d(F_S, F_A) + δ, 0)` where:
- anchor `F_S` = global-average-pooled backbone feat from source batch
- positive `F_T` = same from target batch (we want to pull these together)
- negative `F_A` = same from aux batch (we want to push these apart)
- `d` = L2, `δ` = margin

Implemented as `nn.TripletMarginLoss(margin=delta, p=2)` in `utils/domain_loss.py:triplet_img_loss`.

**Pooling — explicit divergence from DA-Detect**: DA-Detect uses per-sample paired triples (a single dataset yields aligned (S, T, A) triples per index; `data/build.py:32`, `engine/trainer.py:202`, `detector/generalized_rcnn.py:90`). We use independent shuffled loaders + **batch-centroid pooling**: GAP per image, then mean over batch yielding a single `[1, 1024]` representative per domain per iter. The triplet loss is computed on these three centroids.

Rationale for this divergence in v1:
- No need for a paired triplet dataset class (simpler dataloader)
- Robust to short-cycle truncation in `zip(s, t, a)`
- DA-Detect's per-sample triple effectively reduces to one triplet per iter when slicing one feature per domain (`detector/generalized_rcnn.py:90`) — so the *output cardinality* is similar, just the alignment differs
- The cost: we lose semantic pairing (e.g. Foggy Cityscapes is the foggy version of a Cityscapes image; we don't exploit that). This is recorded as a v2 follow-up if results require it.

**Adaptive margin** (opt-in via `--triplet-adaptive`): mirrors DA-Detect's `triplet_img_loss` — if triplet loss hit 0 last iter and margin < max_margin, bump margin by `lr=0.001`. Default off.

---

## 3. Data flow & training loop

### 3.1 Three-dataloader iteration

**Batch-size convention**: `--batch-size B` is **per-domain**. Total images per backbone forward = `B * N_active_domains` (1 for baseline, 2 for `--da-img`, 3 for `--aux`). Effective gradient batch matches this. Pick `B` based on the worst case: 3-domain config on T4 with B=2 is ~6 images of YOLOv5-L at 640×640, which fits in 16GB.

```python
for (s_imgs, s_targets), (t_imgs, _), (a_imgs, _) in zip(src_loader, tgt_loader, aux_loader):
    B_s, B_t, B_a = s_imgs.size(0), t_imgs.size(0), a_imgs.size(0)

    # Single concatenated backbone forward
    all_imgs = torch.cat([s_imgs, t_imgs, a_imgs], dim=0)   # [B_s+B_t+B_a, 3, H, W]
    det_pred, backbone_feat = model(all_imgs)
    # det_pred is a LIST of per-scale tensors in training mode (yolo_GRL.py Detect.forward)
    # backbone_feat is the layer-9 (SPPF) output, shape [B_s+B_t+B_a, 1024, H/32, W/32]

    # Detection loss (source only — only source has labels)
    # Slice each per-scale tensor, not the list itself:
    det_pred_src = [p[:B_s] for p in det_pred]
    loss_det = compute_loss(det_pred_src, s_targets)

    losses = {"loss_det": loss_det}

    # Image-level DA: source + target only (B_s + B_t samples through classifier).
    # Aux features are excluded here — they would conflict with the triplet objective.
    # Labels are dense [B_st, 1, H', W'] matching Classifyy's [B,1,H,W] output —
    # constructed inside advgrl_step via zeros_like(pred) / ones_like(pred) on the
    # detached classifier output (see §2.1 pseudocode).
    if cfg.da_img:
        st_feat = backbone_feat[:B_s + B_t]
        loss_da_image = advgrl_step(st_feat, B_s, classifier_head, advgrl_cfg)
        losses["loss_da_image"] = cfg.da_img_weight * loss_da_image

    # Triplet (uses all three domains; aux only enters via this loss)
    if cfg.triplet_img:
        gap = lambda f: f.mean(dim=[2, 3])                  # [B, 1024]
        F_S = gap(backbone_feat[:B_s]).mean(0, keepdim=True)            # [1, 1024]
        F_T = gap(backbone_feat[B_s:B_s+B_t]).mean(0, keepdim=True)     # [1, 1024]
        F_A = gap(backbone_feat[B_s+B_t:]).mean(0, keepdim=True)        # [1, 1024]
        losses["loss_triplet_img"] = cfg.triplet_img_weight * triplet_img_loss(
            F_S, F_T, F_A, margin=cfg.triplet_margin
        )

    total_loss = sum(losses.values())
    total_loss.backward()
    optimizer.step()
```

### 3.2 Gradient routing

- `loss_det` → updates backbone + neck + detection head (normal path)
- `loss_da_image` → backbone (reversed via GRL with weight `−λ_adv`) + classifier head (normal forward, normal gradient)
- `loss_triplet_img` → backbone only (no domain classifier involvement)

Optimizer is a single instance covering both the detector and the classifier head via separate param groups. EMA tracks the detector only by default (see §4.5 for rationale and the opt-in `--classifier-ema` flag).

### 3.3 Edge cases

- **First iter**: `L_c` may be ~0.69 (random init) → `lambda_adv = min(0.1/0.69, 30) ≈ 0.14`. Healthy.
- **`L_c = 0`**: `eps = 1e-7` guard prevents division by zero. Worst case effective `lambda_adv = lambda_0 * beta = 0.1 * 30 = 3.0` (NOT `beta`). Tests must assert `lambda_adv <= lambda_0 * beta`.
- **Ragged batches**: `zip` drops shortest loader's tail. Acceptable for v1.
- **Single-GPU only**: documented as non-goal.

---

## 4. Hyperparameters & CLI

### 4.1 CLI flags on `train_GRL.py`

All DA flags default to **off** so `train_GRL.py` with no flags = source-only baseline equivalent to `train.py`.

```
# Domain data
--domain-yaml         path/to/city_foggycity.yaml

# Image-level DA
--da-img              store_true                    # enable image-level DANN
--da-img-weight       float, default 1.0
--da-img-grl-weight   float, default 0.1            # lambda_0

# AdvGRL
--advgrl              store_true                    # enable dynamic lambda_adv
--advgrl-threshold    float, default 30.0           # beta
--advgrl-alpha        float, default None           # if None, computed as BCE([0.7,0.3],[1,0]) at startup; pass a float to override

# Auxiliary domain
--aux                 store_true                    # enable 3rd dataloader

# Triplet
--triplet-img         store_true
--triplet-img-weight  float, default 0.1
--triplet-margin      float, default 1.0            # delta
--triplet-adaptive    store_true                    # opt-in adaptive ramp
--triplet-max-margin  float, default 3.0
```

### 4.2 Composability matrix & flag dependencies

Three components, but they are not orthogonal — some flags require others:

| flag | depends on | reason |
|---|---|---|
| `--advgrl` | `--da-img` | AdvGRL modulates the GRL weight on the DA image loss; without DA classifier there is no `L_c` to gate on |
| `--triplet-img` | `--aux` | triplet's negative `F_A` comes from the aux loader |
| `--triplet-adaptive` | `--triplet-img` | adaptive ramp is a refinement of the triplet loss |

`train_GRL.py` validates these at startup: if a flag is set without its dependency, the program exits with a clear error rather than silently no-op'ing. (Auto-enabling dependencies was considered and rejected — explicit beats implicit; the user should know what they're running.)

**Valid configurations:**

| flags | behavior |
|---|---|
| (none) | source-only baseline |
| `--da-img` | original YOLO-G (fixed-λ image-level DANN, S vs T) |
| `--da-img --advgrl` | AdvGRL on (S vs T) |
| `--da-img --aux` | image DANN with S+T through classifier; aux only fills the backbone forward (no DA, no triplet) — sanity config, mostly useful for debugging aux loader |
| `--da-img --aux --triplet-img` | image DANN + triplet on (S, T, A) |
| `--da-img --advgrl --aux --triplet-img` | full Paper 2 method |

### 4.3 Domain YAML schema (`domain/city_foggycity.yaml`)

Backwards-compatible with existing YOLO tooling. `train:` remains the source training set so `train.py`, `val.py`, `val_GRL.py`, and dataset parsing in `utils/dataloaders.py` continue to work unmodified. The `target:` key already exists in the current YAML; `aux:` is new and optional.

```yaml
# Standard YOLO keys (kept — required by val/train infra)
path:   /home/airy/Downloads/cityscapes/yolo_format_8class
train:  images/train          # SOURCE training images (Cityscapes)
val:    images/val_foggy      # validation images (Foggy Cityscapes for headline metric)
nc:     8
names:  ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']

# Domain-adaptation keys
target: images/train_foggy    # TARGET training images (no labels needed)
aux:    images/train_rainy    # AUXILIARY training images (required iff --aux); generated offline

# Used only by tools/gen_rainy_cityscapes.py — not read at train time
rain_mask: /path/to/rain_masks/
```

**Path resolution rule**: `utils/general.py:check_dataset()` only resolves `train`, `val`, `test` against `path:` (see `general.py:416`). It does **not** know about `target:` or `aux:`. `train_GRL.py` therefore performs its own resolution for these keys:

```python
def resolve_da_path(p, root):
    p = Path(p)
    return str(p if p.is_absolute() else Path(root) / p)

target_path = resolve_da_path(data['target'], data['path'])
aux_path    = resolve_da_path(data['aux'],    data['path']) if cfg.aux else None
```

This means YAML authors can write either relative-to-`path` (recommended for portability) or absolute (current `city_foggycity.yaml` style — still supported). Both forms work.

### 4.4 Output artifacts

`runs/train_GRL/<exp>/` contains, in addition to YOLO's standard outputs:
- `da_losses.csv` — per-iter `loss_da_image`, `loss_triplet_img`, `lambda_adv`, `L_c`
- `weights/best.pt`, `last.pt` — checkpoint schema below

### 4.5 Checkpoint schema

The detector and the domain classifier are separate `nn.Module` objects (dumb-model principle, §1.1). They are stored under **separate explicit keys** so `val_GRL.py` and existing eval tooling (`models/experimental.py:attempt_load`) remain unmodified — those tools look for `ckpt['ema']` or `ckpt['model']` and expect a detector.

```python
ckpt = {
    # Detector — keys consumed by existing val/eval tooling, unchanged
    'model':            deepcopy(de_parallel(model)).half(),       # detector only
    'ema':              deepcopy(ema.ema).half(),                  # detector EMA only
    'updates':          ema.updates,
    'optimizer':        optimizer.state_dict(),                    # covers both model + classifier params
    'epoch':            epoch,
    'best_fitness':     best_fitness,
    'opt':              vars(opt),
    'date':             datetime.now().isoformat(),

    # New keys — domain-adaptation state, ignored by eval tools
    'classifier':       classifier_head.state_dict(),              # raw state_dict, not Module
    'classifier_ema':   classifier_ema.ema.state_dict() if classifier_ema else None,
    'advgrl_cfg':       {'lambda_0': ..., 'alpha': ..., 'beta': ...},  # for reproducibility
}
```

**Why this layout:**
- `ckpt['model']` and `ckpt['ema']` keep the existing contract (a detector Module). `val_GRL.py` loads via `attempt_load` and sees exactly what it expects.
- `ckpt['classifier']` is a plain `state_dict()` (not a pickled Module) so loading does not require importing the classifier class — eval tools that don't know about it can safely ignore.
- `ckpt['optimizer']` is a single state covering both detector and classifier params (they share one optimizer with multiple param groups). Resume reconstitutes both.
- `ckpt['advgrl_cfg']` lets `train_GRL.py` warn on resume if hyperparameters changed between runs.

**EMA scope**: the detector EMA mirrors YOLOv5's existing `ModelEMA` and only tracks detector params. A separate `classifier_ema` for the domain classifier is **optional and default-off** — classifier weights move fast and adversarially, and EMA-smoothing them may hurt the dynamic. Add `--classifier-ema` flag if needed later.

---

## 5. Evaluation & notebooks

### 5.1 Evaluation

Detection-only at inference. `val_GRL.py` stays as-is modulo a 2-line patch to strip `backbone_feat` from the model's forward tuple before NMS. Domain classifier is loaded but unused.

**Metrics:**
- mAP@0.5 and mAP@0.5:0.95 on Foggy Cityscapes val (headline)
- mAP on Cityscapes val (sanity)
- Per-class AP

**Command:**
```bash
python val_GRL.py --weights runs/train_GRL/exp/weights/best.pt \
                  --data domain/city_foggycity.yaml \
                  --task val --img 640
```

### 5.2 Ablation runner

`tools/run_ablations.sh`:
```
1. baseline           (no flags)
2. da_img             (--da-img)
3. da_img + advgrl    (--da-img --advgrl)
4. da_img + aux       (--da-img --aux)
5. da_img + triplet   (--da-img --aux --triplet-img)
6. full               (--da-img --advgrl --aux --triplet-img)
```

Sequential. Each run writes to `runs/train_GRL/<name>/`. No result aggregation — user reads `results.csv` per run.

### 5.3 Notebooks (`notebooks/`)

| #   | filename                        | purpose                                                         |
| --- | ------------------------------- | --------------------------------------------------------------- |
| 1   | `00_setup_and_data.ipynb`       | clone repo, mount Drive, download Cityscapes + Foggy Cityscapes |
| 2   | `01_generate_rainy_aux.ipynb`   | run RainMix offline generator (~30 min × 3 splits, one-time)    |
| 3   | `02_train_baseline.ipynb`       | source-only YOLOv5-L (no DA flags)                              |
| 4   | `03_train_yolog_original.ipynb` | `--da-img` only (reproduces Paper 1)                            |
| 5   | `04_train_advgrl_full.ipynb`    | `--da-img --advgrl --aux --triplet-img`                         |
| 6   | `05_evaluate_and_compare.ipynb` | load all checkpoints, run `val_GRL.py`, comparison table        |

Conventions: first cell sets `BASE_DIR` (default `/content/drive/MyDrive/yolog-advgrl/`). Training cells call `!python train_GRL.py` rather than re-implementing the loop. Each training notebook ends with a `val_GRL.py` call.

### 5.4 Sanity protocol

Before full 50-epoch runs:
1. Baseline 10 epochs → loop is healthy, loss decreases
2. `--da-img` 10 epochs → matches existing YOLO-G behavior (verifies dumb-model refactor)
3. Full method 10 epochs → no NaN, `lambda_adv ≤ lambda_0 * β`

---

## 6. Testing & risks

### 6.1 Tests

**Unit (`tests/`, no GPU):**
- `test_advgrl.py` — `compute_lambda_adv` across L_c regimes (0, 0.3, 0.6286, 1.0)
- `test_grl_gradient.py` — gradient sign flipped, magnitude scaled by `−weight`
- `test_triplet_loss.py` — hand-crafted triplets, expected nonzero/zero values
- `test_dumb_model.py` — `_forward_once` returns 2-tuple, backbone_feat shape correct

**Integration smoke (`tests/test_train_smoke.py`, <2 min on GPU):**
- 5 iters on 4-image toy dataset (`tests/fixtures/`)
- All flag combinations sequentially
- Asserts: no NaN/Inf, all loss-dict keys, `lambda_adv ≤ lambda_0 * β`, checkpoint roundtrip

**E2E (manual):** sanity protocol §5.4 + full ablation table.

### 6.2 Risks

| risk | likelihood | mitigation |
|---|---|---|
| Dumb-model refactor breaks existing YOLO-G | medium | sanity step 2 catches this |
| Two-pass adds 30-50% compute overhead | high | accept; classifier is 2 convs, overhead small vs backbone |
| `L_c → 0` makes effective `λ_adv` saturate at `λ_0 * β` = 3.0 | medium | the `min(β, 1/L_c)` cap inside `compute_lambda_adv`; log `lambda_adv` to `da_losses.csv` |
| Triplet collapses (d_pos = d_neg = 0) | medium | notebook 1 preview cells verify aux is domain-shifted; opt-in adaptive ramp (`--triplet-adaptive`) available as second-line defense |
| 3-dataloader memory on Colab T4 (16GB) | medium-high | `--batch-size` is per-domain; effective backbone batch = B × N_domains. For full method (3 domains) on T4, recommend B=2 (total 6 images at 640×640). Document in notebook 4. |
| RainMix offline gen takes ~90 min total | low | one-time; persisted to Drive |
| Multi-GPU unsupported in v1 | low | non-goal |

---

## 7. Open questions

None blocking. All design decisions resolved in collaboration with user:
- λ_0 = 0.1 (matches DA-Detect + existing YOLO-G code) ✓
- α computed at startup from `BCE([0.7,0.3],[1,0])` ✓
- Triplet adaptive default off (opt-in via flag) ✓
- 6 separate notebooks (not merged) ✓
- Ablation runner as shell script (not notebook) ✓

Implementation may surface minor follow-ups (e.g., exact `da_losses.csv` log cadence — per-iter vs per-epoch); those will be resolved during the planning phase.

---

## Appendix A — Reference implementations

- **Paper 1**: `/home/kacchan/VuHung/project/nhandang_project_2/YOLO-G_Improved_YOLO_for_cross-domain_object_detec.pdf`
- **Paper 2**: `/home/kacchan/VuHung/project/nhandang_project_2/2210.15176v1.pdf`
- **DA-Detect official code**: `/home/kacchan/VuHung/project/nhandang_project_2/DA-Detect/`
  - AdvGRL: `maskrcnn_benchmark/modeling/da_heads/da_heads.py:DomainAdaptationModule_triplet.Adv_GRL`
  - Triplet: `maskrcnn_benchmark/modeling/da_heads/loss.py:DALossComputation_Component.triplet_img_loss`
  - RainMix gen: `efficientderain-master/generate_rainy_cityscape.py`
  - Defaults: `maskrcnn_benchmark/config/defaults.py`
- **YOLO-G base**: `/home/kacchan/VuHung/project/nhandang_project_2/yolo-G/`
