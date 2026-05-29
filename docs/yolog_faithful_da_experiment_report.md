# Faithful YOLO-G Domain Adaptation Experiment Report

## Abstract

This report documents the current YOLO-G domain-adaptation study for Cityscapes → Foggy Cityscapes object detection. The work re-implements the original YOLO-G image-level adversarial branch in a toggleable training pipeline, then evaluates AdvGRL, RainMix auxiliary data, and image-level triplet loss as independent extensions. The strongest peak result so far comes from faithful YOLO-G with AdvGRL using `alpha=0.75`, reaching best mAP@0.5 = 0.44556 and best mAP@0.5:0.95 = 0.27256. The most stable final-epoch result comes from faithful YOLO-G with default AdvGRL, reaching final mAP@0.5 = 0.42712 and final mAP@0.5:0.95 = 0.26885. The current faithful full setting with auxiliary RainMix and triplet loss is stable but substantially worse, suggesting the triplet/auxiliary objective should not be scaled up without further ablation.

---

## 1. Research question

The project asks whether DA-Detect-style components can improve YOLO-G cross-domain detection when each component is ported into YOLO-G as an independently toggleable training option.

Main question:

> Can faithful YOLO-G image-level domain adaptation be improved by dynamic AdvGRL strength, an auxiliary RainMix domain, and image-level triplet loss on Cityscapes → Foggy Cityscapes?

Secondary questions:

1. Does the refactored faithful DA path reproduce original YOLO-G behavior?
2. Does AdvGRL improve over fixed GRL weight `0.1`?
3. Does increasing AdvGRL activation through `alpha=0.75` help?
4. Does adding RainMix auxiliary images and triplet loss improve or hurt the faithful setup?
5. Which setting should be used as the current baseline before running more experiments?

---

## 2. Methodology

### 2.1 Base detector

The detector is a YOLOv5-L / YOLO-G fork trained for object detection. The DA-aware entry point is:

```bash
train_GRL.py
```

The matching validator is:

```bash
val_GRL.py
```

The DA-aware model follows a dumb-model pattern:

```python
return det_pred, backbone_feat
```

`det_pred` is used by the standard YOLO detection loss. `backbone_feat` is the SPPF-level backbone feature used by domain adaptation losses outside the model.

This keeps the model simple and moves all DA logic into the training script, so each component can be enabled or disabled by CLI flags.

### 2.2 Faithful YOLO-G image-level domain adaptation

The faithful mode is enabled by:

```bash
--da-img --da-img-faithful
```

This mode matches the original YOLO-G image-level adversarial design more closely than the newer concatenated source/target pipeline.

Faithful behavior:

1. Run source images through the detector.
2. Run target images through the detector in a separate forward pass.
3. Extract source and target SPPF features.
4. Apply gradient reversal to both feature tensors.
5. Feed reversed features into the image-level domain classifier.
6. Use source label `0` and target label `1`.
7. Compute balanced BCE loss:

```text
L_da = 0.5 * BCE(D(GRL(F_source)), 0) + 0.5 * BCE(D(GRL(F_target)), 1)
```

The faithful loss helper is:

```python
da_img_faithful_loss_pair(source_logits, target_logits)
```

The faithful branch differs from the newer `--da-img` path:

| Mode | Forward style | DA labels | DA loss |
|---|---|---|---|
| New `--da-img` | concatenate `[source, target, aux]` | source=1, target=0 | one BCE over concatenated logits |
| Faithful `--da-img --da-img-faithful` | separate source and target forwards | source=0, target=1 | `0.5 * source_BCE + 0.5 * target_BCE` |

### 2.3 Domain classifier head

The image-level classifier head is:

```text
Conv2d(1024 -> 512, kernel=1)
ReLU
Conv2d(512 -> 1, kernel=1)
```

Input shape is the SPPF feature map:

```text
[B, 1024, H, W]
```

Output shape is dense domain logits:

```text
[B, 1, H, W]
```

The classifier predicts domain at every spatial location. BCE-with-logits is applied densely after flattening spatial dimensions.

### 2.4 Fixed GRL strength

Original YOLO-G uses fixed gradient reversal strength:

```text
gamma = -0.1
```

In the refactored faithful path this is represented as:

```text
lambda_0 = 0.1
GRL scale = -lambda_0
```

So fixed faithful DA uses:

```text
GRL scale = -0.1
```

### 2.5 AdvGRL dynamic strength

AdvGRL is enabled by:

```bash
--advgrl
```

AdvGRL replaces the fixed `0.1` GRL magnitude with a dynamic value. The faithful implementation computes detached DA loss `L_c`, then uses it to select adversarial strength:

```text
if L_c <= alpha:
    lambda_adv = lambda_0 * min(beta, 1 / (L_c + eps))
else:
    lambda_adv = lambda_0
```

Defaults:

```text
lambda_0 = 0.1
beta = 30.0
eps = 1e-7
alpha = BCE([0.7, 0.3], [1.0, 0.0]) ≈ 0.6286
```

Effective GRL scale:

```text
GRL scale = -lambda_adv
```

Interpretation:

- If the domain classifier is weak or uncertain, keep base strength `0.1`.
- If the domain classifier becomes confident enough (`L_c <= alpha`), increase adversarial pressure.
- `beta` caps the multiplier so strength cannot grow without bound.

The `alpha=0.75` experiment raises the activation threshold, making AdvGRL active much more often.

### 2.6 RainMix auxiliary domain and triplet loss

Auxiliary RainMix data is enabled by:

```bash
--aux
```

Image-level triplet loss is enabled by:

```bash
--triplet-img
```

For faithful mode, the selected implementation uses three separate forward passes:

1. Source / clear Cityscapes.
2. Target / Foggy Cityscapes.
3. Auxiliary / rainy RainMix.

Triplet semantics:

```text
anchor   = source / clear Cityscapes
positive = target / foggy Cityscapes
negative = aux / rainy RainMix
```

Feature pooling:

```text
F_source = GAP(backbone_feat_source)
F_target = GAP(backbone_feat_target)
F_aux    = GAP(backbone_feat_aux)
```

Triplet objective:

```text
L_triplet = TripletMarginLoss(F_source, F_target, F_aux, margin=1.0)
```

Total loss when all components are active:

```text
L_total = L_det + da_img_weight * L_da + triplet_img_weight * L_triplet
```

In the current full experiment:

```text
da_img_weight = 1.0
triplet_img_weight = 0.03
triplet_margin = 1.0
```

---

## 3. Experimental setup

### 3.1 Dataset

Domain file:

```text
domain/city_foggycity.yaml
```

Dataset roles:

| Split / field | Domain | Role |
|---|---|---|
| `train` | clear Cityscapes | labeled source training data |
| `target` | Foggy Cityscapes train images | unlabeled target DA data |
| `val` | Foggy Cityscapes validation images | evaluation domain |
| `aux` | rainy RainMix images | auxiliary negative domain for triplet loss |

The target domain for evaluation is Foggy Cityscapes. The auxiliary rainy domain is not the validation target; it is used only as an auxiliary negative domain when triplet loss is enabled.

### 3.2 Training configuration

Main 50-epoch experimental runs use:

```text
weights: yolo-G-modal/yolov5l.pt
cfg: yolo-G-modal/configs/domain/yolov5l_GRL.yaml
data: yolo-G-modal/domain/city_foggycity.yaml
hyp: yolo-G-modal/hyps/hyp.scratch-high.yaml
epochs: 50
batch_size: 8
imgsz: 640
optimizer: SGD
cache: ram
da_img_weight: 1.0
da_img_grl_weight: 0.1
da_img_warmup: off
advgrl_threshold: 30.0
```

The source-only sanity baseline is a 10-epoch reference run, so it is useful as a sanity anchor but not a fair 50-epoch ablation point.

### 3.3 Metrics

Primary metrics:

- mAP@0.5
- mAP@0.5:0.95
- precision
- recall

Training metrics are read from:

```text
runs/train/<run_name>/results.csv
```

DA diagnostics are read from:

```text
runs/train/<run_name>/da_losses.csv
```

DA diagnostics include:

- `loss_da_image`
- `loss_triplet_img`
- `lambda_adv`
- `L_c`
- `da_scale`

---

## 4. Experiments

### 4.1 Source-only sanity baseline

Run:

```text
sanity_pr1_baseline
```

Purpose:

- Establish smoke/sanity reference after dumb-model refactor.
- Confirm source-only YOLO-G path can train and validate on Foggy Cityscapes.

Important caveat:

- This run is 10 epochs, not 50 epochs.
- It should not be used as a final comparison against 50-epoch DA runs.

### 4.2 Faithful YOLO-G image-level DA

Run:

```text
city_foggycity_faithful
```

Flags:

```bash
--da-img --da-img-faithful
```

Purpose:

- Establish 50-epoch faithful YOLO-G DA baseline.
- Use fixed original GRL magnitude `0.1`.
- No AdvGRL, no auxiliary domain, no triplet.

### 4.3 Faithful YOLO-G + default AdvGRL

Run:

```text
city_foggycity_advgrl_faithful
```

Flags:

```bash
--da-img --da-img-faithful --advgrl
```

Purpose:

- Test dynamic AdvGRL using default `alpha ≈ 0.6286`.
- Compare final stability against fixed faithful DA.

### 4.4 Faithful YOLO-G + AdvGRL with alpha 0.75

Run:

```text
city_foggycity_advgrl_faithful_0.75alpha
```

Flags:

```bash
--da-img --da-img-faithful --advgrl --advgrl-alpha 0.75
```

Purpose:

- Increase AdvGRL activation frequency.
- Test whether more active adversarial pressure improves peak target-domain mAP.

### 4.5 Faithful full setting with auxiliary RainMix and triplet loss

Run:

```text
city_foggycity_faithful_full
```

Flags:

```bash
--da-img --da-img-faithful --advgrl --aux --triplet-img --triplet-img-weight 0.03
```

Purpose:

- Test full faithful combination: image-level DA, AdvGRL, RainMix aux, and triplet loss.
- Use lower triplet weight `0.03` instead of default `0.1` to reduce instability risk.

---

## 5. Results

### 5.1 Main result table

| Run | Main flags | Final mAP@0.5 | Final mAP@0.5:0.95 | Best mAP@0.5 | Best mAP@0.5:0.95 | Notes |
|---|---|---:|---:|---:|---:|---|
| `sanity_pr1_baseline` | source-only, 10 epochs | 0.37864 | 0.22388 | 0.37864 @ ep9 | 0.22388 @ ep9 | sanity only; not 50-epoch comparable |
| `city_foggycity_faithful` | faithful DA | 0.41195 | 0.25884 | 0.43865 @ ep42 | 0.26948 @ ep41 | stable fixed-GRL baseline |
| `city_foggycity_advgrl_faithful` | faithful DA + default AdvGRL | **0.42712** | **0.26885** | 0.43085 @ ep44 | 0.26885 @ ep49 | best final metrics |
| `city_foggycity_advgrl_faithful_0.75alpha` | faithful DA + AdvGRL alpha=0.75 | 0.39976 | 0.24735 | **0.44556 @ ep44** | **0.27256 @ ep36** | best peak metrics, worse final |
| `city_foggycity_faithful_full` | faithful DA + AdvGRL + aux + triplet | 0.32715 | 0.20160 | 0.33893 @ ep40 | 0.20620 @ ep40 | stable but much worse |

### 5.2 Final metric comparison vs faithful DA

Using `city_foggycity_faithful` as the 50-epoch faithful baseline:

| Run | Δ final mAP@0.5 | Δ final mAP@0.5:0.95 | Δ best mAP@0.5 | Δ best mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| default AdvGRL | +0.01517 | +0.01001 | -0.00780 | -0.00063 |
| AdvGRL alpha=0.75 | -0.01219 | -0.01149 | +0.00691 | +0.00308 |
| full aux+triplet | -0.08480 | -0.05724 | -0.09972 | -0.06328 |

### 5.3 DA diagnostic table

| Run | DA NaNs | Mean DA loss | Max DA loss | Mean lambda | Max lambda | Lambda active % | Mean L_c | Triplet mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `city_foggycity_faithful` | 0 | 0.74673 | 1.98867 | 0.10000 | 0.10000 | 0.000% | 0.74673 | n/a |
| `city_foggycity_advgrl_faithful` | 0 | 0.74882 | 4.26583 | 0.10009 | 0.17051 | 0.135% | 0.74882 | n/a |
| `city_foggycity_advgrl_faithful_0.75alpha` | 0 | 0.74179 | 2.00602 | 0.12875 | 0.17637 | 71.132% | 0.74179 | n/a |
| `city_foggycity_faithful_full` | 0 | 0.74372 | 1.79537 | 0.10003 | 0.16627 | 0.043% | 0.74372 | 0.04465 |

---

## 6. Discussion

### 6.1 Faithful DA is a valid 50-epoch baseline

`city_foggycity_faithful` trains stably for 50 epochs with no DA NaNs. It reaches:

```text
final mAP@0.5      = 0.41195
best mAP@0.5       = 0.43865
final mAP@0.5:0.95 = 0.25884
best mAP@0.5:0.95  = 0.26948
```

This is the current faithful YOLO-G baseline for ablation comparisons.

### 6.2 Default AdvGRL improves final stability

Default AdvGRL gives the best final-epoch result:

```text
final mAP@0.5      = 0.42712
final mAP@0.5:0.95 = 0.26885
```

It improves final mAP over faithful DA by:

```text
+0.01517 mAP@0.5
+0.01001 mAP@0.5:0.95
```

But default AdvGRL barely activates:

```text
lambda active = 0.135%
mean lambda   = 0.10009
```

So this result is best interpreted as a small, stable improvement rather than a strong dynamic-GRL effect.

### 6.3 Alpha 0.75 makes AdvGRL matter, but training drifts late

With `alpha=0.75`, AdvGRL is active for most iterations:

```text
lambda active = 71.132%
mean lambda   = 0.12875
max lambda    = 0.17637
```

This produces the best peak result:

```text
best mAP@0.5      = 0.44556 @ epoch 44
best mAP@0.5:0.95 = 0.27256 @ epoch 36
```

But final metrics drop:

```text
final mAP@0.5      = 0.39976
final mAP@0.5:0.95 = 0.24735
```

Interpretation:

- Raising alpha successfully activates AdvGRL.
- More adversarial pressure can improve peak target performance.
- Late training becomes less stable or over-regularized.
- For model selection, `best.pt` matters more than `last.pt` for this run.

### 6.4 Current aux+triplet setting hurts strongly

The full faithful run is stable but much worse:

```text
best mAP@0.5       = 0.33893
final mAP@0.5      = 0.32715
best mAP@0.5:0.95  = 0.20620
final mAP@0.5:0.95 = 0.20160
```

Compared with faithful DA:

```text
best mAP@0.5       -0.09972
best mAP@0.5:0.95  -0.06328
final mAP@0.5      -0.08480
final mAP@0.5:0.95 -0.05724
```

The run has no DA NaNs:

```text
DA NaNs = 0
mean DA loss = 0.74372
max DA loss  = 1.79537
```

Triplet loss is present:

```text
triplet rows = 18550
triplet mean = 0.04465
triplet max  = 2.89504
```

So the failure is not a numerical explosion. It is more likely objective mismatch: the current triplet/RainMix signal pulls features in a way that hurts foggy-domain detection.

### 6.5 Full run did not really test strong AdvGRL

The full run uses default alpha, so AdvGRL barely activates:

```text
lambda active = 0.043%
mean lambda   = 0.10003
```

Therefore, the drop in `city_foggycity_faithful_full` should be attributed mainly to `--aux --triplet-img`, not to AdvGRL.

---

## 7. Conclusion

Current evidence supports three conclusions:

1. Faithful YOLO-G image-level DA is stable and should remain the main 50-epoch baseline.
2. AdvGRL is promising, but its best setting depends on whether the selection criterion is final epoch or best checkpoint.
3. The current RainMix + triplet formulation is not yet useful and should not be scaled up without isolating the failure mode.

Best current checkpoints:

| Use case | Recommended run | Checkpoint |
|---|---|---|
| Best final stability | `city_foggycity_advgrl_faithful` | `weights/last.pt` or `weights/best.pt` |
| Best peak mAP | `city_foggycity_advgrl_faithful_0.75alpha` | `weights/best.pt` |
| Faithful baseline | `city_foggycity_faithful` | `weights/best.pt` |

Primary recommendation:

> Treat `city_foggycity_advgrl_faithful_0.75alpha/weights/best.pt` as the current best peak model, but treat default faithful AdvGRL as the more stable final-epoch configuration.

---

## 8. Next experiments

Do not continue with full aux+triplet at `triplet_img_weight=0.03` as-is. First isolate the source of degradation.

### 8.1 Aux-only control

Purpose:

- Check whether enabling aux without triplet changes training.
- Expected result should be close to faithful DA, because aux should not affect loss unless triplet is enabled.

Command:

```bash
python train_GRL.py \
  --weights yolo-G-modal/yolov5l.pt \
  --cfg yolo-G-modal/configs/domain/yolov5l_GRL.yaml \
  --data yolo-G-modal/domain/city_foggycity.yaml \
  --hyp yolo-G-modal/hyps/hyp.scratch-high.yaml \
  --epochs 50 \
  --batch-size 8 \
  --img 640 \
  --cache ram \
  --project /content/yolog-runs \
  --name city_foggycity_faithful_aux \
  --da-img \
  --da-img-faithful \
  --aux
```

### 8.2 Lower triplet weight

Purpose:

- Test whether triplet loss can help at smaller magnitude.
- Current `0.03` hurts; try `0.01` before abandoning triplet.

Command:

```bash
python train_GRL.py \
  --weights yolo-G-modal/yolov5l.pt \
  --cfg yolo-G-modal/configs/domain/yolov5l_GRL.yaml \
  --data yolo-G-modal/domain/city_foggycity.yaml \
  --hyp yolo-G-modal/hyps/hyp.scratch-high.yaml \
  --epochs 50 \
  --batch-size 8 \
  --img 640 \
  --cache ram \
  --project /content/yolog-runs \
  --name city_foggycity_faithful_triplet_w001 \
  --da-img \
  --da-img-faithful \
  --aux \
  --triplet-img \
  --triplet-img-weight 0.01
```

### 8.3 AdvGRL alpha sweep

Purpose:

- Find middle ground between default alpha and `0.75`.
- Default is stable but barely active.
- `0.75` is active and gives best peak, but final epoch drops.

Suggested values:

```text
alpha = 0.65
alpha = 0.70
alpha = 0.75
```

Selection criterion:

- Use `best.pt` if optimizing peak target performance.
- Use final epoch metrics if optimizing training stability.

### 8.4 Possible future code change: triplet warmup

If lower triplet weight still hurts, add a separate triplet warmup instead of turning triplet on from iteration 0.

Possible future flag:

```bash
--triplet-warmup {off,gate,ramp}
```

Rationale:

- DA warmup and triplet warmup are different mechanisms.
- Faithful DA currently blocks `--da-img-warmup gate/ramp` to preserve original YOLO-G behavior.
- Triplet warmup could be added independently without changing faithful DA semantics.

---

## 9. Reproducibility notes

### 9.1 Validate best alpha=0.75 checkpoint

```bash
python val_GRL.py \
  --weights /content/yolog-runs/city_foggycity_advgrl_faithful_0.75alpha/weights/best.pt \
  --data yolo-G-modal/domain/city_foggycity.yaml \
  --img 640 \
  --batch-size 8 \
  --project /content/yolog-runs/val \
  --name val_advgrl_faithful_alpha075_best
```

### 9.2 Important artifact paths

| Artifact | Path pattern |
|---|---|
| Training metrics | `runs/train/<run_name>/results.csv` |
| DA diagnostics | `runs/train/<run_name>/da_losses.csv` |
| Training options | `runs/train/<run_name>/opt.yaml` |
| Checkpoints | `runs/train/<run_name>/weights/{best.pt,last.pt}` |
| Validation outputs | `runs/val/<name>/` |

### 9.3 Current run folders used in this report

```text
runs/train/sanity_pr1_baseline
runs/train/city_foggycity_faithful
runs/train/city_foggycity_advgrl_faithful
runs/train/city_foggycity_advgrl_faithful_0.75alpha
runs/train/city_foggycity_faithful_full
```

---

## 10. Limitations

1. The source-only baseline listed here is a 10-epoch sanity run, not a full 50-epoch source-only baseline.
2. Current tables are single-seed results, so small differences should not be over-interpreted.
3. The full aux+triplet run only tests one triplet weight, `0.03`.
4. The alpha=0.75 run improves best checkpoint metrics but not final epoch metrics.
5. The current report does not include a fair full grid of new-pipeline `--da-img` vs faithful `--da-img-faithful` results.

---

## 11. Practical decision before next run

Use this decision rule:

1. For paper-style best-model reporting, evaluate and report:

```text
city_foggycity_advgrl_faithful_0.75alpha/weights/best.pt
```

2. For stable baseline comparison, keep:

```text
city_foggycity_advgrl_faithful
```

3. For faithful original YOLO-G baseline, keep:

```text
city_foggycity_faithful
```

4. For aux/triplet, do not claim improvement yet. Run aux-only and lower triplet-weight controls first.
