# Foreground-Gated Domain Adaptation Design

## Summary

This spec adds foreground/objectness-gated image-level domain adaptation for faithful YOLO-G neck-all training. The feature weights spatial DA BCE by detached YOLO Detect objectness gates so adversarial pressure emphasizes predicted object regions while retaining a configurable background floor.

## Goals

- Add `--da-img-obj-gate` as an opt-in ablation for faithful `neck-all` DA.
- Preserve all existing defaults and non-gated training behavior.
- Keep the model dumb: feature extraction only, no DA heads or losses in `models/yolo_GRL.py`.
- Match the report semantics used by `city_foggycity_advgrl_faithful_neck_all_alpha075_objgate_50ep`.

## Non-Goals

- Do not add class-level DA, instance-level DA, or label-raster foreground masks.
- Do not make objectness gating the default.
- Do not support objectness gating for SPPF or `neck-p4` in this spec.
- Do not change source/target domain labels.
- Do not add gate diagnostics to `da_losses.csv`; the report notes those diagnostics are absent.

## CLI

Add:

```text
--da-img-obj-gate
--da-img-obj-gate-floor 0.05
```

Validation:

- `--da-img-obj-gate` requires `--da-img`.
- `--da-img-obj-gate` requires `--da-img-faithful`.
- `--da-img-obj-gate` requires `--da-feat-layers neck-all`.
- `--da-img-obj-gate-floor` must be non-negative.

## Model Contract

`models/yolo_GRL.py` continues to keep DA logic out of the model. Its forward path returns detection predictions plus a feature dictionary:

```python
return det_pred, {
    "sppf": sppf,
    "neck_p3": neck_p3,
    "neck_p4": neck_p4,
    "neck_p5": neck_p5,
}
```

Feature taps:

- `sppf`: layer 9
- `neck_p3`: layer 17
- `neck_p4`: layer 20
- `neck_p5`: layer 23

This preserves the dumb-model pattern: the model exposes tensors; `train_GRL.py` owns DA heads, GRL, losses, target forwards, and checkpoint classifier state.

## DA Feature Selection

Add `--da-feat-layers` with supported values:

- `sppf`: `["sppf"]`
- `neck-p4`: `["neck_p4"]`
- `neck-all`: `["neck_p3", "neck_p4", "neck_p5"]`

Default is `sppf` to preserve existing behavior.

For selected feature maps, construct one `DAImgHead` per feature map. Compute faithful source/target DA loss per feature map and average across selected maps.

## Objectness Gate Construction

Objectness gates are built from YOLO Detect outputs for the matching source or target forward.

For each selected neck scale:

1. Take Detect objectness logits for that scale.
2. Apply `sigmoid`.
3. Take `max` over anchors.
4. Keep shape `[B, 1, H, W]`.
5. Clamp with minimum `--da-img-obj-gate-floor`.
6. Detach the gate.

The gate for each scale must match the spatial size of that scale's DA logits. If the Detect output layout differs from the feature layout, reshape/permutation happens in a helper with tests.

## Gated DA Loss

For one feature scale:

```text
source_loss = sum(BCE(source_logits, 0) * source_gate) / clamp(sum(source_gate), min=1)
target_loss = sum(BCE(target_logits, 1) * target_gate) / clamp(sum(target_gate), min=1)
scale_loss = 0.5 * source_loss + 0.5 * target_loss
```

For `neck-all`, average `scale_loss` over P3, P4, and P5.

`--advgrl` uses the same gated objective for detached `L_c` and for the GRL-attached loss. The only difference is that the `L_c` pass receives detached feature tensors and the GRL pass receives `gradient_scalar(feature, -lambda_adv)`.

## Training Flow

With:

```text
--da-img --da-img-faithful --advgrl --advgrl-alpha 0.75 --da-feat-layers neck-all --da-img-obj-gate
```

The loop performs:

1. Source forward for detection loss and source features.
2. Target forward for target features and target objectness.
3. Source and target gate creation for P3/P4/P5 from their respective Detect outputs.
4. Detached DA-head pass to compute gated `L_c` and dynamic `lambda_adv` when `--advgrl` is active.
5. GRL-attached gated DA loss using the same gates.
6. DA warmup scale, if enabled, multiplies the final DA image loss as before.
7. Aux/triplet behavior remains unchanged and does not participate in objectness gates.

## Checkpointing

Classifier checkpoint state remains under the existing separate `classifier` key. For multi-scale DA heads, save and load the per-feature DA head state through the same classifier container rather than adding DA modules to the detector model state dict.

## Tests

Add or update tests for:

- `models/yolo_GRL.py` forward returns a feature dictionary with `sppf`, `neck_p3`, `neck_p4`, `neck_p5`.
- `--da-feat-layers` maps to the expected feature names.
- invalid `--da-img-obj-gate` combos fail validation.
- objectness gate helper applies sigmoid, max over anchors, floor clamp, shape conversion, device/dtype preservation, and detach.
- gated BCE normalization divides by gate sum with clamp minimum `1`.
- neck-all gated DA averages P3/P4/P5 scale losses.
- AdvGRL `L_c` and final GRL-attached loss use the same gated objective.
- a short smoke run covers `--da-img --da-img-faithful --advgrl --da-feat-layers neck-all --da-img-obj-gate`.

## Reproducibility Command

```bash
python train_GRL.py \
  --weights yolov5l.pt \
  --cfg configs/domain/yolov5l_GRL.yaml \
  --data domain/city_foggycity.yaml \
  --hyp hyps/hyp.scratch-high.yaml \
  --epochs 50 \
  --batch-size 8 \
  --img 640 \
  --cache ram \
  --name city_foggycity_advgrl_faithful_neck_all_alpha075_objgate_50ep \
  --da-img \
  --da-img-faithful \
  --advgrl \
  --advgrl-alpha 0.75 \
  --da-feat-layers neck-all \
  --da-img-obj-gate
```
