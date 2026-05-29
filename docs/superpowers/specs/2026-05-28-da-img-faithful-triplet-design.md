# DA-Img Faithful Aux/Triplet Design

## Goal

Allow faithful YOLO-G image-level DA mode to run with aux RainMix and image triplet loss while preserving faithful source/target behavior.

Allowed combos:
- `--da-img --da-img-faithful --aux`
- `--da-img --da-img-faithful --aux --triplet-img`
- `--da-img --da-img-faithful --advgrl --aux --triplet-img`

Still blocked:
- `--da-img-faithful --da-img-warmup gate`
- `--da-img-faithful --da-img-warmup ramp`

## Data Flow

Faithful mode keeps separate source and target forwards:

```python
det_pred, backbone_feat = model(imgs)
_, target_backbone_feat = model(t_imgs)
```

When `--aux` is enabled, faithful mode adds a third forward:

```python
_, aux_backbone_feat = model(a_imgs)
```

This avoids pretending that faithful mode is the concatenated new pipeline. It keeps old YOLO-G source/target DA behavior intact and only adds aux features for triplet.

## Losses

Faithful DA classifier loss remains source+target only:

```python
loss_da_image = da_img_faithful_loss_pair(source_logits, target_logits)
```

Labels remain:
- source = `0`
- target/foggy = `1`

Aux/rainy is excluded from DA classifier BCE.

Triplet in faithful mode uses separate feature tensors:

```python
F_S = _gap_mean(backbone_feat)
F_T = _gap_mean(target_backbone_feat)
F_A = _gap_mean(aux_backbone_feat)
loss_triplet = triplet_img_loss(F_S, F_T, F_A, margin=margin)
```

Meaning:
- anchor = source / clear Cityscapes
- positive = target / foggy Cityscapes
- negative = aux / rainy RainMix

## AdvGRL Interaction

If `--advgrl` is also enabled, it still affects only the faithful source/target DA classifier path:
- detached source/target faithful BCE computes `L_c`
- `compute_lambda_adv` computes `lambda_adv`
- GRL applies to source and target DA features

Triplet does not use GRL directly; it backpropagates through source/target/aux features via its own weighted triplet loss.

## Validation Rules

Remove faithful conflicts with `--aux` and `--triplet-img`.

Keep:
- `--da-img-faithful` requires `--da-img`
- `--triplet-img` requires `--aux`
- `--triplet-img` requires `--da-img`
- `--triplet-adaptive` requires `--triplet-img`
- `--da-img-faithful` requires `--da-img-warmup off`

## Tests

Update flag validation tests:
- faithful+aux is allowed past validation
- faithful+triplet without aux still errors via existing `--triplet-img requires --aux`
- faithful+aux+triplet is allowed past validation

Update smoke tests:
- add `--da-img --da-img-faithful --aux`
- add `--da-img --da-img-faithful --aux --triplet-img`
- add `--da-img --da-img-faithful --advgrl --aux --triplet-img`

For faithful+triplet smoke rows, verify `da_losses.csv` has numeric `loss_triplet_img`.

## Self-Review

No placeholders. Scope is one behavior change: faithful mode can now consume aux and triplet through a third forward while preserving source/target faithful DA math. Rainy/foggy order is explicit: target/foggy is positive, aux/rainy is negative.
