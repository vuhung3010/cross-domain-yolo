# DA-Img Faithful AdvGRL Design

## Goal

Allow `--advgrl` to run with `--da-img --da-img-faithful` while preserving faithful YOLO-G image-level DA semantics:
- two separate source/target model forwards
- source domain label `0`
- target domain label `1`
- `0.5 * source_BCE + 0.5 * target_BCE`

## Design

`--da-img-faithful` keeps the two-forward data flow already added in `train_GRL.py`:

```python
det_pred, backbone_feat = model(imgs)
_, target_backbone_feat = model(t_imgs)
```

When `--advgrl` is absent, faithful mode keeps fixed GRL behavior:

```python
lambda_adv = opt.da_img_grl_weight
```

When `--advgrl` is present, faithful mode computes dynamic `lambda_adv` from a detached faithful DA loss:

```python
source_logits_detached = classifier_head(backbone_feat.detach())
target_logits_detached = classifier_head(target_backbone_feat.detach())
L_c = da_img_faithful_loss_pair(source_logits_detached, target_logits_detached).item()
lambda_adv = compute_lambda_adv(L_c, lambda_0=opt.da_img_grl_weight, alpha=alpha, beta=opt.advgrl_threshold)
```

Then it applies GRL to both feature tensors separately:

```python
source_feat_grl = gradient_scalar(backbone_feat, -lambda_adv)
target_feat_grl = gradient_scalar(target_backbone_feat, -lambda_adv)
```

Final faithful DA loss remains:

```python
loss_da_image = da_img_faithful_loss_pair(source_logits, target_logits)
loss = loss + opt.da_img_weight * loss_da_image
```

## Flag Rules

Allowed:
- `--da-img --da-img-faithful`
- `--da-img --da-img-faithful --advgrl`

Still blocked:
- `--da-img-faithful --aux`
- `--da-img-faithful --triplet-img`
- `--da-img-faithful --da-img-warmup gate`
- `--da-img-faithful --da-img-warmup ramp`

`--da-img-faithful` still requires `--da-img`.

## Files

Modify:
- `utils/advgrl.py` — export reusable `compute_lambda_adv` already exists; no change expected.
- `train_GRL.py` — remove faithful/advgrl conflict and add detached faithful `L_c` pass.
- `tests/test_train_grl_flag_validation.py` — remove conflict test for `--advgrl`; add parse/command-level test that combo is accepted past validation.
- `tests/test_train_smoke.py` — add smoke case for `--da-img --da-img-faithful --advgrl`.

## Testing

Run:

```bash
python -m py_compile train_GRL.py
PYTHONPATH=/home/kacchan/VuHung/project/nhandang_project_2/yolo-G pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_domain_loss.py /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_grl_flag_validation.py -v
PYTHONPATH=/home/kacchan/VuHung/project/nhandang_project_2/yolo-G pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_smoke.py -v -s
```

Expected:
- compile exits 0
- unit + flag validation tests pass
- CUDA smoke includes faithful fixed-GRL and faithful AdvGRL cases
- `da_losses.csv` for faithful AdvGRL logs dynamic `lambda_adv`, not always `0.1`

## Self-Review

No placeholders. Scope is one behavior change: allow faithful mode to use dynamic AdvGRL weight while keeping faithful labels and two-pass source/target forwards. Requirements do not conflict with existing extension blocks: aux/triplet/warmup remain disabled in faithful mode.
