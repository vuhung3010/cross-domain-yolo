# DA-Img Faithful AdvGRL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `--advgrl` with `--da-img --da-img-faithful` while preserving faithful YOLO-G two-forward labels/loss.

**Architecture:** Keep faithful mode in `train_GRL.py` as separate source and target model forwards. Add a detached faithful DA loss to compute `L_c`, use existing `compute_lambda_adv`, then apply `gradient_scalar(..., -lambda_adv)` to source and target features separately. Aux, triplet, and DA warmup stay blocked in faithful mode.

**Tech Stack:** Python 3.10+, PyTorch, pytest, YOLO-G `train_GRL.py`.

---

## File Structure

**Modify:**
- `train_GRL.py` — remove faithful/AdvGRL conflict; compute dynamic faithful lambda when `opt.advgrl` is true.
- `tests/test_train_grl_flag_validation.py` — replace conflict test with allowed-combo validation.
- `tests/test_train_smoke.py` — add `--da-img --da-img-faithful --advgrl` smoke case and lambda assertion.

---

## Task 1: Update validation tests first

**Files:**
- Modify: `tests/test_train_grl_flag_validation.py`

- [ ] **Step 1: Replace AdvGRL conflict test**

Replace:

```python
def test_da_img_faithful_conflicts_with_advgrl():
    assert_flag_error(
        ['--da-img', '--da-img-faithful', '--advgrl'],
        '--da-img-faithful conflicts with --advgrl',
    )
```

With:

```python
def test_da_img_faithful_allows_advgrl_past_flag_validation():
    result = run_train_grl_flags('--da-img', '--da-img-faithful', '--advgrl', '--cfg', 'missing.yaml')
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert '--da-img-faithful conflicts with --advgrl' not in output
    assert 'missing.yaml' in output or 'No such file' in output or 'does not exist' in output
```

- [ ] **Step 2: Run validation test and verify red**

```bash
pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_grl_flag_validation.py -v
```

Expected: `test_da_img_faithful_allows_advgrl_past_flag_validation` FAILS because current code still raises `--da-img-faithful conflicts with --advgrl`.

---

## Task 2: Implement faithful AdvGRL dynamic lambda

**Files:**
- Modify: `train_GRL.py`

- [ ] **Step 1: Import `compute_lambda_adv`**

Replace:

```python
from utils.advgrl import advgrl_step, default_alpha
```

With:

```python
from utils.advgrl import advgrl_step, compute_lambda_adv, default_alpha
```

- [ ] **Step 2: Remove faithful/AdvGRL conflict**

Delete this validation block:

```python
    if opt.da_img_faithful and opt.advgrl:
        raise SystemExit('--da-img-faithful conflicts with --advgrl (faithful mode reproduces original YOLO-G only)')
```

- [ ] **Step 3: Update faithful DA branch**

Replace current faithful block:

```python
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
```

With:

```python
                    if opt.da_img_faithful:
                        source_logits_detached = classifier_head(backbone_feat.detach())
                        target_logits_detached = classifier_head(target_backbone_feat.detach())
                        L_c_tensor = da_img_faithful_loss_pair(source_logits_detached, target_logits_detached)
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
                        source_feat_grl = gradient_scalar(backbone_feat, -lambda_adv_this_iter)
                        target_feat_grl = gradient_scalar(target_backbone_feat, -lambda_adv_this_iter)
                        source_logits = classifier_head(source_feat_grl)
                        target_logits = classifier_head(target_feat_grl)
                        loss_da_image = da_img_faithful_loss_pair(source_logits, target_logits)
                        loss_da_image_this_iter = loss_da_image
                        loss = loss + opt.da_img_weight * loss_da_image
```

- [ ] **Step 4: Run validation and compile**

```bash
python -m py_compile /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/train_GRL.py
pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_grl_flag_validation.py -v
```

Expected: compile exits 0; validation tests PASS.

---

## Task 3: Add smoke coverage

**Files:**
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Add faithful AdvGRL smoke parameter**

Add after `daimg_faithful`:

```python
    (['--da-img', '--da-img-faithful', '--advgrl'],                                'daimg_faithful_advgrl'),
```

- [ ] **Step 2: Update faithful CSV assertion**

Replace:

```python
            assert row['lambda_adv'] == '0.1'
            assert row['da_scale'] == '1.0'
```

With:

```python
            assert row['da_scale'] == '1.0'
            if '--advgrl' in flags:
                assert row['lambda_adv'] != '0.1'
            else:
                assert row['lambda_adv'] == '0.1'
```

- [ ] **Step 3: Run smoke**

```bash
PYTHONPATH=/home/kacchan/VuHung/project/nhandang_project_2/yolo-G pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_smoke.py -v -s
```

Expected: PASS, including `smoke_daimg_faithful_advgrl`.

---

## Task 4: Final verification

Run:

```bash
python -m py_compile /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/train_GRL.py
PYTHONPATH=/home/kacchan/VuHung/project/nhandang_project_2/yolo-G pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_domain_loss.py /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_grl_flag_validation.py -v
PYTHONPATH=/home/kacchan/VuHung/project/nhandang_project_2/yolo-G pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_smoke.py -v -s
```

Expected: all pass.

## Self-Review

Spec coverage: allows faithful+AdvGRL, keeps source=0/target=1, computes detached faithful `L_c`, applies dynamic lambda to both faithful feature tensors, keeps aux/triplet/warmup blocked. No placeholders. Type names match existing code.
