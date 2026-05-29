# DA-Img Faithful Aux/Triplet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow faithful YOLO-G image-level DA mode to run with `--aux` and `--triplet-img` by adding a separate aux forward and computing triplet from separate source/target/aux features.

**Architecture:** Faithful mode remains separate source and target forwards for DA classifier loss. When aux is enabled, faithful mode adds `model(a_imgs)` as a third forward only for triplet features. Non-faithful mode keeps existing concatenated `[source | target | aux]` behavior unchanged.

**Tech Stack:** Python 3.10+, PyTorch, pytest, YOLO-G `train_GRL.py`.

---

## File Structure

**Modify:**
- `train_GRL.py` — remove faithful conflicts with aux/triplet; add faithful aux forward; branch triplet feature slicing for faithful mode.
- `tests/test_train_grl_flag_validation.py` — update validation tests so faithful+aux/triplet combos pass validation.
- `tests/test_train_smoke.py` — add faithful aux/triplet smoke cases and verify triplet logging.

---

## Task 1: Update validation tests first

**Files:**
- Modify: `tests/test_train_grl_flag_validation.py`

- [ ] **Step 1: Replace faithful+aux conflict test**

Replace:

```python
def test_da_img_faithful_conflicts_with_aux():
    assert_flag_error(
        ['--da-img', '--da-img-faithful', '--aux'],
        '--da-img-faithful conflicts with --aux',
    )
```

With:

```python
def test_da_img_faithful_allows_aux_past_flag_validation():
    result = run_train_grl_flags('--da-img', '--da-img-faithful', '--aux', '--cfg', 'missing.yaml')
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert '--da-img-faithful conflicts with --aux' not in output
    assert 'missing.yaml' in output or 'No such file' in output or 'does not exist' in output
```

- [ ] **Step 2: Replace faithful+triplet conflict test**

Replace:

```python
def test_da_img_faithful_conflicts_with_triplet_img():
    assert_flag_error(
        ['--da-img', '--da-img-faithful', '--triplet-img'],
        '--da-img-faithful conflicts with --triplet-img',
    )
```

With two tests:

```python
def test_da_img_faithful_triplet_still_requires_aux():
    assert_flag_error(
        ['--da-img', '--da-img-faithful', '--triplet-img'],
        '--triplet-img requires --aux',
    )


def test_da_img_faithful_allows_aux_triplet_past_flag_validation():
    result = run_train_grl_flags('--da-img', '--da-img-faithful', '--aux', '--triplet-img', '--cfg', 'missing.yaml')
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert '--da-img-faithful conflicts with --triplet-img' not in output
    assert 'missing.yaml' in output or 'No such file' in output or 'does not exist' in output
```

- [ ] **Step 3: Run validation test and verify red**

```bash
pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_grl_flag_validation.py -v
```

Expected: new allow tests fail because current code still blocks faithful+aux/triplet.

---

## Task 2: Implement faithful aux/triplet forward path

**Files:**
- Modify: `train_GRL.py`

- [ ] **Step 1: Remove faithful aux/triplet conflict validation**

Delete these blocks:

```python
    if opt.da_img_faithful and opt.aux:
        raise SystemExit('--da-img-faithful conflicts with --aux (faithful mode uses source+target only)')
    if opt.da_img_faithful and opt.triplet_img:
        raise SystemExit('--da-img-faithful conflicts with --triplet-img (faithful mode has no aux negative)')
```

Do not delete:

```python
    if opt.triplet_img and not opt.aux:
        raise SystemExit('--triplet-img requires --aux (triplet negative comes from aux loader)')
```

- [ ] **Step 2: Add faithful aux forward**

Replace current faithful forward block:

```python
                if opt.da_img_faithful:
                    det_pred, backbone_feat = model(imgs)
                    _, target_backbone_feat = model(t_imgs)
                    B_t = t_imgs.size(0)
                    B_a = 0
                    det_pred_src = det_pred
```

With:

```python
                if opt.da_img_faithful:
                    det_pred, backbone_feat = model(imgs)
                    _, target_backbone_feat = model(t_imgs)
                    aux_backbone_feat = None
                    B_t = t_imgs.size(0)
                    B_a = a_imgs.size(0) if a_imgs is not None else 0
                    if a_imgs is not None:
                        _, aux_backbone_feat = model(a_imgs)
                    det_pred_src = det_pred
```

- [ ] **Step 3: Branch triplet feature selection**

Replace current triplet block start:

```python
                if opt.triplet_img:
                    F_S = _gap_mean(backbone_feat[:B_s])
                    F_T = _gap_mean(backbone_feat[B_s:B_s + B_t])
                    F_A = _gap_mean(backbone_feat[B_s + B_t:B_s + B_t + B_a])
```

With:

```python
                if opt.triplet_img:
                    if opt.da_img_faithful:
                        F_S = _gap_mean(backbone_feat)
                        F_T = _gap_mean(target_backbone_feat)
                        F_A = _gap_mean(aux_backbone_feat)
                    else:
                        F_S = _gap_mean(backbone_feat[:B_s])
                        F_T = _gap_mean(backbone_feat[B_s:B_s + B_t])
                        F_A = _gap_mean(backbone_feat[B_s + B_t:B_s + B_t + B_a])
```

No other triplet logic changes. `triplet_img_loss(F_S, F_T, F_A, margin=margin)` remains anchor=source, positive=foggy target, negative=rainy aux.

- [ ] **Step 4: Compile and rerun validation tests**

```bash
python -m py_compile /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/train_GRL.py
pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_grl_flag_validation.py -v
```

Expected: compile exits 0; validation tests pass.

---

## Task 3: Add smoke coverage

**Files:**
- Modify: `tests/test_train_smoke.py`

- [ ] **Step 1: Add faithful aux/triplet smoke cases**

Add after `daimg_faithful_advgrl`:

```python
    (['--da-img', '--da-img-faithful', '--aux'],                                    'daimg_faithful_aux'),
    (['--da-img', '--da-img-faithful', '--aux', '--triplet-img'],                   'daimg_faithful_triplet'),
    (['--da-img', '--da-img-faithful', '--advgrl', '--aux', '--triplet-img'],       'daimg_faithful_full'),
```

- [ ] **Step 2: Assert faithful triplet logs numeric triplet loss**

In the existing DA CSV assertion block, append inside `if '--da-img' in flags:`:

```python
        if '--triplet-img' in flags:
            header = lines[0].split(',')
            first_row = lines[1].split(',')
            row = dict(zip(header, first_row))
            assert row['loss_triplet_img'] != ''
            assert float(row['loss_triplet_img']) >= 0.0
```

- [ ] **Step 3: Run smoke tests**

```bash
PYTHONPATH=/home/kacchan/VuHung/project/nhandang_project_2/yolo-G pytest /home/kacchan/VuHung/project/nhandang_project_2/yolo-G/tests/test_train_smoke.py -v -s
```

Expected: pass, including faithful aux/triplet cases.

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

Spec coverage: faithful+aux/triplet allowed; aux uses third forward; DA classifier still source+target only; triplet order remains source anchor, foggy target positive, rainy aux negative; warmup still blocked. No placeholders. Type names match current code.
