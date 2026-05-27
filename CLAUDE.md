# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

YOLO-G is a YOLOv5-L fork with image-level domain adaptation (DANN/GRL) for cross-domain detection (e.g., Cityscapes → Foggy Cityscapes). Active work (`docs/superpowers/plans/2026-05-25-yolog-advgrl*.md`) is porting AdvGRL improvements from "DA-Detect" (Paper 2) as **independently-toggleable CLI flags** on `train_GRL.py`, one PR per component, with sanity-gate eval after each.

**Two training entry points — pick the right one:**
- `train.py` → vanilla source-only YOLOv5 (no DA). Driven by `train.sh`.
- `train_GRL.py` → DA-aware training. Driven by `train_GRL.sh`. Imports `models.yolo_GRL.Model` (dumb-model variant).

`val_GRL.py` is the matching validator for `train_GRL.py` (it knows the forward signature returns a tuple — see Architecture below).

## Commands

```bash
# Tests (pytest discovers tests/ — fixtures auto-loaded from tests/conftest.py)
pytest                                                # all tests
pytest tests/test_da_warmup.py -v                     # single file
pytest tests/test_advgrl.py::TestComputeLambdaAdv -v  # single class
pytest tests/test_train_smoke.py -v -s                # CUDA end-to-end smoke across flag combos

# Source-only baseline (PR 1 reference): mAP@0.5 = 0.379 on Foggy Cityscapes
python train_GRL.py --weights yolov5l.pt --cfg configs/domain/yolov5l.yaml \
  --data domain/city_foggycity.yaml --hyp hyps/hyp.scratch-high.yaml \
  --epochs 10 --batch-size 8 --img 640 --name sanity_pr1_baseline

# DA training — toggle features individually
python train_GRL.py ... --da-img                                  # image-level DANN
python train_GRL.py ... --da-img --advgrl                         # dynamic AdvGRL weight
python train_GRL.py ... --da-img --da-img-warmup {off,gate,ramp}  # DA-loss warmup
python train_GRL.py ... --da-img --aux                            # RainMix aux domain
python train_GRL.py ... --da-img --aux --triplet-img              # image triplet loss
python train_GRL.py ... --da-img --advgrl --aux --triplet-img     # full method

# Offline RainMix aux generation
python tools/gen_rainy_cityscapes.py --source-images <cityscapes/images/train> \
  --rain-masks <rain-mask-dir> --output <aux-output-dir> --size 640 640

# Validation / inference
python val_GRL.py --weights runs/train/<name>/weights/best.pt --data domain/city_foggycity.yaml
python detect.py --weights runs/train/<name>/weights/best.pt --source <image_folder>
```

The `--cfg` default in `train_GRL.py` points to a non-existent path (`configs/yolov5-standard/yolov5s.yaml`). **Always pass `--cfg configs/domain/yolov5l.yaml` or `--cfg configs/domain/yolov5l_GRL.yaml` explicitly.** Only those domain configs exist. `train_GRL.py` has no `--noplots` flag; use `--project <tmpdir>` in tests to keep generated plots/checkpoints out of the repo.

## Architecture: the "dumb model" pattern

The whole DA refactor (PR 1) was about pulling GRL+classifier OUT of the model and into `train_GRL.py`. This lets PR 2/3/4/5 toggle DA components without touching `models/`.

**Model contract** (`models/yolo_GRL.py:193` `_forward_once`):
```python
return det_pred, backbone_feat   # backbone_feat = SPPF output (layer 9)
```
`val_GRL.py` already handles the tuple shape. Plain `models/yolo.py` still returns just `det_pred` and is used by `train.py`.

**DA components live outside the model:**
- `models/da_classifier.py:DAImgHead` — standalone Conv→ReLU→Conv head producing `[B,1,H,W]` logits. Constructed in `train_GRL.py` only when `--da-img` is set; checkpointed under a separate `classifier` key (not inside the model state dict).
- `utils/domain_grl.py:gradient_scalar` — functional `autograd.Function` that scales the gradient by a Python float on the backward pass. Apply with a **negative** weight for gradient reversal (`gradient_scalar(feat, -lambda_adv)`).
- `utils/domain_loss.py` — `da_img_loss` (BCE-with-logits, dense source/target labels) and `triplet_img_loss`.
- `utils/advgrl.py:advgrl_step` — two-pass routine: pass 1 detached to compute scalar `L_c`, pass 2 GRL-attached with `lambda_adv = lambda_0 * min(beta, 1/L_c)` when `L_c ≤ alpha`, else `lambda_0`. Default `alpha = BCE([0.7,0.3],[1,0]) ≈ 0.6286` per DA-Detect paper.
- `utils/da_warmup.py:compute_da_warmup_scale` — multiplies `loss_da_image` by 0/1 (`gate`) or a linear 0→1 ramp (`ramp`) over the LR-warmup window `nw`. Mode `off` (default) preserves faithful-original PR 2/3 behavior.
- `utils/da_logger.py:DALogger` — per-iter CSV at `runs/.../da_losses.csv` with fields `epoch, iter, loss_det, loss_da_image, loss_triplet_img, lambda_adv, L_c, da_scale`.
- `utils/domain_aux.py` — `resolve_da_path` (resolves `target:`/`aux:` against YAML's `path:`) and `create_target_dataloader` (reuses YOLO's `create_dataloader` so target augmentation matches source).

**Dataset YAML extension:** `domain/city_foggycity.yaml` adds `target:` and `aux:` alongside the standard `train:`/`val:`. The DA loop zips source batches with target batches; when `--aux` is set, aux images are concatenated after target images for triplet training but excluded from image-level DA loss.

**Flag dependency validation** is in `train_GRL.py:734-748` — `--advgrl` requires `--da-img`, `--da-img-warmup != off` requires `--da-img`, `--triplet-img` requires both `--aux` and `--da-img`, `--triplet-adaptive` requires `--triplet-img`. New flags should follow this pattern.

## PR / sanity-gate workflow

Each PR (1–6) gets a 10-epoch sanity/smoke run on Foggy Cityscapes. Early PRs compared mAP to PR 1's 0.379 baseline, but after PR 3.5 the gate is smoke-only: training completes, no NaNs/crashes, losses bounded, and DA invariants hold. Do **not** rank DA variants by 10-epoch mAP; defer real comparison to ablation runs.

Results land in `runs/train/sanity_pr<N>_<name>/`:
- `results.csv` — epoch-level mAP, precision/recall, losses (standard YOLOv5 format).
- `da_losses.csv` — per-iter DA diagnostics written by `DALogger`.

`runs/train/smoke_*` are short debug runs. `runs/train/sanity_*` are the gating runs for code wiring.

PR 2 (`--da-img`) and PR 3 (`--advgrl`) collapsed to mAP@0.5 ≈ 0.030; PR 3.5 warmup showed the 10-epoch DA regime is platform/noise sensitive. Keep `--da-img-warmup off` as the faithful-original default; don't change defaults without updating the plan and methodology doc.

## Plans, specs, and tests

- `docs/superpowers/specs/` — approved design specs.
- `docs/superpowers/plans/` — task-by-task implementation plans driven by the `superpowers:subagent-driven-development` workflow.
- `tests/conftest.py` defines fixtures (`source_manifest`, `target_manifest`, `aux_manifest`, `tiny_backbone_feat`, `device`). Fixture image lists live under `tests/fixtures/{source,target,aux}/` — regenerate via `tests/fixtures/make_fixtures.py`.
- New DA component → add to `utils/` (one file per concern), add a `tests/test_<thing>.py`, then wire into `train_GRL.py` behind a flag with validation in `main()`.

## Things to know before editing

- `train_GRL.py` is 900+ lines and inherits Ultralytics YOLOv5 structure. The DA glue lives roughly in the training step around the forward/loss block and in the CLI/validation block near the bottom. Don't add DA logic to `models/yolo_GRL.py` — keep the model dumb.
- The fork has many unused legacy YOLO variants (`models/Models/*.py`, `models/Detect/`, `utils/loss_ps.py`, etc.) imported from YOLOair. They are not on the DA code path; ignore unless they break imports.
- `requirements.txt` pins `torch>=1.7.0` but the README/upstream environment is `torch==1.12.0` / Python 3.9 / CUDA 11.7 on a single 3090.
