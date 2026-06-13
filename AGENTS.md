# Repository Guidelines

## Project Structure & Module Organization

This repository is a YOLO-G / YOLOv5-L fork for cross-domain object detection and domain adaptation. Core entry points are `train.py` for source-only training and `train_GRL.py` for domain-adaptive training; validation uses `val.py` and `val_GRL.py`. Model code lives in `models/`, including DA variants such as `models/yolo_GRL.py` and `models/da_classifier.py`. Domain-adaptation utilities are in `utils/`, including GRL, loss, auxiliary, and logging helpers. Dataset and experiment configs are in `domain/`, `configs/`, and `hyps/`. Tests are under `tests/`, with fixtures in `tests/fixtures/`. Reports and writing artifacts live in `docs/`, `report/`, and `.claude/worktrees/final-report/report/`.

## Build, Test, and Development Commands

Use Python 3.9 when possible; the upstream environment used Torch 1.12.0 with CUDA 11.7.

```bash
pip install -r requirements.txt
pytest
pytest tests/test_advgrl.py -v
pytest tests/test_obj_gated_da.py -v
python train_GRL.py --weights yolov5l.pt --cfg configs/domain/yolov5l.yaml --data domain/city_foggycity.yaml --epochs 10 --batch-size 8 --img 640 --name sanity_run
python val_GRL.py --weights runs/train/<run>/weights/best.pt --data domain/city_foggycity.yaml
```

Use `train_GRL.py` for DA experiments and `train.py` only for vanilla source-only training.

## Coding Style & Naming Conventions

Follow the existing YOLOv5 Python style: 4-space indentation, `snake_case` functions and variables, `PascalCase` classes, and concise comments only for non-obvious logic. Keep DA behavior in training or utility layers when possible; only change model files when the model contract itself changes. Add focused utility files under `utils/` rather than mixing unrelated concerns.

## Testing Guidelines

Tests use `pytest`. Name files `tests/test_<feature>.py` and functions `test_<behavior>()`. Add targeted tests for DA losses, schedulers, path handling, CLI validation, and feature contracts. Prefer existing fixture data from `tests/fixtures/`; regenerate fixtures only when necessary.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit prefixes such as `feat:`, `fix:`, `test:`, and `docs:`. Keep commits scoped, for example `fix: harden objectness gated da loss`. PRs should describe motivation, changed flags or configs, tests run, and metric or smoke-test outcomes. Do not claim performance gains from short sanity runs alone.

## Agent-Specific Instructions

Do not revert unrelated user changes. Before editing training behavior, inspect `CLAUDE.md` for project constraints and DA architecture notes. Avoid moving DA logic into `models/yolo_GRL.py` unless the model interface requires it.
