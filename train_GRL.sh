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
