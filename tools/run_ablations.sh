#!/usr/bin/env bash
# Sequentially run the six ablation configs. Each writes to runs/train/ablation_<name>/.
# Read each run's results.csv for the final mAP comparison.

set -euo pipefail
set -x

WEIGHTS='yolov5l.pt'
CFG='./configs/domain/yolov5l_GRL.yaml'
DATA='./domain/city_foggycity.yaml'
HYP='data/hyps/hyp.scratch-high.yaml'
EPOCHS=50
BATCH=2
IMG=640

BASE_ARGS="--weights $WEIGHTS --cfg $CFG --data $DATA --epochs $EPOCHS --batch-size $BATCH --img $IMG --hyp $HYP"

# 1. Baseline (source-only)
python train_GRL.py $BASE_ARGS --name ablation_01_baseline

# 2. --da-img only (current PR2+ dumb-model implementation)
python train_GRL.py $BASE_ARGS --name ablation_02_daimg --da-img

# 7. Faithful original YOLO-G image-level DA baseline
python train_GRL.py $BASE_ARGS --name ablation_07_daimg_faithful --da-img --da-img-faithful

# 3. --da-img --advgrl
python train_GRL.py $BASE_ARGS --name ablation_03_advgrl --da-img --advgrl

# 4. --da-img --aux (no triplet)
python train_GRL.py $BASE_ARGS --name ablation_04_aux --da-img --aux

# 5. --da-img --aux --triplet-img
python train_GRL.py $BASE_ARGS --name ablation_05_triplet --da-img --aux --triplet-img

# 6. Full
python train_GRL.py $BASE_ARGS --name ablation_06_full --da-img --advgrl --aux --triplet-img

echo 'All ablations complete. mAP per run:'
for d in runs/train/ablation_*/; do
  echo -n "$d: "
  tail -1 "$d/results.csv" | cut -d, -f9    # mAP@0.5:0.95 column — adjust index if YOLO-G csv differs
done
