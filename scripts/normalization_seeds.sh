#!/usr/bin/env bash
# Normalisation comparison (Tables 8-10 of Lin et al. 2021) at a setting that trains stably
# on URFD (128 hidden nodes, lr 0.001), repeated over several seeds. The seed changes both the
# train/test split (by video) and the weight initialisation, so the spread across seeds shows
# how much a 27-clip test set moves the score.
#
#   bash scripts/normalization_seeds.sh
#   SEEDS="1 2 3 4 5" bash scripts/normalization_seeds.sh
#
# Afterwards: python -m falldet.summarize runs/norm_seeds
set -uo pipefail

POSES=${POSES:-data/poses/urfd}
OUT=${OUT:-runs/norm_seeds}
EPOCHS=${EPOCHS:-200}
HIDDEN=${HIDDEN:-128}
LR=${LR:-0.001}
SEEDS=${SEEDS:-"1 2 3"}
PYTHON=${PYTHON:-python}

declare -A NORM=(
  [none]="preprocess.norm_mode=none preprocess.joints=coco17"
  [minmax]="preprocess.norm_mode=minmax"
  [rp]="preprocess.norm_mode=rp"
  [rp_interp]="preprocess.norm_mode=rp preprocess.interpolate=true"
)
mkdir -p "$OUT"
for seed in $SEEDS; do
  for model in rnn lstm gru; do
    for norm in none minmax rp rp_interp; do
      name="${norm}_${model}_s${seed}"
      [[ -f $OUT/$name/metrics.json ]] && { echo "skip $name"; continue; }
      mkdir -p "$OUT/$name"
      echo "== $name ($(date +%H:%M))"
      # shellcheck disable=SC2086
      if ! $PYTHON -m falldet.train --out "$OUT/$name" data.real_dir=$POSES data.seed=$seed \
          train.epochs=$EPOCHS train.lr=$LR model.name=$model model.$model.hidden=$HIDDEN \
          ${NORM[$norm]} > "$OUT/$name/train.log" 2>&1; then
        echo "FAILED $name, see $OUT/$name/train.log"
        continue
      fi
      grep -E "^accuracy" "$OUT/$name/train.log"
    done
  done
done
echo "all done ($(date +%H:%M))"
