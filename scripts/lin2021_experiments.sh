#!/usr/bin/env bash
# Experiments of Lin et al. 2021 on URFD pose clips, one run directory per setting (39 settings):
#   Tables 8-10  normalisation (none / min-max / RP / RP + interpolation) x (RNN, LSTM, GRU)
#   Table 5      hidden nodes 64 ... 1024 with RP
#   Tables 6-7   learning rate 0.1 / 0.01 / 0.001
#   extra        the paper's plain random split (windows of one video in train and test)
# hidden_512_* and lr_0.01_* are the default setting, so they link to norm_rp_* instead of re-training.
#
#   bash scripts/lin2021_experiments.sh                 # expects data/poses/urfd (see README)
#   POSES=data/poses/urfd EPOCHS=200 bash scripts/lin2021_experiments.sh
#
# Afterwards: python -m falldet.summarize runs/lin2021
set -uo pipefail

POSES=${POSES:-data/poses/urfd}
OUT=${OUT:-runs/lin2021}
EPOCHS=${EPOCHS:-500}
PYTHON=${PYTHON:-python}
common=(data.real_dir=$POSES train.epochs=$EPOCHS)

train() { # <run name> <overrides...>
  local name=$1; shift
  [[ -f $OUT/$name/metrics.json ]] && { echo "skip $name"; return; }
  mkdir -p "$OUT/$name"
  echo "== $name ($(date +%H:%M))"
  if ! $PYTHON -m falldet.train --out "$OUT/$name" "${common[@]}" "$@" > "$OUT/$name/train.log" 2>&1; then
    echo "FAILED $name, see $OUT/$name/train.log"
    return
  fi
  grep -E "^(accuracy|sensitivity)" "$OUT/$name/train.log"
}

same_as() { # <link name> <existing run>
  [[ -e $OUT/$1 ]] || ln -s "$2" "$OUT/$1"
}

declare -A NORM=(
  [none]="preprocess.norm_mode=none preprocess.joints=coco17"
  [minmax]="preprocess.norm_mode=minmax"
  [rp]="preprocess.norm_mode=rp"
  [rp_interp]="preprocess.norm_mode=rp preprocess.interpolate=true"
)
mkdir -p "$OUT"
for model in rnn lstm gru; do
  for norm in rp none minmax rp_interp; do
    # shellcheck disable=SC2086
    train "norm_${norm}_${model}" model.name=$model ${NORM[$norm]}
  done
  for hidden in 64 128 256 1024; do
    train "hidden_${hidden}_${model}" model.name=$model model.$model.hidden=$hidden
  done
  same_as "hidden_512_${model}" "norm_rp_${model}"
  for lr in 0.1 0.001; do
    train "lr_${lr}_${model}" model.name=$model train.lr=$lr
  done
  same_as "lr_0.01_${model}" "norm_rp_${model}"
  train "paper_split_${model}" model.name=$model data.group_split=false
done
echo "all done ($(date +%H:%M))"
