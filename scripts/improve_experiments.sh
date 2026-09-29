#!/usr/bin/env bash
# Improvements beyond Lin et al. 2021, each evaluated by 5-fold cross-validation by video
# (every clip tested once). Each stage adds one change to the previous best, so the effect of
# every change is measured on its own.
#
#   STAGE=A bash scripts/improve_experiments.sh   # balanced 132 windows (the paper's setup)
#   STAGE=B bash scripts/improve_experiments.sh   # more data: every window, stride 10 / 5
#   STAGE=C bash scripts/improve_experiments.sh   # same realistic test set for every training setup
#   STAGE=D bash scripts/improve_experiments.sh   # combinations on top of the best (B2)
#   STAGE=H bash scripts/improve_experiments.sh   # YOLO11m on the original 640x480 frames
#   STAGE=P bash scripts/improve_experiments.sh   # the paper's setting, by video and by window
#
# Afterwards: python -m falldet.summarize runs/improve
set -uo pipefail

OUT=${OUT:-runs/improve}
PYTHON=${PYTHON:-python}
STAGE=${STAGE:-A}
mkdir -p "$OUT"

cv() { # <run name> <overrides...>
  local name=$1; shift
  [[ -f $OUT/$name/cv.json ]] && { echo "skip $name"; return; }
  mkdir -p "$OUT/$name"
  echo "== $name ($(date +%H:%M))"
  if ! $PYTHON -m falldet.cv --out "$OUT/$name" "$@" > "$OUT/$name/cv.log" 2>&1; then
    echo "FAILED $name, see $OUT/$name/cv.log"
    return
  fi
  grep -E "^(accuracy|sensitivity|fold accuracy)" "$OUT/$name/cv.log" | tail -n 3
}

EXTRA="preprocess.extra=[hip_y,hip_vy,neck_vy,aspect,torso]"
BAL="data.real_dir=data/poses/urfd"
SMALL=(model.name=gru model.gru.hidden=128 train.lr=0.001 train.epochs=200)

if [[ $STAGE == A ]]; then
  # A2 / A3 add one change each to A1; A4-A6 swap the model and keep A1's setting
  cv A1_gru128                    $BAL "${SMALL[@]}"
  cv A2_gru128_motion             $BAL "${SMALL[@]}" "$EXTRA"
  cv A3_gru128_motion_val         $BAL "${SMALL[@]}" "$EXTRA" data.val_size=0.15
  cv A4_lstm128                   $BAL model.name=lstm model.lstm.hidden=128 train.lr=0.001 train.epochs=200
  cv A5_bigru128                  $BAL "${SMALL[@]}" model.gru.bidirectional=true
  cv A6_tcn                       $BAL model.name=tcn train.lr=0.001 train.epochs=200
fi

if [[ $STAGE == C ]]; then
  # Same test clips for every run: all windows of the held-out videos (the realistic case).
  # C1 trains like A1 on a balanced subset of the training windows; C2 trains on all of them with
  # A1's schedule, so B1 vs C2 isolates the schedule and C1 vs C2 isolates the extra training data.
  ALL10=data.real_dir=data/poses/urfd_all_s10
  cv C1_all_s10_trainbal_gru128   $ALL10 "${SMALL[@]}" data.train_balance=true
  cv C2_all_s10_fullbatch_gru128  $ALL10 "${SMALL[@]}" train.class_weights=true
  cv C3_all_s10_fullbatch_motion  $ALL10 "${SMALL[@]}" train.class_weights=true "$EXTRA"
fi

if [[ $STAGE == D ]]; then
  # B2 (all windows + motion features) is the best so far; each run adds one change to it
  MB=(train.batch_size=64 train.epochs=80 train.class_weights=true)
  cv D1_all_s5_gru128_motion      data.real_dir=data/poses/urfd_all_s5 "${SMALL[@]}" "${MB[@]}" "$EXTRA"
  cv D2_all_s10_gru128_motion_aug data.real_dir=data/poses/urfd_all_s10 "${SMALL[@]}" "${MB[@]}" "$EXTRA" \
     augment.enabled=true augment.noise_std=0.005
  cv D3_all_s10_bigru128_motion   data.real_dir=data/poses/urfd_all_s10 "${SMALL[@]}" "${MB[@]}" "$EXTRA" \
     model.gru.bidirectional=true
fi

if [[ $STAGE == H ]]; then
  # Closer to the paper's input: YOLO11m on the original 640x480 URFD frames instead of YOLO11n on 320x240.
  # Each run repeats an earlier one (in brackets) with the better poses. "_ws" = split by window (the paper's way).
  MB=(train.batch_size=64 train.epochs=80 train.class_weights=true)
  M=data.real_dir=data/poses/urfd640_m_bal
  cv H3_m640_paper_lstm512_ws     $M model.name=lstm train.epochs=200 data.group_split=false   # [P2]
  cv H2_m640_gru128_ws            $M "${SMALL[@]}" data.group_split=false                      # [E1]
  cv H1_m640_gru128               $M "${SMALL[@]}"                                             # [A1]
  cv H5_m640_all_s10_motion_aug   data.real_dir=data/poses/urfd640_m_all_s10 "${SMALL[@]}" "${MB[@]}" "$EXTRA" \
     augment.enabled=true augment.noise_std=0.005                                              # [D2]
  cv H4_m640_paper_lstm512        $M model.name=lstm train.epochs=200                          # [P1]
  cv H6_m640_cam01_gru128_ws      data.real_dir=data/poses/urfd640_m_cam01_bal "${SMALL[@]}" data.group_split=false  # [G1]
fi

if [[ $STAGE == P ]]; then  # the paper's best setting under cross-validation, for reference (slow on CPU)
  cv P1_paper_lstm512             $BAL model.name=lstm train.epochs=200
  cv P2_paper_lstm512_windowsplit $BAL model.name=lstm train.epochs=200 data.group_split=false
fi

if [[ $STAGE == B ]]; then
  # every window instead of a balanced subset: minibatches and class weights for the imbalance.
  # B1 changes only the data; B2-B4 each add one change to B1.
  MB=(train.batch_size=64 train.epochs=80 train.class_weights=true)
  ALL10=data.real_dir=data/poses/urfd_all_s10
  cv B1_all_s10_gru128            $ALL10 "${SMALL[@]}" "${MB[@]}"
  cv B2_all_s10_gru128_motion     $ALL10 "${SMALL[@]}" "${MB[@]}" "$EXTRA"
  cv B3_all_s5_gru128             data.real_dir=data/poses/urfd_all_s5 "${SMALL[@]}" "${MB[@]}"
  cv B4_all_s10_gru128_aug        $ALL10 "${SMALL[@]}" "${MB[@]}" augment.enabled=true augment.noise_std=0.005
fi

echo "stage $STAGE done ($(date +%H:%M))"
