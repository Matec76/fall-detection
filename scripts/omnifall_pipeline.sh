#!/usr/bin/env bash
# OmniFall experiment (extra data beyond the reference paper):
#   real staged data (official cross-subject split)  ->  Real
#   + OF-Syn synthetic videos (train only)          ->  Real + Synthetic
#   OF-Syn only                                     ->  Synthetic
# for each model in $MODELS, then a test on genuine in-the-wild falls (OOPS).
# Default: fall / no_fall with the Lin et al. 2021 setup (configs/default.yaml, 100-frame windows).
#
#   bash scripts/omnifall_pipeline.sh
#   LABEL_MAP=paper4 CONFIG=configs/juraev2022.yaml WINDOW_SEC=2 MODELS="transformer stacked_lstm" \
#       bash scripts/omnifall_pipeline.sh          # the 4-class setup of Juraev et al. 2022
#   SYN_AGES=elderly_65_plus bash scripts/omnifall_pipeline.sh
#   COMPONENTS="le2i GMDCSA24" WEIGHTS=yolo11n-pose.pt DEVICE=cpu SKIP_OOPS=1 bash scripts/omnifall_pipeline.sh
#
# Pose extraction dominates the run time: use a GPU (DEVICE=0) for the full set.
set -euo pipefail

COMPONENTS=${COMPONENTS:-"le2i caucafall GMDCSA24 up_fall"}
WEIGHTS=${WEIGHTS:-yolo11m-pose.pt}
DEVICE=${DEVICE:-0}
SKIP_OOPS=${SKIP_OOPS:-0}
SKIP_SYN=${SKIP_SYN:-0}
SYN_AGES=${SYN_AGES:-}          # e.g. "elderly_65_plus" to keep only elderly synthetic subjects
LABEL_MAP=${LABEL_MAP:-binary}  # binary | paper4 | core10
CONFIG=${CONFIG:-configs/default.yaml}
WINDOW_SEC=${WINDOW_SEC:-3.333} # 100 frames at 30 fps
MODELS=${MODELS:-"lstm gru"}
seg=(--label-map $LABEL_MAP --window-sec $WINDOW_SEC)
META=data/omnifall
POSES=data/poses/omnifall
# `omnifall prepare` writes videos into its cache as {dataset}/video/{path}.mp4
VIDEO_ROOT=${VIDEO_ROOT:-$HOME/.cache/omnifall/videos}
export OMNIFALL_FFMPEG=${OMNIFALL_FFMPEG:-$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")}

extra=()
[[ $SKIP_SYN == 1 ]] || extra+=(of-syn)
[[ $SKIP_OOPS == 1 ]] || extra+=(OOPS)

echo "== 1. videos (licence notices are accepted with --yes; read them via 'omnifall sources')"
omnifall prepare $COMPONENTS "${extra[@]}" --yes

echo "== 2. segment lists"
python -m falldet.omnifall --components $COMPONENTS "${seg[@]}" --out $META/staged_cs.csv
[[ $SKIP_SYN == 1 ]] || python -m falldet.omnifall --components of-syn --splits train "${seg[@]}" \
    ${SYN_AGES:+--age-groups $SYN_AGES} --out $META/syn_train.csv
[[ $SKIP_OOPS == 1 ]] || python -m falldet.omnifall --components OOPS --splits test "${seg[@]}" --out $META/itw_test.csv

echo "== 3. poses"
sets=(staged_cs)
[[ $SKIP_SYN == 1 ]] || sets+=(syn_train)
[[ $SKIP_OOPS == 1 ]] || sets+=(itw_test)
for name in "${sets[@]}"; do
  python -m falldet.extract_poses --videos "$VIDEO_ROOT" --segments $META/$name.csv \
      --out $POSES/$name --weights "$WEIGHTS" --device "$DEVICE"
done
python -m falldet.pose_quality "${sets[@]/#/$POSES/}"

echo "== 4. training"
common=(data.real_dir=$POSES/staged_cs data.split=predefined train.class_weights=true)
for model in $MODELS; do
  python -m falldet.train --config $CONFIG --out runs/of_${model}_real "${common[@]}" model.name=$model
  if [[ $SKIP_SYN != 1 ]]; then
    python -m falldet.train --config $CONFIG --out runs/of_${model}_real+syn "${common[@]}" model.name=$model \
        data.synthetic_dir=$POSES/syn_train
    python -m falldet.train --config $CONFIG --out runs/of_${model}_syn "${common[@]}" model.name=$model \
        data.synthetic_dir=$POSES/syn_train data.real_train=false
  fi
done

if [[ $SKIP_OOPS != 1 ]]; then
  echo "== 5. in-the-wild test (OOPS)"
  for run in runs/of_*; do
    echo "--- $run"
    python -m falldet.evaluate --run "$run" --pose-dir $POSES/itw_test/test
  done
fi
