#!/usr/bin/env bash
# Research direction: does a fall detector trained on URFD's side camera generalise
#   (a) to a viewpoint it never saw: URFD's ceiling camera (same falls, filmed from above), and
#   (b) to another dataset: GMDCSA24 (other people, three real homes, day and night)?
# Three models are trained on every URFD side-camera window and tested on both, untouched:
#   paper   Lin et al.'s setting: LSTM-512, RP, lr 0.01, balanced training set
#   base    GRU-128 on every window (B1)
#   motion  GRU-128 on every window + whole-body motion features (B2, the best in cross-validation)
#
#   bash scripts/generalization.sh
set -uo pipefail
PY=${PYTHON:-python}
OUT=${OUT:-runs/generalization}
mkdir -p "$OUT"
W=(--weights yolo11n-pose.pt --imgsz 640 --device cpu)

echo "== poses of the ceiling camera ($(date +%H:%M))"
$PY -m falldet.extract_poses --videos data/urfd/videos --segments data/urfd/full_videos_cam1.csv \
    --out data/poses/urfd_full_n_cam1 --resize-width 0 "${W[@]}" 2>&1 | grep -E "Wrote|Error|Traceback"
$PY -m falldet.urfd --out data/urfd --from-poses data/poses/urfd_full_n_cam1 --out-poses data/poses/urfd_cam1_s10 \
    --cams 1 --stride 10 --no-balance

echo "== poses of GMDCSA24 ($(date +%H:%M))"
$PY -m falldet.extract_poses --videos "$HOME/.cache/omnifall/videos" --segments data/omnifall/gmdcsa24_binary.csv \
    --out data/poses/gmdcsa24_binary "${W[@]}" 2>&1 | grep -E "Wrote|Error|Traceback"

EXTRA="preprocess.extra=[hip_y,hip_vy,neck_vy,aspect,torso]"
SMALL=(model.name=gru model.gru.hidden=128 train.lr=0.001)
ALL=(data.real_dir=data/poses/urfd_all_s10 data.test_size=0 data.val_size=0)
train() { # <name> <overrides...>: one model on all training data, no held-out split
  local name=$1; shift
  [[ -f $OUT/$name/last.pt ]] && { echo "skip $name"; return; }
  echo "== train $name ($(date +%H:%M))"
  $PY -m falldet.train --out "$OUT/$name" train.log_every=50 "$@" > "$OUT/$name.log" 2>&1 || echo "FAILED $name"
}
train paper  data.real_dir=data/poses/urfd data.test_size=0 data.val_size=0 model.name=lstm train.epochs=200
train base   "${ALL[@]}" "${SMALL[@]}" train.batch_size=64 train.epochs=80 train.class_weights=true
train motion "${ALL[@]}" "${SMALL[@]}" train.batch_size=64 train.epochs=80 train.class_weights=true "$EXTRA"

for name in paper base motion; do
  for target in cam1 gmdcsa24; do
    case $target in
      cam1) dirs=(data/poses/urfd_cam1_s10) ;;
      gmdcsa24) dirs=(data/poses/gmdcsa24_binary/train data/poses/gmdcsa24_binary/val data/poses/gmdcsa24_binary/test) ;;
    esac
    echo "== $name -> $target"
    $PY -m falldet.evaluate --run "$OUT/$name" --checkpoint last.pt --pose-dir "${dirs[@]}" \
        --json "$OUT/${name}_on_${target}.json" | grep -E "clips|^accuracy|^sensitivity"
  done
done
echo "generalization done ($(date +%H:%M))"
