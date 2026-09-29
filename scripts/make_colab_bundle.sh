#!/usr/bin/env bash
# Pack what Colab needs into one zip: code, configs, scripts, notebook and extracted poses.
#
#   bash scripts/make_colab_bundle.sh                # ~3 MB: enough to train
#   WITH_VIDEOS=1 bash scripts/make_colab_bundle.sh  # + URFD videos (~20 MB) to re-extract poses on the GPU
#
# Upload the zip to Google Drive as MyDrive/falldet/falldet_colab.zip and open
# notebooks/colab_train.ipynb in Colab.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-falldet_colab.zip}
files=(pyproject.toml README.md falldet configs scripts tests notebooks)
[[ -d data/poses ]] && files+=(data/poses)
if [[ ${WITH_VIDEOS:-0} == 1 ]]; then
  files+=(data/urfd/videos data/urfd/segments.csv)
fi
rm -f "$OUT"
zip -qr "$OUT" "${files[@]}" -x '*__pycache__*' '*.pyc'
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
