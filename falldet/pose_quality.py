"""Compare pose estimators by ACV / AMV (paper Fig. 7 and 8), without ground truth.

    python -m falldet.pose_quality data/poses/yolo data/poses/alphapose --conf-thr 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dataset import load_keypoints
from .metrics import pose_quality


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose_dirs", nargs="+", help="one directory of .npz files per estimator")
    ap.add_argument("--conf-thr", type=float, default=0.3)
    args = ap.parse_args()

    print(f"{'estimator':<28}{'videos':>7}{'ACV mean':>10}{'ACV med':>9}{'AMV% mean':>11}{'AMV% med':>10}")
    for d in args.pose_dirs:
        files = sorted(Path(d).rglob("*.npz"))
        if not files:
            print(f"{d:<28}{0:>7}  (no .npz files)")
            continue
        q = [pose_quality(load_keypoints(p), args.conf_thr) for p in files]
        acv = np.array([x["acv"] for x in q])
        amv = np.array([x["amv"] for x in q])
        print(f"{d:<28}{len(files):>7}{acv.mean():>10.3f}{np.median(acv):>9.3f}"
              f"{amv.mean():>11.2f}{np.median(amv):>10.2f}")


if __name__ == "__main__":
    main()
