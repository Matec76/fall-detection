"""K-fold cross-validation by video: every clip is tested exactly once.

With only ~27 test clips per random split, one clip moves the accuracy by ~3.7 points and
different splits disagree by up to ±12 points. Cross-validation pools the test predictions
of all folds, so every clip counts once and comparisons between settings become reliable.

    python -m falldet.cv --out runs/cv/gru128 model.name=gru model.gru.hidden=128 train.lr=0.001
    python -m falldet.cv --out runs/cv/paper_split --folds 5 data.group_split=false   # windows, not videos
    python -m falldet.cv --out runs/cv/gru128 --repeats 3 ...                          # 3 x 5 folds
    python -m falldet.cv --out runs/cv/bal data.train_balance=true ...   # balance the training folds only

``data.train_balance`` keeps as many no-fall as fall clips in each training fold (as the paper
does for its whole dataset) while the test fold keeps every clip, so training on a balanced subset
and training on everything can be compared on the same, realistic test clips.

Writes <out>/cv.json with the pooled metrics (all test predictions together), the mean and
standard deviation over folds, and one run directory per fold.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import load_config, save_config
from .dataset import Sample, _holdout, scan_pose_dir, subsample_per_class
from .metrics import classification_metrics, format_metrics
from .train import run_training


def folds_of(samples: list[Sample], n_folds: int, seed: int, by_video: bool):
    labels = [s.label for s in samples]
    if by_video:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return splitter.split(np.zeros(len(samples)), labels, [s.group for s in samples])
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(samples)), labels)


def pooled_metrics(confusions: list[list[list[int]]], classes: list[str]) -> dict:
    """Metrics of all folds' test predictions together, rebuilt from the confusion matrices."""
    total = np.sum(np.array(confusions), axis=0)
    y_true, y_pred = [], []
    for t in range(len(classes)):
        for p in range(len(classes)):
            y_true += [t] * int(total[t, p])
            y_pred += [p] * int(total[t, p])
    return classification_metrics(y_true, y_pred, classes)


def run_cv(cfg: dict, out: Path, n_folds: int = 5, repeats: int = 1) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    classes, data = list(cfg["classes"]), cfg["data"]
    samples = scan_pose_dir(data["real_dir"], classes)
    by_video = data.get("group_split", True)
    folds, confusions = [], []
    for rep in range(repeats):
        seed = data["seed"] + rep
        for k, (train_idx, test_idx) in enumerate(folds_of(samples, n_folds, seed, by_video)):
            train = [samples[i] for i in train_idx]
            if not by_video:  # each clip is its own group, so the validation split is per clip too
                train = [Sample(s.path, s.label, s.path) for s in train]
            if data.get("train_balance"):
                train = subsample_per_class(train, min(np.bincount([s.label for s in train])), seed)
            train, val = _holdout(train, data.get("val_size", 0.0), seed)
            splits = {"train": train, "val": val, "test": [samples[i] for i in test_idx]}
            fold_cfg = {**cfg, "data": {**data, "seed": seed}}
            name = f"r{rep}_f{k}" if repeats > 1 else f"f{k}"
            print(f"== fold {name}: train {len(train)} val {len(val)} test {len(test_idx)}", flush=True)
            test = run_training(fold_cfg, out / name, splits=splits)["test"]
            folds.append({"fold": name, **{m: test[m] for m in ("accuracy", "sensitivity", "specificity", "f1")
                                           if m in test}})
            confusions.append(test["confusion_matrix"])
    pooled = pooled_metrics(confusions, classes)
    spread = {m: {"mean": statistics.mean(f[m] for f in folds), "sd": statistics.stdev(f[m] for f in folds)}
              for m in folds[0] if m != "fold"}
    result = {"pooled": pooled, "folds": folds, "fold_stats": spread, "n_folds": n_folds, "repeats": repeats,
              "by_video": by_video, "n_samples": len(samples)}
    (out / "cv.json").write_text(json.dumps(result, indent=1))
    print(f"\nPOOLED over {len(folds)} folds ({len(samples)} clips, each tested {repeats}x)\n"
          + format_metrics(pooled, classes)
          + f"\nfold accuracy {spread['accuracy']['mean'] * 100:.1f} ± {spread['accuracy']['sd'] * 100:.1f}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1, help="repeat the k folds with different shuffles")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. model.name=gru")
    args = ap.parse_args()
    cfg = load_config(args.config, ["train.log_every=50", *args.overrides])
    run_cv(cfg, Path(args.out), args.folds, args.repeats)


if __name__ == "__main__":
    main()
