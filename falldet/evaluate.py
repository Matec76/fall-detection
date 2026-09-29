"""Evaluate a trained run on its own test split or on another pose directory.

    python -m falldet.evaluate --run runs/transformer
    python -m falldet.evaluate --run runs/transformer --pose-dir data/poses/other_dataset
    python -m falldet.evaluate --run runs/final --pose-dir data/a data/b --json out.json   # several dirs together
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import PoseSequenceDataset, samples_from_json, scan_pose_dir
from .metrics import classification_metrics, format_metrics
from .models import load_checkpoint
from .train import predict, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory written by falldet.train")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--pose-dir", nargs="+", help="evaluate every clip in these dirs instead of the run's test split")
    ap.add_argument("--json", help="also write the metrics here")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    run = Path(args.run)
    device = resolve_device(args.device)
    model, meta = load_checkpoint(run / args.checkpoint, device)
    classes, pre = meta["classes"], meta["preprocess"]
    if args.pose_dir:
        samples = [s for d in args.pose_dir for s in scan_pose_dir(d, classes)]
    else:
        with open(run / "split.json", encoding="utf-8") as f:
            samples = samples_from_json(json.load(f)["test"])
    ds = PoseSequenceDataset(samples, pre)
    y_true, probs = predict(model, DataLoader(ds, batch_size=256), device)
    metrics = classification_metrics(y_true, probs.argmax(1), classes)
    print(f"{len(ds)} clips\n" + format_metrics(metrics, classes))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"run": str(run), "pose_dirs": args.pose_dir, "test": metrics}, indent=1))


if __name__ == "__main__":
    main()
