"""Train a fall classifier on extracted pose sequences.

    python -m falldet.train --out runs/lstm                       # configs/default.yaml: Lin et al. 2021
    python -m falldet.train --out runs/gru model.name=gru model.gru.hidden=256
    python -m falldet.train --out runs/lstm_minmax preprocess.norm_mode=minmax
    python -m falldet.train --config configs/juraev2022.yaml --out runs/transformer
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .dataset import (PoseSequenceDataset, samples_to_json, scan_pose_dir, split_samples,
                      subsample_per_class)
from .metrics import classification_metrics, format_metrics
from .models import build_model, count_parameters, save_checkpoint
from .preprocess import PREPROCESS_DEFAULTS


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def run_training(cfg: dict, out_dir: str | Path, splits: dict | None = None) -> dict:
    """Train one model; ``splits`` ({"train", "val", "test"} sample lists) overrides the config's split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(cfg["classes"])
    data, tr = cfg["data"], cfg["train"]
    pre = {**PREPROCESS_DEFAULTS, **cfg["preprocess"]}
    torch.manual_seed(data["seed"])

    real = Path(data["real_dir"])
    if splits is not None:
        splits = {k: list(v) for k, v in splits.items()}
    elif data.get("split", "random") == "predefined":  # <real_dir>/{train,val,test}/<class>/*.npz
        splits = {part: scan_pose_dir(real / part, classes) if (real / part).is_dir() else []
                  for part in ("train", "val", "test")}
    else:
        splits = split_samples(scan_pose_dir(real, classes), data["test_size"], data["val_size"], data["seed"],
                               data.get("group_split", True))
    if data.get("synthetic_dir") and "train" in splits:
        synth = Path(data["synthetic_dir"])
        synth = synth / "train" if (synth / "train").is_dir() else synth
        synthetic = subsample_per_class(scan_pose_dir(synth, classes),
                                        data.get("synthetic_per_class"), data["seed"])
        splits["train"] = (splits["train"] if data.get("real_train", True) else []) + synthetic
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump({k: samples_to_json(v) for k, v in splits.items()}, f, indent=1)
    save_config(cfg, out_dir / "config.yaml")

    augment = cfg["augment"] if cfg["augment"].get("enabled") else None
    train_ds = PoseSequenceDataset(splits["train"], pre, augment=augment, seed=data["seed"])
    eval_sets = {k: PoseSequenceDataset(v, pre) for k, v in splits.items() if k != "train" and v}
    batch_size = tr.get("batch_size") or len(train_ds)  # null = full batch, as in the paper
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loaders = {k: DataLoader(v, batch_size=256) for k, v in eval_sets.items()}
    counts = np.bincount([s.label for s in splits["train"]], minlength=len(classes))
    print("samples:", {k: len(v) for k, v in splits.items()}, "| train per class:",
          dict(zip(classes, counts.tolist())))

    device = resolve_device(tr["device"])
    name = cfg["model"]["name"]
    model_kwargs = dict(cfg["model"][name])
    in_dim = train_ds[0][0].shape[1]
    model = build_model(name, in_dim, len(classes), pre["seq_len"], **model_kwargs).to(device)
    print(f"model {name}: {count_parameters(model) / 1e6:.3f}M parameters, device {device}")

    weight = None
    if tr.get("class_weights"):
        weight = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])

    meta = {"model_name": name, "model_kwargs": model_kwargs, "in_dim": in_dim,
            "classes": classes, "preprocess": pre}
    select_on = "val" if "val" in eval_loaders else None
    best_score, history = -1.0, []
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        t0, total_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if tr.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        record = {"epoch": epoch, "loss": total_loss / seen, "train_acc": correct / seen}
        if select_on:
            y_true, probs = predict(model, eval_loaders[select_on], device)
            m = classification_metrics(y_true, probs.argmax(1), classes)
            record.update(val_acc=m["accuracy"], val_f1=m["f1"])
            score = m["f1"]
        else:
            score = epoch  # no validation split: keep the last epoch
        history.append(record)
        if score > best_score:
            best_score = score
            save_checkpoint(out_dir / "best.pt", model, meta)
        log_every = tr.get("log_every", 1)
        if epoch % log_every == 0 or epoch == tr["epochs"]:
            print(f"epoch {epoch:3d}  loss {record['loss']:.4f}  train_acc {record['train_acc']:.3f}"
                  + (f"  val_acc {record['val_acc']:.3f}  val_f1 {record['val_f1']:.3f}" if select_on else "")
                  + f"  ({time.time() - t0:.1f}s)", flush=True)
    save_checkpoint(out_dir / "last.pt", model, meta)

    results = {"history": history}
    if "test" in eval_loaders:
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device,
                                         weights_only=True)["state_dict"])
        y_true, probs = predict(model, eval_loaders["test"], device)
        results["test"] = classification_metrics(y_true, probs.argmax(1), classes)
        print("\nTEST (best checkpoint)\n" + format_metrics(results["test"], classes))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True, help="run directory for checkpoints and metrics")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.epochs=50")
    args = ap.parse_args()
    run_training(load_config(args.config, args.overrides), args.out)


if __name__ == "__main__":
    main()
