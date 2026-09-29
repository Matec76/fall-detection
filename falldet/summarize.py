"""Collect the test metrics of many runs into one table (Markdown or CSV).

    python -m falldet.summarize runs/lin2021
    python -m falldet.summarize runs/lin2021 --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

# bal_acc = (sensitivity + specificity) / 2: comparable between balanced and imbalanced test sets
COLUMNS = ("run", "model", "norm", "hidden", "lr", "accuracy", "acc_sd", "bal_acc", "sensitivity", "specificity",
           "f1", "n_test")


def collect(root: Path) -> list[dict]:
    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        sd = ""
        if (run / "cv.json").exists():  # cross-validation: pooled test predictions of all folds
            cv = json.loads((run / "cv.json").read_text())
            test, sd = cv["pooled"], cv["fold_stats"]["accuracy"]["sd"]
        elif (run / "metrics.json").exists():
            test = json.loads((run / "metrics.json").read_text()).get("test")
        else:
            continue
        if not test:
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        model = cfg["model"]["name"]
        pre = cfg["preprocess"]
        norm = pre.get("norm_mode", "") + ("+interp" if pre.get("interpolate") else "")
        rows.append({"run": run.name, "model": model, "norm": norm,
                     "hidden": cfg["model"].get(model, {}).get("hidden", ""), "lr": cfg["train"]["lr"],
                     "accuracy": test["accuracy"], "acc_sd": sd,
                     "bal_acc": (test["sensitivity"] + test["specificity"]) / 2 if "sensitivity" in test else "",
                     "sensitivity": test.get("sensitivity", ""),
                     "specificity": test.get("specificity", ""), "f1": test["f1"],
                     "n_test": sum(c["support"] for c in test["per_class"].values())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding one sub-directory per run")
    ap.add_argument("--csv", help="also write the table here")
    args = ap.parse_args()

    rows = collect(Path(args.root))
    if not rows:
        raise SystemExit(f"No finished runs under {args.root}")
    pct = lambda v: f"{v * 100:.1f}" if isinstance(v, float) else str(v)  # noqa: E731
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(pct(r[c]) if c in ("accuracy", "acc_sd", "bal_acc", "sensitivity", "specificity", "f1") else str(r[c])
                                for c in COLUMNS) + " |")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
