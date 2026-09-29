"""Turn OmniFall annotations into a segment list for ``falldet.extract_poses``.

OmniFall (https://huggingface.co/datasets/simplexsigil2/omnifall) relabels eight staged
fall datasets, genuine in-the-wild falls (OOPS) and 12k synthetic videos (OF-Syn) with one
16-class taxonomy and official cross-subject (cs) / cross-view (cv) splits. Fetch the
videos with the ``omnifall`` package (``omnifall prepare le2i``); they land at
``{root}/{dataset}/video/{path}.mp4``, which is the layout this script writes paths for.

    # real, staged data with the official cross-subject split
    python -m falldet.omnifall --components le2i caucafall GMDCSA24 up_fall --out data/omnifall/staged_cs.csv
    # synthetic data (OF-Syn), training part only
    python -m falldet.omnifall --components of-syn --splits train --out data/omnifall/syn.csv
    # ... elderly (65+) synthetic subjects only
    python -m falldet.omnifall --components of-syn --splits train --age-groups elderly_65_plus --out data/omnifall/syn_elderly.csv
    # in-the-wild falls, test part only
    python -m falldet.omnifall --components OOPS --splits test --out data/omnifall/itw_test.csv

Every segment becomes one or more samples of exactly --window-sec seconds: shorter
segments are widened around their centre, longer ones are cut into consecutive windows.
This matches the sliding window that ``falldet.demo`` classifies.
"""
from __future__ import annotations

import argparse
import csv
import urllib.error
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main"
COMPONENTS = ("caucafall", "cmdfall", "edf", "GMDCSA24", "le2i", "mcfd", "occu", "up_fall", "OOPS", "of-syn")
AGE_GROUPS = ("toddlers_1_4", "children_5_12", "teenagers_13_17", "young_adults_18_34",
              "middle_aged_35_64", "elderly_65_plus")
OMNIFALL_LABELS = ("walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up",
                   "standing", "other", "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump")

# OmniFall label -> our class name. Labels missing from a map are dropped.
LABEL_MAPS = {
    # the four classes of the paper
    "paper4": {"fall": "falling_down", "stand_up": "standing_up", "lie_down": "lying_down", "walk": "walking"},
    # OmniFall's ten core classes, unchanged
    "core10": {name: name for name in OMNIFALL_LABELS[:10]},
    # fall vs no fall, as in Lin et al. 2021: falling and lying on the floor afterwards are falls
    "binary": {name: "fall" if name in ("fall", "fallen") else "no_fall" for name in OMNIFALL_LABELS},
}


def fetch(rel: str, cache: Path) -> Path | None:
    """Download ``rel`` from the OmniFall Hub repo once; None if it does not exist."""
    path = cache / rel
    if path.exists():
        return path
    try:
        with urllib.request.urlopen(f"{HUB}/{rel}", timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_paths(component: str, split: str, part: str, cache: Path) -> set[str] | None:
    # OF-Syn splits sit directly in splits/syn/<split>/; the others have one folder per
    # component, lower-case for some (labels/GMDCSA24.csv vs splits/cs/gmdcsa24/).
    candidates = [f"splits/{split}/{part}.csv"] if component == "of-syn" else [
        f"splits/{split}/{name}/{part}.csv" for name in dict.fromkeys([component, component.lower()])]
    for rel in candidates:
        path = fetch(rel, cache)
        if path is not None:
            return {row["path"] for row in read_csv(path)}
    return None


def make_windows(start: float, end: float, window: float, max_chunks: int) -> list[tuple[float, float]]:
    """Widen a short segment to ``window`` seconds or cut a long one into windows."""
    if end - start <= window:
        centre = (start + end) / 2
        lo = max(0.0, centre - window / 2)
        return [(lo, lo + window)]
    n = min(int((end - start) // window), max_chunks)
    return [(start + k * window, start + (k + 1) * window) for k in range(n)]


def build_segments(components: list[str], split: str, syn_split: str, parts: list[str],
                   label_map: dict[str, str], window: float, min_sec: float, max_chunks: int,
                   cache: Path, age_groups: list[str] | None = None) -> list[dict]:
    rows = []
    for component in components:
        labels = fetch(f"labels/{component}.csv", cache)
        if labels is None:
            raise SystemExit(f"No OmniFall labels for component {component!r}; choose from {COMPONENTS}")
        split_dir = f"syn/{syn_split}" if component == "of-syn" else split
        members = {}
        for part in parts:
            paths = split_paths(component, split_dir, part, cache)
            if paths is None:
                print(f"warning: {component} has no {split_dir}/{part} split, skipped")
                continue
            members.update(dict.fromkeys(paths, part))
        kept = 0
        for seg in read_csv(labels):
            part = members.get(seg["path"])
            name = OMNIFALL_LABELS[int(seg["label"])]
            start, end = float(seg["start"]), float(seg["end"])
            if part is None or name not in label_map or end - start < min_sec:
                continue
            if age_groups and seg.get("age_group") not in age_groups:  # only OF-Syn has ages
                continue
            subject = int(seg["subject"])
            group = f"{seg['dataset']}/subject{subject}" if subject >= 0 else f"{seg['dataset']}/{seg['path']}"
            for lo, hi in make_windows(start, end, window, max_chunks):
                rows.append({"video": f"{seg['dataset']}/video/{seg['path']}.mp4", "label": label_map[name],
                             "start_sec": f"{lo:.3f}", "end_sec": f"{hi:.3f}", "split": part, "group": group})
                kept += 1
        print(f"{component}: {kept} samples")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", required=True, help=f"any of {', '.join(COMPONENTS)}")
    ap.add_argument("--out", required=True, help="segment CSV to write")
    ap.add_argument("--split", default="cs", choices=["cs", "cv"], help="official split for real data")
    ap.add_argument("--syn-split", default="random", choices=["random", "cross_age", "cross_ethnicity", "cross_bmi"],
                    help="official split for of-syn")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="which parts to keep")
    ap.add_argument("--label-map", default="paper4", choices=sorted(LABEL_MAPS))
    ap.add_argument("--window-sec", type=float, default=2.0, help="length of every sample")
    ap.add_argument("--min-sec", type=float, default=0.3, help="drop segments shorter than this")
    ap.add_argument("--max-chunks", type=int, default=5, help="max windows cut from one long segment")
    ap.add_argument("--age-groups", nargs="+", choices=AGE_GROUPS,
                    help="keep only these OF-Syn age groups (drops components without ages)")
    ap.add_argument("--cache", default="data/omnifall/meta", help="where annotation CSVs are cached")
    args = ap.parse_args()

    rows = build_segments(args.components, args.split, args.syn_split, args.splits, LABEL_MAPS[args.label_map],
                          args.window_sec, args.min_sec, args.max_chunks, Path(args.cache), args.age_groups)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label", "start_sec", "end_sec", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        counts[r["split"], r["label"]] = counts.get((r["split"], r["label"]), 0) + 1
    print(f"wrote {len(rows)} samples to {out}")
    for part in args.splits:
        print(f"  {part:<6}", {label: n for (p, label), n in sorted(counts.items()) if p == part})


if __name__ == "__main__":
    main()
