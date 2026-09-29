"""Prepare the UR Fall Detection dataset (URFD) the way Lin et al. 2021 used it.

URFD (https://fenix.ur.edu.pl/~mkepski/ds/uf.html, CC BY-NC-SA 4.0, non-commercial):
30 falls (two Kinect views) and 40 daily activities (one view) at 30 fps, with a
per-frame posture label for camera 0: -1 not lying, 0 falling, 1 lying on the ground.

    python -m falldet.urfd --out data/urfd
    python -m falldet.extract_poses --videos data/urfd/videos --segments data/urfd/segments.csv \\
        --out data/poses/urfd --resize-width 0

Steps:
  1. download every sequence's mp4 (depth | RGB side by side, RGB is 320x240) or, with
     --rgb-zip, its 640x480 RGB PNG frames (about 60 MB per sequence instead of ~1.3 MB)
  2. write the RGB part as <out>/videos/<sequence>-cam<k>.mp4
  3. cut each sequence into windows of --window frames (100 in the paper) every --stride
     frames and label them (see --fall-rule); daily activities are always no_fall
  4. optionally keep as many no_fall as fall windows (--balance, Table 1 of the paper)

To try other windowings without running the pose model again, extract poses of whole
videos once and cut the windows from them:

    python -m falldet.extract_poses --videos data/urfd/videos --segments data/urfd/full_videos.csv \
        --out data/poses/urfd_full --resize-width 0
    python -m falldet.urfd --from-poses data/poses/urfd_full --out-poses data/poses/urfd_s5_all \
        --stride 5 --no-balance
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"
FALLS = [f"fall-{i:02d}" for i in range(1, 31)]
ADLS = [f"adl-{i:02d}" for i in range(1, 41)]
LABEL_FILES = ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")


def download(name: str, dest: Path, retries: int = 3) -> Path:
    path = dest / name
    if path.exists() and path.stat().st_size > 0:
        return path
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/{name}", timeout=120) as resp:
                data = resp.read()
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.rename(path)
            return path
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return path


def read_labels(raw: Path) -> dict[str, dict[int, int]]:
    """{sequence: {frame (1-based): label}} from the cam0 feature files."""
    labels: dict[str, dict[int, int]] = {}
    for name in LABEL_FILES:
        with open(raw / name, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                labels.setdefault(row[0], {})[int(row[1])] = int(row[2])
    return labels


def write_rgb_video(src: Path, dst: Path, from_zip: bool) -> int:
    """Write the RGB stream of one sequence as an mp4; returns its number of frames."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    if from_zip:
        with zipfile.ZipFile(src) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
            frames = (cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR) for n in names)
            for frame in frames:
                writer = writer or cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                   (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                n += 1
    else:
        cap = cv2.VideoCapture(str(src))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = frame[:, frame.shape[1] // 2:]  # right half; the left half is depth
            writer = writer or cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                               (rgb.shape[1], rgb.shape[0]))
            writer.write(rgb)
            n += 1
        cap.release()
    if writer is not None:
        writer.release()
    return n


def window_label(frame_labels: list[int], is_fall: bool, rule: str) -> str | None:
    """fall / no_fall for one window, or None to drop it."""
    if not is_fall:
        return "no_fall"
    if rule == "sequence":
        return "fall"
    falling, lying = 0 in frame_labels, 1 in frame_labels
    if rule == "event":  # the falling motion itself must be inside the window
        return "fall" if falling else ("no_fall" if not lying else None)
    return "fall" if falling or lying else "no_fall"  # "down"


def make_segments(sequences: list[tuple[str, int, int]], labels: dict[str, dict[int, int]],
                  window: int, stride: int, rule: str) -> list[dict]:
    rows = []
    for seq, cam, n in sequences:
        per_frame = [labels.get(seq, {}).get(t + 1, -1) for t in range(n)]
        starts = list(range(0, max(n - window, 0) + 1, stride))
        for start in starts:
            end = min(start + window, n)
            label = window_label(per_frame[start:end], seq.startswith("fall"), rule)
            if label is not None:
                rows.append({"video": f"{seq}-cam{cam}.mp4", "label": label, "start": start, "end": end,
                             "group": seq})
    return rows


def balance(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    n = min(len(v) for v in by_label.values())
    out = []
    for items in by_label.values():
        out += [items[i] for i in sorted(rng.permutation(len(items))[:n])]
    return out


def cut_pose_windows(full_dir: Path, labels: dict[str, dict[int, int]], out_dir: Path, window: int,
                     stride: int, rule: str, do_balance: bool, seed: int, cams: list[int]) -> dict:
    """Write one .npz per window, cut from whole-video pose files ``<seq>-cam<k>_0_<n>.npz``."""
    clips = {}
    for path in sorted(full_dir.rglob("*.npz")):
        seq_cam = path.stem.rsplit("_", 2)[0]            # e.g. fall-01-cam0
        seq, cam = seq_cam.rsplit("-cam", 1)
        if int(cam) in cams:
            clips[(seq, int(cam))] = path
    sequences = []
    for (seq, cam), path in clips.items():
        with np.load(path) as z:
            sequences.append((seq, cam, len(z["keypoints"])))
    rows = make_segments(sequences, labels, window, stride, rule)
    total = {k: sum(r["label"] == k for r in rows) for k in ("fall", "no_fall")}
    if do_balance:
        rows = balance(rows, seed)
    cache: dict[Path, dict] = {}
    for r in rows:
        seq, cam = r["video"][:-4].rsplit("-cam", 1)
        src = clips[(seq, int(cam))]
        if src not in cache:
            with np.load(src) as z:
                cache[src] = {k: z[k] for k in z.files}
        z = cache[src]
        dst = out_dir / r["label"] / f"{seq}-cam{cam}_{r['start']}_{r['end']}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, keypoints=z["keypoints"][r["start"]:r["end"]], fps=z["fps"], size=z["size"],
                            group=seq, source=z["source"])
    return {"total": total, "kept": {k: sum(r["label"] == k for r in rows) for k in ("fall", "no_fall")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/urfd")
    ap.add_argument("--cams", type=int, nargs="+", default=[0],
                    help="0 = side view (falls + ADL); 1 = ceiling view exists for falls only, so adding it "
                         "lets a model learn 'ceiling view means fall'. The paper used both.")
    ap.add_argument("--rgb-zip", action="store_true", help="use the 640x480 PNG frames instead of the mp4")
    ap.add_argument("--window", type=int, default=100, help="frames per sample (paper: 100)")
    ap.add_argument("--stride", type=int, default=10, help="frames between window starts")
    ap.add_argument("--fall-rule", choices=["down", "event", "sequence"], default="down",
                    help="label of a window from a fall video: down = fall if it contains falling or lying "
                         "frames; event = only if it contains the falling motion (windows of lying only are "
                         "dropped); sequence = every window, as the paper's grouping")
    ap.add_argument("--no-balance", action="store_true", help="keep all windows instead of equal fall/no_fall")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-poses", help="cut windows from whole-video pose files in this directory")
    ap.add_argument("--out-poses", help="with --from-poses: where to write the window pose files")
    args = ap.parse_args()

    out = Path(args.out)
    raw, videos = out / "raw", out / "videos"
    for name in LABEL_FILES:
        download(name, raw)
    labels = read_labels(raw)
    if args.from_poses:
        if not args.out_poses:
            ap.error("--from-poses needs --out-poses")
        counts = cut_pose_windows(Path(args.from_poses), labels, Path(args.out_poses), args.window, args.stride,
                                  args.fall_rule, not args.no_balance, args.seed, args.cams)
        print(f"windows {counts['total']} -> kept {counts['kept']}; wrote {args.out_poses}")
        return

    sequences = [(s, c) for s in FALLS for c in args.cams] + [(s, 0) for s in ADLS]
    built = []
    for seq, cam in tqdm(sequences, desc="sequences"):
        dst = videos / f"{seq}-cam{cam}.mp4"
        if dst.exists():
            cap = cv2.VideoCapture(str(dst))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            name = f"{seq}-cam{cam}-rgb.zip" if args.rgb_zip else f"{seq}-cam{cam}.mp4"
            n = write_rgb_video(download(name, raw), dst, args.rgb_zip)
        built.append((seq, cam, n))

    rows = make_segments(built, labels, args.window, args.stride, args.fall_rule)
    total = {k: sum(r["label"] == k for r in rows) for k in ("fall", "no_fall")}
    if not args.no_balance:
        rows = balance(rows, args.seed)
    with open(out / "segments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label", "start", "end", "group"])
        writer.writeheader()
        writer.writerows(rows)
    kept = {k: sum(r["label"] == k for r in rows) for k in ("fall", "no_fall")}
    print(f"{len(built)} videos in {videos}; windows {total} -> kept {kept}; wrote {out / 'segments.csv'}")


if __name__ == "__main__":
    main()
