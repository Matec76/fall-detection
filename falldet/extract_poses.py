"""Extract 2D pose tracks from videos into ``<out>/[<split>/]<class>/<clip>.npz``.

Two input layouts are supported:
  * trimmed clips:   --videos data/videos       with data/videos/<class>/*.mp4
  * untrimmed video: --videos data/raw --segments segments.csv
                     CSV columns: video,label and either start,end (frame indices, end
                     exclusive) or start_sec,end_sec (seconds). Optional columns:
                     split (train/val/test -> sub-directory) and group (clips of one
                     group never cross splits). ``video`` is relative to --videos.
                     ``falldet.omnifall`` writes this format for OmniFall.

Each video is decoded once, however many segments it has.

Backends:
  * yolo       run an Ultralytics YOLO pose model on every frame (default)
  * alphapose  import results produced by AlphaPose (the paper's best estimator);
               --alphapose-dir must mirror the video paths with .json files
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .pose import YoloPoseEstimator, load_alphapose_json

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def video_info(path: Path) -> tuple[float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, size


def processed_size(size: tuple[int, int], resize_width: int | None) -> tuple[int, int]:
    """Frame size the keypoints refer to after the optional downscale in ``run_yolo``."""
    w, h = size
    if resize_width and w > resize_width:
        return resize_width, round(h * resize_width / w)
    return w, h


def run_yolo(estimator: YoloPoseEstimator, path: Path, start: int, end: int | None,
             step: int, resize_width: int | None) -> np.ndarray:
    """Poses of frames start, start+step, ... before ``end`` (None = end of video)."""
    cap = cv2.VideoCapture(str(path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    estimator.reset()
    frames = []
    t = start
    while end is None or t < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (t - start) % step == 0:
            if resize_width and frame.shape[1] > resize_width:
                h = round(frame.shape[0] * resize_width / frame.shape[1])
                frame = cv2.resize(frame, (resize_width, h), interpolation=cv2.INTER_AREA)
            frames.append(estimator(frame))
        t += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 17, 3), np.float32)


def list_videos(args) -> list[dict]:
    """One entry per video, each carrying its labelled segments."""
    root = Path(args.videos)
    videos: dict[str, dict] = {}
    if args.segments:
        with open(args.segments, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                entry = videos.setdefault(row["video"], {"video": root / row["video"], "rel": Path(row["video"]),
                                                         "segments": []})
                in_sec = "start_sec" in row
                entry["segments"].append({
                    "label": row["label"],
                    "start": float(row["start_sec"] if in_sec else row["start"]),
                    "end": float(row["end_sec"] if in_sec else row["end"]),
                    "unit": "sec" if in_sec else "frame",
                    "split": row.get("split") or "",
                    "group": row.get("group") or Path(row["video"]).with_suffix("").as_posix(),
                })
    else:
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for video in sorted(class_dir.iterdir()):
                if video.suffix.lower() in VIDEO_EXTS:
                    rel = video.relative_to(root)
                    videos[str(rel)] = {"video": video, "rel": rel, "segments": [{
                        "label": class_dir.name, "start": 0, "end": None, "unit": "frame", "split": "",
                        "group": video.stem, "name": video.stem}]}
    if not videos:
        raise FileNotFoundError(f"No videos found under {root}")
    return list(videos.values())


def to_frames(seg: dict, fps: float) -> tuple[int, int | None]:
    if seg["unit"] == "sec":
        return int(round(seg["start"] * fps)), int(round(seg["end"] * fps))
    return int(seg["start"]), None if seg["end"] is None else int(seg["end"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="video root directory")
    ap.add_argument("--out", required=True, help="output directory for .npz pose files")
    ap.add_argument("--segments", help="CSV of labelled segments inside untrimmed videos")
    ap.add_argument("--backend", choices=["yolo", "alphapose"], default="yolo")
    ap.add_argument("--weights", default="yolo11m-pose.pt", help="YOLO pose weights")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0, mps")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--alphapose-dir", help="AlphaPose json root mirroring --videos")
    ap.add_argument("--frame-step", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--resize-width", type=int, default=960,
                    help="downscale wider frames first (paper used 960x540); 0 disables")
    ap.add_argument("--min-frames", type=int, default=4, help="skip clips with fewer kept frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.backend == "alphapose" and not args.alphapose_dir:
        ap.error("--alphapose-dir is required with --backend alphapose")

    videos = list_videos(args)
    estimator = YoloPoseEstimator(args.weights, args.device, imgsz=args.imgsz) if args.backend == "yolo" else None
    out_root = Path(args.out)
    step = args.frame_step
    written = missing = short = 0
    for entry in tqdm(videos, desc="videos"):
        if not entry["video"].exists():
            missing += 1
            continue
        fps, size = video_info(entry["video"])
        stem = entry["rel"].with_suffix("").as_posix().replace("/", "__")
        jobs = []
        for seg in entry["segments"]:
            start, end = to_frames(seg, fps)
            name = seg.get("name") or f"{stem}_{start}_{end}"
            out_path = out_root / seg["split"] / seg["label"] / f"{name}.npz"
            if args.overwrite or not out_path.exists():
                jobs.append((seg, start, end, out_path))
        if not jobs:
            continue

        # Decode only the frame range the pending segments need, once.
        lo = min(j[1] for j in jobs)
        hi = None if any(j[2] is None for j in jobs) else max(j[2] for j in jobs)
        if estimator is not None:
            kpts = run_yolo(estimator, entry["video"], lo, hi, step, args.resize_width or None)
            size = processed_size(size, args.resize_width or None)
        else:
            json_path = (Path(args.alphapose_dir) / entry["rel"]).with_suffix(".json")
            kpts = load_alphapose_json(json_path)[lo:hi:step]

        for seg, start, end, out_path in jobs:
            a = -(-(start - lo) // step)
            b = len(kpts) if end is None else -(-(end - lo) // step)
            clip = kpts[a:b]
            if len(clip) < args.min_frames:  # e.g. segment past the end of the video
                short += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, keypoints=clip.astype(np.float32), fps=fps / step,
                                size=np.array(size), group=seg["group"], source=str(entry["video"]))
            written += 1
    print(f"Wrote {written} pose clips to {out_root}"
          + (f"; {missing} videos not found" if missing else "")
          + (f"; {short} clips skipped (< {args.min_frames} frames)" if short else ""))


if __name__ == "__main__":
    main()
