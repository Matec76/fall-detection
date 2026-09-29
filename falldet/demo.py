"""Real-time fall detection on a video file or webcam.

    python -m falldet.demo --run runs/transformer --source video.mp4 --save out.mp4
    python -m falldet.demo --run runs/transformer --source 0          # webcam

A sliding window of the last --window-sec seconds of poses is classified every
--stride frames. A fall alert fires after --consecutive windows in a row give the
fall class a probability of at least --fall-thr.
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from .constants import SKELETON
from .metrics import FALL_CLASSES
from .models import load_checkpoint
from .pose import YoloPoseEstimator
from .preprocess import prepare_sequence
from .train import resolve_device


def draw_pose(frame: np.ndarray, kpts: np.ndarray, conf_thr: float, color=(0, 255, 0)) -> None:
    for a, b in SKELETON:
        if kpts[a, 2] >= conf_thr and kpts[b, 2] >= conf_thr:
            cv2.line(frame, tuple(map(int, kpts[a, :2])), tuple(map(int, kpts[b, :2])), color, 2)
    for x, y, c in kpts:
        if c >= conf_thr:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory written by falldet.train")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--source", required=True, help="video path or webcam index")
    ap.add_argument("--weights", default="yolo11m-pose.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--window-sec", type=float, default=None,
                    help="seconds of poses per prediction (default: seq_len frames of the model)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fall-class", default=None, help="default: 'fall' or 'falling_down'")
    ap.add_argument("--fall-thr", type=float, default=0.7)
    ap.add_argument("--consecutive", type=int, default=2)
    ap.add_argument("--hold-sec", type=float, default=3.0, help="keep the alert banner on screen")
    ap.add_argument("--save", help="write the annotated video here")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, meta = load_checkpoint(f"{args.run}/{args.checkpoint}", device)
    classes, pre = meta["classes"], meta["preprocess"]
    args.fall_class = args.fall_class or next((c for c in FALL_CLASSES if c in classes), None)
    if args.fall_class not in classes:
        raise SystemExit(f"--fall-class {args.fall_class!r} not in model classes {classes}")
    fall_idx = classes.index(args.fall_class)
    pose_device = None if device.type == "cpu" else ("mps" if device.type == "mps" else 0)
    estimator = YoloPoseEstimator(args.weights, pose_device)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    window = deque(maxlen=int(round(args.window_sec * fps)) if args.window_sec else pre["seq_len"])
    writer = None

    label, prob, streak, alert_until, frame_idx = "...", 0.0, 0, -1.0, 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kpts = estimator(frame)
        window.append(kpts)
        if frame_idx % args.stride == 0 and len(window) >= window.maxlen // 2:
            x = prepare_sequence(np.stack(window), pre, frame_size=(frame.shape[1], frame.shape[0]))
            with torch.no_grad():
                probs = torch.softmax(model(torch.from_numpy(x)[None].to(device)), 1)[0].cpu().numpy()
            label, prob = classes[int(probs.argmax())], float(probs.max())
            streak = streak + 1 if probs[fall_idx] >= args.fall_thr else 0
            video_t = frame_idx / fps
            if streak == args.consecutive:
                print(f"[{video_t:8.2f}s] FALL DETECTED (p={probs[fall_idx]:.2f})")
            if streak >= args.consecutive:
                alert_until = video_t + args.hold_sec

        draw_pose(frame, kpts, pre["conf_thr"])
        cv2.putText(frame, f"{label} {prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if frame_idx / fps <= alert_until:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)
            cv2.putText(frame, "FALL DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("fall detection", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    print(f"{frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-9):.1f} FPS)")


if __name__ == "__main__":
    main()
