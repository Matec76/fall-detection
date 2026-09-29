"""2D human pose estimation backends.

Every backend yields one person per frame as a (K, 3) array of (x, y, confidence)
in pixel coordinates; frames without a detected person are all zeros. When
several people are visible, the one overlapping the previous frame's choice is
kept, otherwise the largest confident detection (a simple single-person tracker).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import NUM_KEYPOINTS


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and an (N, 4) array of xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def select_person(boxes: np.ndarray, scores: np.ndarray, prev_box: np.ndarray | None,
                  iou_thr: float = 0.3) -> int:
    if prev_box is not None:
        ious = box_iou(prev_box, boxes)
        if ious.max() >= iou_thr:
            return int(ious.argmax())
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(scores * areas))


class YoloPoseEstimator:
    """Ultralytics YOLO pose model (single-stage, top-down-like person boxes + keypoints)."""

    def __init__(self, weights: str = "yolo11m-pose.pt", device: str | None = None,
                 det_conf: float = 0.25, imgsz: int = 960):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.det_conf = det_conf
        self.imgsz = imgsz
        self._prev_box: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_box = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(frame, conf=self.det_conf, imgsz=self.imgsz,
                            device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return np.zeros((NUM_KEYPOINTS, 3), np.float32)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()
        i = select_person(boxes, scores, self._prev_box)
        self._prev_box = boxes[i]
        return kpts[i].astype(np.float32)


def _keypoint_box(kpts: np.ndarray) -> np.ndarray:
    vis = kpts[:, 2] > 0
    pts = kpts[vis, :2] if vis.any() else kpts[:, :2]
    return np.array([*pts.min(0), *pts.max(0)], np.float32)


def load_alphapose_json(path: str | Path, num_frames: int | None = None) -> np.ndarray:
    """Read AlphaPose ``alphapose-results.json`` (video mode) into a (T, 17, 3) array.

    The paper's best estimator was AlphaPose; run it separately and import its
    results here. Halpe-26/136 outputs are cut to their first 17 (COCO) joints.
    """
    with open(path, encoding="utf-8") as f:
        detections = json.load(f)
    by_frame: dict[int, list[dict]] = {}
    for det in detections:
        by_frame.setdefault(int(Path(str(det["image_id"])).stem), []).append(det)
    n = num_frames if num_frames is not None else (max(by_frame) + 1 if by_frame else 0)

    out = np.zeros((n, NUM_KEYPOINTS, 3), np.float32)
    prev_box = None
    for t in range(n):
        people = by_frame.get(t)
        if not people:
            continue
        kpts = [np.asarray(p["keypoints"], np.float32).reshape(-1, 3)[:NUM_KEYPOINTS] for p in people]
        boxes = []
        for p, k in zip(people, kpts):
            if "box" in p:  # AlphaPose stores [x, y, w, h]
                x, y, w, h = p["box"]
                boxes.append([x, y, x + w, y + h])
            else:
                boxes.append(_keypoint_box(k))
        boxes = np.asarray(boxes, np.float32)
        scores = np.asarray([p.get("score", 1.0) for p in people], np.float32)
        i = select_person(boxes, scores, prev_box)
        prev_box = boxes[i]
        out[t] = kpts[i]
    return out
