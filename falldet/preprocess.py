"""Turn raw (T, K, 3) keypoint tracks into fixed-length model inputs.

The steps follow Lin et al. 2021 (Sec. 2.3) and can be switched through the
``preprocess`` section of the config:

    joints       coco17 (all 17 YOLO/AlphaPose joints) | body15 (the paper's 15 joints)
    interpolate  linear interpolation of missing joints over time (Sec. 2.3.3)
    norm_mode    none | minmax (Eq. 1) | uniform | rp (relative position, Eq. 2-7)
    use_conf     keep the keypoint confidence as a feature (the paper drops it)
    extra        whole-body motion features appended to every frame (not in the paper):
                 hip_y, hip_vy, neck_vy, aspect, torso (see ``motion_features``)
"""
from __future__ import annotations

import numpy as np

from .constants import BODY15, BODY15_FLIP_PAIRS, BODY15_MID_HIP, FLIP_PAIRS

PREPROCESS_DEFAULTS = {"seq_len": 100, "norm_mode": "rp", "conf_thr": 0.3, "joints": "body15",
                       "interpolate": True, "max_missing": 0.67, "use_conf": False,
                       "rp_size": [640, 480], "rp_scale": True, "extra": []}
EXTRA_DIMS = {"hip_y": 1, "hip_vy": 1, "neck_vy": 1, "aspect": 1, "torso": 2}


def mask_low_confidence(kpts: np.ndarray, conf_thr: float) -> np.ndarray:
    """Zero out keypoints whose confidence is below ``conf_thr`` (treated as missing)."""
    out = kpts.astype(np.float32, copy=True)
    out[out[..., 2] < conf_thr] = 0.0
    return out


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both = (a[:, 2] > 0) & (b[:, 2] > 0)
    out = np.zeros_like(a)
    out[both, :2] = (a[both, :2] + b[both, :2]) / 2
    out[both, 2] = np.minimum(a[both, 2], b[both, 2])
    return out


def to_body15(kpts: np.ndarray) -> np.ndarray:
    """COCO-17 (T, 17, 3) -> the paper's 15 BODY_25 joints (T, 15, 3)."""
    out = np.zeros((len(kpts), len(BODY15), 3), np.float32)
    for j, (_, coco) in enumerate(BODY15):
        if coco is not None:
            out[:, j] = kpts[:, coco]
    out[:, 1] = _midpoint(kpts[:, 5], kpts[:, 6])                # neck
    out[:, BODY15_MID_HIP] = _midpoint(kpts[:, 11], kpts[:, 12])  # mid-hip
    return out


def select_joints(kpts: np.ndarray, joints: str) -> tuple[np.ndarray, tuple, int | None]:
    """Return the chosen joints, their left/right flip pairs and the RP reference joint."""
    if joints == "coco17":
        return kpts, FLIP_PAIRS, None
    if joints == "body15":
        return to_body15(kpts), BODY15_FLIP_PAIRS, BODY15_MID_HIP
    raise ValueError(f"Unknown joints {joints!r}")


def interpolate_missing(kpts: np.ndarray, max_missing: float = 0.67) -> np.ndarray:
    """Fill missing joints by linear interpolation in time (Eq. 8).

    Only gaps between two detected frames are filled, and a joint missing in more than
    ``max_missing`` of the frames is left as it is, as in the paper.
    """
    out = kpts.copy()
    t = np.arange(len(kpts))
    for j in range(kpts.shape[1]):
        vis = kpts[:, j, 2] > 0
        n_vis = int(vis.sum())
        if n_vis < 2 or n_vis == len(kpts) or 1 - n_vis / len(kpts) > max_missing:
            continue
        known = np.flatnonzero(vis)
        gap = ~vis & (t > known[0]) & (t < known[-1])
        for c in range(3):
            out[gap, j, c] = np.interp(t[gap], known, kpts[known, j, c])
    return out


def _ffill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Carry the last valid value forward (and the first one backward); 0 if none is valid."""
    out = np.zeros_like(values)
    if not valid.any():
        return out
    idx = np.where(valid, np.arange(len(values)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = values[idx]
    first = np.flatnonzero(valid)[0]
    out[:first] = values[first]
    return out


def _neck_hip(kpts: np.ndarray, joints: str) -> tuple[np.ndarray, np.ndarray]:
    if joints == "body15":
        return kpts[:, 1], kpts[:, BODY15_MID_HIP]
    return _midpoint(kpts[:, 5], kpts[:, 6]), _midpoint(kpts[:, 11], kpts[:, 12])


def motion_features(kpts: np.ndarray, joints: str, names: list[str],
                    frame_size: tuple[int, int] | None) -> np.ndarray:
    """Per-frame whole-body features that relative-position normalisation removes.

    hip_y    height of the mid-hip in the frame (0 = top, 1 = bottom)
    hip_vy   its change per frame x 10 (positive = moving down, i.e. falling)
    neck_vy  the same for the neck
    aspect   log(height / width) of the skeleton's bounding box (tall > 0, lying < 0)
    torso    sin and cos of the neck-to-hip angle from vertical (upright: 0, 1)
    """
    if not names:
        return np.zeros((len(kpts), 0), np.float32)
    w, h = frame_size or (1.0, 1.0)
    neck, hip = _neck_hip(kpts, joints)
    hip_ok, neck_ok = hip[:, 2] > 0, neck[:, 2] > 0
    hip_y = _ffill(hip[:, 1] / h, hip_ok)
    neck_y = _ffill(neck[:, 1] / h, neck_ok)
    cols = []
    for name in names:
        if name == "hip_y":
            cols.append(hip_y[:, None])
        elif name == "hip_vy":
            cols.append(np.diff(hip_y, prepend=hip_y[:1])[:, None] * 10)
        elif name == "neck_vy":
            cols.append(np.diff(neck_y, prepend=neck_y[:1])[:, None] * 10)
        elif name == "aspect":
            vis = kpts[..., 2] > 0
            asp = np.zeros(len(kpts), np.float32)
            for t in range(len(kpts)):
                if vis[t].sum() >= 2:
                    pts = kpts[t, vis[t], :2] / np.array([w, h])
                    ext = np.maximum(pts.max(0) - pts.min(0), 1e-3)
                    asp[t] = np.clip(np.log(ext[1] / ext[0]), -2, 2)
            cols.append(asp[:, None])
        elif name == "torso":
            ok = hip_ok & neck_ok
            dx, dy = (neck[:, 0] - hip[:, 0]) / w, (neck[:, 1] - hip[:, 1]) / h
            ang = np.arctan2(dx, -dy)
            sin, cos = _ffill(np.sin(ang), ok), _ffill(np.cos(ang), ok)
            cols.append(np.stack([sin, cos], 1))
        else:
            raise ValueError(f"Unknown extra feature {name!r}; choose from {sorted(EXTRA_DIMS)}")
    return np.concatenate(cols, 1).astype(np.float32)


def resample(kpts: np.ndarray, seq_len: int) -> np.ndarray:
    """Uniformly sample ``seq_len`` frames from a clip of any length."""
    if len(kpts) == 0:
        return np.zeros((seq_len, *kpts.shape[1:]), np.float32)
    idx = np.linspace(0, len(kpts) - 1, seq_len).round().astype(int)
    return kpts[idx]


def normalize(kpts: np.ndarray, mode: str = "minmax", frame_size: tuple[int, int] | None = None,
              ref_joint: int | None = None, rp_size: tuple[int, int] = (640, 480),
              rp_scale: bool = True) -> np.ndarray:
    """Normalise x and y coordinates; missing keypoints (confidence 0) stay 0.

    none     pixel coordinates unchanged
    minmax   Eq. (1): scale each axis to [0, 1] over the whole sequence, using detected
             keypoints only (the paper's variant also counted the zeros of missing points)
    uniform  like minmax but one scale for both axes, so the body's aspect ratio survives
    rp       Eq. (2)-(7): resize the frame to ``rp_size`` and move the reference joint
             (mid-hip) of every frame to the image centre; frames without a mid-hip use
             the mean of their detected joints. With ``rp_scale`` the result is divided by
             ``rp_size`` so it lies in [0, 1]: raw pixel values (~100-500) saturate the
             recurrent gates and the models stop learning
    """
    out = kpts.astype(np.float32, copy=True)
    vis = out[..., 2] > 0
    if mode == "none" or not vis.any():
        return out
    if mode == "rp":
        if frame_size is None:
            raise ValueError("norm_mode 'rp' needs the frame size; re-run falldet.extract_poses")
        out[..., 0] *= rp_size[0] / frame_size[0]
        out[..., 1] *= rp_size[1] / frame_size[1]
        centre = np.array(rp_size, np.float32) / 2
        for f in range(len(out)):
            if not vis[f].any():
                continue
            if ref_joint is not None and vis[f, ref_joint]:
                ref = out[f, ref_joint, :2]
            else:
                ref = out[f, vis[f], :2].mean(0)
            out[f, vis[f], :2] += centre - ref
        if rp_scale:
            out[..., :2] /= np.array(rp_size, np.float32)
        return out
    lo = np.array([out[..., a][vis].min() for a in (0, 1)])
    span = np.array([out[..., a][vis].max() for a in (0, 1)]) - lo
    if mode == "uniform":
        span[:] = span.max()
    elif mode != "minmax":
        raise ValueError(f"Unknown norm_mode {mode!r}")
    span = np.maximum(span, 1e-6)
    for a in (0, 1):
        out[..., a] = np.where(vis, (out[..., a] - lo[a]) / span[a], 0.0)
    return out


def flip_horizontal(kpts: np.ndarray, pairs: tuple = FLIP_PAIRS, width: float = 1.0) -> np.ndarray:
    """Mirror keypoints left-right (x -> width - x) and swap left/right joints."""
    out = kpts.copy()
    vis = out[..., 2] > 0
    out[..., 0] = np.where(vis, width - out[..., 0], 0.0)
    for left, right in pairs:
        out[:, [left, right]] = out[:, [right, left]]
    return out


def feature_dim(pre: dict, num_keypoints: int = 17) -> int:
    pre = {**PREPROCESS_DEFAULTS, **pre}
    k = 15 if pre["joints"] == "body15" else num_keypoints
    return k * (3 if pre["use_conf"] else 2) + sum(EXTRA_DIMS[n] for n in pre["extra"])


def prepare_sequence(kpts: np.ndarray, pre: dict, augment: dict | None = None,
                     rng: np.random.Generator | None = None,
                     frame_size: tuple[int, int] | None = None) -> np.ndarray:
    """Full preprocessing pipeline; returns a (seq_len, features) float32 array."""
    pre = {**PREPROCESS_DEFAULTS, **pre}
    seq_len, mode = pre["seq_len"], pre["norm_mode"]
    kpts = mask_low_confidence(kpts, pre["conf_thr"])
    kpts, pairs, ref_joint = select_joints(kpts, pre["joints"])
    if pre["interpolate"]:
        kpts = interpolate_missing(kpts, pre["max_missing"])
    if augment:
        rng = rng or np.random.default_rng()
    if augment and len(kpts) > 1:
        keep = max(2, int(round(len(kpts) * rng.uniform(augment.get("crop_min", 1.0), 1.0))))
        start = rng.integers(0, len(kpts) - keep + 1)
        kpts = kpts[start:start + keep]
    rp_size = tuple(pre["rp_size"])
    kpts = resample(kpts, seq_len)
    extra = motion_features(kpts, pre["joints"], pre["extra"], frame_size)
    kpts = normalize(kpts, mode, frame_size, ref_joint, rp_size, pre["rp_scale"])
    if augment:
        if rng.random() < augment.get("flip_prob", 0.0):
            if "torso" in pre["extra"]:  # mirroring flips the sign of the torso's sideways lean
                col = sum(EXTRA_DIMS[n] for n in pre["extra"][:pre["extra"].index("torso")])
                extra[:, col] *= -1
            width = {"rp": 1.0 if pre["rp_scale"] else rp_size[0],
                     "none": frame_size[0] if frame_size else 1.0}.get(mode, 1.0)
            kpts = flip_horizontal(kpts, pairs, width)
        std = augment.get("noise_std", 0.0)
        if std > 0:
            vis = kpts[..., 2:3] > 0
            kpts[..., :2] += rng.normal(0.0, std, kpts[..., :2].shape).astype(np.float32) * vis
    if not pre["use_conf"]:
        kpts = kpts[..., :2]
    return np.concatenate([kpts.reshape(seq_len, -1), extra], 1).astype(np.float32)
