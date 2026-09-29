"""Pose-sequence datasets and leakage-safe train/val/test splitting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from .preprocess import prepare_sequence


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    group: str  # clips sharing a group (same source video) never cross splits


def load_keypoints(path: str | Path) -> np.ndarray:
    with np.load(path) as z:
        return z["keypoints"].astype(np.float32)


def load_clip(path: str | Path) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Keypoints plus the (width, height) of the frames they were measured in, if stored."""
    with np.load(path) as z:
        size = tuple(int(v) for v in z["size"]) if "size" in z else None
        return z["keypoints"].astype(np.float32), size


def scan_pose_dir(root: str | Path, classes: list[str]) -> list[Sample]:
    """Collect ``<root>/<class>/*.npz`` files written by ``falldet.extract_poses``."""
    root = Path(root)
    samples = []
    for label, name in enumerate(classes):
        for path in sorted((root / name).glob("*.npz")):
            with np.load(path) as z:
                group = str(z["group"]) if "group" in z else path.stem
            samples.append(Sample(str(path), label, group))
    if not samples:
        raise FileNotFoundError(f"No .npz pose files found under {root}/<class>/ for classes {classes}")
    return samples


def _holdout(samples: list[Sample], frac: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Stratified, group-aware split holding out roughly ``frac`` of the samples."""
    if frac <= 0:
        return samples, []
    n_splits = max(2, round(1 / frac))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [s.label for s in samples]
    groups = [s.group for s in samples]
    keep_idx, held_idx = next(splitter.split(np.zeros(len(samples)), labels, groups))
    return [samples[i] for i in keep_idx], [samples[i] for i in held_idx]


def split_samples(samples: list[Sample], test_size: float, val_size: float,
                  seed: int, group_split: bool = True) -> dict[str, list[Sample]]:
    """``group_split=False`` splits clip by clip, like the random 80/20 split of Lin et al.;
    overlapping windows of one video then land in both train and test (optimistic)."""
    if not group_split:
        samples = [Sample(s.path, s.label, s.path) for s in samples]
    train, test = _holdout(samples, test_size, seed)
    train, val = _holdout(train, val_size, seed)
    return {"train": train, "val": val, "test": test}


def subsample_per_class(samples: list[Sample], per_class: int | None, seed: int) -> list[Sample]:
    """Keep at most ``per_class`` random samples of each class (for Fig. 9/10 style sweeps)."""
    if per_class is None:
        return samples
    rng = np.random.default_rng(seed)
    out = []
    for label in sorted({s.label for s in samples}):
        items = [s for s in samples if s.label == label]
        pick = rng.permutation(len(items))[:per_class]
        out.extend(items[i] for i in sorted(pick))
    return out


def samples_to_json(samples: list[Sample]) -> list[dict]:
    return [asdict(s) for s in samples]


def samples_from_json(items: list[dict]) -> list[Sample]:
    return [Sample(**item) for item in items]


class PoseSequenceDataset(Dataset):
    """Loads every clip into memory and returns ((T, features) tensor, label)."""

    def __init__(self, samples: list[Sample], pre: dict, augment: dict | None = None, seed: int = 0):
        self.samples = samples
        self.clips = [load_clip(s.path) for s in samples]
        self.pre = pre
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        kpts, size = self.clips[i]
        x = prepare_sequence(kpts, self.pre, self.augment, self.rng, size)
        return torch.from_numpy(x), self.samples[i].label
