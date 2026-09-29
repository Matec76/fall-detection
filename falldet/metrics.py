"""Evaluation metrics: classification (incl. sensitivity / specificity) and pose quality (ACV / AMV)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


FALL_CLASSES = ("fall", "falling_down")


def classification_metrics(y_true, y_pred, classes: list[str]) -> dict:
    """Accuracy, weighted precision / recall / F1, and, when a fall class exists,
    sensitivity (Eq. 21: falls detected as falls) and specificity (Eq. 22: non-falls
    detected as non-falls), treating every other class as "no fall"."""
    labels = list(range(len(classes)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average="weighted", zero_division=0)
    pc, rc, fc, support = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                          zero_division=0)
    extra = {}
    fall = next((classes.index(c) for c in FALL_CLASSES if c in classes), None)
    if fall is not None:
        t, p_ = np.asarray(y_true) == fall, np.asarray(y_pred) == fall
        extra = {"sensitivity": float((t & p_).sum() / max(t.sum(), 1)),
                 "specificity": float((~t & ~p_).sum() / max((~t).sum(), 1))}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **extra,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "per_class": {
            name: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]),
                   "support": int(support[i])}
            for i, name in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_metrics(m: dict, classes: list[str]) -> str:
    lines = [f"accuracy {m['accuracy'] * 100:6.2f}%   precision {m['precision'] * 100:6.2f}%   "
             f"recall {m['recall'] * 100:6.2f}%   F1 {m['f1'] * 100:6.2f}%"]
    if "sensitivity" in m:
        lines.append(f"sensitivity {m['sensitivity'] * 100:6.2f}%   specificity {m['specificity'] * 100:6.2f}%")
    lines += ["", f"{'class':<16}{'prec':>8}{'recall':>8}{'F1':>8}{'n':>6}"]
    for name in classes:
        c = m["per_class"][name]
        lines.append(f"{name:<16}{c['precision']:>8.3f}{c['recall']:>8.3f}{c['f1']:>8.3f}{c['support']:>6}")
    width = max(len(n) for n in classes)
    lines += ["", "confusion matrix (rows = true, cols = predicted)"]
    lines.append(" " * (width + 1) + " ".join(f"{i:>5}" for i in range(len(classes))))
    for i, row in enumerate(m["confusion_matrix"]):
        lines.append(f"{classes[i]:<{width}} " + " ".join(f"{v:>5}" for v in row))
    return "\n".join(lines)


def pose_quality(kpts: np.ndarray, conf_thr: float = 0.3) -> dict:
    """Average confidence (ACV, Eq. 2-3) and average % missing keypoints (AMV, Eq. 4-5)
    of one video's (T, K, 3) keypoints. A keypoint is missing if its confidence < conf_thr.
    """
    conf = kpts[..., 2]
    ic = conf.mean(axis=1)                          # Eq. 2, per frame
    im = (conf < conf_thr).mean(axis=1) * 100.0     # Eq. 4, per frame
    return {"acv": float(ic.mean()) if len(ic) else 0.0,   # Eq. 3
            "amv": float(im.mean()) if len(im) else 100.0}  # Eq. 5
