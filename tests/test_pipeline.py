"""Smoke tests that run without any real data or pose model."""
import json

import numpy as np
import pytest
import torch

from falldet.config import load_config
from falldet.dataset import scan_pose_dir, split_samples
from falldet.metrics import classification_metrics, pose_quality
from falldet.models import build_model, count_parameters, load_checkpoint
from falldet.pose import load_alphapose_json, select_person
from falldet.preprocess import (feature_dim, flip_horizontal, interpolate_missing, normalize,
                                prepare_sequence, to_body15)
from falldet.train import run_training

CLASSES = ["falling_down", "standing_up", "lying_down", "walking"]
LIN = "configs/default.yaml"          # Lin et al. 2021: fall / no_fall, body15, RP, LSTM-512
JURAEV = "configs/juraev2022.yaml"    # Juraev et al. 2022: 4 classes, Transformer / stacked LSTM


def fake_clip(label: int, rng: np.random.Generator, n: int = 40) -> np.ndarray:
    """Standing person that drops (fall), rises, lies still or moves sideways."""
    base = np.stack([rng.uniform(-20, 20, 17), np.linspace(0, 170, 17)], 1)
    t = np.linspace(0, 1, n)[:, None]
    shift = {0: np.c_[0 * t, 150 * t], 1: np.c_[0 * t, 150 * (1 - t)],
             2: np.c_[0 * t, 150 + 0 * t], 3: np.c_[300 * t, 0 * t]}[label]
    xy = 400 + base[None] + shift[:, None] + rng.normal(0, 2, (n, 17, 2))
    return np.concatenate([xy, rng.uniform(0.5, 1, (n, 17, 1))], 2).astype(np.float32)


def fake_fall_clip(fall: bool, rng: np.random.Generator, n: int = 40) -> np.ndarray:
    """A body that tips over from standing to lying (fall) or stays upright while it moves.
    Relative-position normalisation removes translation, so a fall must change the pose."""
    body = np.stack([rng.uniform(-20, 20, 17), np.linspace(-85, 85, 17)], 1)
    angle = np.linspace(0, np.pi / 2, n) if fall else np.zeros(n)
    rot = np.stack([np.stack([np.cos(angle), -np.sin(angle)], -1), np.stack([np.sin(angle), np.cos(angle)], -1)], -2)
    xy = np.einsum("tij,kj->tki", rot, body) + rng.uniform(200, 500, 2) + rng.normal(0, 2, (n, 17, 2))
    if not fall:
        xy += np.linspace(0, rng.uniform(-150, 150), n)[:, None, None]
    return np.concatenate([xy, rng.uniform(0.5, 1, (n, 17, 1))], 2).astype(np.float32)


def write_clips(root, names_by_label: dict, per_class: int = 12):
    """names_by_label maps a class folder to a function (rng) -> (T, 17, 3) keypoints."""
    rng = np.random.default_rng(0)
    for name, make in names_by_label.items():
        (root / name).mkdir(parents=True)
        for i in range(per_class):
            np.savez(root / name / f"{name}_{i}.npz", keypoints=make(rng),
                     size=np.array([960, 720]), group=f"{name}_{i // 2}")
    return root


@pytest.fixture
def pose_root(tmp_path):
    return write_clips(tmp_path / "four", {name: (lambda rng, i=i: fake_clip(i, rng)) for i, name in enumerate(CLASSES)})


@pytest.fixture
def binary_root(tmp_path):
    return write_clips(tmp_path / "binary", {"fall": lambda rng: fake_fall_clip(True, rng),
                                             "no_fall": lambda rng: fake_fall_clip(False, rng)}, per_class=18)


@pytest.mark.parametrize("config,name,expected", [
    (JURAEV, "transformer", 0.221e6), (JURAEV, "stacked_lstm", 0.061e6),
    (LIN, "lstm", 4 * 512 * (30 + 512 + 2) + 512 * 2 + 2)])
def test_models_match_paper_size(config, name, expected):
    cfg = load_config(config)
    in_dim = feature_dim(cfg["preprocess"])
    model = build_model(name, in_dim, 2, 25, **cfg["model"][name])
    assert model(torch.randn(3, 25, in_dim)).shape == (3, 2)
    assert abs(count_parameters(model) - expected) / expected < 0.15


def test_normalize_and_flip():
    kpts = np.random.default_rng(1).uniform(100, 500, (10, 17, 3)).astype(np.float32)
    kpts[..., 2] = 1.0
    kpts[0, 3] = 0.0  # a missing keypoint
    out = normalize(kpts)
    vis = out[..., 2] > 0
    assert out[..., :2][vis].min() == pytest.approx(0) and out[..., :2][vis].max() == pytest.approx(1)
    assert (out[0, 3] == 0).all()
    flipped = flip_horizontal(out)
    np.testing.assert_allclose(flipped[:, 5, 0], 1 - out[:, 6, 0])
    np.testing.assert_allclose(flip_horizontal(flipped), out, atol=1e-6)


def test_body15_and_relative_position():
    kpts = fake_clip(3, np.random.default_rng(3), n=6)
    kpts[2, 11, 2] = 0.0                       # left hip missing -> no mid-hip in frame 2
    b15 = to_body15(kpts)
    assert b15.shape == (6, 15, 3)
    np.testing.assert_allclose(b15[0, 1, :2], (kpts[0, 5, :2] + kpts[0, 6, :2]) / 2)   # neck
    assert b15[2, 8, 2] == 0
    rp = normalize(b15, "rp", frame_size=(1280, 960), ref_joint=8, rp_scale=False)
    np.testing.assert_allclose(rp[[0, 1, 3, 4, 5], 8, :2], 320.0 * np.array([1, 0.75]) * np.ones((5, 2)), atol=1e-4)
    scale = np.array([0.5, 0.5])                # 1280x960 -> 640x480
    np.testing.assert_allclose(rp[0, 0, :2] - rp[0, 8, :2], (b15[0, 0, :2] - b15[0, 8, :2]) * scale, atol=1e-4)


def test_interpolate_missing():
    kpts = np.zeros((10, 2, 3), np.float32)
    kpts[:, 0] = np.c_[np.arange(10), np.arange(10) * 2, np.ones(10)]
    kpts[3:6, 0] = 0                            # a gap -> filled
    kpts[[1, 8], 1] = [[5, 5, 1], [9, 9, 1]]    # joint 1 missing in 80% of frames -> untouched
    out = interpolate_missing(kpts, max_missing=0.67)
    np.testing.assert_allclose(out[3:6, 0, :2], [[3, 6], [4, 8], [5, 10]])
    np.testing.assert_array_equal(out[:, 1], kpts[:, 1])


def test_prepare_sequence_shapes():
    kpts = fake_clip(0, np.random.default_rng(2), n=7)
    aug = {"flip_prob": 1.0, "crop_min": 0.8, "noise_std": 0.01}
    juraev = load_config(JURAEV)["preprocess"]
    lin = load_config(LIN)["preprocess"]
    assert prepare_sequence(kpts, juraev).shape == (25, 51)
    assert prepare_sequence(kpts, juraev, augment=aug).shape == (25, 51)
    assert prepare_sequence(np.zeros((0, 17, 3), np.float32), juraev).shape == (25, 51)
    assert prepare_sequence(kpts, lin, frame_size=(640, 480)).shape == (100, 30)
    assert prepare_sequence(kpts, {**lin, "interpolate": True}, augment=aug, frame_size=(640, 480)).shape == (100, 30)


def test_sensitivity_specificity():
    m = classification_metrics([0, 0, 0, 1, 1], [0, 0, 1, 1, 0], ["fall", "no_fall"])
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(1 / 2)


def test_urfd_windows():
    from falldet.urfd import balance, make_segments, window_label
    assert window_label([-1, -1], True, "down") == "no_fall"
    assert window_label([-1, 0, 1], True, "event") == "fall"
    assert window_label([1, 1], True, "event") is None and window_label([1, 1], True, "down") == "fall"
    assert window_label([1, 1], False, "sequence") == "no_fall"
    labels = {"fall-01": {t: (-1 if t <= 100 else 0 if t <= 120 else 1) for t in range(1, 161)}}
    rows = make_segments([("fall-01", 0, 160), ("adl-01", 0, 90)], labels, 100, 20, "down")
    assert [(r["start"], r["label"]) for r in rows] == [(0, "no_fall"), (20, "fall"), (40, "fall"),
                                                         (60, "fall"), (0, "no_fall")]
    assert rows[-1]["end"] == 90 and rows[0]["group"] == "fall-01"
    kept = balance(rows, seed=0)
    assert sorted(r["label"] for r in kept) == ["fall", "fall", "no_fall", "no_fall"]


def test_pose_quality():
    kpts = np.zeros((4, 17, 3), np.float32)
    kpts[:2, :, 2] = 0.8
    q = pose_quality(kpts, conf_thr=0.3)
    assert q["acv"] == pytest.approx(0.4) and q["amv"] == pytest.approx(50.0)


def test_select_person_prefers_previous_track():
    boxes = np.array([[0, 0, 100, 200], [300, 0, 350, 60]], np.float32)
    scores = np.array([0.9, 0.9], np.float32)
    assert select_person(boxes, scores, None) == 0
    assert select_person(boxes, scores, np.array([305, 5, 350, 60], np.float32)) == 1


def test_alphapose_json(tmp_path):
    kp = np.ones((17, 3)).ravel().tolist()
    dets = [{"image_id": "0.jpg", "keypoints": kp, "score": 2.0, "box": [0, 0, 10, 10]},
            {"image_id": "2.jpg", "keypoints": kp, "score": 2.0}]
    path = tmp_path / "alphapose-results.json"
    path.write_text(json.dumps(dets))
    out = load_alphapose_json(path)
    assert out.shape == (3, 17, 3) and out[1].sum() == 0 and out[2, 0, 2] == 1


def test_split_keeps_groups_apart(pose_root):
    splits = split_samples(scan_pose_dir(pose_root, CLASSES), 0.25, 0.1, seed=0)
    groups = [{s.group for s in v} for v in splits.values()]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])
    assert all(splits.values())
    per_clip = split_samples(scan_pose_dir(pose_root, CLASSES), 0.25, 0.0, seed=0, group_split=False)
    assert len({s.path for s in per_clip["test"]}) == len(per_clip["test"])


@pytest.mark.parametrize("model_name", ["rnn", "lstm", "gru"])
def test_lin_training_end_to_end(binary_root, tmp_path, model_name):
    cfg = load_config(LIN, [f"data.real_dir={binary_root}", "train.epochs=60", "train.device=cpu",
                            f"model.name={model_name}", f"model.{model_name}.hidden=64"])
    out = tmp_path / "run"
    results = run_training(cfg, out)
    assert results["test"]["accuracy"] > 0.7 and "sensitivity" in results["test"]
    model, meta = load_checkpoint(out / "best.pt")
    assert meta["classes"] == ["fall", "no_fall"] and meta["preprocess"]["joints"] == "body15"


@pytest.mark.parametrize("model_name", ["transformer", "stacked_lstm"])
def test_juraev_training_end_to_end(pose_root, tmp_path, model_name):
    cfg = load_config(JURAEV, [f"data.real_dir={pose_root}", "data.split=random", f"data.synthetic_dir={pose_root}",
                               "data.synthetic_per_class=3", "train.epochs=150",
                               "train.batch_size=16", "train.device=cpu",
                               f"model.name={model_name}"])
    out = tmp_path / "run"
    results = run_training(cfg, out)
    assert results["test"]["accuracy"] > 0.5  # 4 classes, chance = 0.25
    model, meta = load_checkpoint(out / "best.pt")
    assert meta["classes"] == CLASSES and meta["model_name"] == model_name


def test_make_windows():
    from falldet.omnifall import make_windows
    assert make_windows(1.0, 1.5, 2.0, 5) == [(0.25, 2.25)]    # widened around the centre
    assert make_windows(0.0, 0.4, 2.0, 5) == [(0.0, 2.0)]      # clamped at t = 0
    assert make_windows(0.0, 7.0, 2.0, 5) == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    assert len(make_windows(0.0, 100.0, 2.0, 5)) == 5


def test_omnifall_segments_from_cached_annotations(tmp_path):
    from falldet.omnifall import LABEL_MAPS, build_segments
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "le2i.csv").write_text(
        "path,label,start,end,subject,cam,dataset\n"
        "Home_01/video_1,0,0.0,4.1,3,1,le2i\n"     # walk, 4.1 s -> 2 windows
        "Home_01/video_1,1,4.1,5.0,3,1,le2i\n"     # fall
        "Home_01/video_1,2,5.0,9.0,3,1,le2i\n"     # fallen -> not in paper4, dropped
        "Home_01/video_2,7,0.0,1.0,4,1,le2i\n")    # stand_up, but video not in any split
    for part, paths in {"train": ["Home_01/video_1"], "val": [], "test": []}.items():
        (tmp_path / "splits" / "cs" / "le2i").mkdir(parents=True, exist_ok=True)
        (tmp_path / "splits" / "cs" / "le2i" / f"{part}.csv").write_text("path\n" + "".join(p + "\n" for p in paths))
    rows = build_segments(["le2i"], "cs", "random", ["train", "val", "test"], LABEL_MAPS["paper4"],
                          2.0, 0.3, 5, tmp_path)
    assert [r["label"] for r in rows] == ["walking", "walking", "falling_down"]
    assert rows[2]["start_sec"] == "3.550" and rows[2]["end_sec"] == "5.550"
    assert {r["split"] for r in rows} == {"train"} and rows[0]["group"] == "le2i/subject3"
    assert rows[0]["video"] == "le2i/video/Home_01/video_1.mp4"


def test_extract_segments_and_predefined_split(tmp_path, monkeypatch):
    import cv2
    import sys
    from falldet import extract_poses
    video = tmp_path / "videos" / "ds" / "video" / "a.mp4"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for _ in range(100):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    rng = np.random.default_rng(0)
    dets = [{"image_id": f"{t}.jpg", "keypoints": fake_clip(t % 4, rng, 1)[0].ravel().tolist(), "score": 1.0}
            for t in range(100)]
    ap_json = tmp_path / "ap" / "ds" / "video" / "a.json"
    ap_json.parent.mkdir(parents=True)
    ap_json.write_text(json.dumps(dets))
    seg_csv = tmp_path / "segments.csv"
    rows = ["video,label,start_sec,end_sec,split,group"]
    for part, (lo, hi) in {"train": (0.0, 2.0), "val": (1.0, 3.0), "test": (2.0, 4.0)}.items():
        rows += [f"ds/video/a.mp4,{c},{lo},{hi},{part},g" for c in CLASSES]
    rows.append("ds/video/missing.mp4,walking,0,1,train,g")
    seg_csv.write_text("\n".join(rows) + "\n")
    out = tmp_path / "poses"
    monkeypatch.setattr(sys, "argv", ["x", "--videos", str(tmp_path / "videos"), "--segments", str(seg_csv),
                                      "--out", str(out), "--backend", "alphapose", "--alphapose-dir",
                                      str(tmp_path / "ap"), "--frame-step", "2"])
    extract_poses.main()
    clip = np.load(out / "val" / "walking" / "ds__video__a_25_75.npz")
    assert clip["keypoints"].shape == (25, 17, 3) and float(clip["fps"]) == 12.5

    cfg = load_config("configs/juraev2022.yaml", [f"data.real_dir={out}", "data.split=predefined",
                                               "train.epochs=1", "train.device=cpu"])
    results = run_training(cfg, tmp_path / "run")
    split = json.loads((tmp_path / "run" / "split.json").read_text())
    assert {k: len(v) for k, v in split.items()} == {"train": 4, "val": 4, "test": 4}
    assert "test" in results


def test_omnifall_age_filter(tmp_path):
    from falldet.omnifall import LABEL_MAPS, build_segments
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "of-syn.csv").write_text(
        "path,label,start,end,subject,cam,dataset,age_group\n"
        "fall/a,1,0.0,1.0,-1,-1,of-syn,elderly_65_plus\n"
        "fall/b,1,0.0,1.0,-1,-1,of-syn,toddlers_1_4\n")
    (tmp_path / "splits" / "syn" / "random").mkdir(parents=True)
    (tmp_path / "splits" / "syn" / "random" / "train.csv").write_text("path\nfall/a\nfall/b\n")
    args = (["of-syn"], "cs", "random", ["train"], LABEL_MAPS["paper4"], 2.0, 0.3, 5, tmp_path)
    assert len(build_segments(*args)) == 2
    rows = build_segments(*args, age_groups=["elderly_65_plus"])
    assert [r["video"] for r in rows] == ["of-syn/video/fall/a.mp4"]
