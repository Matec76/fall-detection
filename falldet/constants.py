"""Shared constants: keypoint layouts (COCO-17 and the 15 joints of Lin et al. 2021)."""

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_KEYPOINTS = len(COCO_KEYPOINTS)

# (left, right) index pairs swapped by a horizontal flip.
FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

# Limb connections used when drawing a skeleton.
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (0, 1), (0, 2), (1, 3), (2, 4),
)

# The 15 OpenPose BODY_25 joints kept by Lin et al. (Sec. 2.3.2): nose, neck, shoulders,
# elbows, wrists, mid-hip, hips, knees and ankles, in BODY_25 order. Neck and mid-hip do
# not exist in COCO-17 and are built as midpoints (``None`` entries below).
BODY15 = (
    ("nose", 0), ("neck", None), ("right_shoulder", 6), ("right_elbow", 8), ("right_wrist", 10),
    ("left_shoulder", 5), ("left_elbow", 7), ("left_wrist", 9), ("mid_hip", None),
    ("right_hip", 12), ("right_knee", 14), ("right_ankle", 16),
    ("left_hip", 11), ("left_knee", 13), ("left_ankle", 15),
)
BODY15_FLIP_PAIRS = ((2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14))
BODY15_MID_HIP = 8

