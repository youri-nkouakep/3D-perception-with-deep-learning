"""KITTI car AP (R40, IoU 0.7) for BEV and 3D boxes.

A re-implementation of the official protocol: same difficulty levels, same ignored objects
(vans, objects that do not meet the difficulty, detections too small in the image),
40 recall points. Matching is greedy by score, so numbers can differ slightly from the devkit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.boxes import corners_3d, iou_3d, iou_bev
from perception.datasets.kitti import Calib, Labels

DIFFICULTIES = {"easy": (40, 0, 0.15), "moderate": (25, 1, 0.30), "hard": (25, 2, 0.50)}


@dataclass
class Frame:
    labels: Labels
    calib: Calib
    boxes: np.ndarray
    scores: np.ndarray


def image_heights(boxes: np.ndarray, calib: Calib) -> np.ndarray:
    """Height in pixels of each box once projected in the camera image."""
    if len(boxes) == 0:
        return np.zeros(0)
    corners = corners_3d(boxes).reshape(-1, 3)
    uv = calib.rect_to_image(calib.lidar_to_rect(corners)).reshape(len(boxes), 8, 2)
    v = np.clip(uv[..., 1], 0, 375)
    return v.max(axis=1) - v.min(axis=1)


def _frame_matches(frame: Frame, difficulty: str, iou_fn, threshold: float = 0.7):
    min_h, max_occ, max_trunc = DIFFICULTIES[difficulty]
    lab = frame.labels
    heights = lab.bbox[:, 3] - lab.bbox[:, 1]
    is_car = lab.types == "Car"
    valid = is_car & (lab.occluded <= max_occ) & (lab.truncated <= max_trunc) & (heights >= min_h)
    ignored = (is_car & ~valid) | (lab.types == "Van")
    gt_idx = np.flatnonzero(valid | ignored)
    gt_boxes = lab.boxes[gt_idx]

    order = np.argsort(-frame.scores)
    det, scores = frame.boxes[order], frame.scores[order]
    small = image_heights(det, frame.calib) < min_h
    iou = iou_fn(det, gt_boxes)
    taken = np.zeros(len(gt_idx), dtype=bool)

    tp_scores, fp_scores = [], []
    for k in range(len(det)):
        cand = np.where(~taken & (iou[k] >= threshold), iou[k], -1) if len(gt_idx) else np.array([])
        j = int(np.argmax(cand)) if len(cand) and cand.max() >= threshold else -1
        if j >= 0:
            taken[j] = True
            if valid[gt_idx[j]]:
                tp_scores.append(scores[k])
        elif not small[k]:
            fp_scores.append(scores[k])
    return tp_scores, fp_scores, int(valid.sum())


def average_precision(frames: list[Frame], difficulty: str = "moderate", metric: str = "3d") -> float:
    iou_fn = iou_3d if metric == "3d" else iou_bev
    tps, fps, n_gt = [], [], 0
    for f in frames:
        tp, fp, n = _frame_matches(f, difficulty, iou_fn)
        tps += tp
        fps += fp
        n_gt += n
    if n_gt == 0:
        return 0.0
    scores = np.array(tps + fps)
    is_tp = np.array([1] * len(tps) + [0] * len(fps))
    order = np.argsort(-scores, kind="stable")
    is_tp = is_tp[order]
    tp_cum = np.cumsum(is_tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.arange(1, len(is_tp) + 1)
    ap = 0.0
    for r in np.linspace(1 / 40, 1, 40):
        p = precision[recall >= r - 1e-9]
        ap += p.max() if len(p) else 0.0
    return 100 * ap / 40


def evaluate(frames: list[Frame]) -> dict[str, dict[str, float]]:
    return {m: {d: round(average_precision(frames, d, m), 2) for d in DIFFICULTIES} for m in ("bev", "3d")}
