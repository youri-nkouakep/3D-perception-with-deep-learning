"""Geometry for 3D boxes (x, y, z, l, w, h, yaw) in the LiDAR frame."""

from __future__ import annotations

import numpy as np


def bev_corners(boxes: np.ndarray) -> np.ndarray:
    """(N, 7) -> (N, 4, 2) corners seen from above, counter-clockwise."""
    x, y, l, w, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    local = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
    dx = local[None, :, 0] * l[:, None]
    dy = local[None, :, 1] * w[:, None]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack([x[:, None] + dx * c - dy * s, y[:, None] + dx * s + dy * c], axis=-1)


def corners_3d(boxes: np.ndarray) -> np.ndarray:
    """(N, 7) -> (N, 8, 3)."""
    bev = bev_corners(boxes)
    z0 = (boxes[:, 2] - boxes[:, 5] / 2)[:, None, None]
    z1 = (boxes[:, 2] + boxes[:, 5] / 2)[:, None, None]
    bottom = np.concatenate([bev, np.broadcast_to(z0, (len(boxes), 4, 1))], axis=-1)
    top = np.concatenate([bev, np.broadcast_to(z1, (len(boxes), 4, 1))], axis=-1)
    return np.concatenate([bottom, top], axis=1)


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: intersection of two convex counter-clockwise polygons."""
    output = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inputs, output = output, []
        if not inputs:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        for j in range(len(inputs)):
            p, q = inputs[j], inputs[(j + 1) % len(inputs)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                output.append(p)
            if sp * sq < 0:
                t = sp / (sp - sq)
                output.append(p + t * (q - p))
    return np.array(output)


def iou_bev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _iou(a, b, three_d=False)


def iou_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _iou(a, b, three_d=True)


def _iou(a: np.ndarray, b: np.ndarray, three_d: bool) -> np.ndarray:
    out = np.zeros((len(a), len(b)))
    if len(a) == 0 or len(b) == 0:
        return out
    ca, cb = bev_corners(a), bev_corners(b)
    # cheap circle test before the polygon clipping
    ra = np.hypot(a[:, 3], a[:, 4]) / 2
    rb = np.hypot(b[:, 3], b[:, 4]) / 2
    dist = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    for i, j in zip(*np.nonzero(dist < ra[:, None] + rb[None, :])):
        inter = _polygon_area(_clip(ca[i], cb[j]))
        area_a, area_b = a[i, 3] * a[i, 4], b[j, 3] * b[j, 4]
        if three_d:
            z_overlap = max(0.0, min(a[i, 2] + a[i, 5] / 2, b[j, 2] + b[j, 5] / 2)
                            - max(a[i, 2] - a[i, 5] / 2, b[j, 2] - b[j, 5] / 2))
            inter *= z_overlap
            area_a *= a[i, 5]
            area_b *= b[j, 5]
        out[i, j] = inter / (area_a + area_b - inter)
    return out
