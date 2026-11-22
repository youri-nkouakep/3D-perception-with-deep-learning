"""KITTI 3D object detection: calibration, labels, LiDAR scans and a Dataset for cars."""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
IMAGE_SHAPE = (375, 1242)
# x forward, y left, z up (LiDAR frame), meters
PC_RANGE = np.array([0.0, -32.0, -3.0, 64.0, 32.0, 1.0], dtype=np.float32)


class HttpRangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests, so zipfile can pull single members out of a remote archive."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf) -> int:
        end = min(self.pos + len(buf), self.size)
        if end <= self.pos:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


@dataclass
class Calib:
    P2: np.ndarray
    R0: np.ndarray
    Tr: np.ndarray

    @classmethod
    def from_text(cls, text: str) -> "Calib":
        vals = {}
        for line in text.splitlines():
            if ":" in line:
                key, rest = line.split(":", 1)
                vals[key.strip()] = np.array(rest.split(), dtype=np.float64)
        return cls(vals["P2"].reshape(3, 4), vals["R0_rect"].reshape(3, 3), vals["Tr_velo_to_cam"].reshape(3, 4))

    def lidar_to_rect(self, pts: np.ndarray) -> np.ndarray:
        cam = pts[:, :3] @ self.Tr[:, :3].T + self.Tr[:, 3]
        return cam @ self.R0.T

    def rect_to_lidar(self, pts: np.ndarray) -> np.ndarray:
        cam = pts @ np.linalg.inv(self.R0).T
        return (cam - self.Tr[:, 3]) @ np.linalg.inv(self.Tr[:, :3]).T

    def rect_to_image(self, pts: np.ndarray) -> np.ndarray:
        uvw = pts @ self.P2[:, :3].T + self.P2[:, 3]
        return uvw[:, :2] / uvw[:, 2:3]


def crop_to_camera(points: np.ndarray, calib: Calib, shape=IMAGE_SHAPE) -> np.ndarray:
    """Keep the points seen by the front camera, the only area with labels."""
    rect = calib.lidar_to_rect(points)
    uv = calib.rect_to_image(rect)
    keep = (rect[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    return points[keep]


@dataclass
class Labels:
    types: np.ndarray
    truncated: np.ndarray
    occluded: np.ndarray
    bbox: np.ndarray       # (N, 4) image box, pixels
    boxes: np.ndarray      # (N, 7) LiDAR frame: x, y, z (center), l, w, h, yaw


def parse_labels(text: str, calib: Calib) -> Labels:
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    types = np.array([r[0] for r in rows])
    num = np.array([[float(v) for v in r[1:15]] for r in rows], dtype=np.float64).reshape(-1, 14)
    h, w, l = num[:, 7], num[:, 8], num[:, 9]
    # KITTI gives the bottom center in the rectified camera frame (y points down)
    center_rect = num[:, 10:13] - np.stack([np.zeros_like(h), h / 2, np.zeros_like(h)], axis=1)
    center = calib.rect_to_lidar(center_rect)
    yaw = -num[:, 13] - np.pi / 2
    boxes = np.concatenate([center, np.stack([l, w, h, yaw], axis=1)], axis=1).astype(np.float32)
    return Labels(types, num[:, 0], num[:, 1].astype(int), num[:, 3:7], boxes)


def read_split(file: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(file).read_text().splitlines() if ln.strip()]


def global_augment(points: np.ndarray, boxes: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        points[:, 1] *= -1
        boxes[:, 1] *= -1
        boxes[:, 6] *= -1
    angle = rng.uniform(-np.pi / 8, np.pi / 8)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rot.T
    boxes[:, :2] = boxes[:, :2] @ rot.T
    boxes[:, 6] += angle
    scale = rng.uniform(0.95, 1.05)
    points[:, :3] *= scale
    boxes[:, :6] *= scale
    return points, boxes


def in_range(xyz: np.ndarray, pc_range=PC_RANGE) -> np.ndarray:
    return np.all((xyz >= pc_range[:3]) & (xyz < pc_range[3:]), axis=1)


class KittiCars(Dataset):
    """One item: LiDAR points (N, 4) and the car boxes (M, 7) inside the detection range."""

    def __init__(self, root: str | Path, split: str, train: bool = False):
        self.root = Path(root)
        self.ids = read_split(self.root / "splits" / f"{split}.txt")
        self.train = train
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.ids)

    def calib(self, idx: str) -> Calib:
        return Calib.from_text((self.root / "calib" / f"{idx}.txt").read_text())

    def labels(self, idx: str) -> Labels:
        return parse_labels((self.root / "label_2" / f"{idx}.txt").read_text(), self.calib(idx))

    def __getitem__(self, i: int):
        idx = self.ids[i]
        points = np.load(self.root / "points" / f"{idx}.npy").astype(np.float32)
        labels = self.labels(idx)
        boxes = labels.boxes[labels.types == "Car"].copy()
        if self.train:
            points, boxes = global_augment(points, boxes, self._rng)
        points = points[in_range(points[:, :3])]
        boxes = boxes[in_range(boxes[:, :3])]
        return torch.from_numpy(points), torch.from_numpy(boxes), idx


def collate(batch):
    points, boxes, ids = zip(*batch)
    return list(points), list(boxes), list(ids)
