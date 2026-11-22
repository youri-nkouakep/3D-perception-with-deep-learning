"""NYU Depth v2 (labeled subset): 1,449 indoor RGB images with Kinect depth, official 795/654 split."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

# the RGB frames have a white border left by the calibration, this is the valid area
CROP = (slice(44, 471), slice(40, 601))
MAX_DEPTH = 10.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare_frame(rgb: np.ndarray, depth: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Crop the border and resize to size = (height, width)."""
    h, w = size
    rgb = cv2.resize(rgb[CROP], (w, h), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth[CROP], (w, h), interpolation=cv2.INTER_NEAREST)
    return rgb, depth.astype(np.float32)


def preprocess_nyu(mat_file: str | Path, splits_file: str | Path, out_dir: str | Path,
                   size: tuple[int, int] = (240, 320)) -> list[Path]:
    import h5py
    from scipy.io import loadmat
    from tqdm import tqdm

    splits = loadmat(splits_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with h5py.File(mat_file, "r") as f:
        images, depths = f["images"], f["depths"]
        for split, key in (("train", "trainNdxs"), ("test", "testNdxs")):
            idx = splits[key].ravel() - 1
            rgbs = np.empty((len(idx), *size, 3), dtype=np.uint8)
            deps = np.empty((len(idx), *size), dtype=np.float32)
            for k, i in enumerate(tqdm(idx, desc=f"NYU {split}")):
                # stored as (C, W, H) in the .mat file
                rgb = np.ascontiguousarray(images[i].transpose(2, 1, 0))
                depth = np.ascontiguousarray(depths[i].T)
                rgbs[k], deps[k] = prepare_frame(rgb, depth, size)
            out = out_dir / f"nyu_{split}_{size[0]}x{size[1]}.npz"
            np.savez(out, rgb=rgbs, depth=deps.astype(np.float16))
            outputs.append(out)
    return outputs


class NYUDepth(Dataset):
    def __init__(self, npz_file: str | Path, train: bool = False):
        data = np.load(npz_file)
        self.rgb = data["rgb"]
        self.depth = data["depth"]
        self.train = train
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, i: int):
        rgb = self.rgb[i].astype(np.float32) / 255
        depth = self.depth[i].astype(np.float32)
        if self.train:
            if self._rng.random() < 0.5:
                rgb, depth = rgb[:, ::-1], depth[:, ::-1]
            gain = self._rng.uniform(0.8, 1.2, size=3).astype(np.float32)
            rgb = np.clip(rgb * gain * self._rng.uniform(0.8, 1.2), 0, 1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(depth))[None])
