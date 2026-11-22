"""ModelNet40: .off meshes -> normalized point clouds -> PyTorch Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


def read_off(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertices (V, 3) and triangles (F, 3) of an OFF file."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    header = lines[0]
    if not header.startswith("OFF"):
        raise ValueError("Invalid OFF file: header must start with 'OFF'")

    # some ModelNet files glue the header to the counts: "OFF490 518 0"
    rest = header[3:].strip()
    counts_line, body = (rest, lines[1:]) if rest else (lines[1], lines[2:])
    n_verts, n_faces = (int(x) for x in counts_line.split()[:2])

    verts = np.array([ln.split()[:3] for ln in body[:n_verts]], dtype=np.float64)

    tris = []
    for ln in body[n_verts : n_verts + n_faces]:
        vals = [int(x) for x in ln.split()]
        k, idx = vals[0], vals[1 : 1 + vals[0]]
        for i in range(1, k - 1):
            tris.append([idx[0], idx[i], idx[i + 1]])
    faces = np.array(tris, dtype=np.int64).reshape(-1, 3)
    return verts, faces


def sample_surface(
    verts: np.ndarray, faces: np.ndarray, n_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample points uniformly on the surface, weighting each face by its area."""
    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = areas.sum()
    if total <= 0:
        idx = rng.choice(len(verts), n_points, replace=len(verts) < n_points)
        return verts[idx].astype(np.float32)

    chosen = rng.choice(len(faces), n_points, p=areas / total)
    u, v = rng.random(n_points), rng.random(n_points)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = tri[chosen, 0], tri[chosen, 1], tri[chosen, 2]
    pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    return pts.astype(np.float32)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    scale = np.linalg.norm(centered, axis=1).max()
    return (centered / scale if scale > 0 else centered).astype(np.float32)


@dataclass
class _Job:
    path: Path
    label: int
    n_points: int
    seed: int


def _process(job: _Job) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(job.seed)
    verts, faces = read_off(job.path.read_text())
    pts = normalize_unit_sphere(sample_surface(verts, faces, job.n_points, rng))
    return pts, job.label


def list_classes(root: Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


def preprocess_split(
    root: str | Path, split: str, out_file: str | Path, n_points: int = 1024, workers: int = 4
) -> Path:
    """Convert every mesh of a split ('train' or 'test') and save them to a .npz file."""
    from tqdm import tqdm

    root, out_file = Path(root), Path(out_file)
    classes = list_classes(root)
    files = [
        (f, label)
        for label, cls in enumerate(classes)
        for f in sorted((root / cls / split).glob("*.off"))
    ]
    jobs = [_Job(f, label, n_points, seed=i) for i, (f, label) in enumerate(files)]

    with Pool(workers) as pool:
        results = list(
            tqdm(pool.imap(_process, jobs, chunksize=8), total=len(jobs), desc=f"ModelNet40 {split}")
        )

    points = np.stack([r[0] for r in results])
    labels = np.array([r[1] for r in results], dtype=np.int64)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, points=points, labels=labels, classes=np.array(classes))
    return out_file


def augment(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = rng.uniform(2 / 3, 3 / 2, size=3)
    shift = rng.uniform(-0.2, 0.2, size=3)
    jitter = np.clip(rng.normal(0, 0.01, size=points.shape), -0.05, 0.05)
    return (points * scale + shift + jitter).astype(np.float32)


class ModelNet40(Dataset):
    def __init__(self, npz_file: str | Path, train: bool = False, n_points: int | None = None):
        data = np.load(npz_file)
        self.points = data["points"]
        self.labels = data["labels"]
        self.classes = [str(c) for c in data["classes"]]
        self.train = train
        self.n_points = n_points or self.points.shape[1]
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        pts = self.points[i]
        if self.train:
            idx = self._rng.permutation(len(pts))[: self.n_points]
            pts = augment(pts[idx], self._rng)
        else:
            pts = pts[: self.n_points]
        return torch.from_numpy(pts), int(self.labels[i])
