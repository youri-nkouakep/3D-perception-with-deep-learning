import numpy as np
import pytest

from perception.datasets.modelnet import (
    ModelNet40,
    normalize_unit_sphere,
    preprocess_split,
    read_off,
    sample_surface,
)

# unit cube with quad faces, to test triangulation
CUBE = """OFF
8 6 0
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
4 0 1 2 3
4 4 5 6 7
4 0 1 5 4
4 2 3 7 6
4 1 2 6 5
4 0 3 7 4
"""


def test_read_off_triangulates_quads():
    verts, faces = read_off(CUBE)
    assert verts.shape == (8, 3)
    assert faces.shape == (12, 3)


def test_read_off_glued_header():
    verts, faces = read_off(CUBE.replace("OFF\n8 6 0", "OFF8 6 0"))
    assert verts.shape == (8, 3) and faces.shape == (12, 3)


def test_samples_lie_on_cube_surface():
    verts, faces = read_off(CUBE)
    pts = sample_surface(verts, faces, 2000, np.random.default_rng(0))
    assert pts.shape == (2000, 3)
    on_face = np.isclose(pts, 0, atol=1e-6) | np.isclose(pts, 1, atol=1e-6)
    assert on_face.any(axis=1).all()


def test_normalize_unit_sphere():
    pts = normalize_unit_sphere(np.random.default_rng(0).normal(5, 3, (500, 3)))
    assert np.allclose(pts.mean(0), 0, atol=1e-5)
    assert np.isclose(np.linalg.norm(pts, axis=1).max(), 1, atol=1e-5)


def test_preprocess_and_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    for cls in ("box", "cube"):
        for split in ("train", "test"):
            d = tmp_path / "raw" / cls / split
            d.mkdir(parents=True)
            (d / f"{cls}_0001.off").write_text(CUBE)
    out = preprocess_split(tmp_path / "raw", "train", tmp_path / "train.npz", n_points=256, workers=1)
    ds = ModelNet40(out, train=True)
    assert ds.classes == ["box", "cube"] and len(ds) == 2
    pts, label = ds[1]
    assert isinstance(pts, torch.Tensor) and pts.shape == (256, 3) and label == 1
