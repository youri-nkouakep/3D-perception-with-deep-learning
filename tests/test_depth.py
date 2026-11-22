import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from perception.datasets.nyu import NYUDepth, prepare_frame  # noqa: E402
from perception.metrics import depth_metrics, silog_loss  # noqa: E402
from perception.models import DepthUNet  # noqa: E402


def test_unet_output_shape_and_range():
    model = DepthUNet(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 240, 320))
    assert out.shape == (2, 1, 240, 320)
    assert out.min() >= 0 and out.max() <= 10


def test_perfect_prediction_metrics():
    target = torch.rand(1, 32, 32) * 5 + 0.5
    m = depth_metrics(target.clone(), target)
    assert m["abs_rel"] == pytest.approx(0) and m["rmse"] == pytest.approx(0)
    assert m["delta1"] == 1.0


def test_silog_ignores_global_scale():
    """A prediction off by a constant factor is almost free with lambda close to 1."""
    target = torch.rand(1, 32, 32) * 5 + 0.5
    assert silog_loss(target * 2, target, lam=1.0).item() == pytest.approx(0, abs=1e-2)
    assert silog_loss(target * 2, target, lam=0.0).item() > 1


def test_invalid_pixels_are_ignored():
    target = torch.rand(1, 16, 16) * 5 + 0.5
    pred = target.clone()
    target[0, :4] = 0
    pred[0, :4] = 9
    assert depth_metrics(pred, target)["abs_rel"] == pytest.approx(0)


def test_prepare_frame_and_dataset(tmp_path):
    rgb, depth = prepare_frame(np.zeros((480, 640, 3), np.uint8), np.ones((480, 640), np.float32), (60, 80))
    assert rgb.shape == (60, 80, 3) and depth.shape == (60, 80)

    np.savez(tmp_path / "nyu.npz", rgb=np.stack([rgb] * 2), depth=np.stack([depth] * 2).astype(np.float16))
    x, y = NYUDepth(tmp_path / "nyu.npz", train=True)[0]
    assert x.shape == (3, 60, 80) and y.shape == (1, 60, 80)
