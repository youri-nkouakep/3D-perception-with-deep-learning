import numpy as np
import pytest

torch = pytest.importorskip("torch")

from perception.boxes import iou_3d, iou_bev  # noqa: E402
from perception.datasets.kitti import Calib, crop_to_camera, parse_labels  # noqa: E402
from perception.kitti_eval import Frame, evaluate  # noqa: E402
from perception.models.pointpillars import (  # noqa: E402
    PointPillarsCenter,
    build_targets,
    decode,
    detection_loss,
)

# calibration of KITTI training frame 000000
CALIB = """P0: 7.070493e+02 0.000000e+00 6.040814e+02 0.000000e+00 0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.070493e+02 0.000000e+00 6.040814e+02 4.575831e+01 0.000000e+00 7.070493e+02 1.805066e+02 -3.454157e-01 0.000000e+00 0.000000e+00 1.000000e+00 4.981016e-03
R0_rect: 9.999128e-01 1.009263e-02 -8.511932e-03 -1.012729e-02 9.999406e-01 -4.037671e-03 8.470675e-03 4.123522e-03 9.999556e-01
Tr_velo_to_cam: 6.927964e-03 -9.999722e-01 -2.757829e-03 -2.457729e-02 -1.162982e-03 2.749836e-03 -9.999955e-01 -6.127237e-02 9.999753e-01 6.931141e-03 -1.143899e-03 -3.321029e-01
"""
LABEL = "Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59\n"


def box(x, y, yaw=0.0, z=-1.0, l=4.0, w=2.0, h=1.5):
    return np.array([[x, y, z, l, w, h, yaw]], dtype=np.float32)


def test_iou_identical_disjoint_and_rotated():
    a = box(10, 0)
    assert iou_bev(a, a)[0, 0] == pytest.approx(1)
    assert iou_3d(a, box(30, 0))[0, 0] == 0
    sq = box(0, 0, l=2, w=2)
    # a square rotated by 45 degrees on itself: intersection is a regular octagon
    octagon = 8 * (np.sqrt(2) - 1)
    assert iou_bev(sq, box(0, 0, yaw=np.pi / 4, l=2, w=2))[0, 0] == pytest.approx(octagon / (8 - octagon), rel=1e-6)
    # same box raised by half its height: 0.75 m of overlap out of 2.25 m
    assert iou_3d(a, box(10, 0, z=-0.25))[0, 0] == pytest.approx(0.75 / 2.25)


def test_calib_roundtrip_and_label_to_lidar():
    calib = Calib.from_text(CALIB)
    pts = np.random.default_rng(0).normal(0, 10, (50, 3))
    assert np.allclose(calib.rect_to_lidar(calib.lidar_to_rect(pts)), pts, atol=1e-6)

    lab = parse_labels(LABEL, calib)
    x, y, z, l, w, h, yaw = lab.boxes[0]
    assert x == pytest.approx(47.0, abs=0.2) and abs(y) < 1.5
    assert (l, w, h) == pytest.approx((3.64, 1.67, 1.65))
    assert yaw == pytest.approx(1.59 - np.pi / 2, abs=1e-6)

    in_front = np.array([[20.0, 0.0, -1.0, 0.5]])
    behind = np.array([[-20.0, 0.0, -1.0, 0.5]])
    assert len(crop_to_camera(np.vstack([in_front, behind]), calib)) == 1


def test_model_shapes_and_loss():
    model = PointPillarsCenter().eval()
    pts = [torch.rand(500, 4) * torch.tensor([60, 60, 3, 1]) - torch.tensor([0, 30, 2, 0]) for _ in range(2)]
    out = model(pts)
    assert out["heatmap"].shape[:2] == (2, 1) and out["reg"].shape[:2] == (2, 8)
    loss, _, _ = detection_loss(out, [torch.from_numpy(box(20, 3)), torch.zeros(0, 7)])
    assert torch.isfinite(loss)


def test_targets_decode_back_to_boxes():
    """Decoding perfect network outputs must give back the original boxes."""
    gt = np.vstack([box(20.3, 4.1, yaw=0.7), box(41.9, -10.2, yaw=-2.0, l=4.4)])
    heat, reg, ind, batch = build_targets([torch.from_numpy(gt)])
    logits = torch.logit(heat.clamp(1e-6, 1 - 1e-6))
    reg_map = torch.zeros(1, 8, *heat.shape[-2:])
    reg_map.flatten(2)[0, :, ind] = reg.T
    boxes, scores = decode({"heatmap": logits, "reg": reg_map}, threshold=0.5)[0]
    assert len(boxes) == 2
    boxes = boxes[np.argsort(boxes[:, 0])]
    assert np.allclose(boxes, gt, atol=1e-4)


def test_perfect_detections_give_full_ap():
    calib = Calib.from_text(CALIB)
    lab = parse_labels(LABEL, calib)
    frames = [Frame(lab, calib, lab.boxes.copy(), np.array([0.9]))]
    res = evaluate(frames)
    assert res["3d"]["moderate"] == pytest.approx(100)
    frames = [Frame(lab, calib, np.zeros((0, 7), np.float32), np.zeros(0))]
    assert evaluate(frames)["bev"]["easy"] == 0
