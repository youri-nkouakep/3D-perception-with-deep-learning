import pytest

torch = pytest.importorskip("torch")

from perception.models import PointNet, orthogonality_loss, pointnet_loss  # noqa: E402


def test_output_shape():
    model = PointNet(n_classes=40).eval()
    logits, feat_mat = model(torch.randn(4, 1024, 3))
    assert logits.shape == (4, 40)
    assert feat_mat.shape == (4, 64, 64)


def test_permutation_invariance():
    """Shuffling the points must not change the prediction."""
    torch.manual_seed(0)
    model = PointNet(n_classes=10).eval()
    pts = torch.randn(2, 512, 3)
    perm = torch.randperm(512)
    with torch.no_grad():
        a, _ = model(pts)
        b, _ = model(pts[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_tnets_start_as_identity():
    model = PointNet(n_classes=5).eval()
    with torch.no_grad():
        _, feat_mat = model(torch.randn(2, 128, 3))
    assert torch.allclose(feat_mat, torch.eye(64).expand(2, 64, 64))
    assert orthogonality_loss(feat_mat).item() == pytest.approx(0.0, abs=1e-6)


def test_can_overfit_tiny_batch():
    """Sanity check: the model must be able to memorize 8 samples."""
    torch.manual_seed(0)
    model = PointNet(n_classes=4, feature_transform=False)
    pts, labels = torch.randn(8, 256, 3), torch.arange(8) % 4
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(60):
        logits, feat_mat = model(pts)
        loss = pointnet_loss(logits, labels, feat_mat)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(pts)[0].argmax(1) == labels).float().mean().item()
    assert acc == 1.0
