"""Light PointPillars backbone with a CenterPoint-style head, for cars in KITTI.

Points are grouped into vertical pillars on a bird's-eye-view grid, each pillar is encoded by a
small PointNet, and a 2D CNN predicts a heatmap of car centers plus the box parameters.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from perception.datasets.kitti import PC_RANGE

PILLAR = 0.32
STRIDE = 2
GRID = (int(round((PC_RANGE[4] - PC_RANGE[1]) / PILLAR)), int(round((PC_RANGE[3] - PC_RANGE[0]) / PILLAR)))
OUT_SIZE = (GRID[0] // STRIDE, GRID[1] // STRIDE)  # (rows along y, cols along x)
CELL = PILLAR * STRIDE


class PillarEncoder(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.channels = channels
        self.linear = nn.Linear(9, channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, points: list[torch.Tensor]) -> torch.Tensor:
        h, w = GRID
        device = self.linear.weight.device
        idx, feats = [], []
        for b, pts in enumerate(points):
            pts = pts.to(device)
            ix = ((pts[:, 0] - PC_RANGE[0]) / PILLAR).long().clamp(0, w - 1)
            iy = ((pts[:, 1] - PC_RANGE[1]) / PILLAR).long().clamp(0, h - 1)
            idx.append(b * h * w + iy * w + ix)
            centers = torch.stack([(ix.float() + 0.5) * PILLAR + PC_RANGE[0],
                                   (iy.float() + 0.5) * PILLAR + PC_RANGE[1]], dim=1)
            feats.append((pts, centers))
        cell = torch.cat(idx)
        pts = torch.cat([f[0] for f in feats])
        centers = torch.cat([f[1] for f in feats])

        pillars, inv = torch.unique(cell, return_inverse=True)
        counts = torch.zeros(len(pillars), device=device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
        mean = torch.zeros(len(pillars), 3, device=device).index_add_(0, inv, pts[:, :3]) / counts[:, None]
        x = torch.cat([pts, pts[:, :3] - mean[inv], pts[:, :2] - centers], dim=1)
        x = F.relu(self.bn(self.linear(x)))

        pooled = torch.zeros(len(pillars), self.channels, device=device)
        pooled = pooled.scatter_reduce(0, inv[:, None].expand(-1, self.channels), x, "amax", include_self=False)
        canvas = torch.zeros(len(points) * h * w, self.channels, device=device)
        canvas[pillars] = pooled
        return canvas.view(len(points), h, w, self.channels).permute(0, 3, 1, 2).contiguous()


def _block(c_in: int, c_out: int, stride: int, n: int) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    for _ in range(n):
        layers += [nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _up(c_in: int, c_out: int, factor: int) -> nn.Sequential:
    conv = nn.ConvTranspose2d(c_in, c_out, factor, factor, bias=False) if factor > 1 else nn.Conv2d(c_in, c_out, 1, bias=False)
    return nn.Sequential(conv, nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class PointPillarsCenter(nn.Module):
    """Output dict at 1/STRIDE resolution: heatmap (B,1,H,W) logits and reg (B,8,H,W).

    reg channels: dx, dy (offset inside the cell), z, log(l), log(w), log(h), sin(yaw), cos(yaw).
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.encoder = PillarEncoder(channels)
        self.b1 = _block(channels, 64, 2, 1)
        self.b2 = _block(64, 128, 2, 2)
        self.b3 = _block(128, 128, 2, 2)
        self.up1, self.up2, self.up3 = _up(64, 64, 1), _up(128, 64, 2), _up(128, 64, 4)
        self.shared = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.heatmap = nn.Conv2d(64, 1, 1)
        self.reg = nn.Conv2d(64, 8, 1)
        # start with a low car probability everywhere so the focal loss is stable
        nn.init.constant_(self.heatmap.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, points: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        x1 = self.b1(self.encoder(points))
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x = self.shared(torch.cat([self.up1(x1), self.up2(x2), self.up3(x3)], dim=1))
        return {"heatmap": self.heatmap(x), "reg": self.reg(x)}


def _gaussian_radius(length: float, width: float, min_overlap: float = 0.1) -> float:
    """CenterNet rule: the largest radius that still keeps min_overlap IoU with the true box."""
    a1, b1 = 1, length + width
    c1 = width * length * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (length + width)
    c2 = (1 - min_overlap) * width * length
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (length + width)
    c3 = (min_overlap - 1) * width * length
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def build_targets(boxes: list[torch.Tensor]):
    """Heatmap with a Gaussian on each car center, and the regression target at that center cell."""
    h, w = OUT_SIZE
    heat = np.zeros((len(boxes), 1, h, w), dtype=np.float32)
    reg, ind, batch = [], [], []
    ys, xs = np.ogrid[:h, :w]
    for b, bx in enumerate(boxes):
        for box in bx.numpy():
            cx = (box[0] - PC_RANGE[0]) / CELL
            cy = (box[1] - PC_RANGE[1]) / CELL
            ix, iy = int(cx), int(cy)
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            r = max(2, int(_gaussian_radius(box[3] / CELL, box[4] / CELL)))
            g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * (r / 3) ** 2))
            heat[b, 0] = np.maximum(heat[b, 0], g)
            reg.append([cx - ix, cy - iy, box[2], *np.log(box[3:6]), np.sin(box[6]), np.cos(box[6])])
            ind.append(iy * w + ix)
            batch.append(b)
    return (torch.from_numpy(heat), torch.tensor(reg, dtype=torch.float32).view(-1, 8),
            torch.tensor(ind, dtype=torch.long), torch.tensor(batch, dtype=torch.long))


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced focal loss from CornerNet / CenterNet."""
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg_weight = (1 - target) ** 4
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * neg_weight * (1 - pos)
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def detection_loss(out: dict[str, torch.Tensor], boxes: list[torch.Tensor], reg_weight: float = 2.0):
    heat, reg, ind, batch = build_targets(boxes)
    device = out["heatmap"].device
    hm_loss = focal_loss(out["heatmap"], heat.to(device))
    if len(ind) == 0:
        return hm_loss, hm_loss.detach(), torch.zeros(())
    pred = out["reg"].flatten(2)[batch.to(device), :, ind.to(device)]
    reg_loss = F.l1_loss(pred, reg.to(device))
    return hm_loss + reg_weight * reg_loss, hm_loss.detach(), reg_loss.detach()


@torch.no_grad()
def decode(out: dict[str, torch.Tensor], max_boxes: int = 50, threshold: float = 0.1):
    """Return one (boxes (K, 7), scores (K,)) pair per sample, as numpy arrays."""
    heat = torch.sigmoid(out["heatmap"])
    peaks = heat * (F.max_pool2d(heat, 3, 1, 1) == heat)
    h, w = OUT_SIZE
    results = []
    for b in range(len(heat)):
        scores, ind = peaks[b].flatten().topk(max_boxes)
        keep = scores > threshold
        scores, ind = scores[keep], ind[keep]
        r = out["reg"][b].flatten(1)[:, ind].T
        iy, ix = (ind // w).float(), (ind % w).float()
        x = (ix + r[:, 0]) * CELL + PC_RANGE[0]
        y = (iy + r[:, 1]) * CELL + PC_RANGE[1]
        dims = r[:, 3:6].clamp(max=3).exp()
        yaw = torch.atan2(r[:, 6], r[:, 7])
        boxes = torch.stack([x, y, r[:, 2], dims[:, 0], dims[:, 1], dims[:, 2], yaw], dim=1)
        results.append((boxes.cpu().numpy(), scores.cpu().numpy()))
    return results
