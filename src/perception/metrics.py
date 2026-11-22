"""Depth loss and the standard NYU Depth v2 metrics."""

from __future__ import annotations

import torch


def valid_mask(target: torch.Tensor, max_depth: float = 10.0) -> torch.Tensor:
    return (target > 1e-3) & (target <= max_depth)


def silog_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al., 2014)."""
    mask = valid_mask(target)
    d = torch.log(pred[mask].clamp(min=1e-3)) - torch.log(target[mask])
    return 10 * torch.sqrt(((d ** 2).mean() - lam * d.mean() ** 2).clamp(min=0) + 1e-8)


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """AbsRel, RMSE (m), log10 and the delta < 1.25^k accuracies, over valid pixels."""
    mask = valid_mask(target)
    p, t = pred[mask].clamp(min=1e-3), target[mask]
    ratio = torch.maximum(p / t, t / p)
    return {
        "abs_rel": ((p - t).abs() / t).mean().item(),
        "rmse": torch.sqrt(((p - t) ** 2).mean()).item(),
        "log10": (torch.log10(p) - torch.log10(t)).abs().mean().item(),
        "delta1": (ratio < 1.25).float().mean().item(),
        "delta2": (ratio < 1.25 ** 2).float().mean().item(),
        "delta3": (ratio < 1.25 ** 3).float().mean().item(),
    }
