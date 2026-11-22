"""Evaluate the best depth model on the NYU test set and build the README figures.

    python scripts/evaluate_depth.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from perception.datasets import NYUDepth
from perception.datasets.nyu import IMAGENET_MEAN, IMAGENET_STD
from perception.metrics import depth_metrics
from perception.models import DepthUNet

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_curves(log_file: Path, out: Path):
    rows = list(csv.DictReader(log_file.open()))
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, title in zip(axes, ["train_loss", "abs_rel", "delta1"],
                              ["Training loss (SILog)", "Test AbsRel (lower is better)",
                               "Test δ < 1.25 (higher is better)"]):
        y = [float(r[key]) for r in rows]
        ax.plot(ep, y, color=BLUE, lw=2)
        ax.annotate(f"{y[-1]:.3f}", (ep[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
        if key == "abs_rel":
            b = int(np.argmin(y))
            ax.margins(y=0.15)
            ax.plot(ep[b], y[b], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2)
            ax.annotate(f"kept: {y[b]:.3f}", (ep[b], y[b]), xytext=(0, -16), textcoords="offset points",
                        ha="center", color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, ep[-1] + 2)
        ax.grid(axis="y", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_examples(rgb, gt, pred, out: Path):
    n = len(rgb)
    fig, axes = plt.subplots(n, 4, figsize=(13, 2.6 * n))
    for i in range(n):
        img = np.clip(rgb[i].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        mask = gt[i] > 1e-3
        vmax = gt[i][mask].max()
        err = np.where(mask, np.abs(pred[i] - gt[i]), np.nan)
        panels = [(img, None, None, "Input image"),
                  (gt[i], "viridis_r", (0, vmax), "Ground truth (Kinect)"),
                  (pred[i], "viridis_r", (0, vmax), "Prediction"),
                  (err, "magma", (0, 1), "Absolute error (0 to 1 m)")]
        for ax, (im, cmap, lim, title) in zip(axes[i], panels):
            ax.imshow(im, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(title, loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("models/depth_best.pt"))
    p.add_argument("--log", type=Path, default=Path("models/depth_log.csv"))
    p.add_argument("--data", type=Path, default=Path("data/processed/nyu_test_240x320.npz"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument("--examples", type=int, nargs="+", default=[0, 120, 260, 400, 560])
    args = p.parse_args()

    model = DepthUNet(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    test_set = NYUDepth(args.data)

    totals, n = {}, 0
    with torch.no_grad():
        for rgb, depth in DataLoader(test_set, batch_size=8):
            pred = model(rgb)
            for i in range(len(rgb)):
                for k, v in depth_metrics(pred[i], depth[i]).items():
                    totals[k] = totals.get(k, 0.0) + v
            n += len(rgb)
        results = {k: round(v / n, 4) for k, v in totals.items()}

        samples = [test_set[i] for i in args.examples]
        rgb = torch.stack([s[0] for s in samples])
        gt = torch.stack([s[1] for s in samples])
        pred = model(rgb)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "depth_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_curves(args.log, args.out / "depth_curves.png")
    plot_examples(rgb.numpy(), gt[:, 0].numpy(), pred[:, 0].numpy(), args.out / "depth_examples.png")
    for k, v in results.items():
        print(f"{k:>8}: {v:.3f}")


if __name__ == "__main__":
    main()
