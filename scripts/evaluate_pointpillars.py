"""Evaluate the best PointPillars model on the KITTI val split and build the README figures.

    python scripts/evaluate_pointpillars.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from perception.boxes import bev_corners
from perception.datasets.kitti import PC_RANGE, KittiCars, collate
from perception.kitti_eval import Frame, evaluate
from perception.models.pointpillars import PointPillarsCenter, decode

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_curves(log_file: Path, out: Path):
    rows = list(csv.DictReader(log_file.open()))
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    ax = axes[0]
    for key, label, color in [("heatmap_loss", "heatmap", BLUE), ("reg_loss", "box", ORANGE)]:
        y = [float(r[key]) for r in rows]
        ax.plot(ep, y, color=color, lw=2)
        ax.annotate(label, (ep[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=color, fontsize=9)
    ax.set_title("Training loss", loc="left", color=INK)

    ax = axes[1]
    evals = [r for r in rows if r["3d_moderate"]]
    e = [int(r["epoch"]) for r in evals]
    for key, label, color in [("bev_moderate", "BEV", BLUE), ("3d_moderate", "3D", ORANGE)]:
        y = [float(r[key]) for r in evals]
        ax.plot(e, y, "-o", color=color, lw=2, ms=6, mec=SURFACE, mew=1.5)
        ax.annotate(f"{label} {y[-1]:.1f}", (e[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=color, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Val AP moderate, car, IoU 0.7 (%)", loc="left", color=INK)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, ep[-1] + 3)
        ax.grid(axis="y", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_bev(samples, out: Path):
    fig, axes = plt.subplots(1, len(samples), figsize=(4.2 * len(samples), 4.6))
    for ax, (idx, points, gt, pred) in zip(axes, samples):
        ax.scatter(points[:, 1], points[:, 0], s=0.3, c=INK_2, alpha=0.35, linewidths=0)
        for boxes, color in [(gt, AQUA), (pred, ORANGE)]:
            for c in bev_corners(boxes):
                c = np.vstack([c, c[:1]])
                ax.plot(c[:, 1], c[:, 0], color=color, lw=1.6)
        ax.set_xlim(PC_RANGE[4], PC_RANGE[1])
        ax.set_ylim(PC_RANGE[0], 50)
        ax.set_aspect("equal")
        ax.set_title(f"scene {idx}", loc="left", color=INK)
        ax.set_xlabel("y (m)")
    axes[0].set_ylabel("x, forward (m)")
    axes[0].plot([], [], color=AQUA, lw=2, label="ground truth")
    axes[0].plot([], [], color=ORANGE, lw=2, label="prediction")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("models/pointpillars_best.pt"))
    p.add_argument("--log", type=Path, default=Path("models/pointpillars_log.csv"))
    p.add_argument("--data", type=Path, default=Path("data/processed/kitti"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument("--examples", type=int, nargs="+", default=[400, 1200, 2050, 3525])
    p.add_argument("--min-score", type=float, default=0.3, help="score threshold for the figure only")
    args = p.parse_args()

    model = PointPillarsCenter()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    val = KittiCars(args.data, "val")

    frames, preds = [], {}
    with torch.no_grad():
        for points, _, ids in DataLoader(val, batch_size=4, collate_fn=collate):
            for idx, (boxes, scores) in zip(ids, decode(model(points))):
                frames.append(Frame(val.labels(idx), val.calib(idx), boxes, scores))
                preds[idx] = boxes[scores >= args.min_score]
    results = evaluate(frames)

    samples = []
    for i in args.examples:
        points, gt, idx = val[i]
        samples.append((idx, points.numpy(), gt.numpy(), preds[idx]))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pointpillars_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_curves(args.log, args.out / "pointpillars_curves.png")
    plot_bev(samples, args.out / "pointpillars_bev.png")
    print(f"{'':>4} {'easy':>7} {'moderate':>9} {'hard':>7}")
    for m, v in results.items():
        print(f"{m:>4} {v['easy']:7.2f} {v['moderate']:9.2f} {v['hard']:7.2f}")


if __name__ == "__main__":
    main()
