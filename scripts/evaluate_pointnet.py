"""Évalue le meilleur modèle PointNet et génère les figures du README.

    python scripts/evaluate_pointnet.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

from perception.datasets import ModelNet40
from perception.models import PointNet

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


def predict(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for pts, y in loader:
            preds.append(model(pts)[0].argmax(1).numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def plot_curves(log_file: Path, out: Path):
    rows = list(csv.DictReader(log_file.open()))
    ep = [int(r["epoch"]) for r in rows]
    col = lambda k: [float(r[k]) * (100 if "acc" in k else 1) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    series = [("train_acc", "Entraînement", BLUE), ("test_acc", "Test", ORANGE),
              ("test_mean_class_acc", "Test, moyenne par classe", AQUA)]
    for key, name, color in series:
        y = col(key)
        ax1.plot(ep, y, color=color, lw=2, label=name)
        ax1.annotate(f"{y[-1]:.1f} %", (ep[-1], y[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", color=INK_2, fontsize=9)
    best = int(np.argmax(col("test_acc")))
    ax1.plot(ep[best], col("test_acc")[best], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2)
    ax1.annotate(f"meilleur : {col('test_acc')[best]:.1f} %", (ep[best], col("test_acc")[best]),
                 xytext=(0, 10), textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    ax1.set_title("Précision (%)", loc="left", color=INK)
    ax1.set_xlabel("Époque")
    ax1.legend(frameon=False, loc="lower right")

    ax2.plot(ep, col("train_loss"), color=BLUE, lw=2)
    ax2.set_title("Loss d'entraînement", loc="left", color=INK)
    ax2.set_xlabel("Époque")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xlim(1, ep[-1] + 2)
        for x in (10.5, 20.5):
            ax.axvline(x, color=INK_2, lw=0.8, ls=":")
    ax2.text(10.7, ax2.get_ylim()[1], "lr ÷ 2", color=INK_2, fontsize=8, va="top")
    ax2.text(20.7, ax2.get_ylim()[1], "lr ÷ 2", color=INK_2, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list[str], out: Path):
    norm = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, "#cfe1f7", "#6fa6e8", BLUE, "#123e75"])
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm * 100, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title("Matrice de confusion sur le test (% par vraie classe)", loc="left", color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_per_class(acc: np.ndarray, support: np.ndarray, classes: list[str], out: Path):
    order = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(8, 10))
    y = np.arange(len(classes))
    ax.barh(y, acc[order] * 100, height=0.7, color=BLUE)
    ax.set_yticks(y, [f"{classes[i]} ({support[i]})" for i in order], fontsize=8)
    for yi, i in zip(y, order):
        ax.text(acc[i] * 100 + 1, yi, f"{acc[i] * 100:.0f}", va="center", fontsize=7, color=INK_2)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Précision (%)")
    ax.set_title("Précision par classe (nombre d'exemples de test)", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("models/pointnet_best.pt"))
    p.add_argument("--log", type=Path, default=Path("models/pointnet_log.csv"))
    p.add_argument("--data", type=Path, default=Path("data/processed/modelnet40_test_1024.npz"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    feature_transform = ckpt["args"].get("no_feature_transform") != "True"
    model = PointNet(len(classes), feature_transform=feature_transform)
    model.load_state_dict(ckpt["model"])

    test_set = ModelNet40(args.data)
    preds, labels = predict(model, DataLoader(test_set, batch_size=64))

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    support = cm.sum(axis=1)
    per_class = np.diag(cm) / support

    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:10], off.shape))[0]
    confusions = [{"vraie": classes[i], "predite": classes[j], "nombre": int(off[i, j]),
                   "part": round(off[i, j] / support[i], 3)} for i, j in top]

    results = {
        "precision_globale": round(float((preds == labels).mean()), 4),
        "precision_moyenne_par_classe": round(float(per_class.mean()), 4),
        "par_classe": {c: round(float(a), 3) for c, a in zip(classes, per_class)},
        "confusions_principales": confusions,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pointnet_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    plot_curves(args.log, args.out / "pointnet_curves.png")
    plot_confusion(cm, classes, args.out / "pointnet_confusion.png")
    plot_per_class(per_class, support, classes, args.out / "pointnet_per_class.png")

    print(f"Précision globale : {results['precision_globale']:.3f}")
    print(f"Moyenne par classe : {results['precision_moyenne_par_classe']:.3f}")
    print("Confusions principales :")
    for c in confusions:
        print(f"  {c['vraie']:>12} -> {c['predite']:<12} {c['nombre']:3d}  ({c['part'] * 100:.0f} %)")


if __name__ == "__main__":
    main()
