"""Affiche des nuages de points ModelNet40.

    python scripts/visualize_modelnet40.py                  # grille d'exemples -> PNG
    python scripts/visualize_modelnet40.py --open3d --index 0  # vue 3D interactive
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=Path("data/processed/modelnet40_train_1024.npz"))
    p.add_argument("--out", type=Path, default=Path("docs/modelnet40_samples.png"))
    p.add_argument("--open3d", action="store_true")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    data = np.load(args.file)
    points, labels, classes = data["points"], data["labels"], data["classes"]

    if args.open3d:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[args.index]))
        pcd.paint_uniform_color([0.1, 0.5, 0.9])
        o3d.visualization.draw_geometries([pcd], window_name=str(classes[labels[args.index]]))
        return

    # un exemple pour 12 classes différentes
    rng = np.random.default_rng(1)
    picks = [rng.choice(np.flatnonzero(labels == c)) for c in rng.choice(len(classes), 12, replace=False)]
    fig = plt.figure(figsize=(14, 10))
    for k, i in enumerate(picks):
        ax = fig.add_subplot(3, 4, k + 1, projection="3d")
        x, y, z = points[i].T
        ax.scatter(x, y, z, s=1.5, c=z, cmap="viridis")
        ax.set_title(str(classes[labels[i]]))
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
