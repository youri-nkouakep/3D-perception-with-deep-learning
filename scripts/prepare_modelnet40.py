"""Convert the ModelNet40 meshes (.off) into cached point clouds.

Usage:
    python scripts/prepare_modelnet40.py --root data/raw/ModelNet40 --out data/processed
"""

import argparse
from pathlib import Path

from perception.datasets.modelnet import preprocess_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/ModelNet40"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for split in ("train", "test"):
        out = args.out / f"modelnet40_{split}_{args.n_points}.npz"
        preprocess_split(args.root, split, out, n_points=args.n_points, workers=args.workers)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
