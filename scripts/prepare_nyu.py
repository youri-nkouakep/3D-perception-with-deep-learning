"""Extract the NYU Depth v2 labeled set into train/test .npz files.

Usage:
    python scripts/prepare_nyu.py --root data/raw/nyu --out data/processed
"""

import argparse
from pathlib import Path

from perception.datasets.nyu import preprocess_nyu


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw/nyu"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=320)
    args = p.parse_args()

    for out in preprocess_nyu(args.root / "nyu_depth_v2_labeled.mat", args.root / "splits.mat",
                              args.out, size=(args.height, args.width)):
        print(f"-> {out}")


if __name__ == "__main__":
    main()
