"""Train the depth U-Net on NYU Depth v2 and save the best model.

Usage:
    python scripts/train_depth.py --epochs 20
    python scripts/train_depth.py --epochs 1 --limit 32   # quick check
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from perception.datasets import NYUDepth
from perception.metrics import depth_metrics, silog_loss
from perception.models import DepthUNet

METRICS = ["abs_rel", "rmse", "log10", "delta1", "delta2", "delta3"]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals, n = dict.fromkeys(METRICS, 0.0), 0
    for rgb, depth in loader:
        pred = model(rgb.to(device))
        for i in range(len(rgb)):
            for k, v in depth_metrics(pred[i], depth[i].to(device)).items():
                totals[k] += v
        n += len(rgb)
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--size", default="240x320")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--limit", type=int, default=None, help="use only N samples (debug)")
    p.add_argument("--resume", action="store_true", help="resume from models/depth_last.pt")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    train_set = NYUDepth(args.data / f"nyu_train_{args.size}.npz", train=True)
    test_set = NYUDepth(args.data / f"nyu_test_{args.size}.npz")
    if args.limit:
        train_set = Subset(train_set, range(args.limit))
        test_set = Subset(test_set, range(max(args.limit // 4, 1)))
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, 8)

    model = DepthUNet(pretrained=True).to(device)
    # the pretrained encoder learns 10x slower than the new decoder
    encoder = [p for n, p in model.named_parameters() if n.startswith(("stem", "enc"))]
    decoder = [p for n, p in model.named_parameters() if not n.startswith(("stem", "enc"))]
    opt = torch.optim.AdamW([{"params": encoder, "lr": args.lr / 10}, {"params": decoder}],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"device={device}  train={len(train_set)}  test={len(test_set)}  "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    args.out.mkdir(parents=True, exist_ok=True)
    log_path, last_path = args.out / "depth_log.csv", args.out / "depth_last.pt"
    best, start = float("inf"), 1
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        best, start = ckpt["best"], ckpt["epoch"] + 1
        print(f"Resuming at epoch {start} (best AbsRel so far: {best:.3f})")

    with log_path.open("a" if start > 1 else "w", newline="") as f:
        log = csv.writer(f)
        if start == 1:
            log.writerow(["epoch", "train_loss", *METRICS, "seconds"])
        for epoch in range(start, args.epochs + 1):
            model.train()
            t0, losses = time.time(), []
            for rgb, depth in train_loader:
                loss = silog_loss(model(rgb.to(device)), depth.to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()

            m = evaluate(model, test_loader, device)
            dt = time.time() - t0
            log.writerow([epoch, np.mean(losses), *(m[k] for k in METRICS), round(dt, 1)])
            f.flush()
            flag = ""
            if m["abs_rel"] < best:
                best = m["abs_rel"]
                torch.save({"model": model.state_dict(), "args": {k: str(v) for k, v in vars(args).items()}},
                           args.out / "depth_best.pt")
                flag = "  <- best"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "epoch": epoch, "best": best}, last_path)
            print(f"epoch {epoch:3d} | loss {np.mean(losses):.3f} | AbsRel {m['abs_rel']:.3f} | "
                  f"RMSE {m['rmse']:.3f} m | d1 {m['delta1']:.3f} | {dt:.0f}s{flag}", flush=True)

    print(f"Best test AbsRel: {best:.3f}  (log: {log_path})")


if __name__ == "__main__":
    main()
