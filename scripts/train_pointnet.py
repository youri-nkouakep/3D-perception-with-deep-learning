"""Train PointNet on ModelNet40 and save the best model.

Usage:
    python scripts/train_pointnet.py --epochs 50
    python scripts/train_pointnet.py --epochs 2 --limit 512   # quick check
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from perception.datasets import ModelNet40
from perception.models import PointNet, pointnet_loss


@torch.no_grad()
def evaluate(model, loader, n_classes, device):
    model.eval()
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    for pts, labels in loader:
        preds = model(pts.to(device))[0].argmax(1).cpu().numpy()
        labels = labels.numpy()
        np.add.at(total, labels, 1)
        np.add.at(correct, labels, preds == labels)
    overall = correct.sum() / total.sum()
    mean_class = np.mean(correct[total > 0] / total[total > 0])
    return overall, mean_class


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--n-points", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-step", type=int, default=20,
                   help="halve the learning rate every N epochs")
    p.add_argument("--resume", action="store_true",
                   help="resume from models/pointnet_last.pt after an interruption")
    p.add_argument("--no-feature-transform", action="store_true",
                   help="disable the 64x64 T-Net (about 2x faster on CPU)")
    p.add_argument("--limit", type=int, default=None, help="use only N samples (debug)")
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    train_set = ModelNet40(args.data / f"modelnet40_train_{args.n_points}.npz", train=True)
    test_set = ModelNet40(args.data / f"modelnet40_test_{args.n_points}.npz", train=False)
    classes = train_set.classes
    if args.limit:
        rng = np.random.default_rng(0)
        train_set = Subset(train_set, rng.permutation(len(train_set))[: args.limit])
        test_set = Subset(test_set, rng.permutation(len(test_set))[: args.limit // 4])

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers)
    test_loader = DataLoader(test_set, 64, num_workers=args.workers)

    model = PointNet(len(classes), feature_transform=not args.no_feature_transform).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=0.5)
    print(f"device={device}  train={len(train_set)}  test={len(test_set)}  "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "pointnet_log.csv"
    last_path = args.out / "pointnet_last.pt"
    best, start = 0.0, 1
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        best, start = ckpt["best"], ckpt["epoch"] + 1
        print(f"Resuming at epoch {start} (best accuracy so far: {best:.3f})")

    with log_path.open("a" if start > 1 else "w", newline="") as f:
        log = csv.writer(f)
        if start == 1:
            log.writerow(["epoch", "train_loss", "train_acc", "test_acc", "test_mean_class_acc", "seconds"])
        for epoch in range(start, args.epochs + 1):
            model.train()
            t0, losses, hits, seen = time.time(), [], 0, 0
            for pts, labels in train_loader:
                pts, labels = pts.to(device), labels.to(device)
                logits, feat_mat = model(pts)
                loss = pointnet_loss(logits, labels, feat_mat, label_smoothing=0.2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                hits += (logits.argmax(1) == labels).sum().item()
                seen += len(labels)
            sched.step()

            acc, mean_cls = evaluate(model, test_loader, len(classes), device)
            dt = time.time() - t0
            log.writerow([epoch, np.mean(losses), hits / seen, acc, mean_cls, round(dt, 1)])
            f.flush()
            flag = ""
            if acc > best:
                best = float(acc)
                torch.save({"model": model.state_dict(), "classes": classes,
                            "args": {k: str(v) for k, v in vars(args).items()}},
                           args.out / "pointnet_best.pt")
                flag = "  <- best"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "epoch": epoch, "best": best}, last_path)
            print(f"epoch {epoch:3d} | loss {np.mean(losses):.3f} | train {hits / seen:.3f} | "
                  f"test {acc:.3f} (class avg {mean_cls:.3f}) | {dt:.0f}s{flag}", flush=True)

    print(f"Best test accuracy: {best:.3f}  (log: {log_path})")


if __name__ == "__main__":
    main()
