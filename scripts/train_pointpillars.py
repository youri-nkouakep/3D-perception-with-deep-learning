"""Train the PointPillars car detector on the KITTI train split, evaluate on val.

Usage:
    python scripts/train_pointpillars.py --epochs 20
    python scripts/train_pointpillars.py --epochs 1 --limit 40   # quick check
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from perception.datasets.kitti import KittiCars, collate
from perception.kitti_eval import Frame, evaluate
from perception.models.pointpillars import PointPillarsCenter, decode, detection_loss


@torch.no_grad()
def run_eval(model, dataset, base: KittiCars, device, batch_size: int = 4):
    model.eval()
    frames = []
    for points, _, ids in DataLoader(dataset, batch_size, collate_fn=collate):
        for idx, (boxes, scores) in zip(ids, decode(model(points))):
            frames.append(Frame(base.labels(idx), base.calib(idx), boxes, scores))
    return evaluate(frames)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/kitti"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="use only N samples (debug)")
    p.add_argument("--resume", action="store_true", help="resume from models/pointpillars_last.pt")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    train_set = KittiCars(args.data, "train", train=True)
    val_base = KittiCars(args.data, "val")
    val_set = val_base
    if args.limit:
        train_set = Subset(train_set, range(args.limit))
        val_set = Subset(val_base, range(max(args.limit // 2, 1)))
    loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)

    model = PointPillarsCenter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader),
                                                pct_start=0.3)
    print(f"device={device}  train={len(train_set)}  val={len(val_set)}  "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    args.out.mkdir(parents=True, exist_ok=True)
    log_path, last_path = args.out / "pointpillars_log.csv", args.out / "pointpillars_last.pt"
    best, start = -1.0, 1
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        best, start = ckpt["best"], ckpt["epoch"] + 1
        print(f"Resuming at epoch {start} (best moderate 3D AP so far: {best:.2f})")

    with log_path.open("a" if start > 1 else "w", newline="") as f:
        log = csv.writer(f)
        if start == 1:
            log.writerow(["epoch", "loss", "heatmap_loss", "reg_loss",
                          "bev_easy", "bev_moderate", "bev_hard", "3d_easy", "3d_moderate", "3d_hard", "seconds"])
        for epoch in range(start, args.epochs + 1):
            model.train()
            t0, losses = time.time(), []
            for points, boxes, _ in loader:
                loss, hm, reg = detection_loss(model(points), boxes)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                opt.step()
                sched.step()
                losses.append([loss.item(), hm.item(), reg.item()])
            mean = np.mean(losses, axis=0)

            ap = None
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                ap = run_eval(model, val_set, val_base, device)
            dt = time.time() - t0
            cells = [ap[m][d] for m in ("bev", "3d") for d in ("easy", "moderate", "hard")] if ap else [""] * 6
            log.writerow([epoch, *mean, *cells, round(dt, 1)])
            f.flush()

            msg = f"epoch {epoch:3d} | loss {mean[0]:.3f} (heatmap {mean[1]:.3f}, box {mean[2]:.3f})"
            if ap:
                msg += f" | AP3D moderate {ap['3d']['moderate']:.2f} | APBEV moderate {ap['bev']['moderate']:.2f}"
                if ap["3d"]["moderate"] > best:
                    best = ap["3d"]["moderate"]
                    torch.save({"model": model.state_dict()}, args.out / "pointpillars_best.pt")
                    msg += "  <- best"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "epoch": epoch, "best": best}, last_path)
            print(f"{msg} | {dt:.0f}s", flush=True)

    print(f"Best moderate 3D AP: {best:.2f}  (log: {log_path})")


if __name__ == "__main__":
    main()
