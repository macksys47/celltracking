from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CellDINOPairDataset, collate_one, move_sample_to_device
from .model import CellDINOTrackingLoss, CellDINOTrackingModel


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train standalone CellDINO tracking-only project.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", default="/group/dingyi/ct/data")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--target-size", nargs=2, type=int, default=[512, 512])
    p.add_argument("--max-objects", type=int, default=2048)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--num-feature-levels", type=int, default=3)
    p.add_argument("--train-backbone", action="store_true")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=248)
    p.add_argument("--device", default="cuda")
    return p


def make_optimizer(model, lr: float, lr_backbone: float, weight_decay: float):
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.backbone"):
            backbone_params.append(param)
        else:
            other_params.append(param)
    groups = [{"params": other_params, "lr": lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def run_epoch(model, criterion, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    for sample in loader:
        sample = move_sample_to_device(sample, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            losses = criterion(model(sample), sample)
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir or f"/group/dingyi/ct/results/celldino_tracking/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2)
    target_size = (int(args.target_size[0]), int(args.target_size[1]))
    train_set = CellDINOPairDataset(Path(args.data_dir), args.dataset, "train", target_size, args.max_objects)
    val_set = CellDINOPairDataset(Path(args.data_dir), args.dataset, "val", target_size, args.max_objects)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_one)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_one)
    model = CellDINOTrackingModel(args.embedding_dim, args.heads, args.backbone, args.train_backbone, args.num_feature_levels).to(device)
    criterion = CellDINOTrackingLoss().to(device)
    optimizer = make_optimizer(model, args.lr, args.lr_backbone, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    best = float("inf")
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "split", "loss", "loss_pair", "loss_division", "loss_mitosis"])
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, criterion, train_loader, device, optimizer)
        val_metrics = run_epoch(model, criterion, val_loader, device)
        scheduler.step()
        with (out_dir / "metrics.csv").open("a", newline="") as f:
            writer = csv.writer(f)
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                writer.writerow([epoch, split] + [f"{metrics.get(k, 0.0):.8f}" for k in ("loss", "loss_pair", "loss_division", "loss_mitosis")])
        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_loss": val_metrics.get("loss", float("inf"))}
        torch.save(ckpt, out_dir / "checkpoint.pth")
        if ckpt["val_loss"] < best:
            best = ckpt["val_loss"]
            torch.save(ckpt, out_dir / "checkpoint_best.pth")
        print(f"epoch {epoch:03d} train={train_metrics.get('loss', 0.0):.6f} val={val_metrics.get('loss', 0.0):.6f}")


if __name__ == "__main__":
    main()

