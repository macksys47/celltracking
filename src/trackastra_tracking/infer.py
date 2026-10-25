from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .data import CTCSequence, discover_sequences, move_sample_to_device
from .linking import Edge, build_tracks, greedy_link, write_edges, write_results
from .model import TrackastraModel


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Infer standalone Trackastra-style association project.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", default="/group/dingyi/ct/data")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--window-size", type=int, default=6)
    p.add_argument("--target-size", nargs=2, type=int, default=[512, 512])
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--link-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    return p


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    model = TrackastraModel(
        embedding_dim=int(a.get("embedding_dim", 256)),
        layers=int(a.get("layers", 6)),
        heads=int(a.get("heads", 4)),
        dim_feedforward=int(a.get("dim_feedforward", 512)),
        backbone=str(a.get("backbone", "resnet50")),
        train_backbone=bool(a.get("train_backbone", False)),
        num_feature_levels=int(a.get("num_feature_levels", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, a


def infer_sequence(model, sequence: CTCSequence, args, device: torch.device, output_root: Path) -> None:
    target_size = (int(args.target_size[0]), int(args.target_size[1]))
    starts = sequence.valid_window_starts(args.window_size)
    if not starts:
        starts = [0]
    aggregate = defaultdict(lambda: [0.0, 0])
    for start in starts:
        sample = sequence.build_window(start, args.window_size, target_size, args.max_tokens, deterministic=True)
        sample = move_sample_to_device(sample, device)
        with torch.no_grad():
            scores = model(sample)["assoc_logits"].sigmoid()
        if sample.frame_indices.numel() == 0:
            continue
        max_frame = int(sample.frame_indices.max().item())
        for local_frame in range(max_frame):
            source_idx = torch.nonzero(sample.frame_indices == local_frame, as_tuple=False).flatten()
            target_idx = torch.nonzero(sample.frame_indices == local_frame + 1, as_tuple=False).flatten()
            if source_idx.numel() == 0 or target_idx.numel() == 0:
                continue
            edge_scores = scores.index_select(0, source_idx).index_select(1, target_idx).detach().cpu().reshape(-1).tolist()
            source_nodes = sample.node_ids.index_select(0, source_idx).detach().cpu().repeat_interleave(target_idx.numel()).tolist()
            target_nodes = sample.node_ids.index_select(0, target_idx).detach().cpu().repeat(source_idx.numel()).tolist()
            for src, dst, score in zip(source_nodes, target_nodes, edge_scores):
                values = aggregate[(int(src), int(dst))]
                values[0] += float(score)
                values[1] += 1
    edges = [Edge(src, dst, total / count) for (src, dst), (total, count) in aggregate.items()]
    selected = greedy_link(edges, args.link_threshold)
    node_to_track, tracks = build_tracks(sequence, selected)
    out_dir = output_root / f"{sequence.seq_name}_RES"
    write_results(sequence, node_to_track, tracks, out_dir)
    write_edges(edges, out_dir / "trackastra_edges.csv")


def main() -> None:
    args = parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = Path(args.checkpoint or f"/group/dingyi/ct/results/trackastra_tracking/{args.dataset}/checkpoint_best.pth")
    output_dir = Path(args.output_dir or f"/group/dingyi/ct/results/trackastra_tracking/{args.dataset}/{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, train_args = load_model(checkpoint, device)
    json.dump({"infer_args": vars(args), "train_args": train_args}, (output_dir / "infer_config.json").open("w"), indent=2)
    for seq_dir in discover_sequences(Path(args.data_dir), args.dataset, args.split):
        print(f"infer trackastra {args.dataset}/{args.split}/{seq_dir.name}")
        infer_sequence(model, CTCSequence(seq_dir), args, device, output_dir)


if __name__ == "__main__":
    main()
