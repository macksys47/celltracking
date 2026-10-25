from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .data import CTCSequence, discover_sequences, move_sample_to_device
from .linking import write_pair_scores, write_results
from .model import CellDINOTrackingModel


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Infer standalone CellDINO tracking-only project.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", default="/group/dingyi/ct/data")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--target-size", nargs=2, type=int, default=[512, 512])
    p.add_argument("--max-objects", type=int, default=4096)
    p.add_argument("--link-threshold", type=float, default=0.5)
    p.add_argument("--link-mode", choices=["greedy", "global"], default="greedy")
    p.add_argument("--division-threshold", type=float, default=0.5)
    p.add_argument("--mitosis-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    return p


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    model = CellDINOTrackingModel(
        embedding_dim=int(a.get("embedding_dim", 256)),
        heads=int(a.get("heads", 4)),
        backbone=str(a.get("backbone", "resnet50")),
        train_backbone=bool(a.get("train_backbone", False)),
        num_feature_levels=int(a.get("num_feature_levels", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, a


def infer_sequence(model, sequence: CTCSequence, args, device: torch.device, output_root: Path) -> None:
    node_to_track: Dict[int, int] = {}
    tracks: Dict[int, Tuple[int, int, int, int]] = {}
    active_track_by_node: Dict[int, int] = {}
    next_track = 1
    score_rows = []
    target_size = (int(args.target_size[0]), int(args.target_size[1]))

    for start in sequence.valid_pair_starts():
        sample = sequence.build_pair(start, target_size, args.max_objects, deterministic=True)
        sample = move_sample_to_device(sample, device)
        with torch.no_grad():
            out = model(sample)
        prev_idx = out["prev_idx"]
        curr_idx = out["curr_idx"]
        if prev_idx.numel() == 0:
            continue
        pair_scores = out["pair_logits"].sigmoid()
        div_scores = out["division_logits"].sigmoid()
        mitosis_scores = out["mitosis_logits"].sigmoid()
        mitosis_by_prev = {}
        for row_idx, (p, a, b) in enumerate(out["mitosis_pairs"].tolist()):
            score = float(mitosis_scores[row_idx].cpu())
            if score > mitosis_by_prev.get(p, (-1.0, -1, -1))[0]:
                mitosis_by_prev[p] = (score, a, b)

        used_curr = set()
        used_prev = set()
        for p in range(prev_idx.numel()):
            prev_node = int(sample.node_ids[prev_idx[p]].item())
            if prev_node not in node_to_track:
                node_to_track[prev_node] = next_track
                tracks[next_track] = (next_track, int(sample.prev_frame), int(sample.prev_frame), 0)
                next_track += 1
            prev_track = node_to_track[prev_node]
            mitosis = mitosis_by_prev.get(p)
            if mitosis and float(div_scores[p].cpu()) >= args.division_threshold and mitosis[0] >= args.mitosis_threshold:
                child_indices = [mitosis[1], mitosis[2]]
                used_prev.add(p)
            elif args.link_mode == "greedy":
                ranked = torch.argsort(pair_scores[p], descending=True).tolist()
                child_indices = [ranked[0]] if ranked and float(pair_scores[p, ranked[0]].cpu()) >= args.link_threshold else []
            else:
                child_indices = []
            for c in child_indices:
                if c in used_curr:
                    continue
                curr_node = int(sample.node_ids[curr_idx[c]].item())
                score_rows.append((prev_node, curr_node, float(pair_scores[p, c].cpu())))
                used_curr.add(c)
                if len(child_indices) >= 2:
                    track_id = next_track
                    next_track += 1
                    tracks[track_id] = (track_id, int(sample.curr_frame), int(sample.curr_frame), prev_track)
                else:
                    track_id = prev_track
                    old = tracks[track_id]
                    tracks[track_id] = (old[0], old[1], int(sample.curr_frame), old[3])
                node_to_track[curr_node] = track_id

        if args.link_mode == "global":
            prev_candidates = [p for p in range(prev_idx.numel()) if p not in used_prev]
            curr_candidates = [c for c in range(curr_idx.numel()) if c not in used_curr]
            if prev_candidates and curr_candidates:
                scores = pair_scores.detach().cpu().numpy()
                sub_scores = scores[np.ix_(prev_candidates, curr_candidates)]
                row_ind, col_ind = linear_sum_assignment(-sub_scores)
                for row, col in zip(row_ind, col_ind):
                    p = prev_candidates[int(row)]
                    c = curr_candidates[int(col)]
                    score = float(scores[p, c])
                    if score < args.link_threshold:
                        continue
                    if c in used_curr:
                        continue
                    prev_node = int(sample.node_ids[prev_idx[p]].item())
                    curr_node = int(sample.node_ids[curr_idx[c]].item())
                    prev_track = node_to_track[prev_node]
                    score_rows.append((prev_node, curr_node, score))
                    used_curr.add(c)
                    old = tracks[prev_track]
                    tracks[prev_track] = (old[0], old[1], int(sample.curr_frame), old[3])
                    node_to_track[curr_node] = prev_track

        for c in range(curr_idx.numel()):
            curr_node = int(sample.node_ids[curr_idx[c]].item())
            if curr_node in node_to_track:
                continue
            node_to_track[curr_node] = next_track
            tracks[next_track] = (next_track, int(sample.curr_frame), int(sample.curr_frame), 0)
            next_track += 1

    out_dir = output_root / f"{sequence.seq_name}_RES"
    write_results(sequence, node_to_track, tracks, out_dir)
    write_pair_scores(score_rows, out_dir / "cell_dino_pair_scores.csv")


def main() -> None:
    args = parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = Path(args.checkpoint or f"/group/dingyi/ct/results/celldino_tracking/{args.dataset}/checkpoint_best.pth")
    output_dir = Path(args.output_dir or f"/group/dingyi/ct/results/celldino_tracking/{args.dataset}/{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, train_args = load_model(checkpoint, device)
    json.dump({"infer_args": vars(args), "train_args": train_args}, (output_dir / "infer_config.json").open("w"), indent=2)
    for seq_dir in discover_sequences(Path(args.data_dir), args.dataset, args.split):
        print(f"infer celldino {args.dataset}/{args.split}/{seq_dir.name}")
        infer_sequence(model, CTCSequence(seq_dir), args, device, output_dir)


if __name__ == "__main__":
    main()
