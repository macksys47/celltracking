#!/usr/bin/env python3
"""COLA duplicate-trajectory and intermediate-layer analysis."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_case_stats as fstats


DATASETS = {
    "3T3": "3t3",
    "HEK293": "hek",
    "HeLa": "hela",
    "PC-3": "pc",
    "RAW264": "raw",
}
LAYERS = [0, 1, 2, 3]
LOSS_COMPONENTS = ["bbox", "giou"]
COLA_NAME = "CoMOT"


def save_figure(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        return pickle.load(f)


def epoch_mean(metrics: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in metrics:
        return np.array([])
    arr = np.asarray(metrics[key], dtype=float)
    valid_epoch = ~np.isnan(arr).all(axis=-1)
    means = np.full(arr.shape[0], np.nan, dtype=float)
    if valid_epoch.any():
        means[valid_epoch] = np.nanmean(arr[valid_epoch], axis=-1)
    return means


def final_valid(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid[-1]) if valid.size else np.nan


def first_valid(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid[0]) if valid.size else np.nan


def analyze_cola_layers(root: Path, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    curves: Dict[Tuple[str, str, str, int], np.ndarray] = {}

    for dataset in DATASETS:
        dataset_dir = root / dataset
        split_metrics = {
            "train": load_metrics(dataset_dir / "metrics_train.pkl"),
            "val": load_metrics(dataset_dir / "metrics_val.pkl"),
        }
        for split, metrics in split_metrics.items():
            for layer in LAYERS:
                for component in LOSS_COMPONENTS:
                    key = f"{COLA_NAME}_loss_{component}_{layer}"
                    means = epoch_mean(metrics, key)
                    curves[(dataset, split, component, layer)] = means
                    first = first_valid(means)
                    final = final_valid(means)
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "component": component,
                            "layer": layer,
                            "valid_epochs": int(np.sum(~np.isnan(means))),
                            "first_epoch_mean": round(first, 6) if not np.isnan(first) else "",
                            "final_epoch_mean": round(final, 6) if not np.isnan(final) else "",
                            "min_epoch_mean": round(float(np.nanmin(means)), 6) if np.any(~np.isnan(means)) else "",
                            "relative_reduction": round((first - final) / first, 6) if first and not np.isnan(first) and not np.isnan(final) else "",
                        }
                    )

    layer_df = pd.DataFrame(rows)
    layer_df.to_csv(out_dir / "cola_intermediate_layer_loss_summary.csv", index=False)
    plot_cola_layer_curves(curves, out_dir)
    plot_cola_layer_heatmaps(layer_df, out_dir)
    copy_existing_cola_plots(root, out_dir)
    return layer_df


def plot_cola_layer_curves(curves: Dict[Tuple[str, str, str, int], np.ndarray], out_dir: Path) -> None:
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(LAYERS)))
    fig, axes = plt.subplots(len(DATASETS), len(LOSS_COMPONENTS), figsize=(13, 12), sharex=True)
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, component in enumerate(LOSS_COMPONENTS):
            ax = axes[row_idx, col_idx]
            for layer_idx, layer in enumerate(LAYERS):
                train = curves.get((dataset, "train", component, layer), np.array([]))
                val = curves.get((dataset, "val", component, layer), np.array([]))
                if train.size:
                    ax.plot(np.arange(1, len(train) + 1), train, color=colors[layer_idx], linewidth=1.1, alpha=0.55, linestyle="--")
                if val.size:
                    ax.plot(np.arange(1, len(val) + 1), val, color=colors[layer_idx], linewidth=1.6, label=f"Layer {layer}")
            if row_idx == 0:
                ax.set_title(f"COLA {component.upper()} loss")
            if col_idx == 0:
                ax.set_ylabel(dataset)
            ax.grid(color="#dddddd", linewidth=0.35)
            ax.spines[["top", "right"]].set_visible(False)
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("Epoch")
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("COLA Intermediate-Layer Loss Curves\nsolid: validation, dashed: training", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir / "cola_intermediate_layer_loss_curves")


def plot_cola_layer_heatmaps(layer_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(LOSS_COMPONENTS), figsize=(10.5, 4.1), constrained_layout=True)
    for ax, component in zip(axes, LOSS_COMPONENTS):
        sub = layer_df[(layer_df["split"] == "val") & (layer_df["component"] == component)].copy()
        mat = np.full((len(DATASETS), len(LAYERS)), np.nan)
        for i, dataset in enumerate(DATASETS):
            for j, layer in enumerate(LAYERS):
                row = sub[(sub["dataset"] == dataset) & (sub["layer"] == layer)]
                if not row.empty and row["final_epoch_mean"].iloc[0] != "":
                    mat[i, j] = float(row["final_epoch_mean"].iloc[0])
        im = ax.imshow(mat, aspect="auto", cmap="mako" if "mako" in plt.colormaps() else "viridis")
        ax.set_title(f"Final Val {component.upper()} Loss")
        ax.set_xticks(range(len(LAYERS)), [f"L{l}" for l in LAYERS])
        ax.set_yticks(range(len(DATASETS)), list(DATASETS))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("COLA Layer-Wise Final Validation Loss", fontsize=13, fontweight="bold")
    save_figure(fig, out_dir / "cola_layer_final_loss_heatmap")


def copy_existing_cola_plots(root: Path, out_dir: Path) -> None:
    plot_dir = out_dir / "copied_aux_loss_CoMOT_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        src = root / dataset / "aux_loss_CoMOT_plot.png"
        if src.exists():
            dst = plot_dir / f"{dataset}_aux_loss_CoMOT_plot.png"
            dst.write_bytes(src.read_bytes())


def sequence_dirs(root: Path, dataset: str) -> List[str]:
    ctc = root / dataset / "test" / "CTC"
    return sorted(path.name.replace("_RES", "") for path in ctc.glob("*_RES"))


def build_match_histories(
    root: Path, data_root: Path, dataset: str, sequence: str
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, int]], Dict[int, Dict[str, int]], Dict[int, Dict[str, int]]]:
    data_name = DATASETS[dataset]
    gt_dir = data_root / data_name / "test" / f"{sequence}_GT" / "TRA"
    pred_dir = root / dataset / "test" / "CTC" / f"{sequence}_RES"
    gt_files = fstats.mask_files(gt_dir, "man_track")
    pred_files = fstats.mask_files(pred_dir, "mask")
    gt_tracks = fstats.parse_track_file(gt_dir / "man_track.txt")
    pred_tracks = fstats.parse_track_file(pred_dir / "res_track.txt")
    gt_to_pred_history: Dict[int, Dict[int, int]] = defaultdict(dict)
    pred_to_gt_history: Dict[int, Dict[int, int]] = defaultdict(dict)
    for frame in sorted(set(gt_files) & set(pred_files)):
        gt = tifffile.imread(str(gt_files[frame]))
        pred = tifffile.imread(str(pred_files[frame]))
        _, _, _, pairs = fstats.compute_pair_stats(gt, pred)
        gt_to_pred, pred_to_gt, _ = fstats.greedy_match(pairs, fstats.IOU_THRESHOLD)
        for gt_id, pred_id in gt_to_pred.items():
            gt_to_pred_history[gt_id][frame] = pred_id
            pred_to_gt_history[pred_id][frame] = gt_id
    return gt_to_pred_history, pred_to_gt_history, gt_tracks, pred_tracks


def analyze_duplicate_trajectories(root: Path, data_root: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for dataset in DATASETS:
        summary = defaultdict(int)
        unique_pred_counts = []
        unique_gt_counts = []
        for sequence in sequence_dirs(root, dataset):
            gt_hist, pred_hist, gt_tracks, pred_tracks = build_match_histories(root, data_root, dataset, sequence)
            for gt_id, info in sorted(gt_tracks.items()):
                frames = range(info["start"], info["end"] + 1)
                matches = {frame: gt_hist.get(gt_id, {}).get(frame) for frame in frames}
                matched_pred_ids = [pred_id for pred_id in matches.values() if pred_id is not None]
                unique_pred_ids = sorted(set(matched_pred_ids))
                frame_to_pred = ";".join(f"{frame}:{pred_id}" for frame, pred_id in matches.items() if pred_id is not None)
                unique_pred_counts.append(len(unique_pred_ids))
                duplicate = int(len(unique_pred_ids) >= 2)
                summary["gt_tracks"] += 1
                summary["gt_duplicate_tracks"] += duplicate
                summary["gt_unmatched_tracks"] += int(len(unique_pred_ids) == 0)
                gt_rows.append(
                    {
                        "dataset": dataset,
                        "sequence": sequence,
                        "gt_track_id": gt_id,
                        "start": info["start"],
                        "end": info["end"],
                        "track_frames": len(list(frames)),
                        "matched_frames": len(matched_pred_ids),
                        "unique_pred_tracks": len(unique_pred_ids),
                        "pred_track_ids": ",".join(map(str, unique_pred_ids)),
                        "frame_to_pred": frame_to_pred,
                        "duplicate_or_fragmented": duplicate,
                        "missing_ratio": round(1 - len(matched_pred_ids) / max(1, info["end"] - info["start"] + 1), 6),
                    }
                )

            for pred_id, info in sorted(pred_tracks.items()):
                frames = range(info["start"], info["end"] + 1)
                matches = {frame: pred_hist.get(pred_id, {}).get(frame) for frame in frames}
                matched_gt_ids = [gt_id for gt_id in matches.values() if gt_id is not None]
                unique_gt_ids = sorted(set(matched_gt_ids))
                unique_gt_counts.append(len(unique_gt_ids))
                multi_gt = int(len(unique_gt_ids) >= 2)
                summary["pred_tracks"] += 1
                summary["pred_multi_gt_tracks"] += multi_gt
                summary["pred_unmatched_tracks"] += int(len(unique_gt_ids) == 0)
                pred_rows.append(
                    {
                        "dataset": dataset,
                        "sequence": sequence,
                        "pred_track_id": pred_id,
                        "start": info["start"],
                        "end": info["end"],
                        "track_frames": len(list(frames)),
                        "matched_frames": len(matched_gt_ids),
                        "unique_gt_tracks": len(unique_gt_ids),
                        "gt_track_ids": ",".join(map(str, unique_gt_ids)),
                        "multi_gt_or_merge": multi_gt,
                        "unmatched_ratio": round(1 - len(matched_gt_ids) / max(1, info["end"] - info["start"] + 1), 6),
                    }
                )

        gt_total = max(1, summary["gt_tracks"])
        pred_total = max(1, summary["pred_tracks"])
        summary_rows.append(
            {
                "dataset": dataset,
                "gt_tracks": summary["gt_tracks"],
                "gt_duplicate_tracks": summary["gt_duplicate_tracks"],
                "gt_duplicate_rate": round(summary["gt_duplicate_tracks"] / gt_total, 6),
                "gt_unmatched_tracks": summary["gt_unmatched_tracks"],
                "gt_unmatched_rate": round(summary["gt_unmatched_tracks"] / gt_total, 6),
                "mean_unique_pred_per_gt": round(float(np.mean(unique_pred_counts)) if unique_pred_counts else 0.0, 6),
                "max_unique_pred_per_gt": int(max(unique_pred_counts) if unique_pred_counts else 0),
                "pred_tracks": summary["pred_tracks"],
                "pred_multi_gt_tracks": summary["pred_multi_gt_tracks"],
                "pred_multi_gt_rate": round(summary["pred_multi_gt_tracks"] / pred_total, 6),
                "pred_unmatched_tracks": summary["pred_unmatched_tracks"],
                "pred_unmatched_rate": round(summary["pred_unmatched_tracks"] / pred_total, 6),
                "mean_unique_gt_per_pred": round(float(np.mean(unique_gt_counts)) if unique_gt_counts else 0.0, 6),
                "max_unique_gt_per_pred": int(max(unique_gt_counts) if unique_gt_counts else 0),
            }
        )

    gt_df = pd.DataFrame(gt_rows)
    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows)
    gt_df.to_csv(out_dir / "gt_to_pred_duplicate_tracks.csv", index=False)
    pred_df.to_csv(out_dir / "pred_to_gt_multi_gt_tracks.csv", index=False)
    summary_df.to_csv(out_dir / "duplicate_trajectory_summary.csv", index=False)
    plot_duplicate_summary(summary_df, out_dir)
    plot_top_duplicate_timeline(gt_df, out_dir)
    return summary_df, gt_df, pred_df


def plot_duplicate_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(summary_df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    bars = [
        ("gt_duplicate_rate", "GT -> multiple pred", "#ff8c1a"),
        ("gt_unmatched_rate", "GT unmatched", "#ff4d4d"),
        ("pred_multi_gt_rate", "Pred -> multiple GT", "#23b7ff"),
        ("pred_unmatched_rate", "Pred unmatched", "#7f7f7f"),
    ]
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(bars))
    for offset, (col, label, color) in zip(offsets, bars):
        vals = summary_df[col].to_numpy(dtype=float)
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        for xi, value in zip(x + offset, vals):
            ax.text(xi, value + 0.012, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x, summary_df["dataset"])
    ax.set_ylim(0, max(0.12, min(1.0, summary_df[["gt_duplicate_rate", "gt_unmatched_rate", "pred_multi_gt_rate", "pred_unmatched_rate"]].to_numpy().max() * 1.35)))
    ax.set_ylabel("Track proportion")
    ax.set_title("Duplicate / Ambiguous Trajectory Rates", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.35)
    save_figure(fig, out_dir / "duplicate_trajectory_rates")


def segments_from_ids(start: int, end: int, ids: List[Optional[int]]) -> List[Tuple[int, int, Optional[int]]]:
    segments: List[Tuple[int, int, Optional[int]]] = []
    current = ids[0] if ids else None
    seg_start = start
    for frame, value in zip(range(start + 1, end + 1), ids[1:]):
        if value != current:
            segments.append((seg_start, frame - 1, current))
            current = value
            seg_start = frame
    segments.append((seg_start, end, current))
    return segments


def plot_top_duplicate_timeline(gt_df: pd.DataFrame, out_dir: Path) -> None:
    top = gt_df.sort_values(["unique_pred_tracks", "missing_ratio", "track_frames"], ascending=[False, False, False]).head(12)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(12.5, max(4.0, 0.45 * len(top) + 1.0)))
    cmap = plt.get_cmap("tab20")
    color_cache: Dict[str, object] = {}
    for y, (_, row) in enumerate(top.iterrows()):
        start, end = int(row["start"]), int(row["end"])
        frame_to_pred: Dict[int, Optional[int]] = {}
        for pair in str(row.get("frame_to_pred", "")).split(";"):
            if ":" not in pair:
                continue
            frame_str, pred_str = pair.split(":", 1)
            frame_to_pred[int(frame_str)] = int(pred_str)
        ids = [frame_to_pred.get(frame) for frame in range(start, end + 1)]
        for seg_start, seg_end, pred_id in segments_from_ids(start, end, ids):
            seg_width = seg_end - seg_start + 1
            if pred_id is None:
                ax.broken_barh([(seg_start, seg_width)], (y - 0.3, 0.6), facecolors="#eeeeee", edgecolors="#cccccc", linewidth=0.2)
                continue
            key = f"{row['dataset']}-{pred_id}"
            if key not in color_cache:
                color_cache[key] = cmap(len(color_cache) % 20)
            ax.broken_barh([(seg_start, seg_width)], (y - 0.3, 0.6), facecolors=color_cache[key], edgecolors="white", linewidth=0.3)
            if seg_width >= 2:
                ax.text(seg_start + seg_width / 2, y, str(pred_id), ha="center", va="center", fontsize=6)
    labels = [f"{r.dataset}-{r.sequence} GT {int(r.gt_track_id)} ({int(r.unique_pred_tracks)} pred)" for r in top.itertuples()]
    ax.set_yticks(np.arange(len(top)), labels, fontsize=8)
    ax.set_xlabel("Frame span (segments show different predicted track ids)")
    ax.set_title("Top Duplicate / Fragmented GT Trajectories", fontsize=13, fontweight="bold")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=plt.get_cmap("tab20")(0), edgecolor="white", label="Matched predicted track id"),
            mpatches.Patch(facecolor="#eeeeee", edgecolor="#cccccc", label="Unmatched gap"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.35)
    save_figure(fig, out_dir / "top_duplicate_gt_track_timeline")


def analyze_query_maps(root: Path, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(9.5, 12))
    for row_idx, dataset in enumerate(DATASETS):
        qdir = root / dataset / "test" / "CTC" / "two_stage"
        for layer_idx in range(3):
            path = qdir / f"enc_queries_{layer_idx:02d}.npy"
            ax = axes[row_idx, layer_idx]
            ax.set_axis_off()
            if path.exists():
                arr = np.load(path)
                rows.append(
                    {
                        "dataset": dataset,
                        "query_map": path.name,
                        "shape": "x".join(map(str, arr.shape)),
                        "sum": round(float(arr.sum()), 6),
                        "max": round(float(arr.max()), 6),
                        "active_cells": int((arr > 0).sum()),
                        "active_fraction": round(float((arr > 0).mean()), 6),
                    }
                )
                im = ax.imshow(arr, cmap="magma")
                ax.set_title(f"{dataset} q{layer_idx}\nactive={(arr > 0).sum()}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
    fig.suptitle("Two-Stage Encoder Query Activation Maps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir / "encoder_query_activation_maps")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "encoder_query_activation_summary.csv", index=False)
    return df


def write_report(out_dir: Path, duplicate_df: pd.DataFrame, layer_df: pd.DataFrame, query_df: pd.DataFrame) -> None:
    dup_lines = []
    for row in duplicate_df.itertuples(index=False):
        dup_lines.append(
            f"- {row.dataset}: GT duplicate/fragmented={row.gt_duplicate_tracks}/{row.gt_tracks} "
            f"({row.gt_duplicate_rate:.1%}), pred multi-GT={row.pred_multi_gt_tracks}/{row.pred_tracks} "
            f"({row.pred_multi_gt_rate:.1%}), pred unmatched={row.pred_unmatched_tracks}/{row.pred_tracks} "
            f"({row.pred_unmatched_rate:.1%})"
        )
    layer_lines = []
    for dataset in DATASETS:
        sub = layer_df[(layer_df["dataset"] == dataset) & (layer_df["split"] == "val") & (layer_df["component"].isin(LOSS_COMPONENTS))]
        vals = []
        for component in LOSS_COMPONENTS:
            comp = sub[sub["component"] == component].sort_values("layer")
            if not comp.empty:
                vals.append(f"{component}: " + ", ".join(f"L{int(r.layer)}={float(r.final_epoch_mean):.2f}" for r in comp.itertuples() if r.final_epoch_mean != ""))
        layer_lines.append(f"- {dataset}: " + "; ".join(vals))
    query_lines = []
    for dataset in DATASETS:
        sub = query_df[query_df["dataset"] == dataset]
        active = ", ".join(f"{r.query_map}:{int(r.active_cells)}" for r in sub.itertuples())
        query_lines.append(f"- {dataset}: {active}")

    text = f"""# COLA 重复轨迹与中间层分析

## 说明

这里的 COLA 对应代码与日志中的 `CoMOT` 中间层监督分支。由于当前推理输出没有保存每一层 decoder 的预测 mask，本目录采用两类可复现证据：

1. 使用最终 CTC 结果与 GT TRA mask 的 IoU>=0.5 逐帧匹配，统计重复/碎片轨迹：一个 GT track 匹配到多个 predicted track ids，视为 duplicate/fragmented trajectory；一个 predicted track 匹配到多个 GT track ids，视为 merge/ambiguous trajectory。
2. 使用 `metrics_train.pkl` 和 `metrics_val.pkl` 中保存的 `CoMOT_loss_bbox_{{0..3}}` 与 `CoMOT_loss_giou_{{0..3}}`，分析 COLA 中间层监督在不同 decoder auxiliary layer 上的收敛情况。

## 重复轨迹统计

{chr(10).join(dup_lines)}

## COLA 中间层 final validation loss

{chr(10).join(layer_lines)}

## Encoder query 激活

{chr(10).join(query_lines)}

## 生成文件

- `duplicate_trajectory_summary.csv`：数据集级重复/碎片轨迹统计。
- `gt_to_pred_duplicate_tracks.csv`：GT track 到 predicted track ids 的逐轨迹明细。
- `pred_to_gt_multi_gt_tracks.csv`：predicted track 到 GT track ids 的逐轨迹明细。
- `duplicate_trajectory_rates.png/.pdf`：重复轨迹、未匹配轨迹、多 GT 轨迹比例。
- `top_duplicate_gt_track_timeline.png/.pdf`：最严重 duplicate/fragmented GT tracks 的 timeline。
- `cola_intermediate_layer_loss_summary.csv`：COLA/CoMOT 中间层 loss 统计。
- `cola_intermediate_layer_loss_curves.png/.pdf`：各数据集各中间层 train/val loss 曲线。
- `cola_layer_final_loss_heatmap.png/.pdf`：layer-wise final validation loss 热图。
- `encoder_query_activation_maps.png/.pdf`：现有 two-stage encoder query 激活图。
- `encoder_query_activation_summary.csv`：query 激活统计。
- `copied_aux_loss_CoMOT_plots/`：原始训练脚本生成的 CoMOT auxiliary loss 图拷贝。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/group/dingyi/ct/r1-50-66-512-288-mask-30"))
    parser.add_argument("--data-root", type=Path, default=Path("/group/dingyi/ct/data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.root / "cola_duplicate_intermediate_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_df = analyze_cola_layers(args.root, out_dir)
    duplicate_df, _, _ = analyze_duplicate_trajectories(args.root, args.data_root, out_dir)
    query_df = analyze_query_maps(args.root, out_dir)
    write_report(out_dir, duplicate_df, layer_df, query_df)
    manifest = {
        "root": str(args.root),
        "data_root": str(args.data_root),
        "out_dir": str(out_dir),
        "iou_threshold": fstats.IOU_THRESHOLD,
        "datasets": list(DATASETS),
        "cola_name_in_code": COLA_NAME,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
