#!/usr/bin/env python3
"""Create failure-case figures from CTC-style tracking outputs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_case_stats as stats


DATASET_INFO = {
    "3T3": {"data": "3t3", "short": "3T3"},
    "HEK293": {"data": "hek", "short": "HEK293"},
    "HeLa": {"data": "hela", "short": "HeLa"},
    "PC-3": {"data": "pc", "short": "PC-3"},
    "RAW264": {"data": "raw", "short": "RAW264"},
}

SELECTED_CASES = [
    ("HEK293", "01", 67, "division_link", "division/link"),
    ("RAW264", "03", 21, "identity_switch", "identity switch"),
    ("PC-3", "01", 3, "identity_switch", "identity switch"),
    ("HeLa", "03", 34, "missed_detection", "missed detection"),
    ("3T3", "01", 62, "split_merge", "split/merge"),
]

COLORS = {
    "gt": "#00c853",
    "pred": "#ff3fbf",
    "fn": "#ff2d2d",
    "fp": "#ffd23f",
    "id": "#ff8c1a",
    "split_merge": "#23b7ff",
    "division": "#8a5cff",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bbox(value: str) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 4:
        return None
    y1, x1, y2, x2 = map(int, parts)
    if y2 <= y1 or x2 <= x1:
        return None
    return y1, x1, y2, x2


def bbox_for_label(mask: np.ndarray, label_id: int) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask == label_id)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def union_bboxes(bboxes: Iterable[Optional[Tuple[int, int, int, int]]]) -> Optional[Tuple[int, int, int, int]]:
    valid = [bbox for bbox in bboxes if bbox is not None]
    if not valid:
        return None
    return (
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    )


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    shape: Tuple[int, int],
    pad: int = 72,
    min_size: int = 150,
    max_size: int = 230,
) -> Tuple[int, int, int, int]:
    y1, x1, y2, x2 = bbox
    cy = (y1 + y2) // 2
    cx = (x1 + x2) // 2
    size = max(y2 - y1 + 2 * pad, x2 - x1 + 2 * pad, min_size)
    size = min(size, max_size, max(shape))
    half = int(math.ceil(size / 2))
    yy1 = max(0, cy - half)
    xx1 = max(0, cx - half)
    yy2 = min(shape[0], yy1 + 2 * half)
    xx2 = min(shape[1], xx1 + 2 * half)
    yy1 = max(0, yy2 - 2 * half)
    xx1 = max(0, xx2 - 2 * half)
    return yy1, xx1, yy2, xx2


def intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def paths_for(root: Path, data_root: Path, dataset: str, sequence: str, frame: int) -> Tuple[Path, Path, Path]:
    data_name = DATASET_INFO[dataset]["data"]
    raw = data_root / data_name / "test" / sequence / f"t{frame:03d}.tif"
    gt = data_root / data_name / "test" / f"{sequence}_GT" / "TRA" / f"man_track{frame:03d}.tif"
    pred = root / dataset / "test" / "CTC" / f"{sequence}_RES" / f"mask{frame:03d}.tif"
    return raw, gt, pred


def load_image(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path)).astype(np.float32)
    if img.ndim > 2:
        img = img[..., 0]
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max()) + 1.0
    return np.clip((img - p1) / (p99 - p1), 0, 1)


def overlay_masks(raw: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.repeat(raw[..., None], 3, axis=2)
    gt_boundary = find_boundaries(gt > 0, mode="outer")
    pred_boundary = find_boundaries(pred > 0, mode="outer")
    rgb[gt_boundary] = (0.0, 0.78, 0.32)
    rgb[pred_boundary] = (1.0, 0.22, 0.78)
    return rgb


def event_color(event_type: str) -> str:
    if event_type == "false_negative":
        return COLORS["fn"]
    if event_type == "false_positive":
        return COLORS["fp"]
    if event_type in {"id_switch", "reid_after_gap"}:
        return COLORS["id"]
    if event_type in {"split_candidate", "merge_candidate"}:
        return COLORS["split_merge"]
    if event_type == "division_error":
        return COLORS["division"]
    return "#ffffff"


def event_bboxes(row: Dict[str, str]) -> List[Tuple[int, int, int, int]]:
    bboxes = []
    for key in ("bbox_gt_y1x1y2x2", "bbox_pred_y1x1y2x2"):
        bbox = parse_bbox(row.get(key, ""))
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def candidate_bboxes_from_events(
    root: Path,
    data_root: Path,
    dataset: str,
    sequence: str,
    center_frame: int,
    dominant_type: str,
) -> List[Tuple[int, int, int, int]]:
    rows = read_csv_rows(root / dataset / "failure_cases" / "failure_events.csv")
    by_type = {
        "identity_switch": {"id_switch", "reid_after_gap"},
        "split_merge": {"split_candidate", "merge_candidate"},
        "missed_detection": {"false_negative"},
        "false_positive": {"false_positive"},
    }
    target_types = by_type.get(dominant_type, set())
    candidates = []
    for row in rows:
        if row["sequence"] != sequence:
            continue
        frame = int(row["frame"])
        if abs(frame - center_frame) > 2:
            continue
        if target_types and row["event_type"] not in target_types:
            continue
        score = 10 if frame == center_frame else 5 - abs(frame - center_frame)
        for bbox in event_bboxes(row):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            candidates.append((score, area, bbox))
    if candidates:
        candidates.sort(reverse=True)
        return [candidates[0][2]]

    fallback = []
    for row in rows:
        if row["sequence"] != sequence or abs(int(row["frame"]) - center_frame) > 2:
            continue
        fallback.extend(event_bboxes(row))
    return fallback[:1]


def candidate_bboxes_from_divisions(
    root: Path,
    data_root: Path,
    dataset: str,
    sequence: str,
    center_frame: int,
) -> List[Tuple[int, int, int, int]]:
    rows = read_csv_rows(root / dataset / "failure_cases" / "division_stats.csv")
    candidates = [row for row in rows if row["sequence"] == sequence and row["status"] != "correct" and abs(int(row["frame"]) - center_frame) <= 2]
    if not candidates:
        return []
    row = sorted(candidates, key=lambda r: abs(int(r["frame"]) - center_frame))[0]
    frame = int(row["frame"])
    _, gt_path, pred_path = paths_for(root, data_root, dataset, sequence, frame)
    gt = tifffile.imread(str(gt_path)) if gt_path.exists() else None
    pred = tifffile.imread(str(pred_path)) if pred_path.exists() else None
    bboxes = []
    for key in ("child_gt_id", "parent_gt_id"):
        value = row.get(key, "")
        if gt is not None and value:
            bbox = bbox_for_label(gt, int(value))
            if bbox is not None:
                bboxes.append(bbox)
    for key in ("child_pred_id", "parent_pred_id", "pred_parent_of_child"):
        value = row.get(key, "")
        if pred is not None and value:
            bbox = bbox_for_label(pred, int(value))
            if bbox is not None:
                bboxes.append(bbox)
    return bboxes


def crop_for_case(root: Path, data_root: Path, dataset: str, sequence: str, center_frame: int, dominant_type: str) -> Tuple[int, int, int, int]:
    raw_path, _, _ = paths_for(root, data_root, dataset, sequence, center_frame)
    shape = tifffile.imread(str(raw_path)).shape[:2]
    if dominant_type == "division_link":
        bboxes = candidate_bboxes_from_divisions(root, data_root, dataset, sequence, center_frame)
    else:
        bboxes = candidate_bboxes_from_events(root, data_root, dataset, sequence, center_frame, dominant_type)
    bbox = union_bboxes(bboxes)
    if bbox is None:
        h, w = shape
        bbox = (h // 2 - 60, w // 2 - 60, h // 2 + 60, w // 2 + 60)
    return expand_bbox(bbox, shape)


def draw_event_boxes(
    ax: plt.Axes,
    root: Path,
    dataset: str,
    sequence: str,
    frame: int,
    roi: Tuple[int, int, int, int],
) -> None:
    rows = read_csv_rows(root / dataset / "failure_cases" / "failure_events.csv")
    y1, x1, y2, x2 = roi
    crop_bbox = roi
    drawn = 0
    for row in rows:
        if row["sequence"] != sequence or int(row["frame"]) != frame:
            continue
        color = event_color(row["event_type"])
        for bbox in event_bboxes(row):
            if not intersects(crop_bbox, bbox):
                continue
            by1, bx1, by2, bx2 = bbox
            by1, bx1 = max(by1, y1), max(bx1, x1)
            by2, bx2 = min(by2, y2), min(bx2, x2)
            if by2 <= by1 or bx2 <= bx1:
                continue
            rect = patches.Rectangle(
                (bx1 - x1, by1 - y1),
                bx2 - bx1,
                by2 - by1,
                fill=False,
                edgecolor=color,
                linewidth=1.2,
            )
            ax.add_patch(rect)
            drawn += 1
            if drawn >= 10:
                return


def save_figure(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_distribution(root: Path, out_dir: Path) -> None:
    df = pd.read_csv(root / "failure_case_dataset_summary.csv")
    df["id_switch"] = df["id_switches"] + df["reid_after_gap"]
    df["split_merge"] = df["split_candidates"] + df["merge_candidates"]
    df["division_errors"] = df["gt_child_tracks"] - df["correct_division_child_links"]
    cols = [
        ("false_negatives", "FN", COLORS["fn"]),
        ("false_positives", "FP", COLORS["fp"]),
        ("id_switch", "IDSW/ReID", COLORS["id"]),
        ("split_merge", "Split/Merge", COLORS["split_merge"]),
        ("fragmented_gt_tracks", "Fragmented", "#7f7f7f"),
        ("division_errors", "Division", COLORS["division"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for col, label, color in cols:
        axes[0].bar(x, df[col], bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.4)
        bottom += df[col].to_numpy()
    axes[0].set_xticks(x, df["dataset"], rotation=0)
    axes[0].set_ylabel("Error count")
    axes[0].set_title("A. Failure-Type Counts")
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    normalized = pd.DataFrame({label: df[col] for col, label, _ in cols})
    normalized = normalized.div(normalized.sum(axis=1), axis=0).fillna(0)
    bottom = np.zeros(len(df))
    for _, label, color in cols:
        axes[1].bar(x, normalized[label], bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.4)
        bottom += normalized[label].to_numpy()
    axes[1].set_xticks(x, df["dataset"], rotation=0)
    axes[1].set_ylabel("Proportion")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("B. Normalized Composition")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Failure Case Statistics Across Datasets", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir / "failure_error_distribution")


def make_success_failure_ratio(root: Path, out_dir: Path) -> None:
    df = pd.read_csv(root / "failure_case_dataset_summary.csv")
    ratio_rows = []
    for _, row in df.iterrows():
        instance_success = int(row["tp"])
        instance_failure = int(row["false_negatives"]) + int(row["false_positives"])
        track_success = int(row["gt_tracks_with_match"]) - int(row["fragmented_gt_tracks"])
        track_failure = int(row["gt_tracks_never_matched"]) + int(row["fragmented_gt_tracks"])
        division_success = int(row["correct_division_child_links"])
        division_failure = int(row["gt_child_tracks"]) - int(row["correct_division_child_links"])

        def rate(success: int, failure: int) -> float:
            denom = success + failure
            return success / denom if denom else np.nan

        ratio_rows.append(
            {
                "dataset": row["dataset"],
                "instance_success": instance_success,
                "instance_failure": instance_failure,
                "instance_success_rate": round(rate(instance_success, instance_failure), 6),
                "track_success": track_success,
                "track_failure": track_failure,
                "track_success_rate": round(rate(track_success, track_failure), 6),
                "division_success": division_success,
                "division_failure": division_failure,
                "division_success_rate": round(rate(division_success, division_failure), 6),
            }
        )
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(root / "failure_success_ratio.csv", index=False)

    datasets = ratio_df["dataset"].tolist()
    categories = [
        ("Inst.", "instance_success_rate"),
        ("Track", "track_success_rate"),
        ("Div.", "division_success_rate"),
    ]
    success_color = "#2fbf71"
    failure_color = "#ff4d4d"
    group_gap = 0.9
    bar_width = 0.82
    group_width = len(categories) + group_gap
    positions = []
    category_labels = []
    group_centers = []
    values = []
    for group_idx, dataset in enumerate(datasets):
        base = group_idx * group_width
        group_centers.append(base + 1)
        for cat_idx, (label, col) in enumerate(categories):
            positions.append(base + cat_idx)
            category_labels.append(label)
            values.append(float(ratio_df.loc[ratio_df["dataset"] == dataset, col].iloc[0]))

    fig, ax = plt.subplots(figsize=(12.2, 4.9))
    for xi, label, value in zip(positions, category_labels, values):
        ax.bar(xi, value, bar_width, color=success_color, edgecolor="white", linewidth=0.5)
        ax.bar(xi, 1 - value, bar_width, bottom=value, color=failure_color, edgecolor="white", linewidth=0.5)
        if value >= 0.12:
            ax.text(xi, value / 2, f"S {value * 100:.1f}%", ha="center", va="center", fontsize=7, rotation=90)
        else:
            ax.text(xi, 0.05, f"S {value * 100:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90)
        failure = 1 - value
        if failure >= 0.12:
            ax.text(xi, value + failure / 2, f"F {failure * 100:.1f}%", ha="center", va="center", fontsize=7, rotation=90)

    ax.set_xticks(group_centers, datasets)
    for xi, label in zip(positions, category_labels):
        ax.text(xi, -0.09, label, ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Proportion")
    ax.set_title("Success vs. Failure Ratios", fontsize=13, fontweight="bold")
    ax.legend(
        handles=[
            patches.Patch(facecolor=success_color, edgecolor="none", label="Success"),
            patches.Patch(facecolor=failure_color, edgecolor="none", label="Failure"),
        ],
        frameon=False,
        loc="upper right",
        ncol=2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    ax.text(0.0, -0.22, "Instance: TP vs FN+FP; Track: matched non-fragmented GT tracks vs unmatched/fragmented; Division: correct vs incorrect child links.", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "failure_success_ratio")


def make_heatmaps(root: Path, out_dir: Path) -> None:
    datasets = list(DATASET_INFO)
    all_scores = []
    frames_by_dataset = {}
    for dataset in datasets:
        df = pd.read_csv(root / dataset / "failure_cases" / "frame_stats.csv")
        frames_by_dataset[dataset] = df
        all_scores.extend(df["failure_score"].tolist())
    vmax = max(1, np.percentile(all_scores, 98))

    fig, axes = plt.subplots(len(datasets), 1, figsize=(11.5, 8.2), sharex=False)
    images = []
    for ax, dataset in zip(axes, datasets):
        df = frames_by_dataset[dataset]
        sequences = sorted(df["sequence"].astype(str).unique())
        max_frame = int(df["frame"].max())
        mat = np.full((len(sequences), max_frame + 1), np.nan)
        for i, seq in enumerate(sequences):
            seq_df = df[df["sequence"].astype(str) == seq]
            mat[i, seq_df["frame"].to_numpy(dtype=int)] = seq_df["failure_score"].to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        images.append(im)
        ax.set_yticks(np.arange(len(sequences)), sequences)
        ax.set_ylabel(dataset)
        ax.set_xlim(-0.5, max_frame + 0.5)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Per-Frame Failure Score Heatmaps", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(images[0], ax=axes, orientation="vertical", fraction=0.018, pad=0.015)
    cbar.set_label("Failure score")
    save_figure(fig, out_dir / "failure_frame_heatmaps")


def make_representative_crops(root: Path, data_root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(len(SELECTED_CASES), 5, figsize=(15.5, 12.5))
    legend_handles = [
        patches.Patch(facecolor="none", edgecolor=COLORS["gt"], label="GT contour"),
        patches.Patch(facecolor="none", edgecolor=COLORS["pred"], label="Prediction contour"),
        patches.Patch(facecolor="none", edgecolor=COLORS["fn"], label="FN"),
        patches.Patch(facecolor="none", edgecolor=COLORS["fp"], label="FP"),
        patches.Patch(facecolor="none", edgecolor=COLORS["id"], label="ID switch/ReID"),
        patches.Patch(facecolor="none", edgecolor=COLORS["split_merge"], label="Split/Merge"),
    ]
    for row_idx, (dataset, sequence, center_frame, dominant_type, label) in enumerate(SELECTED_CASES):
        roi = crop_for_case(root, data_root, dataset, sequence, center_frame, dominant_type)
        y1, x1, y2, x2 = roi
        for col_idx, frame in enumerate(range(center_frame - 2, center_frame + 3)):
            ax = axes[row_idx, col_idx]
            raw_path, gt_path, pred_path = paths_for(root, data_root, dataset, sequence, frame)
            ax.set_axis_off()
            if not raw_path.exists() or not gt_path.exists() or not pred_path.exists():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                continue
            raw = load_image(raw_path)[y1:y2, x1:x2]
            gt = tifffile.imread(str(gt_path))[y1:y2, x1:x2]
            pred = tifffile.imread(str(pred_path))[y1:y2, x1:x2]
            ax.imshow(overlay_masks(raw, gt, pred))
            draw_event_boxes(ax, root, dataset, sequence, frame, roi)
            if row_idx == 0:
                rel = frame - center_frame
                ax.set_title(f"t{rel:+d}\nframe {frame}", fontsize=9)
            if col_idx == 0:
                ax.text(
                    -0.05,
                    0.5,
                    f"{dataset}-{sequence}\n{label}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    rotation=90,
                )
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Representative Failure-Case Crops", fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.08, hspace=0.08, wspace=0.04)
    save_figure(fig, out_dir / "failure_representative_crops")


def match_history_for_sequence(root: Path, data_root: Path, dataset: str, sequence: str) -> Dict[int, Dict[int, int]]:
    data_name = DATASET_INFO[dataset]["data"]
    gt_dir = data_root / data_name / "test" / f"{sequence}_GT" / "TRA"
    pred_dir = root / dataset / "test" / "CTC" / f"{sequence}_RES"
    gt_files = stats.mask_files(gt_dir, "man_track")
    pred_files = stats.mask_files(pred_dir, "mask")
    history: Dict[int, Dict[int, int]] = defaultdict(dict)
    for frame in sorted(set(gt_files) & set(pred_files)):
        gt = tifffile.imread(str(gt_files[frame]))
        pred = tifffile.imread(str(pred_files[frame]))
        _, _, _, pairs = stats.compute_pair_stats(gt, pred)
        gt_to_pred, _, _ = stats.greedy_match(pairs, stats.IOU_THRESHOLD)
        for gt_id, pred_id in gt_to_pred.items():
            history[gt_id][frame] = pred_id
    return history


def contiguous_segments(values: Dict[int, int], start: int, end: int) -> List[Tuple[int, int, Optional[int]]]:
    segments: List[Tuple[int, int, Optional[int]]] = []
    current_value: Optional[int] = None
    segment_start = start
    for frame in range(start, end + 1):
        value = values.get(frame)
        if frame == start:
            current_value = value
            segment_start = frame
        elif value != current_value:
            segments.append((segment_start, frame - 1, current_value))
            current_value = value
            segment_start = frame
    segments.append((segment_start, end, current_value))
    return segments


def make_timeline(root: Path, data_root: Path, out_dir: Path) -> None:
    event_candidates = []
    for dataset in DATASET_INFO:
        path = root / dataset / "failure_cases" / "failure_events.csv"
        for row in read_csv_rows(path):
            if row["event_type"] in {"id_switch", "reid_after_gap"} and row["gt_id"]:
                event_candidates.append(row)
    if not event_candidates:
        return
    event_candidates.sort(key=lambda r: (r["dataset"] != "RAW264", r["dataset"] != "HEK293", -int(r["frame"])))

    selected = []
    seen = set()
    for row in event_candidates:
        key = (row["dataset"], row["sequence"], int(row["gt_id"]))
        if key not in seen:
            selected.append(key)
            seen.add(key)
        if len(selected) == 8:
            break

    histories_by_sequence: Dict[Tuple[str, str], Dict[int, Dict[int, int]]] = {}
    track_info_by_sequence: Dict[Tuple[str, str], Dict[int, Dict[str, int]]] = {}
    pred_ids = set()
    rows_to_plot = []
    for dataset, sequence, gt_id in selected:
        seq_key = (dataset, sequence)
        if seq_key not in histories_by_sequence:
            histories_by_sequence[seq_key] = match_history_for_sequence(root, data_root, dataset, sequence)
            data_name = DATASET_INFO[dataset]["data"]
            track_info_by_sequence[seq_key] = stats.parse_track_file(data_root / data_name / "test" / f"{sequence}_GT" / "TRA" / "man_track.txt")
        info = track_info_by_sequence[seq_key].get(gt_id)
        if not info:
            continue
        history = histories_by_sequence[seq_key].get(gt_id, {})
        pred_ids.update(history.values())
        rows_to_plot.append((dataset, sequence, gt_id, info["start"], info["end"], history))

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    pred_color = {pred_id: colors[i % len(colors)] for i, pred_id in enumerate(sorted(pred_ids))}

    fig, ax = plt.subplots(figsize=(12.5, max(3.2, 0.48 * len(rows_to_plot) + 1.2)))
    for y, (dataset, sequence, gt_id, start, end, history) in enumerate(rows_to_plot):
        for seg_start, seg_end, pred_id in contiguous_segments(history, start, end):
            width = seg_end - seg_start + 1
            if pred_id is None:
                ax.broken_barh([(seg_start, width)], (y - 0.32, 0.64), facecolors="#eeeeee", edgecolors="#cccccc", linewidth=0.2)
            else:
                ax.broken_barh([(seg_start, width)], (y - 0.32, 0.64), facecolors=pred_color[pred_id], edgecolors="white", linewidth=0.2)
                if width >= 4:
                    ax.text(seg_start + width / 2, y, str(pred_id), ha="center", va="center", fontsize=6, color="black")
        values = [history.get(frame) for frame in range(start, end + 1)]
        for offset in range(1, len(values)):
            if values[offset] is not None and values[offset - 1] is not None and values[offset] != values[offset - 1]:
                ax.vlines(start + offset - 0.5, y - 0.38, y + 0.38, color=COLORS["fn"], linewidth=1.6)
    ax.set_yticks(np.arange(len(rows_to_plot)), [f"{d}-{s} GT {g}" for d, s, g, *_ in rows_to_plot], fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_title("Trajectory Fragmentation and ID-Switch Timelines", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.8, len(rows_to_plot) - 0.2)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)
    save_figure(fig, out_dir / "failure_track_timeline")


def write_index(out_dir: Path) -> None:
    text = """# Failure Case Figures

Generated files:

- `failure_error_distribution.png/.pdf`: overall error composition across datasets.
- `failure_success_ratio.png/.pdf`: success/failure ratios for instance, track, and division levels.
- `failure_frame_heatmaps.png/.pdf`: frame-level failure-score heatmaps.
- `failure_representative_crops.png/.pdf`: five representative multi-frame crops with GT/prediction overlays.
- `failure_track_timeline.png/.pdf`: fragmented track / ID-switch timeline examples.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/group/dingyi/ct/r1-50-66-512-288-mask-30"))
    parser.add_argument("--data-root", type=Path, default=Path("/group/dingyi/ct/data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.root / "failure_case_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_distribution(args.root, out_dir)
    make_success_failure_ratio(args.root, out_dir)
    make_heatmaps(args.root, out_dir)
    make_representative_crops(args.root, args.data_root, out_dir)
    make_timeline(args.root, args.data_root, out_dir)
    write_index(out_dir)


if __name__ == "__main__":
    main()
