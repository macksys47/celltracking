#!/usr/bin/env python3
"""Failure-case statistics for CTC-style tracking outputs.

The script compares predicted *_RES masks with GT TRA masks, writes per-dataset
failure tables, and emits a concise visualization plan for manuscript figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tifffile


DATASETS = {
    "3t3": ("3T3", "3t3"),
    "hek": ("HEK293", "hek"),
    "hela": ("HeLa", "hela"),
    "pc": ("PC-3", "pc"),
    "raw": ("RAW264", "raw"),
}

IOU_THRESHOLD = 0.5
OVERLAP_FRACTION_THRESHOLD = 0.10
MIN_OVERLAP_PIXELS = 8
TOP_WINDOWS_PER_SEQUENCE = 8
WINDOW_RADIUS = 2
NMS_RADIUS = 3


def read_mask(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def parse_frame_num(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame number from {path}")
    return int(digits[-3:])


def mask_files(directory: Path, prefix: str) -> Dict[int, Path]:
    return {parse_frame_num(path): path for path in sorted(directory.glob(f"{prefix}*.tif"))}


def parse_track_file(path: Path) -> Dict[int, Dict[str, int]]:
    tracks: Dict[int, Dict[str, int]] = {}
    if not path.exists():
        return tracks
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            track_id, start, end, parent = map(int, parts[:4])
            tracks[track_id] = {
                "start": start,
                "end": end,
                "parent": parent,
            }
    return tracks


def label_areas(mask: np.ndarray) -> Dict[int, int]:
    labels, counts = np.unique(mask, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts) if int(label) > 0}


def label_bboxes(mask: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return {}
    labels = mask[ys, xs].astype(np.int64)
    order = np.argsort(labels, kind="mergesort")
    labels = labels[order]
    ys = ys[order]
    xs = xs[order]
    bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    for start, end in zip(starts, ends):
        label = int(labels[start])
        bboxes[label] = (
            int(ys[start:end].min()),
            int(xs[start:end].min()),
            int(ys[start:end].max()) + 1,
            int(xs[start:end].max()) + 1,
        )
    return bboxes


def bbox_to_str(bbox: Optional[Tuple[int, int, int, int]]) -> str:
    if bbox is None:
        return ""
    y1, x1, y2, x2 = bbox
    return f"{y1}:{x1}:{y2}:{x2}"


def compute_pair_stats(
    gt: np.ndarray, pred: np.ndarray
) -> Tuple[
    Dict[int, int],
    Dict[int, int],
    Dict[Tuple[int, int], int],
    List[Tuple[float, int, int, int]],
]:
    gt_areas = label_areas(gt)
    pred_areas = label_areas(pred)
    intersections: Dict[Tuple[int, int], int] = {}
    valid = (gt > 0) & (pred > 0)
    if np.any(valid):
        pred_base = int(pred.max()) + 1
        codes = gt[valid].astype(np.int64) * pred_base + pred[valid].astype(np.int64)
        unique_codes, counts = np.unique(codes, return_counts=True)
        for code, count in zip(unique_codes, counts):
            gt_id = int(code // pred_base)
            pred_id = int(code % pred_base)
            if gt_id > 0 and pred_id > 0:
                intersections[(gt_id, pred_id)] = int(count)

    pairs: List[Tuple[float, int, int, int]] = []
    for (gt_id, pred_id), inter in intersections.items():
        union = gt_areas[gt_id] + pred_areas[pred_id] - inter
        iou = inter / union if union else 0.0
        pairs.append((float(iou), gt_id, pred_id, inter))
    return gt_areas, pred_areas, intersections, pairs


def greedy_match(pairs: Iterable[Tuple[float, int, int, int]], threshold: float) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, float]]:
    gt_to_pred: Dict[int, int] = {}
    pred_to_gt: Dict[int, int] = {}
    gt_to_iou: Dict[int, float] = {}
    for iou, gt_id, pred_id, _ in sorted(pairs, reverse=True):
        if iou < threshold:
            continue
        if gt_id in gt_to_pred or pred_id in pred_to_gt:
            continue
        gt_to_pred[gt_id] = pred_id
        pred_to_gt[pred_id] = gt_id
        gt_to_iou[gt_id] = iou
    return gt_to_pred, pred_to_gt, gt_to_iou


def significant_overlaps(
    intersections: Dict[Tuple[int, int], int],
    gt_areas: Dict[int, int],
    pred_areas: Dict[int, int],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    by_gt: Dict[int, List[int]] = defaultdict(list)
    by_pred: Dict[int, List[int]] = defaultdict(list)
    for (gt_id, pred_id), inter in intersections.items():
        if inter < MIN_OVERLAP_PIXELS:
            continue
        gt_fraction = inter / gt_areas[gt_id]
        pred_fraction = inter / pred_areas[pred_id]
        if gt_fraction >= OVERLAP_FRACTION_THRESHOLD:
            by_gt[gt_id].append(pred_id)
        if pred_fraction >= OVERLAP_FRACTION_THRESHOLD:
            by_pred[pred_id].append(gt_id)
    return by_gt, by_pred


def best_overlap(
    label_id: int,
    pairs: Iterable[Tuple[float, int, int, int]],
    kind: str,
) -> Tuple[str, float]:
    best_id = ""
    best_iou = 0.0
    for iou, gt_id, pred_id, _ in pairs:
        if kind == "gt" and gt_id == label_id and iou > best_iou:
            best_id = str(pred_id)
            best_iou = iou
        elif kind == "pred" and pred_id == label_id and iou > best_iou:
            best_id = str(gt_id)
            best_iou = iou
    return best_id, best_iou


def parent_prediction_for(gt_id: int, start_frame: int, gt_history: Dict[int, Dict[int, int]]) -> Optional[int]:
    frames = gt_history.get(gt_id, {})
    for frame in range(start_frame - 1, -1, -1):
        if frame in frames:
            return frames[frame]
    return None


def child_prediction_for(gt_id: int, start_frame: int, gt_history: Dict[int, Dict[int, int]]) -> Optional[int]:
    frames = gt_history.get(gt_id, {})
    for frame in range(start_frame, start_frame + 3):
        if frame in frames:
            return frames[frame]
    return None


def event_row(
    dataset: str,
    sequence: str,
    frame: int,
    event_type: str,
    gt_id: str = "",
    pred_id: str = "",
    bbox_gt: str = "",
    bbox_pred: str = "",
    iou: str = "",
    related_ids: str = "",
    note: str = "",
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "sequence": sequence,
        "frame": frame,
        "event_type": event_type,
        "gt_id": gt_id,
        "pred_id": pred_id,
        "bbox_gt_y1x1y2x2": bbox_gt,
        "bbox_pred_y1x1y2x2": bbox_pred,
        "iou": iou,
        "related_ids": related_ids,
        "note": note,
    }


def analyze_sequence(
    dataset_key: str,
    display_name: str,
    data_name: str,
    sequence: str,
    pred_res_dir: Path,
    gt_sequence_dir: Path,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    gt_tra_dir = gt_sequence_dir / f"{sequence}_GT" / "TRA"
    pred_files = mask_files(pred_res_dir, "mask")
    gt_files = mask_files(gt_tra_dir, "man_track")
    frames = sorted(set(pred_files) | set(gt_files))
    gt_tracks = parse_track_file(gt_tra_dir / "man_track.txt")
    pred_tracks = parse_track_file(pred_res_dir / "res_track.txt")

    frame_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    matched_history: Dict[int, Dict[int, int]] = defaultdict(dict)
    matched_iou_history: Dict[int, Dict[int, float]] = defaultdict(dict)
    pred_matched_frames: Dict[int, set] = defaultdict(set)
    id_switch_rows: List[Dict[str, object]] = []

    prev_pred_by_gt: Dict[int, int] = {}
    prev_frame_by_gt: Dict[int, int] = {}
    totals = Counter()

    for frame in frames:
        gt = read_mask(gt_files[frame]) if frame in gt_files else np.zeros_like(read_mask(pred_files[frame]))
        pred = read_mask(pred_files[frame]) if frame in pred_files else np.zeros_like(read_mask(gt_files[frame]))

        gt_areas, pred_areas, intersections, pairs = compute_pair_stats(gt, pred)
        gt_to_pred, pred_to_gt, gt_to_iou = greedy_match(pairs, IOU_THRESHOLD)
        by_gt, by_pred = significant_overlaps(intersections, gt_areas, pred_areas)
        gt_bboxes = label_bboxes(gt)
        pred_bboxes = label_bboxes(pred)

        gt_ids = set(gt_areas)
        pred_ids = set(pred_areas)
        false_negatives = sorted(gt_ids - set(gt_to_pred))
        false_positives = sorted(pred_ids - set(pred_to_gt))
        split_ids = sorted(gt_id for gt_id, pred_ids_list in by_gt.items() if len(set(pred_ids_list)) >= 2)
        merge_ids = sorted(pred_id for pred_id, gt_ids_list in by_pred.items() if len(set(gt_ids_list)) >= 2)

        id_switches = 0
        reid_after_gap = 0
        for gt_id, pred_id in sorted(gt_to_pred.items()):
            matched_history[gt_id][frame] = pred_id
            matched_iou_history[gt_id][frame] = gt_to_iou[gt_id]
            pred_matched_frames[pred_id].add(frame)
            if gt_id in prev_pred_by_gt and prev_pred_by_gt[gt_id] != pred_id:
                gap = frame - prev_frame_by_gt[gt_id]
                if gap == 1:
                    event_type = "id_switch"
                    id_switches += 1
                else:
                    event_type = "reid_after_gap"
                    reid_after_gap += 1
                row = event_row(
                    dataset=display_name,
                    sequence=sequence,
                    frame=frame,
                    event_type=event_type,
                    gt_id=str(gt_id),
                    pred_id=str(pred_id),
                    bbox_gt=bbox_to_str(gt_bboxes.get(gt_id)),
                    bbox_pred=bbox_to_str(pred_bboxes.get(pred_id)),
                    iou=f"{gt_to_iou[gt_id]:.6f}",
                    related_ids=f"prev_pred={prev_pred_by_gt[gt_id]};prev_frame={prev_frame_by_gt[gt_id]};gap={gap}",
                )
                event_rows.append(row)
                id_switch_rows.append(row)
            prev_pred_by_gt[gt_id] = pred_id
            prev_frame_by_gt[gt_id] = frame

        for gt_id in false_negatives:
            best_pred_id, best_iou = best_overlap(gt_id, pairs, "gt")
            event_rows.append(
                event_row(
                    dataset=display_name,
                    sequence=sequence,
                    frame=frame,
                    event_type="false_negative",
                    gt_id=str(gt_id),
                    pred_id=best_pred_id,
                    bbox_gt=bbox_to_str(gt_bboxes.get(gt_id)),
                    iou=f"{best_iou:.6f}" if best_iou else "",
                )
            )

        for pred_id in false_positives:
            best_gt_id, best_iou = best_overlap(pred_id, pairs, "pred")
            event_rows.append(
                event_row(
                    dataset=display_name,
                    sequence=sequence,
                    frame=frame,
                    event_type="false_positive",
                    gt_id=best_gt_id,
                    pred_id=str(pred_id),
                    bbox_pred=bbox_to_str(pred_bboxes.get(pred_id)),
                    iou=f"{best_iou:.6f}" if best_iou else "",
                )
            )

        for gt_id in split_ids:
            related = sorted(set(by_gt[gt_id]))
            event_rows.append(
                event_row(
                    dataset=display_name,
                    sequence=sequence,
                    frame=frame,
                    event_type="split_candidate",
                    gt_id=str(gt_id),
                    bbox_gt=bbox_to_str(gt_bboxes.get(gt_id)),
                    related_ids="pred_ids=" + ",".join(map(str, related)),
                )
            )

        for pred_id in merge_ids:
            related = sorted(set(by_pred[pred_id]))
            event_rows.append(
                event_row(
                    dataset=display_name,
                    sequence=sequence,
                    frame=frame,
                    event_type="merge_candidate",
                    pred_id=str(pred_id),
                    bbox_pred=bbox_to_str(pred_bboxes.get(pred_id)),
                    related_ids="gt_ids=" + ",".join(map(str, related)),
                )
            )

        frame_score = (
            len(false_negatives)
            + len(false_positives)
            + 2 * len(split_ids)
            + 2 * len(merge_ids)
            + 3 * id_switches
            + 2 * reid_after_gap
        )
        frame_rows.append(
            {
                "dataset": display_name,
                "sequence": sequence,
                "frame": frame,
                "gt_instances": len(gt_ids),
                "pred_instances": len(pred_ids),
                "tp": len(gt_to_pred),
                "false_negatives": len(false_negatives),
                "false_positives": len(false_positives),
                "split_candidates": len(split_ids),
                "merge_candidates": len(merge_ids),
                "id_switches": id_switches,
                "reid_after_gap": reid_after_gap,
                "failure_score": frame_score,
            }
        )

        totals.update(
            {
                "gt_instances": len(gt_ids),
                "pred_instances": len(pred_ids),
                "tp": len(gt_to_pred),
                "false_negatives": len(false_negatives),
                "false_positives": len(false_positives),
                "split_candidates": len(split_ids),
                "merge_candidates": len(merge_ids),
                "id_switches": id_switches,
                "reid_after_gap": reid_after_gap,
            }
        )

    track_rows: List[Dict[str, object]] = []
    fragmented_tracks = 0
    gt_tracks_with_match = 0
    total_matched_gt_frames = 0
    for gt_id, info in sorted(gt_tracks.items()):
        start = info["start"]
        end = info["end"]
        track_frames = [frame for frame in range(start, end + 1) if frame in gt_files]
        pred_ids = [matched_history.get(gt_id, {}).get(frame) for frame in track_frames]
        matched_pred_ids = [pred_id for pred_id in pred_ids if pred_id is not None]
        unique_pred_ids = sorted(set(matched_pred_ids))
        if unique_pred_ids:
            gt_tracks_with_match += 1
        if len(unique_pred_ids) > 1:
            fragmented_tracks += 1
        total_matched_gt_frames += len(matched_pred_ids)
        track_rows.append(
            {
                "dataset": display_name,
                "sequence": sequence,
                "gt_track_id": gt_id,
                "start": start,
                "end": end,
                "parent": info["parent"],
                "track_frames": len(track_frames),
                "matched_frames": len(matched_pred_ids),
                "missing_frames": len(track_frames) - len(matched_pred_ids),
                "matched_ratio": round(len(matched_pred_ids) / len(track_frames), 6) if track_frames else 0.0,
                "unique_pred_tracks": len(unique_pred_ids),
                "pred_track_ids": ",".join(map(str, unique_pred_ids)),
                "fragmented": int(len(unique_pred_ids) > 1),
            }
        )

    pred_tracks_with_match = sum(1 for pred_id in pred_tracks if pred_matched_frames.get(pred_id))
    pred_tracks_never_matched = max(0, len(pred_tracks) - pred_tracks_with_match)
    gt_tracks_never_matched = max(0, len(gt_tracks) - gt_tracks_with_match)

    division_rows: List[Dict[str, object]] = []
    division_counts = Counter()
    for child_gt_id, child_info in sorted(gt_tracks.items()):
        parent_gt_id = child_info["parent"]
        if parent_gt_id <= 0:
            continue
        start = child_info["start"]
        division_counts["gt_child_tracks"] += 1
        child_pred = child_prediction_for(child_gt_id, start, matched_history)
        parent_pred = parent_prediction_for(parent_gt_id, start, matched_history)
        pred_parent = pred_tracks.get(child_pred, {}).get("parent") if child_pred is not None else None
        if child_pred is None:
            status = "child_unmatched"
        elif parent_pred is None:
            status = "parent_unmatched"
        elif pred_parent == parent_pred:
            status = "correct"
        elif pred_parent in (None, 0):
            status = "missing_parent_link"
        else:
            status = "wrong_parent_link"
        division_counts[status] += 1
        division_rows.append(
            {
                "dataset": display_name,
                "sequence": sequence,
                "frame": start,
                "child_gt_id": child_gt_id,
                "parent_gt_id": parent_gt_id,
                "child_pred_id": "" if child_pred is None else child_pred,
                "parent_pred_id": "" if parent_pred is None else parent_pred,
                "pred_parent_of_child": "" if pred_parent is None else pred_parent,
                "status": status,
            }
        )
        if status != "correct":
            event_rows.append(
                event_row(
                    dataset=display_name,
                    sequence=sequence,
                    frame=start,
                    event_type="division_error",
                    gt_id=str(child_gt_id),
                    pred_id="" if child_pred is None else str(child_pred),
                    related_ids=f"parent_gt={parent_gt_id};parent_pred={parent_pred};pred_parent={pred_parent}",
                    note=status,
                )
            )

    for row in frame_rows:
        row["division_errors"] = 0
    division_errors_by_frame = Counter(row["frame"] for row in division_rows if row["status"] != "correct")
    for row in frame_rows:
        errors = division_errors_by_frame[int(row["frame"])]
        row["division_errors"] = errors
        row["failure_score"] = int(row["failure_score"]) + 4 * errors

    top_windows = select_top_windows(frame_rows)

    precision = totals["tp"] / (totals["tp"] + totals["false_positives"]) if (totals["tp"] + totals["false_positives"]) else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["false_negatives"]) if (totals["tp"] + totals["false_negatives"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    gt_child_tracks = division_counts["gt_child_tracks"]
    correct_divisions = division_counts["correct"]
    division_error_rate = 1 - correct_divisions / gt_child_tracks if gt_child_tracks else 0.0

    summary = {
        "dataset": display_name,
        "dataset_key": dataset_key,
        "sequence": sequence,
        "frames": len(frames),
        "gt_instances": totals["gt_instances"],
        "pred_instances": totals["pred_instances"],
        "tp": totals["tp"],
        "false_negatives": totals["false_negatives"],
        "false_positives": totals["false_positives"],
        "precision_iou50": round(precision, 6),
        "recall_iou50": round(recall, 6),
        "f1_iou50": round(f1, 6),
        "fn_rate": round(totals["false_negatives"] / totals["gt_instances"], 6) if totals["gt_instances"] else 0.0,
        "fp_per_gt": round(totals["false_positives"] / totals["gt_instances"], 6) if totals["gt_instances"] else 0.0,
        "split_candidates": totals["split_candidates"],
        "merge_candidates": totals["merge_candidates"],
        "id_switches": totals["id_switches"],
        "reid_after_gap": totals["reid_after_gap"],
        "gt_tracks": len(gt_tracks),
        "gt_tracks_with_match": gt_tracks_with_match,
        "gt_tracks_never_matched": gt_tracks_never_matched,
        "fragmented_gt_tracks": fragmented_tracks,
        "pred_tracks": len(pred_tracks),
        "pred_tracks_with_match": pred_tracks_with_match,
        "pred_tracks_never_matched": pred_tracks_never_matched,
        "matched_gt_track_frame_ratio": round(total_matched_gt_frames / totals["gt_instances"], 6) if totals["gt_instances"] else 0.0,
        "gt_child_tracks": gt_child_tracks,
        "correct_division_child_links": correct_divisions,
        "division_child_unmatched": division_counts["child_unmatched"],
        "division_parent_unmatched": division_counts["parent_unmatched"],
        "division_missing_parent_link": division_counts["missing_parent_link"],
        "division_wrong_parent_link": division_counts["wrong_parent_link"],
        "division_error_rate": round(division_error_rate, 6),
        "mean_frame_failure_score": round(float(np.mean([row["failure_score"] for row in frame_rows])) if frame_rows else 0.0, 6),
        "max_frame_failure_score": max([int(row["failure_score"]) for row in frame_rows], default=0),
    }
    return summary, frame_rows, event_rows, track_rows, division_rows + top_windows


def dominant_failure_type(row: Dict[str, object]) -> str:
    if int(row.get("division_errors", 0)) > 0:
        return "division_link"
    if int(row.get("id_switches", 0)) + int(row.get("reid_after_gap", 0)) > 0:
        return "identity_switch"
    if int(row.get("split_candidates", 0)) + int(row.get("merge_candidates", 0)) > 0:
        return "split_merge"
    if int(row.get("false_negatives", 0)) >= int(row.get("false_positives", 0)):
        return "missed_detection"
    return "false_positive"


def select_top_windows(frame_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    chosen: List[Dict[str, object]] = []
    candidates = sorted(frame_rows, key=lambda row: (int(row["failure_score"]), int(row["false_negatives"]) + int(row["false_positives"])), reverse=True)
    for row in candidates:
        if int(row["failure_score"]) <= 0:
            continue
        frame = int(row["frame"])
        if any(
            row["dataset"] == selected["dataset"]
            and row["sequence"] == selected["sequence"]
            and abs(frame - int(selected["center_frame"])) <= NMS_RADIUS
            for selected in chosen
        ):
            continue
        chosen.append(
            {
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                "center_frame": frame,
                "start_frame": max(0, frame - WINDOW_RADIUS),
                "end_frame": frame + WINDOW_RADIUS,
                "failure_score": row["failure_score"],
                "dominant_failure_type": dominant_failure_type(row),
                "false_negatives": row["false_negatives"],
                "false_positives": row["false_positives"],
                "split_candidates": row["split_candidates"],
                "merge_candidates": row["merge_candidates"],
                "id_switches": row["id_switches"],
                "reid_after_gap": row["reid_after_gap"],
                "division_errors": row.get("division_errors", 0),
            }
        )
        if len(chosen) >= TOP_WINDOWS_PER_SEQUENCE:
            break
    return chosen


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_top_windows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["failure_score"]),
            int(row["false_negatives"]) + int(row["false_positives"]),
            str(row["dataset"]),
            str(row["sequence"]),
            int(row["center_frame"]),
        ),
        reverse=True,
    )


def aggregate_dataset(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)

    int_columns = [
        "frames",
        "gt_instances",
        "pred_instances",
        "tp",
        "false_negatives",
        "false_positives",
        "split_candidates",
        "merge_candidates",
        "id_switches",
        "reid_after_gap",
        "gt_tracks",
        "gt_tracks_with_match",
        "gt_tracks_never_matched",
        "fragmented_gt_tracks",
        "pred_tracks",
        "pred_tracks_with_match",
        "pred_tracks_never_matched",
        "gt_child_tracks",
        "correct_division_child_links",
        "division_child_unmatched",
        "division_parent_unmatched",
        "division_missing_parent_link",
        "division_wrong_parent_link",
    ]
    dataset_rows: List[Dict[str, object]] = []
    for dataset, items in sorted(grouped.items()):
        agg = {"dataset": dataset, "sequences": len(items)}
        for col in int_columns:
            agg[col] = sum(int(item[col]) for item in items)
        precision = agg["tp"] / (agg["tp"] + agg["false_positives"]) if (agg["tp"] + agg["false_positives"]) else 0.0
        recall = agg["tp"] / (agg["tp"] + agg["false_negatives"]) if (agg["tp"] + agg["false_negatives"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        agg["precision_iou50"] = round(precision, 6)
        agg["recall_iou50"] = round(recall, 6)
        agg["f1_iou50"] = round(f1, 6)
        agg["fn_rate"] = round(agg["false_negatives"] / agg["gt_instances"], 6) if agg["gt_instances"] else 0.0
        agg["fp_per_gt"] = round(agg["false_positives"] / agg["gt_instances"], 6) if agg["gt_instances"] else 0.0
        gt_child_tracks = agg["gt_child_tracks"]
        correct_divisions = agg["correct_division_child_links"]
        agg["division_error_rate"] = round(1 - correct_divisions / gt_child_tracks, 6) if gt_child_tracks else 0.0
        agg["mean_frame_failure_score"] = round(float(np.average([float(item["mean_frame_failure_score"]) for item in items], weights=[int(item["frames"]) for item in items])), 6)
        agg["max_frame_failure_score"] = max(int(item["max_frame_failure_score"]) for item in items)
        dataset_rows.append(agg)
    return dataset_rows


def write_plan(path: Path, dataset_summary_rows: List[Dict[str, object]]) -> None:
    rows = sorted(dataset_summary_rows, key=lambda row: float(row["fn_rate"]) + float(row["fp_per_gt"]) + 0.02 * int(row["id_switches"]), reverse=True)
    priority_lines = []
    for row in rows:
        priority_lines.append(
            f"- {row['dataset']}：FN={row['false_negatives']}，FP={row['false_positives']}，"
            f"IDSW={row['id_switches']}，fragmentation={row['fragmented_gt_tracks']}/{row['gt_tracks']}，"
            f"division error rate={row['division_error_rate']}"
        )
    priority_block = "\n".join(priority_lines)
    text = f"""# Failure Case 可视化方案

## 目标

基于 `r1-50-66-512-288-mask-30` 下的 3T3、HEK293、HeLa、PC-3 和 RAW264 五个数据集，制作一组紧凑的 failure-case 图和补充统计表，用来回应 decision letter 中对失败案例分析的要求。图中重点展示方法仍会失败的场景：漏检、误检、分裂/粘连、ID switch、轨迹碎片化和有丝分裂连接错误。

## 统计口径

逐帧读取预测的 CTC `*_RES/maskXXX.tif` 和对应 GT `TRA/man_trackXXX.tif`，使用 IoU >= {IOU_THRESHOLD} 的贪心一对一匹配统计 TP、FN 和 FP。若一个 GT/预测实例与另一侧至少两个实例存在显著重叠，则记为 split/merge candidate；显著重叠阈值为 overlap fraction >= {OVERLAP_FRACTION_THRESHOLD} 且至少 {MIN_OVERLAP_PIXELS} 个像素。

当前数据集层面的 failure 优先级如下：

{priority_block}

## 推荐主图

Panel A：五个数据集的 failure 类型分布。建议使用堆叠柱状图展示 FN、FP、ID switch、split/merge、fragmentation 和 division-link error，让读者先看到整体错误构成。

Panel B：逐帧 failure heatmap。横轴为 frame index，纵轴为 sequence，颜色为 `failure_score`。这个图用于解释为什么选择这些可视化帧，避免看起来像 cherry-picking。

Panel C：代表性多帧局部图。每类主要 failure 从 `failure_case_top_windows.csv` 中挑选一个高分窗口，以中心帧前后各 2 帧组成 5-frame crop。建议 GT 轮廓用绿色，预测轮廓用洋红色，FN 用红框，FP 用黄框，ID switch 用橙框，预测轨迹连接用青色实线，GT 轨迹连接用绿色虚线。

Panel D：轨迹 timeline。选择一个 ID switch 或 fragmentation 案例，每行表示一个 GT track，不同颜色片段表示匹配到的不同 predicted track id；未匹配帧留空，ID switch 位置用红色竖线标记。

Panel E：有丝分裂/连接错误示例。根据 `division_stats.csv` 中非 correct 的行，围绕 parent 和 child 在 t-1、t、t+1 取 crop，并标注错误类型：child unmatched、parent unmatched、missing parent link 或 wrong parent link。

## 候选案例选择规则

优先从 `failure_case_top_windows.csv` 中选图。主文建议选 4 到 6 个例子，尽量覆盖不同 failure 类型和不同数据集：

- 一个密集场景下的漏检或误检案例，优先从 FN/FP 负担最大的 dataset 中选择；
- 一个 PC-3 或 RAW264 的 split/merge 案例；
- 一个 ID switch 或 re-identification-after-gap 案例；
- 一个 mitosis/linking error 案例；
- 一个整体跟踪较好但局部仍有明显 failure 的 success-adjacent 案例。

## 已生成文件

- `failure_case_dataset_summary.csv`：数据集级总表。
- `failure_case_sequence_summary.csv`：sequence 级总表。
- `failure_case_top_windows.csv`：按 failure score 排序的可视化候选窗口。
- `<DATASET>/failure_cases/frame_stats.csv`：逐帧 failure 计数和 failure score。
- `<DATASET>/failure_cases/failure_events.csv`：事件级记录，包含对象 id 和 bbox。
- `<DATASET>/failure_cases/track_failure_stats.csv`：GT track 级匹配情况和 fragmentation。
- `<DATASET>/failure_cases/division_stats.csv`：child track 的 division/link 状态。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/group/dingyi/ct/r1-50-66-512-288-mask-30"))
    parser.add_argument("--data-root", type=Path, default=Path("/group/dingyi/ct/data"))
    args = parser.parse_args()

    root = args.root
    data_root = args.data_root
    all_sequence_summaries: List[Dict[str, object]] = []
    all_top_windows: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {
        "root": str(root),
        "data_root": str(data_root),
        "iou_threshold": IOU_THRESHOLD,
        "overlap_fraction_threshold": OVERLAP_FRACTION_THRESHOLD,
        "min_overlap_pixels": MIN_OVERLAP_PIXELS,
        "datasets": {},
    }

    for dataset_key, (display_name, data_name) in DATASETS.items():
        pred_ctc_dir = root / display_name / "test" / "CTC"
        dataset_out_dir = root / display_name / "failure_cases"
        sequence_summaries: List[Dict[str, object]] = []
        dataset_frame_rows: List[Dict[str, object]] = []
        dataset_event_rows: List[Dict[str, object]] = []
        dataset_track_rows: List[Dict[str, object]] = []
        dataset_division_rows: List[Dict[str, object]] = []
        dataset_top_windows: List[Dict[str, object]] = []

        for pred_res_dir in sorted(pred_ctc_dir.glob("*_RES")):
            sequence = pred_res_dir.name.replace("_RES", "")
            gt_sequence_dir = data_root / data_name / "test"
            summary, frame_rows, event_rows, track_rows, division_and_window_rows = analyze_sequence(
                dataset_key=dataset_key,
                display_name=display_name,
                data_name=data_name,
                sequence=sequence,
                pred_res_dir=pred_res_dir,
                gt_sequence_dir=gt_sequence_dir,
            )
            division_rows = [row for row in division_and_window_rows if "status" in row]
            top_windows = [row for row in division_and_window_rows if "center_frame" in row]
            sequence_summaries.append(summary)
            dataset_frame_rows.extend(frame_rows)
            dataset_event_rows.extend(event_rows)
            dataset_track_rows.extend(track_rows)
            dataset_division_rows.extend(division_rows)
            dataset_top_windows.extend(top_windows)

        write_csv(dataset_out_dir / "sequence_summary.csv", sequence_summaries)
        write_csv(dataset_out_dir / "frame_stats.csv", dataset_frame_rows)
        write_csv(dataset_out_dir / "failure_events.csv", dataset_event_rows)
        write_csv(dataset_out_dir / "track_failure_stats.csv", dataset_track_rows)
        write_csv(dataset_out_dir / "division_stats.csv", dataset_division_rows)
        dataset_top_windows = sort_top_windows(dataset_top_windows)
        write_csv(dataset_out_dir / "top_failure_windows.csv", dataset_top_windows)

        all_sequence_summaries.extend(sequence_summaries)
        all_top_windows.extend(dataset_top_windows)
        manifest["datasets"][display_name] = {
            "sequences": len(sequence_summaries),
            "output_dir": str(dataset_out_dir),
        }

    dataset_summary_rows = aggregate_dataset(all_sequence_summaries)
    write_csv(root / "failure_case_sequence_summary.csv", all_sequence_summaries)
    write_csv(root / "failure_case_dataset_summary.csv", dataset_summary_rows)
    write_csv(root / "failure_case_top_windows.csv", sort_top_windows(all_top_windows))
    write_plan(root / "failure_case_visualization_plan.md", dataset_summary_rows)
    (root / "failure_case_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
