#!/usr/bin/env python3
"""Evaluate all CellDINO tracking outputs with local CTC metrics."""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import tifffile

from run_ablation_metrics import (
    FINAL_KEYS,
    _ensure_gt_bbox_file,
    _extract_frame_number,
    _format_value,
    _jsonable,
    _mean_results,
    _prepend_metrics_package,
    _write_metric_csv,
    _write_rows_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["3t3", "hek", "hela", "pc", "raw"]


def _track_confusion(labels_ref: list, labels_comp: list, mapped_ref: list, mapped_comp: list) -> np.ndarray:
    from ctc_metrics.utils.representations import track_confusion_matrix

    return track_confusion_matrix(labels_ref, labels_comp, mapped_ref, mapped_comp)


def _nonzero_track_pairs(track_intersection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = track_intersection[1:, 1:]
    rows, cols = np.nonzero(positive)
    values = positive[rows, cols]
    return rows + 1, cols + 1, values


def _lineage_clique_getter(tracks: np.ndarray, max_label: int):
    parent = {int(row[0]): int(row[3]) for row in tracks}
    children: dict[int, list[int]] = {label: [] for label in range(1, max_label + 1)}
    for child, parent_id in parent.items():
        if 1 <= child <= max_label and 1 <= parent_id <= max_label:
            children.setdefault(parent_id, []).append(child)

    @lru_cache(maxsize=None)
    def descendants(label: int) -> tuple[int, ...]:
        seen: set[int] = set()
        stack = list(children.get(label, ()))
        while stack:
            child = stack.pop()
            if child in seen:
                continue
            seen.add(child)
            stack.extend(children.get(child, ()))
        return tuple(sorted(seen))

    @lru_cache(maxsize=None)
    def ancestors(label: int) -> tuple[int, ...]:
        seen: set[int] = set()
        current = int(parent.get(label, 0))
        while 1 <= current <= max_label and current not in seen:
            seen.add(current)
            current = int(parent.get(current, 0))
        return tuple(sorted(seen))

    @lru_cache(maxsize=None)
    def clique(label: int) -> np.ndarray:
        values = {label}
        values.update(ancestors(label))
        values.update(descendants(label))
        return np.asarray(sorted(value for value in values if 1 <= value <= max_label), dtype=np.int64)

    return clique


def _fast_hota(labels_ref: list, labels_comp: list, mapped_ref: list, mapped_comp: list) -> dict[str, float]:
    track_intersection = _track_confusion(labels_ref, labels_comp, mapped_ref, mapped_comp)
    rows, cols, values = _nonzero_track_pairs(track_intersection)
    row_sum = track_intersection[1:, :].sum(axis=1)
    col_sum = track_intersection[:, 1:].sum(axis=0)

    hota_score = 0.0
    for i, j, tpa in zip(rows, cols, values):
        fna = row_sum[i - 1] - tpa
        fpa = col_sum[j - 1] - tpa
        hota_score += tpa * (tpa / (tpa + fna + fpa))

    tp = track_intersection[1:, 1:].sum()
    fp = track_intersection[0, 1:].sum()
    fn = track_intersection[1:, 0].sum()
    return {"HOTA": float(np.sqrt(hota_score / (tp + fp + fn)))}


def _fast_chota(
    ref_tracks: np.ndarray,
    comp_tracks: np.ndarray,
    labels_ref: list,
    labels_comp: list,
    mapped_ref: list,
    mapped_comp: list,
) -> dict[str, float]:
    max_ref = int(np.max(np.concatenate(labels_ref)))
    max_comp = int(np.max(np.concatenate(labels_comp)))
    track_intersection = _track_confusion(labels_ref, labels_comp, mapped_ref, mapped_comp)
    rows, cols, values = _nonzero_track_pairs(track_intersection)
    ref_clique = _lineage_clique_getter(ref_tracks, max_ref)
    comp_clique = _lineage_clique_getter(comp_tracks, max_comp)

    clique_pair_sum: dict[tuple[int, int], float] = {}
    clique_ref_total: dict[int, float] = {}
    clique_comp_total: dict[int, float] = {}

    chota_score = 0.0
    for i, j, value in zip(rows, cols, values):
        ref_key = int(i)
        comp_key = int(j)
        ref_ids = ref_clique(ref_key)
        comp_ids = comp_clique(comp_key)
        pair_key = (ref_key, comp_key)
        if pair_key not in clique_pair_sum:
            clique_pair_sum[pair_key] = float(track_intersection[np.ix_(ref_ids, comp_ids)].sum())
        if ref_key not in clique_ref_total:
            clique_ref_total[ref_key] = float(track_intersection[ref_ids, :].sum())
        if comp_key not in clique_comp_total:
            clique_comp_total[comp_key] = float(track_intersection[:, comp_ids].sum())
        tpa = clique_pair_sum[pair_key]
        fna = clique_ref_total[ref_key] - tpa
        fpa = clique_comp_total[comp_key] - tpa
        chota_score += float(value) * (tpa / (tpa + fna + fpa))

    tp = track_intersection[1:, 1:].sum()
    fp = track_intersection[0, 1:].sum()
    fn = track_intersection[1:, 0].sum()
    return {"CHOTA": float(np.sqrt(chota_score / (tp + fp + fn)))}


def _fast_idf1(labels_ref: list, labels_comp: list, mapped_ref: list, mapped_comp: list) -> dict[str, float]:
    track_intersection = _track_confusion(labels_ref, labels_comp, mapped_ref, mapped_comp)
    max_ref = track_intersection.shape[0] - 1
    max_comp = track_intersection.shape[1] - 1
    intersections = track_intersection[1:, 1:]

    costs = np.zeros((max_ref, max_comp + max_ref), dtype=np.float64)
    costs[:, :max_comp] = -intersections
    row_ind, col_ind = linear_sum_assignment(costs)
    matched = col_ind < max_comp
    idtp = float(intersections[row_ind[matched], col_ind[matched]].sum())

    total_gt = float(track_intersection[1:, :].sum())
    total_comp = float(track_intersection[:, 1:].sum())
    idfn = total_gt - idtp
    idfp = total_comp - idtp
    idp = idtp / (idtp + idfp) if idtp + idfp > 0 else 0.0
    idr = idtp / (idtp + idfn) if idtp + idfn > 0 else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn > 0 else 0.0

    return {"IDF1": idf1, "IDP": idp, "IDR": idr, "IDTP": idtp, "IDFP": idfp, "IDFN": idfn}


def _patch_fast_association_metrics() -> None:
    import ctc_metrics.scripts.evaluate as evaluate_module

    evaluate_module.hota = _fast_hota
    evaluate_module.chota = _fast_chota
    evaluate_module.idf1 = _fast_idf1


def _write_final_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows_csv(path, rows, ["dataset", "sequence", *FINAL_KEYS])


def _ensure_res_bbox_file(res_dir: Path) -> Path:
    bbox_file = res_dir / "res.txt"
    mask_files = sorted(res_dir.glob("mask*.tif"), key=_extract_frame_number)
    if not mask_files:
        mask_files = sorted(res_dir.glob("*.tif"), key=_extract_frame_number)
    if not mask_files:
        raise FileNotFoundError(f"No result mask files found in {res_dir}")

    newest_mask_mtime = max(path.stat().st_mtime for path in mask_files)
    if bbox_file.exists() and bbox_file.stat().st_mtime >= newest_mask_mtime:
        return bbox_file

    with bbox_file.open("w", encoding="utf-8") as handle:
        for frame_index, mask_path in enumerate(mask_files, start=1):
            mask = tifffile.imread(mask_path)
            if mask.ndim > 2:
                mask = mask[..., 0]
            labels = np.unique(mask)
            labels = labels[labels != 0]
            for label in np.sort(labels):
                ys, xs = np.where(mask == label)
                if xs.size == 0:
                    continue
                x_min = int(xs.min())
                x_max = int(xs.max())
                y_min = int(ys.min())
                y_max = int(ys.max())
                handle.write(
                    f"{frame_index},{int(label)},{x_min},{y_min},"
                    f"{x_max - x_min + 1},{y_max - y_min + 1}\n"
                )
    return bbox_file


def _resolve_tracking_root(path: str) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def _default_metrics_subdir(split: str) -> str:
    return "metrics" if split == "test" else f"metrics_{split}"


def evaluate_dataset(
    dataset: str,
    num_threads: int,
    metric_names: list[str] | None,
    tracking_root: Path,
    split: str,
    metrics_subdir: str,
) -> dict[str, Any]:
    _prepend_metrics_package()
    _patch_fast_association_metrics()
    from ctc_metrics.scripts.evaluate import evaluate_sequence

    result_root = tracking_root / dataset
    ctc_root = result_root / split
    metrics_dir = result_root / metrics_subdir
    prepared_gt_root = result_root / f"metrics_gt_{split}"
    source_test_root = REPO_ROOT / "data" / dataset / split

    if not ctc_root.exists():
        raise FileNotFoundError(f"Missing CellDINO result root: {ctc_root}")

    res_dirs = sorted(path for path in ctc_root.iterdir() if path.is_dir() and path.name.endswith("_RES"))
    sequence_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    successful_results: list[dict[str, Any]] = []
    failed_sequences: list[dict[str, str]] = []
    missing_sequences: list[str] = []

    for res_dir in res_dirs:
        sequence = res_dir.name.removesuffix("_RES")
        res_track = res_dir / "res_track.txt"

        if not res_track.exists():
            missing_sequences.append(sequence)
            sequence_rows.append({"dataset": dataset, "sequence": sequence, "status": "missing"})
            continue

        source_gt = source_test_root / f"{sequence}_GT"
        prepared_gt = prepared_gt_root / f"{sequence}_GT"
        try:
            _ensure_res_bbox_file(res_dir)
            _ensure_gt_bbox_file(source_gt, prepared_gt)
            results = evaluate_sequence(
                res=str(res_dir),
                gt=str(prepared_gt),
                metrics=metric_names,
                threads=num_threads,
            )
        except Exception as exc:
            failed_sequences.append({"sequence": sequence, "error": repr(exc)})
            sequence_rows.append(
                {"dataset": dataset, "sequence": sequence, "status": "failed", "error": repr(exc)}
            )
            continue

        metric_name = f"{dataset}-{sequence}"
        _write_metric_csv(metrics_dir / f"{metric_name}.csv", results)
        successful_results.append(results)
        full_row = {"dataset": dataset, "sequence": metric_name, "status": "ok", **_jsonable(results)}
        final_row = {"dataset": dataset, "sequence": metric_name, **{key: results.get(key) for key in FINAL_KEYS}}
        sequence_rows.append(full_row)
        final_rows.append(final_row)

    if sequence_rows:
        keys: list[str] = []
        for row in sequence_rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        _write_rows_csv(metrics_dir / "metrics_summary.csv", sequence_rows, keys)

    _write_final_summary(metrics_dir / "metrics_summary_final.csv", final_rows)

    mean_row = {
        "dataset": dataset,
        "complete": bool(res_dirs) and len(successful_results) == len(res_dirs),
        "expected_sequences": len(res_dirs),
        "evaluated_sequences": len(successful_results),
        **_mean_results(successful_results),
    }
    status = {
        **mean_row,
        "split": split,
        "missing_sequences": missing_sequences,
        "failed_sequences": failed_sequences,
        "metrics_dir": str(metrics_dir),
        "summary_final_path": str(metrics_dir / "metrics_summary_final.csv"),
    }
    (metrics_dir / "status.json").write_text(json.dumps(_jsonable(status), indent=2), encoding="utf-8")
    return _jsonable(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tracking-root", default="results/celldino_tracking")
    parser.add_argument("--split", default="test")
    parser.add_argument("--metrics-subdir", default=None)
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Evaluate every metric exposed by /group/dingyi/metrics instead of only final table metrics.",
    )
    args = parser.parse_args()
    metric_names = None if args.all_metrics else FINAL_KEYS
    tracking_root = _resolve_tracking_root(args.tracking_root)
    metrics_subdir = args.metrics_subdir or _default_metrics_subdir(args.split)

    summaries = []
    all_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"== {dataset} ==")
        status = evaluate_dataset(dataset, args.threads, metric_names, tracking_root, args.split, metrics_subdir)
        summaries.append(status)
        print(json.dumps(status, indent=2))

        dataset_summary = Path(status["summary_final_path"])
        with dataset_summary.open("r", encoding="utf-8", newline="") as handle:
            all_rows.extend(csv.DictReader(handle, delimiter=";"))

    result_root = tracking_root
    summary_name = "metrics_summary_final.csv" if args.split == "test" else f"metrics_summary_{args.split}_final.csv"
    means_name = "dataset_means_final.csv" if args.split == "test" else f"dataset_means_{args.split}_final.csv"
    _write_final_summary(result_root / summary_name, all_rows)
    _write_rows_csv(
        result_root / means_name,
        summaries,
        ["dataset", "split", "complete", "expected_sequences", "evaluated_sequences", *FINAL_KEYS],
    )
    print(f"Wrote {result_root / summary_name}")
    print(f"Wrote {result_root / means_name}")


if __name__ == "__main__":
    main()
