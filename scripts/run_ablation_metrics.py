#!/usr/bin/env python3
"""Evaluate DQ/COLA ablation inference outputs with local CTC metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = Path("/group/dingyi/metrics")
FINAL_KEYS = ["CHOTA", "TRA", "DET", "MOTA", "HOTA", "IDF1", "LNK"]
DEFAULT_DATASETS = [
    "3t3-ab-dq0-cola0",
    "3t3-ab-dq0-cola1",
    "3t3-ab-dq1-cola0",
    "hek-ab-dq0-cola0",
    "hek-ab-dq0-cola1",
    "hek-ab-dq1-cola0",
    "hela-ab-dq0-cola0",
    "hela-ab-dq0-cola1",
    "hela-ab-dq1-cola0",
    "pc-ab-dq0-cola0",
    "pc-ab-dq0-cola1",
    "pc-ab-dq1-cola0",
]


def _prepend_metrics_package() -> None:
    metrics_root = str(METRICS_ROOT)
    if metrics_root not in sys.path:
        sys.path.insert(0, metrics_root)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(val) for val in value]
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _extract_frame_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def _write_metric_csv(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(results.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(keys)
        writer.writerow([_format_value(results.get(key)) for key in keys])

    final_path = path.with_name(path.stem + "_final.csv")
    with final_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(FINAL_KEYS)
        writer.writerow([_format_value(results.get(key)) for key in FINAL_KEYS])


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in keys})


def _mean_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {key: None for key in FINAL_KEYS}

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    mean_row: dict[str, Any] = {}
    for key in keys:
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                values.append(float(value))
        mean_row[key] = float(np.mean(values)) if values else None
    return mean_row


def _ensure_gt_bbox_file(source_gt_dir: Path, prepared_gt_dir: Path) -> Path:
    prepared_gt_dir.mkdir(parents=True, exist_ok=True)
    prepared_tra = prepared_gt_dir / "TRA"
    prepared_tra.mkdir(exist_ok=True)

    source_track = source_gt_dir / "TRA" / "man_track.txt"
    prepared_track = prepared_tra / "man_track.txt"
    if not prepared_track.exists() or prepared_track.stat().st_mtime < source_track.stat().st_mtime:
        shutil.copy2(source_track, prepared_track)

    bbox_file = prepared_gt_dir / "gt.txt"
    source_mask_dir = source_gt_dir / "SEG"
    if not source_mask_dir.exists():
        source_mask_dir = source_gt_dir / "TRA"

    mask_files = sorted(source_mask_dir.glob("*.tif"), key=_extract_frame_number)
    if not mask_files:
        raise FileNotFoundError(f"No GT mask files found in {source_gt_dir}")

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


def _base_dataset(dataset: str) -> str:
    if "-ab-" not in dataset:
        return dataset
    return dataset.split("-ab-", 1)[0]


def evaluate_dataset(dataset: str, num_threads: int) -> dict[str, Any]:
    _prepend_metrics_package()
    from ctc_metrics.scripts.evaluate import evaluate_sequence

    result_root = REPO_ROOT / "results" / dataset
    ctc_root = result_root / "test" / "CTC"
    metrics_dir = result_root / "metrics"
    prepared_gt_root = result_root / "metrics_gt"
    base_dataset = _base_dataset(dataset)
    source_test_root = REPO_ROOT / "data" / base_dataset / "test"

    res_dirs = sorted(path for path in ctc_root.iterdir() if path.is_dir())
    sequence_rows: list[dict[str, Any]] = []
    successful_results: list[dict[str, Any]] = []
    missing_sequences: list[str] = []
    failed_sequences: list[dict[str, str]] = []

    for res_dir in res_dirs:
        sequence = res_dir.name
        res_txt = res_dir / "res.txt"
        res_track = res_dir / "res_track.txt"
        row_prefix = {"dataset": dataset, "base_dataset": base_dataset, "sequence": sequence}

        if not res_txt.exists() or not res_track.exists():
            missing_sequences.append(sequence)
            sequence_rows.append({**row_prefix, "status": "missing"})
            continue

        source_gt = source_test_root / f"{sequence}_GT"
        prepared_gt = prepared_gt_root / f"{sequence}_GT"
        try:
            _ensure_gt_bbox_file(source_gt, prepared_gt)
            results = evaluate_sequence(
                res=str(res_dir),
                gt=str(prepared_gt),
                metrics=None,
                threads=num_threads,
            )
        except Exception as exc:
            failed_sequences.append({"sequence": sequence, "error": repr(exc)})
            sequence_rows.append({**row_prefix, "status": "failed", "error": repr(exc)})
            continue

        _write_metric_csv(metrics_dir / f"{sequence}.csv", results)
        successful_results.append(results)
        sequence_rows.append({**row_prefix, "status": "ok", **_jsonable(results)})

    mean_row = _mean_results(successful_results)
    mean_row_with_meta = {
        "dataset": dataset,
        "base_dataset": base_dataset,
        "sequence": "MEAN",
        "status": "complete" if len(successful_results) == len(res_dirs) else "incomplete",
        **mean_row,
    }

    metric_keys: list[str] = []
    for row in sequence_rows + [mean_row_with_meta]:
        for key in row:
            if key not in metric_keys:
                metric_keys.append(key)
    _write_rows_csv(metrics_dir / "summary.csv", sequence_rows + [mean_row_with_meta], metric_keys)

    final_keys = ["dataset", "base_dataset", "sequence", "status", *FINAL_KEYS]
    _write_rows_csv(
        metrics_dir / "summary_final.csv",
        [{key: row.get(key) for key in final_keys} for row in sequence_rows + [mean_row_with_meta]],
        final_keys,
    )

    status = {
        "dataset": dataset,
        "base_dataset": base_dataset,
        "expected_sequences": len(res_dirs),
        "evaluated_sequences": len(successful_results),
        "missing_sequences": missing_sequences,
        "failed_sequences": failed_sequences,
        "complete": len(successful_results) == len(res_dirs),
        "metrics_dir": str(metrics_dir),
        "summary": _jsonable(mean_row),
    }
    (metrics_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    summaries = []
    for dataset in args.datasets:
        print(f"== {dataset} ==")
        status = evaluate_dataset(dataset, args.threads)
        summaries.append(status)
        print(json.dumps(status, indent=2))

    rows = []
    for status in summaries:
        row = {
            "dataset": status["dataset"],
            "base_dataset": status["base_dataset"],
            "complete": status["complete"],
            **{key: status["summary"].get(key) for key in FINAL_KEYS},
        }
        rows.append(row)

    out_path = REPO_ROOT / "results" / "dq_cola_ablation_metrics_summary_final.csv"
    _write_rows_csv(out_path, rows, ["dataset", "base_dataset", "complete", *FINAL_KEYS])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
