#!/usr/bin/env python3
"""Evaluate LODO inference outputs with the local CTC metrics package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = Path("/group/dingyi/metrics")
DEFAULT_DATASETS = [
    "lodo_test_3t3",
    "lodo_test_hek",
    "lodo_test_hela",
    "lodo_test_pc",
    "lodo_test_raw",
]
FINAL_KEYS = ["CHOTA", "TRA", "DET", "MOTA", "HOTA", "IDF1", "LNK"]


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


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in keys})


def _extract_frame_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


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


def _test_entries(dataset: str) -> list[dict[str, Any]]:
    manifest_path = REPO_ROOT / "data" / dataset / "split_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = [entry for entry in manifest if entry["generated_split"] == "test"]
    return sorted(entries, key=lambda entry: entry["generated_sequence"])


def evaluate_dataset(dataset: str, num_threads: int) -> dict[str, Any]:
    _prepend_metrics_package()
    from ctc_metrics.scripts.evaluate import evaluate_sequence

    result_root = REPO_ROOT / "results" / dataset
    ctc_root = result_root / "test" / "CTC"
    metrics_dir = result_root / "metrics"
    prepared_gt_root = result_root / "metrics_gt"
    entries = _test_entries(dataset)

    sequence_rows: list[dict[str, Any]] = []
    successful_results: list[dict[str, Any]] = []
    missing_sequences: list[str] = []
    failed_sequences: list[dict[str, str]] = []

    for entry in entries:
        sequence = entry["generated_sequence"]
        res_dir = ctc_root / sequence
        res_txt = res_dir / "res.txt"
        res_track = res_dir / "res_track.txt"
        row_prefix = {
            "dataset": dataset,
            "sequence": sequence,
            "source_cell_type": entry["source_cell_type"],
            "source_split": entry["source_split"],
            "source_sequence": entry["source_sequence"],
        }

        if not res_txt.exists() or not res_track.exists():
            missing_sequences.append(sequence)
            sequence_rows.append({**row_prefix, "status": "missing"})
            continue

        source_gt = Path(entry["source_gt_dir"])
        prepared_gt = prepared_gt_root / f"{sequence}_GT"

        try:
            _ensure_gt_bbox_file(source_gt, prepared_gt)
            results = evaluate_sequence(
                res=str(res_dir),
                gt=str(prepared_gt),
                metrics=None,
                threads=num_threads,
            )
        except Exception as exc:  # Keep going so one bad sequence does not hide the rest.
            failed_sequences.append({"sequence": sequence, "error": repr(exc)})
            sequence_rows.append({**row_prefix, "status": "failed", "error": repr(exc)})
            continue

        _write_metric_csv(metrics_dir / f"{sequence}.csv", results)
        successful_results.append(results)
        sequence_rows.append({**row_prefix, "status": "ok", **_jsonable(results)})

    mean_row = _mean_results(successful_results)
    mean_row_with_meta = {
        "dataset": dataset,
        "sequence": "MEAN",
        "source_cell_type": "",
        "source_split": "",
        "source_sequence": "",
        "status": "complete" if len(successful_results) == len(entries) else "incomplete",
        **mean_row,
    }

    metric_keys: list[str] = []
    for row in sequence_rows:
        for key in row:
            if key not in metric_keys:
                metric_keys.append(key)
    for key in mean_row_with_meta:
        if key not in metric_keys:
            metric_keys.append(key)

    _write_rows_csv(metrics_dir / "summary.csv", sequence_rows + [mean_row_with_meta], metric_keys)
    final_keys = [
        "dataset",
        "sequence",
        "source_cell_type",
        "source_split",
        "source_sequence",
        "status",
        *FINAL_KEYS,
    ]
    _write_rows_csv(
        metrics_dir / "summary_final.csv",
        [
            {
                key: row.get(key)
                for key in final_keys
            }
            for row in sequence_rows + [mean_row_with_meta]
        ],
        final_keys,
    )

    status = {
        "dataset": dataset,
        "expected_sequences": len(entries),
        "evaluated_sequences": len(successful_results),
        "missing_sequences": missing_sequences,
        "failed_sequences": failed_sequences,
        "complete": len(successful_results) == len(entries),
        "metrics_dir": str(metrics_dir),
        "summary": mean_row,
    }
    with (metrics_dir / "status.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(status), handle, indent=2)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run metrics for LODO test results.")
    parser.add_argument("datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("-n", "--num-threads", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONPATH", str(METRICS_ROOT))

    statuses = []
    for dataset in args.datasets:
        print(f"== {dataset} ==", flush=True)
        status = evaluate_dataset(dataset, args.num_threads)
        statuses.append(status)
        print(json.dumps(_jsonable(status), indent=2), flush=True)

    summary_rows = []
    for status in statuses:
        row = {
            "dataset": status["dataset"],
            "status": "complete" if status["complete"] else "incomplete",
            "expected_sequences": status["expected_sequences"],
            "evaluated_sequences": status["evaluated_sequences"],
            "missing_sequences": ",".join(status["missing_sequences"]),
        }
        row.update(status["summary"])
        summary_rows.append(row)

    summary_keys = [
        "dataset",
        "status",
        "expected_sequences",
        "evaluated_sequences",
        "missing_sequences",
    ]
    for row in summary_rows:
        for key in row:
            if key not in summary_keys:
                summary_keys.append(key)

    _write_rows_csv(REPO_ROOT / "results" / "lodo_metrics_summary.csv", summary_rows, summary_keys)
    summary_final_keys = [
        "dataset",
        "status",
        "expected_sequences",
        "evaluated_sequences",
        "missing_sequences",
        *FINAL_KEYS,
    ]
    _write_rows_csv(
        REPO_ROOT / "results" / "lodo_metrics_summary_final.csv",
        [{key: row.get(key) for key in summary_final_keys} for row in summary_rows],
        summary_final_keys,
    )


if __name__ == "__main__":
    main()
