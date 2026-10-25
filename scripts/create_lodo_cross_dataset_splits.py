#!/usr/bin/env python3
"""Create leave-one-cell-type-out datasets, configs, and run scripts.

For each held-out cell type, the generated dataset contains:
- train: 8 videos, 2 from each of the other four cell types
- val: 4 videos, 1 from each of the other four cell types
- test: 4 videos, all from the held-out cell type

The CTC split directories are symlinked to the source videos by default to avoid
duplicating the raw data. COCO train/val annotations are materialized because the
training loader consumes COCO-style files.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_ROOT = ROOT / "data"
CFGS_ROOT = ROOT / "cfgs"
RESULTS_ROOT = ROOT / "results"
TEMPLATE_CFG = CFGS_ROOT / "train_3t3.yaml"

sys.path.insert(0, str(SCRIPT_DIR))
import utils_coco as utils  # noqa: E402


CELL_TYPES = ("3t3", "hek", "hela", "pc", "raw")
DATASET_PREFIX = "lodo_test"
SEED = 20260725


@dataclass(frozen=True)
class SourceVideo:
    cell_type: str
    split: str
    seq: str
    seq_dir: Path
    gt_dir: Path

    @property
    def source_id(self) -> str:
        return f"{self.cell_type}:{self.split}/{self.seq}"


@dataclass(frozen=True)
class AssignedVideo:
    generated_dataset: str
    generated_split: str
    generated_seq: str
    source: SourceVideo


def seq_sort_key(path: Path) -> int:
    return int(path.name)


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame number from {path}")
    return int(match.group(1))


def discover_videos(data_root: Path) -> Dict[str, List[SourceVideo]]:
    pools: Dict[str, List[SourceVideo]] = {}
    for cell_type in CELL_TYPES:
        videos: List[SourceVideo] = []
        for split in ("test", "val", "train"):
            split_dir = data_root / cell_type / split
            if not split_dir.exists():
                continue
            for seq_dir in sorted(
                [p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                key=seq_sort_key,
            ):
                gt_dir = split_dir / f"{seq_dir.name}_GT"
                if not gt_dir.exists():
                    raise FileNotFoundError(f"Missing GT directory for {seq_dir}: {gt_dir}")
                if not list(seq_dir.glob("*.tif")):
                    raise FileNotFoundError(f"No tif frames found in {seq_dir}")
                videos.append(
                    SourceVideo(
                        cell_type=cell_type,
                        split=split,
                        seq=seq_dir.name,
                        seq_dir=seq_dir,
                        gt_dir=gt_dir,
                    )
                )
        if len(videos) < 4:
            raise RuntimeError(f"{cell_type} has fewer than 4 available videos")
        pools[cell_type] = videos
    return pools


def choose_assignments(pools: Dict[str, List[SourceVideo]], seed: int) -> Dict[str, List[AssignedVideo]]:
    rng = random.Random(seed)
    all_assignments: Dict[str, List[AssignedVideo]] = {}

    for heldout in CELL_TYPES:
        generated_dataset = f"{DATASET_PREFIX}_{heldout}"
        assignments: List[AssignedVideo] = []

        train_counter = 1
        val_counter = 1
        test_counter = 1

        for cell_type in CELL_TYPES:
            candidates = list(pools[cell_type])
            rng.shuffle(candidates)
            if cell_type == heldout:
                for source in candidates[:4]:
                    assignments.append(
                        AssignedVideo(
                            generated_dataset=generated_dataset,
                            generated_split="test",
                            generated_seq=f"{test_counter:02d}",
                            source=source,
                        )
                    )
                    test_counter += 1
            else:
                selected = candidates[:3]
                for source in selected[:2]:
                    assignments.append(
                        AssignedVideo(
                            generated_dataset=generated_dataset,
                            generated_split="train",
                            generated_seq=f"{train_counter:02d}",
                            source=source,
                        )
                    )
                    train_counter += 1
                assignments.append(
                    AssignedVideo(
                        generated_dataset=generated_dataset,
                        generated_split="val",
                        generated_seq=f"{val_counter:02d}",
                        source=selected[2],
                    )
                )
                val_counter += 1

        all_assignments[generated_dataset] = assignments
    return all_assignments


def safe_rmtree(path: Path, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to rebuild it")
    shutil.rmtree(path)


def link_or_copy(source: Path, target: Path, copy_data: bool) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Target already exists: {target}")
    if copy_data:
        shutil.copytree(source, target)
    else:
        target.symlink_to(source, target_is_directory=True)


def materialize_ctc_dataset(
    dataset_name: str,
    assignments: Sequence[AssignedVideo],
    data_root: Path,
    overwrite: bool,
    copy_data: bool,
) -> None:
    dataset_dir = data_root / dataset_name
    safe_rmtree(dataset_dir, overwrite)
    dataset_dir.mkdir(parents=True, exist_ok=False)

    for split in ("train", "val", "test"):
        (dataset_dir / split).mkdir()

    for assignment in assignments:
        split_dir = dataset_dir / assignment.generated_split
        seq_dir = split_dir / assignment.generated_seq
        gt_dir = split_dir / f"{assignment.generated_seq}_GT"
        link_or_copy(assignment.source.seq_dir, seq_dir, copy_data)
        link_or_copy(assignment.source.gt_dir, gt_dir, copy_data)


def read_track_file(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                rows.append([int(value) for value in parts])
    if not rows:
        raise ValueError(f"Empty track file: {path}")
    return np.asarray(rows, dtype=np.int64)


def make_coco_for_split(dataset_dir: Path, split: str, dataset_name: str) -> None:
    coco_dir = dataset_dir / "COCO"
    img_dir = coco_dir / split / "img"
    gt_out_dir = coco_dir / split / "gt"
    anno_dir = coco_dir / "annotations" / split
    man_track_dir = coco_dir / "man_track" / split
    for path in (img_dir, gt_out_dir, anno_dir, man_track_dir):
        path.mkdir(parents=True, exist_ok=True)
    for folder in (img_dir, gt_out_dir):
        for old in folder.glob("*.tif"):
            old.unlink()
    for old in anno_dir.glob("*.json"):
        old.unlink()
    for old in man_track_dir.glob("*.txt"):
        old.unlink()

    image_id = 0
    annotation_id = 0
    images = []
    annotations = []
    max_num_of_cells = 0

    sequence_dirs = sorted(
        [p for p in (dataset_dir / split).iterdir() if p.is_dir() and p.name.isdigit()],
        key=seq_sort_key,
    )

    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        fps = sorted(sequence_dir.glob("*.tif"), key=frame_number)
        if len(fps) < 4:
            continue

        track_file = read_track_file(sequence_dir.parent / f"{sequence_name}_GT" / "TRA" / "man_track.txt")
        for frame_path in fps:
            frame = frame_number(frame_path)
            image = read_image_uint8(frame_path)
            gt_path = sequence_dir.parent / f"{sequence_name}_GT" / "SEG" / f"man_seg{frame:03d}.tif"
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt is None:
                raise FileNotFoundError(f"Could not read GT mask: {gt_path}")

            cell_ids = np.unique(gt)
            cell_ids = cell_ids[cell_ids != 0]
            max_num_of_cells = max(max_num_of_cells, int(len(cell_ids)))
            if len(cell_ids) == 0:
                cell_ids = [-1]

            for cell_id in cell_ids:
                annotation = create_bbox_annotation(gt, int(cell_id), image_id, annotation_id, sequence_name)
                annotations.append(annotation)
                annotation_id += 1

            file_name = f"CTC_{sequence_name}_frame_{frame:03d}.tif"
            images.append(
                {
                    "license": 1,
                    "file_name": file_name,
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                    "id": image_id,
                    "ctc_id": sequence_name,
                    "frame_id": frame,
                    "seq_length": len(fps),
                }
            )
            image_id += 1
            cv2.imwrite(str(img_dir / file_name), image)
            cv2.imwrite(str(gt_out_dir / file_name), gt)

        np.savetxt(man_track_dir / f"{sequence_name}.txt", track_file, fmt="%d")

    metadata = {
        "annotations": annotations,
        "images": images,
        "categories": [{"id": 1, "name": "cell"}],
        "licenses": [{"id": 1, "name": "MIT license"}],
        "info": {
            "description": "Leave-one-cell-type-out mixed cell tracking dataset generated from existing CTC splits",
            "version": "1.0",
            "year": 2026,
        },
        "sequences": "cells",
        "max_num_of_cells": max_num_of_cells,
    }
    with (anno_dir / "anno.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, cls=utils.NpEncoder)


def read_image_uint8(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    max_value = float(np.max(image))
    if max_value > 0:
        image = image / max_value
    return (255 * image).astype(np.uint8)


def create_bbox_annotation(
    mask: np.ndarray,
    cell_id: int,
    image_id: int,
    annotation_id: int,
    sequence_name: str,
) -> Dict[str, object]:
    if cell_id <= 0:
        return {
            "id": annotation_id,
            "image_id": image_id,
            "bbox": [],
            "area": 0,
            "category_id": 1,
            "track_id": [],
            "dataset_name": sequence_name,
            "ignore": 0,
            "iscrowd": 0,
            "empty": True,
            "edge": None,
        }

    ys, xs = np.nonzero(mask == cell_id)
    if ys.size == 0:
        raise ValueError(f"Cell id {cell_id} is absent from image {image_id}")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    edge = (
        x_min == 0
        or y_min == 0
        or x_max == mask.shape[1] - 1
        or y_max == mask.shape[0] - 1
    )
    return {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": [x_min, y_min, width, height],
        "area": int(ys.size),
        "category_id": 1,
        "track_id": cell_id,
        "dataset_name": sequence_name,
        "ignore": 0,
        "iscrowd": 0,
        "empty": False,
        "edge": edge,
    }


def make_coco_dataset(dataset_name: str, data_root: Path) -> None:
    dataset_dir = data_root / dataset_name
    coco_dir = dataset_dir / "COCO"
    safe_rmtree(coco_dir, overwrite=True)
    for split in ("train", "val"):
        make_coco_for_split(dataset_dir, split, dataset_name)


def write_manifest(dataset_name: str, assignments: Sequence[AssignedVideo], data_root: Path) -> None:
    dataset_dir = data_root / dataset_name
    rows = []
    for assignment in sorted(assignments, key=lambda item: (item.generated_split, item.generated_seq)):
        rows.append(
            {
                "generated_dataset": dataset_name,
                "generated_split": assignment.generated_split,
                "generated_sequence": assignment.generated_seq,
                "source_cell_type": assignment.source.cell_type,
                "source_split": assignment.source.split,
                "source_sequence": assignment.source.seq,
                "source_sequence_dir": str(assignment.source.seq_dir),
                "source_gt_dir": str(assignment.source.gt_dir),
            }
        )

    with (dataset_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (dataset_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    summary = {
        "dataset": dataset_name,
        "train_count": sum(row["generated_split"] == "train" for row in rows),
        "val_count": sum(row["generated_split"] == "val" for row in rows),
        "test_count": sum(row["generated_split"] == "test" for row in rows),
        "train_by_cell_type": count_by_cell(rows, "train"),
        "val_by_cell_type": count_by_cell(rows, "val"),
        "test_by_cell_type": count_by_cell(rows, "test"),
    }
    with (dataset_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def count_by_cell(rows: Iterable[Dict[str, str]], split: str) -> Dict[str, int]:
    counts = {cell_type: 0 for cell_type in CELL_TYPES}
    for row in rows:
        if row["generated_split"] == split:
            counts[row["source_cell_type"]] += 1
    return counts


def write_cfg(dataset_name: str) -> None:
    text = TEMPLATE_CFG.read_text(encoding="utf-8")
    text = re.sub(r"^dataset:\s*.*$", f"dataset: {dataset_name}", text, flags=re.MULTILINE)
    text = re.sub(
        r"^output_dir:\s*.*$",
        f"output_dir: {RESULTS_ROOT / dataset_name}",
        text,
        flags=re.MULTILINE,
    )
    (CFGS_ROOT / f"train_{dataset_name}.yaml").write_text(text, encoding="utf-8")


def write_shell_scripts(dataset_names: Sequence[str]) -> None:
    for dataset_name in dataset_names:
        train_script = ROOT / f"run_train_{dataset_name}.sh"
        infer_script = ROOT / f"run_inf_{dataset_name}.sh"
        train_script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    'PYTHON="${PYTHON:-/home/dingyi/miniconda3/envs/ct/bin/python}"',
                    'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
                    f"EXP={dataset_name}",
                    "",
                    '"${PYTHON}" /group/dingyi/ct/src/train.py with dataset="${EXP}" >"train_${EXP}.log" 2>&1',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        infer_script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    'PYTHON="${PYTHON:-/home/dingyi/miniconda3/envs/ct/bin/python}"',
                    'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
                    f"EXP={dataset_name}",
                    "",
                    '"${PYTHON}" /group/dingyi/ct/src/pipeline.py with dataset="${EXP}" >"inf_${EXP}.log" 2>&1',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        train_script.chmod(0o755)
        infer_script.chmod(0o755)

    all_names = " ".join(dataset_names)
    (ROOT / "run_train_lodo_all.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"for EXP in {all_names}; do",
                '  echo "Training ${EXP}"',
                '  bash "/group/dingyi/ct/run_train_${EXP}.sh"',
                "done",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (ROOT / "run_inf_lodo_all.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"for EXP in {all_names}; do",
                '  echo "Inferring ${EXP}"',
                '  bash "/group/dingyi/ct/run_inf_${EXP}.sh"',
                "done",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (ROOT / "run_train_lodo_all.sh").chmod(0o755)
    (ROOT / "run_inf_lodo_all.sh").chmod(0o755)


def verify_dataset(dataset_name: str, data_root: Path) -> None:
    dataset_dir = data_root / dataset_name
    summary = json.loads((dataset_dir / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["train_count"] == 8, summary
    assert summary["val_count"] == 4, summary
    assert summary["test_count"] == 4, summary

    heldout = dataset_name.replace(f"{DATASET_PREFIX}_", "")
    assert summary["test_by_cell_type"][heldout] == 4, summary
    for cell_type in CELL_TYPES:
        if cell_type == heldout:
            assert summary["train_by_cell_type"][cell_type] == 0, summary
            assert summary["val_by_cell_type"][cell_type] == 0, summary
        else:
            assert summary["train_by_cell_type"][cell_type] == 2, summary
            assert summary["val_by_cell_type"][cell_type] == 1, summary

    for split in ("train", "val", "test"):
        expected = {"train": 8, "val": 4, "test": 4}[split]
        seqs = sorted(p for p in (dataset_dir / split).iterdir() if p.is_dir() and p.name.isdigit())
        gts = sorted(p for p in (dataset_dir / split).iterdir() if p.is_dir() and p.name.endswith("_GT"))
        assert len(seqs) == expected, (dataset_name, split, len(seqs))
        assert len(gts) == expected, (dataset_name, split, len(gts))

    for split in ("train", "val"):
        anno = dataset_dir / "COCO" / "annotations" / split / "anno.json"
        data = json.loads(anno.read_text(encoding="utf-8"))
        assert data["images"], anno
        assert data["annotations"], anno


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-data", action="store_true", help="Copy CTC directories instead of symlinking them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = discover_videos(args.data_root)
    all_assignments = choose_assignments(pools, args.seed)
    dataset_names = list(all_assignments.keys())

    for dataset_name, assignments in all_assignments.items():
        materialize_ctc_dataset(
            dataset_name=dataset_name,
            assignments=assignments,
            data_root=args.data_root,
            overwrite=args.overwrite,
            copy_data=args.copy_data,
        )
        make_coco_dataset(dataset_name, args.data_root)
        write_manifest(dataset_name, assignments, args.data_root)
        write_cfg(dataset_name)
        verify_dataset(dataset_name, args.data_root)

    write_shell_scripts(dataset_names)

    print("Generated datasets:")
    for dataset_name in dataset_names:
        print(f"  {args.data_root / dataset_name}")
    print("Generated cfg files:")
    for dataset_name in dataset_names:
        print(f"  {CFGS_ROOT / ('train_' + dataset_name + '.yaml')}")
    print("Generated run scripts:")
    for dataset_name in dataset_names:
        print(f"  {ROOT / ('run_train_' + dataset_name + '.sh')}")
        print(f"  {ROOT / ('run_inf_' + dataset_name + '.sh')}")


if __name__ == "__main__":
    main()
