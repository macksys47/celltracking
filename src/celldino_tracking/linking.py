from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .data import CTCSequence


def write_results(sequence: CTCSequence, node_to_track: Dict[int, int], tracks: Dict[int, Tuple[int, int, int, int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in sequence.frames:
        label_to_track = {
            record.label: node_to_track[record.node_id]
            for record in sequence.records_by_frame.get(frame, [])
            if record.node_id in node_to_track
        }
        Image.fromarray(sequence.relabel_mask(frame, label_to_track).astype(np.uint16)).save(output_dir / f"mask{frame:03d}.tif")
    rows = np.array([tracks[k] for k in sorted(tracks)], dtype=np.int64)
    if rows.size == 0:
        rows = np.zeros((0, 4), dtype=np.int64)
    np.savetxt(output_dir / "res_track.txt", rows, fmt="%d")


def write_pair_scores(rows: Sequence[Tuple[int, int, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_node", "target_node", "score"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.6f}"])

