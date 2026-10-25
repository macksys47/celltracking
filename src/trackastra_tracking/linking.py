from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .data import CTCSequence


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    score: float


def greedy_link(edges: Iterable[Edge], threshold: float) -> List[Edge]:
    child_parent: Dict[int, int] = {}
    parent_children: Dict[int, List[int]] = defaultdict(list)
    selected: List[Edge] = []
    for edge in sorted(edges, key=lambda e: e.score, reverse=True):
        if edge.score < threshold:
            break
        if edge.target in child_parent:
            continue
        if len(parent_children[edge.source]) >= 2:
            continue
        child_parent[edge.target] = edge.source
        parent_children[edge.source].append(edge.target)
        selected.append(edge)
    return selected


def build_tracks(sequence: CTCSequence, edges: Sequence[Edge]):
    parent_of = {edge.target: edge.source for edge in edges}
    children_of: Dict[int, List[int]] = defaultdict(list)
    for edge in edges:
        children_of[edge.source].append(edge.target)
    node_to_track: Dict[int, int] = {}
    tracks: Dict[int, Tuple[int, int, int, int]] = {}
    next_track = 1
    for frame in sequence.frames:
        for record in sequence.records_by_frame.get(frame, []):
            parent = parent_of.get(record.node_id)
            if parent is not None and parent in node_to_track:
                parent_track = node_to_track[parent]
                if len(children_of[parent]) >= 2:
                    track_id = next_track
                    next_track += 1
                    tracks[track_id] = (track_id, frame, frame, parent_track)
                else:
                    track_id = parent_track
                    old = tracks[track_id]
                    tracks[track_id] = (old[0], old[1], frame, old[3])
            else:
                track_id = next_track
                next_track += 1
                tracks[track_id] = (track_id, frame, frame, 0)
            node_to_track[record.node_id] = track_id
    return node_to_track, tracks


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


def write_edges(edges: Sequence[Edge], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_node", "target_node", "score"])
        for edge in sorted(edges, key=lambda e: (e.source, e.target)):
            writer.writerow([edge.source, edge.target, f"{edge.score:.6f}"])

