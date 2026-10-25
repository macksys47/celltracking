from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class ObjectRecord:
    node_id: int
    frame: int
    label: int
    parent_id: int
    center: Tuple[float, float]
    box: Tuple[float, float, float, float]
    area: float
    mean_intensity: float


@dataclass
class WindowSample:
    frames: torch.Tensor
    centers: torch.Tensor
    boxes: torch.Tensor
    shallow: torch.Tensor
    frame_indices: torch.Tensor
    track_ids: torch.Tensor
    parent_ids: torch.Tensor
    node_ids: torch.Tensor
    seq_name: str
    start_frame: int


def frame_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"Cannot parse frame number from {path}")
    return int(matches[-1])


def read_raw_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.float32)


def read_label_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int64)


def image_to_tensor(path: Path, target_size: Tuple[int, int]) -> torch.Tensor:
    image = read_raw_image(path)
    max_value = float(image.max())
    if max_value > 0:
        image = image / max_value
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(image).resize((target_size[1], target_size[0]), Image.BILINEAR).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_track_table(path: Path) -> Dict[int, int]:
    parents: Dict[int, int] = {}
    if not path.exists():
        return parents
    raw = np.loadtxt(path, dtype=np.int64)
    if raw.size == 0:
        return parents
    raw = raw.reshape(1, -1) if raw.ndim == 1 else raw
    for row in raw:
        if row.shape[0] >= 4:
            parents[int(row[0])] = int(row[3])
    return parents


def discover_sequences(data_dir: Path, dataset: str, split: str) -> List[Path]:
    root = data_dir / dataset / split
    if not root.exists():
        raise FileNotFoundError(f"Missing split directory: {root}")
    return sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)], key=lambda p: p.name)


class CTCSequence:
    def __init__(self, seq_dir: Path):
        self.seq_dir = seq_dir
        self.seq_name = seq_dir.name
        gt_dir = seq_dir.parent / f"{seq_dir.name}_GT"
        self.tra_dir = gt_dir / "TRA"
        self.seg_dir = gt_dir / "SEG"
        self.parent_by_label = load_track_table(self.tra_dir / "man_track.txt")
        self.image_paths = sorted(seq_dir.glob("*.tif"), key=frame_number)
        if not self.image_paths:
            raise FileNotFoundError(f"No tif images in {seq_dir}")
        self.frame_to_image = {frame_number(path): path for path in self.image_paths}
        self.frames = sorted(self.frame_to_image)
        self.frame_to_mask = self._collect_masks()
        self.records_by_frame: Dict[int, List[ObjectRecord]] = {}
        self.records_by_node: Dict[int, ObjectRecord] = {}
        self._load_records()

    def _collect_masks(self) -> Dict[int, Path]:
        masks: Dict[int, Path] = {}
        for root in (self.tra_dir, self.seg_dir):
            if root.exists():
                for path in sorted(root.glob("*.tif"), key=frame_number):
                    masks.setdefault(frame_number(path), path)
        if not masks:
            raise FileNotFoundError(f"No GT label masks for {self.seq_dir}")
        return masks

    def _load_records(self) -> None:
        node_id = 0
        for frame in self.frames:
            mask_path = self.frame_to_mask.get(frame)
            if mask_path is None:
                self.records_by_frame[frame] = []
                continue
            mask = read_label_image(mask_path)
            raw = read_raw_image(self.frame_to_image[frame])
            max_value = max(float(raw.max()), 1.0)
            height, width = mask.shape[:2]
            records: List[ObjectRecord] = []
            for label_value in np.unique(mask):
                label = int(label_value)
                if label == 0:
                    continue
                ys, xs = np.where(mask == label)
                if xs.size == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                cx = (x0 + x1 + 1) * 0.5 / width
                cy = (y0 + y1 + 1) * 0.5 / height
                bw = (x1 - x0 + 1) / width
                bh = (y1 - y0 + 1) / height
                record = ObjectRecord(
                    node_id=node_id,
                    frame=frame,
                    label=label,
                    parent_id=self.parent_by_label.get(label, 0),
                    center=(float(cx), float(cy)),
                    box=(float(cx), float(cy), float(bw), float(bh)),
                    area=float(xs.size) / float(width * height),
                    mean_intensity=float(raw[ys, xs].mean() / max_value),
                )
                records.append(record)
                self.records_by_node[node_id] = record
                node_id += 1
            self.records_by_frame[frame] = records

    def valid_window_starts(self, window_size: int) -> List[int]:
        return list(range(max(len(self.frames) - window_size + 1, 0)))

    def build_window(self, start_index: int, window_size: int, target_size: Tuple[int, int], max_tokens: int, deterministic: bool = False) -> WindowSample:
        window_frames = self.frames[start_index:start_index + window_size]
        frames = torch.stack([image_to_tensor(self.frame_to_image[frame], target_size) for frame in window_frames], dim=0)
        records: List[ObjectRecord] = []
        frame_indices: List[int] = []
        for local_idx, frame in enumerate(window_frames):
            frame_records = self.records_by_frame.get(frame, [])
            records.extend(frame_records)
            frame_indices.extend([local_idx] * len(frame_records))
        if len(records) > max_tokens:
            keep = np.arange(max_tokens) if deterministic else np.sort(np.random.permutation(len(records))[:max_tokens])
            records = [records[i] for i in keep]
            frame_indices = [frame_indices[i] for i in keep]
        return make_window_sample(records, frame_indices, frames, self.seq_name, window_frames[0])

    def relabel_mask(self, frame: int, label_to_track: Dict[int, int]) -> np.ndarray:
        source = read_label_image(self.frame_to_mask[frame])
        out = np.zeros_like(source, dtype=np.uint16)
        for label, track_id in label_to_track.items():
            out[source == int(label)] = int(track_id)
        return out


def make_window_sample(records: Sequence[ObjectRecord], frame_indices: Sequence[int], frames: torch.Tensor, seq_name: str, start_frame: int) -> WindowSample:
    if records:
        centers = torch.tensor([r.center for r in records], dtype=torch.float32)
        boxes = torch.tensor([r.box for r in records], dtype=torch.float32)
        frame_tensor = torch.tensor(frame_indices, dtype=torch.long)
        denom = max(int(frame_tensor.max().item()), 1)
        time_norm = frame_tensor.float().unsqueeze(1) / float(denom)
        shallow = torch.cat(
            [
                centers,
                boxes[:, 2:],
                torch.tensor([[r.area, r.mean_intensity] for r in records], dtype=torch.float32),
                time_norm,
            ],
            dim=1,
        )
        track_ids = torch.tensor([r.label for r in records], dtype=torch.long)
        parent_ids = torch.tensor([r.parent_id for r in records], dtype=torch.long)
        node_ids = torch.tensor([r.node_id for r in records], dtype=torch.long)
    else:
        centers = torch.zeros((0, 2), dtype=torch.float32)
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        shallow = torch.zeros((0, 7), dtype=torch.float32)
        frame_tensor = torch.zeros((0,), dtype=torch.long)
        track_ids = torch.zeros((0,), dtype=torch.long)
        parent_ids = torch.zeros((0,), dtype=torch.long)
        node_ids = torch.zeros((0,), dtype=torch.long)
    return WindowSample(frames, centers, boxes, shallow, frame_tensor, track_ids, parent_ids, node_ids, seq_name, start_frame)


class TrackastraWindowDataset(Dataset):
    def __init__(self, data_dir: Path, dataset: str, split: str, window_size: int, target_size: Tuple[int, int], max_tokens: int):
        self.sequences = [CTCSequence(path) for path in discover_sequences(data_dir, dataset, split)]
        self.window_size = window_size
        self.target_size = target_size
        self.max_tokens = max_tokens
        self.samples: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            for start in seq.valid_window_starts(window_size):
                self.samples.append((seq_idx, start))
        if not self.samples:
            raise RuntimeError(f"No windows for {dataset}/{split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> WindowSample:
        seq_idx, start = self.samples[index]
        return self.sequences[seq_idx].build_window(start, self.window_size, self.target_size, self.max_tokens)


def move_sample_to_device(sample: WindowSample, device: torch.device) -> WindowSample:
    for name in ("frames", "centers", "boxes", "shallow", "frame_indices", "track_ids", "parent_ids", "node_ids"):
        setattr(sample, name, getattr(sample, name).to(device))
    return sample


def collate_one(batch: Sequence[WindowSample]) -> WindowSample:
    if len(batch) != 1:
        raise ValueError("Trackastra project expects batch_size=1.")
    return batch[0]

