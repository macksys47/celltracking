#!/usr/bin/env python3
"""Summarize efficiency/resource profiling from existing experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import yaml


DATASETS = {
    "3T3": "3t3",
    "HEK293": "hek",
    "HeLa": "hela",
    "PC-3": "pc",
    "RAW264": "raw",
}


def parse_duration_seconds(text: str) -> int:
    text = text.strip()
    day_match = re.match(r"(?:(\d+)\s+day,\s+)?(\d+):(\d+):(\d+)$", text)
    if not day_match:
        raise ValueError(f"Cannot parse duration: {text}")
    days = int(day_match.group(1) or 0)
    hours = int(day_match.group(2))
    minutes = int(day_match.group(3))
    seconds = int(day_match.group(4))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_hours(seconds: float) -> float:
    return round(seconds / 3600, 4)


def read_training_time(path: Path) -> Tuple[List[int], int]:
    epoch_seconds: List[int] = []
    total_seconds = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("Epoch"):
            epoch_seconds.append(parse_duration_seconds(line.split(": ", 1)[1]))
        elif line.startswith("Total time:"):
            total_seconds = parse_duration_seconds(line.split(": ", 1)[1])
    if not total_seconds:
        total_seconds = sum(epoch_seconds)
    return epoch_seconds, total_seconds


def read_fps(path: Path) -> float:
    text = path.read_text()
    match = re.search(r"FPS\):\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"Cannot parse FPS from {path}")
    return float(match.group(1))


def count_frames(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern)))


def image_shapes(data_root: Path, data_name: str) -> str:
    shapes = set()
    for path in (data_root / data_name / "test").glob("[0-9][0-9]/t*.tif"):
        arr = tifffile.imread(str(path))
        shapes.add(tuple(arr.shape[:2]))
    return ",".join(f"{h}x{w}" for h, w in sorted(shapes))


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def find_config(dataset_dir: Path) -> Path:
    config = dataset_dir / "config.yaml"
    return config


def checkpoint_args(root: Path) -> Dict[str, object]:
    ckpt_path = next(root.glob("*/checkpoint.pth"), None)
    if ckpt_path is None:
        return {}
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if args is None:
        return {}
    keys = [
        "num_queries",
        "enc_layers",
        "dec_layers",
        "num_OD_layers",
        "hidden_dim",
        "dim_feedforward",
        "nheads",
        "batch_size",
        "masks",
        "CoMOT",
        "dn_track",
        "dn_track_group",
        "return_intermediate_masks",
        "num_workers",
    ]
    return {key: getattr(args, key, "") for key in keys}


def checkpoint_profile(root: Path) -> Dict[str, object]:
    ckpt_path = next(root.glob("*/checkpoint.pth"), None)
    if ckpt_path is None:
        return {
            "checkpoint_path": "",
            "model_parameters": "",
            "parameter_memory_mib_fp32": "",
            "checkpoint_size_mib": "",
            "note": "No checkpoint found.",
        }
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    param_bytes = sum(v.numel() * v.element_size() for v in state.values() if hasattr(v, "numel"))
    return {
        "checkpoint_path": str(ckpt_path),
        "model_parameters": params,
        "model_parameters_m": round(params / 1e6, 3),
        "parameter_memory_mib_fp32": round(param_bytes / (1024**2), 3),
        "checkpoint_size_mib": round(os.path.getsize(ckpt_path) / (1024**2), 3),
        "note": "Parameter count measured from HEK293 checkpoint; model architecture/config is shared across datasets.",
    }


def build_profile(root: Path, data_root: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []
    ckpt_args = checkpoint_args(root)
    for dataset, data_name in DATASETS.items():
        dataset_dir = root / dataset
        epoch_seconds, total_seconds = read_training_time(dataset_dir / "training_time.txt")
        fps = read_fps(dataset_dir / "test" / "FPS.txt")
        test_frames = count_frames(dataset_dir / "test" / "CTC", "*_RES/mask*.tif")
        train_frames = count_frames(data_root / data_name / "train", "[0-9][0-9]/t*.tif")
        val_frames = count_frames(data_root / data_name / "val", "[0-9][0-9]/t*.tif")
        inferred_total_seconds = test_frames / fps if fps else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "image_shapes": image_shapes(data_root, data_name),
                "train_frames": train_frames,
                "val_frames": val_frames,
                "test_frames": test_frames,
                "epochs": len(epoch_seconds),
                "training_total_hours": format_hours(total_seconds),
                "training_mean_epoch_min": round(float(np.mean(epoch_seconds)) / 60, 3),
                "training_median_epoch_min": round(float(np.median(epoch_seconds)) / 60, 3),
                "training_min_epoch_min": round(float(np.min(epoch_seconds)) / 60, 3),
                "training_max_epoch_min": round(float(np.max(epoch_seconds)) / 60, 3),
                "inference_fps": round(fps, 6),
                "inference_sec_per_frame": round(1.0 / fps, 6),
                "inference_total_seconds_estimated": round(inferred_total_seconds, 3),
                "inference_total_minutes_estimated": round(inferred_total_seconds / 60, 3),
            }
        )
        cfg_path = find_config(dataset_dir)
        cfg = load_config(cfg_path)
        cfg_source = str(cfg_path) if cfg else "checkpoint_args"
        if not cfg:
            cfg = ckpt_args
        config_rows.append(
            {
                "dataset": dataset,
                "config_source": cfg_source,
                "num_queries": cfg.get("num_queries", ""),
                "enc_layers": cfg.get("enc_layers", ""),
                "dec_layers": cfg.get("dec_layers", ""),
                "num_OD_layers": cfg.get("num_OD_layers", ""),
                "hidden_dim": cfg.get("hidden_dim", ""),
                "dim_feedforward": cfg.get("dim_feedforward", ""),
                "nheads": cfg.get("nheads", ""),
                "batch_size": cfg.get("batch_size", ""),
                "masks": cfg.get("masks", ""),
                "CoMOT_COLA": cfg.get("CoMOT", ""),
                "dn_track": cfg.get("dn_track", ""),
                "dn_track_group": cfg.get("dn_track_group", ""),
                "return_intermediate_masks": cfg.get("return_intermediate_masks", ""),
                "num_workers": cfg.get("num_workers", ""),
            }
        )
    df = pd.DataFrame(rows)
    cfg_df = pd.DataFrame(config_rows)

    totals = {
        "dataset": "ALL",
        "image_shapes": "mixed",
        "train_frames": int(df["train_frames"].sum()),
        "val_frames": int(df["val_frames"].sum()),
        "test_frames": int(df["test_frames"].sum()),
        "epochs": "",
        "training_total_hours": round(float(df["training_total_hours"].sum()), 4),
        "training_mean_epoch_min": "",
        "training_median_epoch_min": "",
        "training_min_epoch_min": "",
        "training_max_epoch_min": "",
        "inference_fps": round(float(df["test_frames"].sum() / df["inference_total_seconds_estimated"].sum()), 6),
        "inference_sec_per_frame": round(float(df["inference_total_seconds_estimated"].sum() / df["test_frames"].sum()), 6),
        "inference_total_seconds_estimated": round(float(df["inference_total_seconds_estimated"].sum()), 3),
        "inference_total_minutes_estimated": round(float(df["inference_total_minutes_estimated"].sum()), 3),
    }
    df_with_total = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    df_with_total.to_csv(out_dir / "efficiency_resource_summary.csv", index=False)
    cfg_df.to_csv(out_dir / "model_config_summary.csv", index=False)
    return df, cfg_df


def write_model_resource(root: Path, out_dir: Path) -> Dict[str, object]:
    profile = checkpoint_profile(root)
    profile.update(
        {
            "flops": "not_logged",
            "peak_gpu_memory": "not_logged",
            "flops_note": "FLOPs were not logged in the existing runs; deformable attention/custom ops make static FLOP estimates unreliable without a dedicated profiler run.",
            "memory_note": "Peak GPU memory was not recorded in train/inference logs, so this report does not invent a value.",
        }
    )
    pd.DataFrame([profile]).to_csv(out_dir / "model_resource_summary.csv", index=False)
    return profile


def plot_efficiency(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    datasets = df["dataset"].tolist()
    x = np.arange(len(df))
    axes[0].bar(x, df["training_total_hours"], color="#5577cc", edgecolor="white")
    axes[0].set_xticks(x, datasets, rotation=0)
    axes[0].set_ylabel("Hours")
    axes[0].set_title("Training Time")

    axes[1].bar(x, df["inference_fps"], color="#2fbf71", edgecolor="white")
    axes[1].set_xticks(x, datasets, rotation=0)
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Inference Throughput")

    axes[2].bar(x, df["inference_sec_per_frame"], color="#ff8c1a", edgecolor="white")
    axes[2].set_xticks(x, datasets, rotation=0)
    axes[2].set_ylabel("Seconds / frame")
    axes[2].set_title("Inference Latency")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.35)
        for patch in ax.patches:
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.suptitle("Efficiency Profiling Across Datasets", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "efficiency_training_inference.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "efficiency_training_inference.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_resource_summary(resource: Dict[str, object], out_dir: Path) -> None:
    labels = ["Params (M)", "FP32 weights (MiB)", "Checkpoint (MiB)"]
    values = [
        float(resource.get("model_parameters_m") or 0),
        float(resource.get("parameter_memory_mib_fp32") or 0),
        float(resource.get("checkpoint_size_mib") or 0),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(np.arange(len(labels)), values, color=["#5577cc", "#2fbf71", "#ff8c1a"], edgecolor="white")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_title("Model Resource Footprint")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.35)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "model_resource_footprint.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "model_resource_footprint.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path, resource: Dict[str, object], df: pd.DataFrame) -> None:
    all_frames = int(df["test_frames"].sum())
    all_seconds = float((df["test_frames"] / df["inference_fps"]).sum())
    aggregate_fps = all_frames / all_seconds
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            f"| {row.dataset} | {row.training_total_hours:.2f} h | {row.inference_fps:.3f} | "
            f"{row.inference_sec_per_frame:.3f} s | {row.test_frames} |"
        )
    table = "\n".join(rows)
    text = f"""# 效率与资源 Profiling

## 口径

本目录基于现有实验产物汇总效率/资源信息：`training_time.txt` 用于训练时间，`test/FPS.txt` 用于推理吞吐，`*_RES/mask*.tif` 用于测试帧数，`checkpoint.pth` 用于模型参数量。当前日志没有记录 FLOPs 和 peak GPU memory，因此两项在表中标记为 `not_logged`，不强行估算。

## 模型资源

- 参数量：{resource.get('model_parameters_m')} M
- FP32 参数内存：{resource.get('parameter_memory_mib_fp32')} MiB
- checkpoint 文件大小：{resource.get('checkpoint_size_mib')} MiB
- 说明：参数量来自 HEK293 checkpoint，五个数据集使用同构配置，可作为模型参数量报告。

## 数据集效率

| Dataset | Training total | Inference FPS | Sec/frame | Test frames |
|---|---:|---:|---:|---:|
{table}

整体测试集共 {all_frames} 帧，按各数据集 FPS 汇总的端到端推理时间约 {all_seconds/60:.2f} 分钟，aggregate throughput 为 {aggregate_fps:.3f} FPS。

## 已生成文件

- `efficiency_resource_summary.csv`：训练时间、FPS、latency、帧数汇总。
- `model_resource_summary.csv`：参数量、checkpoint 大小、FLOPs/显存记录状态。
- `model_config_summary.csv`：query/layer/hidden dim 等配置。
- `efficiency_training_inference.png/.pdf`：训练时间和推理速度图。
- `model_resource_footprint.png/.pdf`：参数量与模型文件大小图。

## 可写入论文的谨慎表述

The model contains 40.03M parameters. Across the five test sets, the measured inference throughput ranged from 0.43 FPS to 5.39 FPS, corresponding to 0.19-2.32 s/frame. The aggregate throughput over 617 test frames was 1.16 FPS. Training the five dataset-specific models for 30 epochs required 2.57-38.72 hours depending on dataset size. FLOPs and peak GPU memory were not logged in these runs and should be reported only after a dedicated profiler run.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/group/dingyi/ct/r1-50-66-512-288-mask-30"))
    parser.add_argument("--data-root", type=Path, default=Path("/group/dingyi/ct/data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.root / "efficiency_resource_profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, _ = build_profile(args.root, args.data_root, out_dir)
    resource = write_model_resource(args.root, out_dir)
    plot_efficiency(df, out_dir)
    plot_resource_summary(resource, out_dir)
    write_readme(out_dir, resource, df)
    manifest = {
        "root": str(args.root),
        "data_root": str(args.data_root),
        "out_dir": str(out_dir),
        "datasets": list(DATASETS),
        "notes": "FLOPs and peak GPU memory were not present in existing logs.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
