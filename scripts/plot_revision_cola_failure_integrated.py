#!/usr/bin/env python3
"""Build the integrated revision Figure 8 without the PC-3 data or crop panel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

Path("/tmp/ct_mpl_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ct_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PAPER_DIR = ROOT / "jbhi2026_v1.5" / "jbhi2026_v1.5"
FIG_DIR = PAPER_DIR / "figures"
MAIN_ROOT = ROOT / "r1-50-66-512-288-mask-30"
COLA_DIR = MAIN_ROOT / "cola_duplicate_intermediate_analysis"

DATASETS = ["3T3", "HEK293", "HeLa", "RAW264"]
DATASET_LABELS = {
    "3T3": "3T3",
    "HEK293": "HEK293",
    "HeLa": "HeLa-S3",
    "RAW264": "RAW264",
}

COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "gray": "#7f7f7f",
    "cyan": "#3ba3cf",
}

FAILURE_COLORS = {
    "fn": "#d62728",
    "fp": "#1f77b4",
    "split_merge": "#ff7f0e",
    "fragment": "#9a9a9a",
    "id": "#9467bd",
    "division": "#2ca02c",
}

DATASET_COLORS = {
    "3T3": "#1b9e77",
    "HEK293": "#d95f02",
    "HeLa": "#7570b3",
    "RAW264": "#e7298a",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "font.weight": "bold",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.35,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dataset_labels() -> list[str]:
    return [DATASET_LABELS[name] for name in DATASETS]


def style_tick_labels(ax: plt.Axes) -> None:
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")


def style_boxed_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.35)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", width=1.25, length=4.5, color="black")
    style_tick_labels(ax)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#cfcfcf", linestyle="--", linewidth=0.65, alpha=0.85)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def style_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_linewidth(0)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def heatmap_text_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "white"
    norm = (value - vmin) / (vmax - vmin)
    return "black" if norm > 0.68 else "white"


def plot_trajectory_ambiguity(ax: plt.Axes) -> None:
    duplicate = pd.read_csv(COLA_DIR / "duplicate_trajectory_summary.csv")
    duplicate = duplicate.set_index("dataset").reindex(DATASETS)
    x = np.arange(len(DATASETS))
    bars = [
        ("gt_duplicate_rate", "GT to multiple pred", COLORS["orange"]),
        ("pred_multi_gt_rate", "Pred to multiple GT", COLORS["cyan"]),
        ("pred_unmatched_rate", "Pred unmatched", "#9a9a9a"),
    ]
    width = 0.23
    max_value = 0.0
    for idx, (col, label, color) in enumerate(bars):
        vals = duplicate[col].to_numpy(dtype=float)
        max_value = max(max_value, float(np.nanmax(vals)))
        pos = x + (idx - 1) * width
        ax.bar(pos, vals, width=width, label=label, color=color, edgecolor="black", linewidth=0.75)
        for xi, value in zip(pos, vals):
            if value >= 0.045:
                ax.text(xi, value + 0.006, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=8.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels())
    ax.set_ylabel("Track proportion")
    ax.set_title("COLA trajectory ambiguity", pad=8)
    ax.set_ylim(0, max(0.18, max_value * 1.34))
    style_boxed_axis(ax, "y")
    style_legend(ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(0.99, 0.99)))


def plot_layer_heatmap(ax: plt.Axes, data: pd.DataFrame, component: str, title: str):
    sub = data[(data["split"] == "val") & (data["component"] == component)].copy()
    matrix = (
        sub.pivot(index="dataset", columns="layer", values="final_epoch_mean")
        .reindex(DATASETS)
        .reindex(columns=[0, 1, 2, 3])
    )
    arr = matrix.to_numpy(dtype=float)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"L{i}" for i in range(4)])
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels(dataset_labels())
    ax.set_title(title, pad=8)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            value = arr[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=heatmap_text_color(value, vmin, vmax),
                fontsize=9.2,
                fontweight="bold",
            )
    style_boxed_axis(ax, None)
    ax.tick_params(length=4.5)
    return im


def normalized_frame_failure_matrix(n_bins: int = 100) -> np.ndarray:
    rows = []
    for dataset in DATASETS:
        df = pd.read_csv(MAIN_ROOT / dataset / "failure_cases" / "frame_stats.csv")
        seq_rows = []
        for _, sub in df.groupby("sequence"):
            sub = sub.sort_values("frame")
            frames = sub["frame"].to_numpy(dtype=float)
            scores = sub["failure_score"].to_numpy(dtype=float)
            denom = max(float(frames.max() - frames.min()), 1.0)
            bins = np.round((frames - frames.min()) / denom * (n_bins - 1)).astype(int)
            values = np.full(n_bins, np.nan)
            for idx in range(n_bins):
                selected = scores[bins == idx]
                if selected.size:
                    values[idx] = float(np.mean(selected))
            seq_rows.append(values)
        stack = np.vstack(seq_rows)
        counts = np.sum(~np.isnan(stack), axis=0)
        sums = np.nansum(stack, axis=0)
        row = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
        rows.append(row)
    return np.vstack(rows)


def plot_success_failure_heatmap(ax: plt.Axes) -> None:
    ratio = pd.read_csv(MAIN_ROOT / "failure_success_ratio.csv")
    ratio = ratio.set_index("dataset").reindex(DATASETS)
    row_defs = [
        ("Instance", "instance_success_rate"),
        ("Track", "track_success_rate"),
        ("Division", "division_success_rate"),
    ]
    arr = np.vstack([ratio[col].to_numpy(dtype=float) for _, col in row_defs])
    ax.imshow(arr, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(dataset_labels())
    ax.set_yticks(np.arange(len(row_defs)))
    ax.set_yticklabels([name for name, _ in row_defs])
    ax.set_title("Success/failure (%)", pad=8)
    ax.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_defs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            success = int(round(arr[i, j] * 100))
            failure = 100 - success
            text_color = "white" if arr[i, j] >= 0.62 else "#222222"
            ax.text(j, i, f"{success}/{failure}", ha="center", va="center", color=text_color, fontsize=10.5)
    style_boxed_axis(ax, None)


def plot_failure_composition(ax: plt.Axes) -> None:
    df = pd.read_csv(MAIN_ROOT / "failure_case_dataset_summary.csv")
    df = df.set_index("dataset").reindex(DATASETS)
    components = [
        ("false_negatives", "FN", FAILURE_COLORS["fn"]),
        ("false_positives", "FP", FAILURE_COLORS["fp"]),
        ("split_merge", "Split/Merge", FAILURE_COLORS["split_merge"]),
        ("fragmented_gt_tracks", "Fragment", FAILURE_COLORS["fragment"]),
        ("id_switch", "ID/ReID", FAILURE_COLORS["id"]),
        ("division_errors", "Division", FAILURE_COLORS["division"]),
    ]
    values = pd.DataFrame(index=df.index)
    values["split_merge"] = df["split_candidates"] + df["merge_candidates"]
    values["id_switch"] = df["id_switches"] + df["reid_after_gap"]
    values["division_errors"] = df["gt_child_tracks"] - df["correct_division_child_links"]
    values["false_negatives"] = df["false_negatives"]
    values["false_positives"] = df["false_positives"]
    values["fragmented_gt_tracks"] = df["fragmented_gt_tracks"]
    normalized = values[[col for col, _, _ in components]].clip(lower=0)
    normalized = normalized.div(normalized.sum(axis=1), axis=0).fillna(0)
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS))
    for col, label, color in components:
        vals = normalized[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.6, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels(), rotation=20, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Failure composition", pad=8)
    ax.set_ylim(0, 1)
    style_boxed_axis(ax, "y")
    style_legend(ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18)))


def plot_failure_frame_heatmap(ax: plt.Axes):
    mat = normalized_frame_failure_matrix(100)
    vmax = max(1.0, float(np.nanpercentile(mat, 98)))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels(dataset_labels())
    ax.set_xticks([0, 49, 99])
    ax.set_xticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("Normalized frame position")
    ax.set_title("Frame-level failure score", pad=8)
    style_boxed_axis(ax, None)
    return im


def text_color_for_rgb(rgb: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.60 else "white"


def draw_header(ax: plt.Axes, x0: float, x1: float, title: str) -> None:
    ax.text((x0 + x1) / 2, 4.56, title, ha="center", va="bottom", fontsize=10.8, fontweight="bold")


def draw_cell(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    color: tuple[float, float, float, float] | str,
    text: str,
    fontsize: float = 8.8,
) -> None:
    ax.add_patch(
        plt.Rectangle(
            (x0, y - 0.33),
            x1 - x0,
            0.66,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
        )
    )
    rgba = color if not isinstance(color, str) else (1.0, 1.0, 1.0, 1.0)
    ax.text(
        (x0 + x1) / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=text_color_for_rgb(rgba),
    )


def draw_heat_cells(
    ax: plt.Axes,
    data: pd.DataFrame,
    component: str,
    x0: float,
    x1: float,
    row_y: dict[str, float],
) -> None:
    cmap = plt.get_cmap("viridis")
    sub = data[(data["split"] == "val") & (data["component"] == component)].copy()
    matrix = (
        sub.pivot(index="dataset", columns="layer", values="final_epoch_mean")
        .reindex(DATASETS)
        .reindex(columns=[0, 1, 2, 3])
    )
    arr = matrix.to_numpy(dtype=float)
    norm = plt.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
    gap = 0.12
    width = (x1 - x0 - gap * 3) / 4
    for layer_idx in range(4):
        left = x0 + layer_idx * (width + gap)
        ax.text(left + width / 2, 4.24, f"L{layer_idx}", ha="center", va="center", fontsize=8.7, fontweight="bold")
        for dataset in DATASETS:
            value = float(matrix.loc[dataset, layer_idx])
            color = cmap(norm(value))
            draw_cell(ax, left, left + width, row_y[dataset], color, f"{value:.2f}", fontsize=8.3)


def draw_success_cells(ax: plt.Axes, x0: float, x1: float, row_y: dict[str, float]) -> None:
    ratio = pd.read_csv(MAIN_ROOT / "failure_success_ratio.csv")
    ratio = ratio.set_index("dataset").reindex(DATASETS)
    cmap = plt.get_cmap("Greens")
    norm = plt.Normalize(vmin=0.55, vmax=1.0)
    cols = [
        ("I", "instance_success_rate"),
        ("T", "track_success_rate"),
        ("D", "division_success_rate"),
    ]
    gap = 0.15
    width = (x1 - x0 - gap * 2) / 3
    for idx, (short, col) in enumerate(cols):
        left = x0 + idx * (width + gap)
        ax.text(left + width / 2, 4.24, short, ha="center", va="center", fontsize=8.7, fontweight="bold")
        for dataset in DATASETS:
            success = float(ratio.loc[dataset, col])
            color = cmap(norm(success))
            text = f"{int(round(success * 100))}/{int(round((1 - success) * 100))}"
            draw_cell(ax, left, left + width, row_y[dataset], color, text, fontsize=8.3)


def draw_ambiguity_bars(ax: plt.Axes, x0: float, x1: float, row_y: dict[str, float]) -> None:
    duplicate = pd.read_csv(COLA_DIR / "duplicate_trajectory_summary.csv")
    duplicate = duplicate.set_index("dataset").reindex(DATASETS)
    metrics = [
        ("gt_duplicate_rate", "GT", COLORS["orange"]),
        ("pred_multi_gt_rate", "Pred", COLORS["cyan"]),
        ("pred_unmatched_rate", "Unm.", "#9a9a9a"),
    ]
    max_value = float(duplicate[[name for name, _, _ in metrics]].max().max()) * 100
    scale_max = max(10.0, np.ceil(max_value / 2) * 2)
    label_x = x0 + 0.2
    bar_x0 = x0 + 3.2
    bar_x1 = x1 - 1.8
    for idx, (_, label, color) in enumerate(metrics):
        ax.add_patch(plt.Rectangle((x0 + idx * 4.2, 4.17), 0.55, 0.12, facecolor=color, edgecolor="black", linewidth=0.5))
        ax.text(x0 + idx * 4.2 + 0.72, 4.23, label, ha="left", va="center", fontsize=8.1, fontweight="bold")
    for dataset in DATASETS:
        for offset, (col, label, color) in zip([0.22, 0.0, -0.22], metrics):
            value = float(duplicate.loc[dataset, col]) * 100
            y = row_y[dataset] + offset
            ax.text(label_x, y, label, ha="left", va="center", fontsize=7.9, fontweight="bold")
            ax.add_patch(
                plt.Rectangle(
                    (bar_x0, y - 0.055),
                    bar_x1 - bar_x0,
                    0.11,
                    facecolor="#f4f4f4",
                    edgecolor="#c7c7c7",
                    linewidth=0.45,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (bar_x0, y - 0.055),
                    (bar_x1 - bar_x0) * value / scale_max,
                    0.11,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.25,
                )
            )
            ax.text(bar_x1 + 0.25, y, f"{value:.1f}", ha="left", va="center", fontsize=7.9, fontweight="bold")
    ax.text(bar_x0, 0.30, "0", ha="center", va="center", fontsize=7.8, fontweight="bold")
    ax.text(bar_x1, 0.30, f"{scale_max:.0f}%", ha="center", va="center", fontsize=7.8, fontweight="bold")


def draw_failure_composition(ax: plt.Axes, x0: float, x1: float, row_y: dict[str, float]) -> None:
    df = pd.read_csv(MAIN_ROOT / "failure_case_dataset_summary.csv")
    df = df.set_index("dataset").reindex(DATASETS)
    components = [
        ("false_negatives", "FN", FAILURE_COLORS["fn"]),
        ("false_positives", "FP", FAILURE_COLORS["fp"]),
        ("split_merge", "S/M", FAILURE_COLORS["split_merge"]),
        ("fragmented_gt_tracks", "Frag", FAILURE_COLORS["fragment"]),
        ("id_switch", "ID", FAILURE_COLORS["id"]),
        ("division_errors", "Div", FAILURE_COLORS["division"]),
    ]
    values = pd.DataFrame(index=df.index)
    values["split_merge"] = df["split_candidates"] + df["merge_candidates"]
    values["id_switch"] = df["id_switches"] + df["reid_after_gap"]
    values["division_errors"] = df["gt_child_tracks"] - df["correct_division_child_links"]
    values["false_negatives"] = df["false_negatives"]
    values["false_positives"] = df["false_positives"]
    values["fragmented_gt_tracks"] = df["fragmented_gt_tracks"]
    normalized = values[[col for col, _, _ in components]].clip(lower=0)
    normalized = normalized.div(normalized.sum(axis=1), axis=0).fillna(0)
    for dataset in DATASETS:
        left = x0
        for col, _, color in components:
            width = (x1 - x0) * float(normalized.loc[dataset, col])
            ax.add_patch(
                plt.Rectangle(
                    (left, row_y[dataset] - 0.20),
                    width,
                    0.40,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.45,
                )
            )
            left += width
        ax.add_patch(
            plt.Rectangle(
                (x0, row_y[dataset] - 0.20),
                x1 - x0,
                0.40,
                facecolor="none",
                edgecolor="black",
                linewidth=0.65,
            )
        )
    legend_step = (x1 - x0) / len(components)
    for idx, (_, label, color) in enumerate(components):
        lx = x0 + idx * legend_step + 0.05
        ax.add_patch(plt.Rectangle((lx, 4.18), 0.42, 0.12, facecolor=color, edgecolor="none"))
        ax.text(lx + 0.52, 4.24, label, ha="left", va="center", fontsize=7.0, fontweight="bold")


def draw_temporal_strips(ax: plt.Axes, x0: float, x1: float, row_y: dict[str, float]) -> None:
    mat = normalized_frame_failure_matrix(100)
    cmap = plt.get_cmap("magma")
    norm = plt.Normalize(vmin=0, vmax=max(1.0, float(np.nanpercentile(mat, 98))))
    dx = (x1 - x0) / mat.shape[1]
    for row_idx, dataset in enumerate(DATASETS):
        y = row_y[dataset]
        for bin_idx, value in enumerate(mat[row_idx]):
            color = cmap(norm(float(value)))
            ax.add_patch(
                plt.Rectangle(
                    (x0 + bin_idx * dx, y - 0.27),
                    dx + 0.002,
                    0.54,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0,
                )
            )
        ax.add_patch(
            plt.Rectangle(
                (x0, y - 0.27),
                x1 - x0,
                0.54,
                facecolor="none",
                edgecolor="black",
                linewidth=0.65,
            )
        )
    ax.text(x0, 0.30, "0%", ha="left", va="center", fontsize=7.8, fontweight="bold")
    ax.text((x0 + x1) / 2, 0.30, "50%", ha="center", va="center", fontsize=7.8, fontweight="bold")
    ax.text(x1, 0.30, "100%", ha="right", va="center", fontsize=7.8, fontweight="bold")


def draw_integrated_matrix() -> plt.Figure:
    layer = pd.read_csv(COLA_DIR / "cola_intermediate_layer_loss_summary.csv")
    row_y = {dataset: 3.73 - idx * 0.88 for idx, dataset in enumerate(DATASETS)}
    group_spans = [
        (0.8, 6.4, "Dataset"),
        (7.0, 24.4, "Trajectory ambiguity (%)"),
        (25.7, 38.7, "BBox loss"),
        (40.1, 53.1, "GIoU loss"),
        (54.5, 68.8, "Success/failure (%)"),
        (70.2, 84.7, "Failure mix"),
        (86.1, 99.1, "Temporal score"),
    ]

    fig, ax = plt.subplots(figsize=(17.0, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 4.85)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 4.08), 100, 0.62, facecolor="#f1f3f5", edgecolor="none"))
    for idx, dataset in enumerate(DATASETS):
        y = row_y[dataset]
        if idx % 2 == 0:
            ax.add_patch(plt.Rectangle((0.2, y - 0.39), 99.1, 0.78, facecolor="#fafafa", edgecolor="none"))
        ax.plot([0.4, 99.2], [y - 0.39, y - 0.39], color="#dddddd", linewidth=0.65)
        ax.text(3.7, y, DATASET_LABELS[dataset], ha="center", va="center", fontsize=10.5, fontweight="bold")

    for x0, x1, title in group_spans:
        draw_header(ax, x0, x1, title)
        ax.plot([x0 - 0.35, x0 - 0.35], [0.42, 4.68], color="#b8b8b8", linewidth=0.9)
    ax.plot([99.2, 99.2], [0.42, 4.68], color="#b8b8b8", linewidth=0.9)
    ax.plot([0.4, 99.2], [4.08, 4.08], color="#b8b8b8", linewidth=0.9)
    ax.plot([0.4, 99.2], [0.42, 0.42], color="#b8b8b8", linewidth=0.9)

    draw_ambiguity_bars(ax, 7.0, 24.4, row_y)
    draw_heat_cells(ax, layer, "bbox", 25.7, 38.7, row_y)
    draw_heat_cells(ax, layer, "giou", 40.1, 53.1, row_y)
    draw_success_cells(ax, 54.5, 68.8, row_y)
    draw_failure_composition(ax, 70.2, 84.7, row_y)
    draw_temporal_strips(ax, 86.1, 99.1, row_y)

    return fig


def minmax_to_100(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        return np.full_like(values, 50.0)
    return (values - vmin) / (vmax - vmin) * 100.0


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def failure_mix_values() -> pd.DataFrame:
    df = pd.read_csv(MAIN_ROOT / "failure_case_dataset_summary.csv")
    df = df.set_index("dataset").reindex(DATASETS)
    values = pd.DataFrame(index=df.index)
    values["FN"] = df["false_negatives"]
    values["FP"] = df["false_positives"]
    values["S/M"] = df["split_candidates"] + df["merge_candidates"]
    values["Frag"] = df["fragmented_gt_tracks"]
    values["ID"] = df["id_switches"] + df["reid_after_gap"]
    values["Div"] = df["gt_child_tracks"] - df["correct_division_child_links"]
    values = values.clip(lower=0)
    return values.div(values.sum(axis=1), axis=0).fillna(0)


def add_donut(
    ax: plt.Axes,
    center: tuple[float, float],
    fractions: Sequence[float],
    colors: Sequence[str],
    size: int = 40,
) -> None:
    drawing = DrawingArea(size, size, clip=False)
    start = 90.0
    radius = size / 2.0
    for fraction, color in zip(fractions, colors):
        if fraction <= 0:
            continue
        end = start + float(fraction) * 360.0
        drawing.add_artist(
            mpatches.Wedge(
                (radius, radius),
                radius,
                start,
                end,
                facecolor=color,
                edgecolor="white",
                linewidth=0.65,
            )
        )
        start = end
    drawing.add_artist(mpatches.Circle((radius, radius), radius * 0.42, facecolor="white", edgecolor="white"))
    ax.add_artist(AnnotationBbox(drawing, center, frameon=False, box_alignment=(0.5, 0.5)))


def draw_integrated_chart() -> plt.Figure:
    duplicate = pd.read_csv(COLA_DIR / "duplicate_trajectory_summary.csv")
    duplicate = duplicate.set_index("dataset").reindex(DATASETS)
    layer = pd.read_csv(COLA_DIR / "cola_intermediate_layer_loss_summary.csv")

    giou = (
        layer[(layer["split"] == "val") & (layer["component"] == "giou")]
        .pivot(index="dataset", columns="layer", values="final_epoch_mean")
        .reindex(DATASETS)
        .reindex(columns=[0, 1, 2, 3])
    )
    giou_scaled = pd.DataFrame(minmax_to_100(giou.to_numpy()), index=giou.index, columns=giou.columns)

    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.21)
    ax.set_xlim(-0.45, 14.2)
    ax.set_ylim(-6, 112)

    blocks = [
        (-0.30, 1.40, "Trajectory ambiguity"),
        (2.35, 6.55, "Layerwise GIoU loss"),
        (7.55, 14.00, "Failure mix"),
    ]
    for idx, (x0, x1, title) in enumerate(blocks):
        ax.axvspan(x0, x1, color="#f2f5f7" if idx % 2 == 0 else "#fbfbfb", zorder=0)
        ax.text((x0 + x1) / 2, 108.5, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.axvline(x1, color="#c7c7c7", linewidth=0.9, zorder=1)

    ax.grid(axis="y", color="#d7d7d7", linestyle="--", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ambiguity_x = np.array([0.0, 1.0])
    giou_x = np.array([2.9, 3.9, 4.9, 5.9])

    for dataset in DATASETS:
        color = DATASET_COLORS[dataset]
        amb = duplicate.loc[dataset, ["gt_duplicate_rate", "pred_multi_gt_rate"]].to_numpy(dtype=float)
        amb_scaled = np.clip(amb * 100.0 / 10.0 * 100.0, 0, 100)

        ax.plot(ambiguity_x, amb_scaled, color=color, marker="o", linewidth=2.3, markersize=5.4)
        ax.plot(giou_x, giou_scaled.loc[dataset].to_numpy(dtype=float), color=color, marker="o", linewidth=2.3, markersize=5.4)

    mix = failure_mix_values()
    mix_cols = ["FN", "FP", "S/M", "Frag", "ID", "Div"]
    mix_colors = [
        FAILURE_COLORS["fn"],
        FAILURE_COLORS["fp"],
        FAILURE_COLORS["split_merge"],
        FAILURE_COLORS["fragment"],
        FAILURE_COLORS["id"],
        FAILURE_COLORS["division"],
    ]
    donut_centers = [8.25, 9.85, 11.45, 13.05]
    for dataset, x in zip(DATASETS, donut_centers):
        add_donut(ax, (x, 58), mix.loc[dataset, mix_cols].to_numpy(dtype=float), mix_colors, size=52)
        ax.text(
            x,
            20,
            DATASET_LABELS[dataset],
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=DATASET_COLORS[dataset],
        )
        ax.text(
            x,
            91,
            f"FN {mix.loc[dataset, 'FN'] * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color=FAILURE_COLORS["fn"],
        )

    ticks = [
        (0, "GT\nmulti"),
        (1, "Pred\nmulti"),
        (2.9, "L0"),
        (3.9, "L1"),
        (4.9, "L2"),
        (5.9, "L3"),
    ]
    ax.set_xticks([pos for pos, _ in ticks])
    ax.set_xticklabels([label for _, label in ticks])
    ax.set_ylabel("Scaled ambiguity/loss value (0-100)")
    style_tick_labels(ax)
    ax.tick_params(axis="both", which="major", width=1.15, length=4.2)

    dataset_handles = [
        Line2D([0], [0], color=DATASET_COLORS[dataset], marker="o", linewidth=2.4, markersize=5.2, label=DATASET_LABELS[dataset])
        for dataset in DATASETS
    ]
    mix_handles = [mpatches.Patch(facecolor=color, label=label) for label, color in zip(mix_cols, mix_colors)]
    legend1 = ax.legend(handles=dataset_handles, loc="upper left", bbox_to_anchor=(0.005, -0.22), ncol=4, frameon=False)
    style_legend(legend1)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=mix_handles, loc="upper right", bbox_to_anchor=(0.995, -0.22), ncol=6, frameon=False, columnspacing=0.9)
    style_legend(legend2)
    return fig


def main() -> None:
    apply_style()
    fig = draw_integrated_chart()
    save_figure(fig, "revision_cola_failure_integrated")
    print(f"Wrote {FIG_DIR / 'revision_cola_failure_integrated.pdf'}")


if __name__ == "__main__":
    main()
