#!/usr/bin/env python3
"""Statistical analysis from existing inference outputs only.

This script does not train or run model inference. It evaluates the saved
CTC-style predictions from the main experiment when a cached metric table is
missing, then uses existing CellDINO/Trackastra metric summaries for
sequence-level confidence intervals and paired significance tests.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - scipy is expected in this environment
    wilcoxon = None


CT_ROOT = Path("/group/dingyi/ct")
METRICS_ROOT = Path("/group/dingyi/metrics")
MAIN_ROOT = CT_ROOT / "r1-50-66-512-288-mask-30"
OUTPUT_DIR = MAIN_ROOT / "statistical_significance_no_new_runs"

METRICS = ["CHOTA", "TRA", "DET", "MOTA", "HOTA", "IDF1", "LNK"]
PLOT_METRICS = ["CHOTA", "HOTA", "IDF1", "MOTA"]
MAIN_METHOD = "Ours"
BOOTSTRAP_SAMPLES = 20000
RANDOM_SEED = 20260725

sys.path.insert(0, str(METRICS_ROOT))
from ctc_metrics.scripts.evaluate import evaluate_sequence  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    display_name: str
    dataset: str


DATASETS: Tuple[DatasetSpec, ...] = (
    DatasetSpec("3T3", "3t3"),
    DatasetSpec("HEK293", "hek"),
    DatasetSpec("HeLa", "hela"),
    DatasetSpec("PC-3", "pc"),
    DatasetSpec("RAW264", "raw"),
)

BASELINE_SUMMARIES: Dict[str, Path] = {
    "CellDINO": CT_ROOT / "results" / "celldino_tracking" / "metrics_summary_final.csv",
    "Trackastra": CT_ROOT / "results" / "trackastra_tracking" / "metrics_summary_final.csv",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame number from {path}")
    return int(match.group(1))


def write_bbox_cache_from_masks(mask_dir: Path, out_file: Path) -> int:
    """Write frame,id,x,y,w,h rows from CTC mask*.tif files."""
    mask_files = sorted(mask_dir.glob("mask*.tif"), key=frame_number)
    if not mask_files:
        raise FileNotFoundError(f"No mask*.tif files found in {mask_dir}")

    rows_written = 0
    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for mask_path in mask_files:
            frame = frame_number(mask_path)
            mask = tifffile.imread(mask_path)
            if mask.ndim > 2:
                mask = mask[..., 0]

            labels = np.unique(mask)
            labels = labels[labels != 0]
            for label in labels:
                ys, xs = np.nonzero(mask == label)
                if ys.size == 0:
                    continue
                x_min = int(xs.min())
                x_max = int(xs.max())
                y_min = int(ys.min())
                y_max = int(ys.max())
                writer.writerow(
                    [
                        frame,
                        int(label),
                        x_min,
                        y_min,
                        x_max - x_min + 1,
                        y_max - y_min + 1,
                    ]
                )
                rows_written += 1
    return rows_written


def write_metric_csv(path: Path, row: Dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS, delimiter=";")
        writer.writeheader()
        writer.writerow({metric: row[metric] for metric in METRICS})


def evaluate_main_outputs(out_dir: Path) -> pd.DataFrame:
    """Evaluate saved main predictions and return sequence-level metrics."""
    eval_input_dir = out_dir / "main_eval_inputs"
    metric_dir = out_dir / "main_ctc_metrics"
    ensure_dir(eval_input_dir)
    ensure_dir(metric_dir)

    rows: List[Dict[str, object]] = []
    cache_rows: List[Dict[str, object]] = []

    for spec in DATASETS:
        pred_ctc_dir = MAIN_ROOT / spec.display_name / "test" / "CTC"
        if not pred_ctc_dir.exists():
            raise FileNotFoundError(f"Missing prediction CTC directory: {pred_ctc_dir}")

        for pred_res_dir in sorted(pred_ctc_dir.glob("*_RES")):
            seq_id = pred_res_dir.name.replace("_RES", "")
            sequence = f"{spec.dataset}-{seq_id}"
            gt_dir = METRICS_ROOT / "data" / spec.dataset / "gt" / f"{seq_id}_GT"
            if not gt_dir.exists():
                raise FileNotFoundError(f"Missing metric GT directory: {gt_dir}")
            if not (gt_dir / "gt.txt").exists():
                raise FileNotFoundError(f"Missing GT bbox cache: {gt_dir / 'gt.txt'}")

            seq_eval_dir = eval_input_dir / spec.dataset / seq_id
            ensure_dir(seq_eval_dir)
            shutil.copy2(pred_res_dir / "res_track.txt", seq_eval_dir / "res_track.txt")
            bbox_rows = write_bbox_cache_from_masks(pred_res_dir, seq_eval_dir / "res.txt")
            cache_rows.append(
                {
                    "dataset": spec.dataset,
                    "sequence": sequence,
                    "source_res_dir": str(pred_res_dir),
                    "eval_res_dir": str(seq_eval_dir),
                    "gt_dir": str(gt_dir),
                    "bbox_rows": bbox_rows,
                }
            )

            results = evaluate_sequence(
                res=str(seq_eval_dir),
                gt=str(gt_dir),
                metrics=METRICS,
                threads=1,
            )
            row: Dict[str, object] = {
                "method": MAIN_METHOD,
                "dataset": spec.dataset,
                "sequence": sequence,
                "source": str(pred_res_dir),
            }
            for metric in METRICS:
                row[metric] = float(results[metric])
            rows.append(row)

            write_metric_csv(metric_dir / f"{sequence}_final.csv", row)

    df = pd.DataFrame(rows).sort_values(["dataset", "sequence"])
    df.to_csv(out_dir / "main_metrics_summary_final.csv", sep=";", index=False)
    pd.DataFrame(cache_rows).to_csv(out_dir / "main_eval_input_manifest.csv", index=False)
    return df


def load_baseline_summary(method: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline metric summary: {path}")
    df = pd.read_csv(path, sep=";")
    required = {"dataset", "sequence", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["dataset", "sequence", *METRICS]].copy()
    df.insert(0, "method", method)
    df["source"] = str(path)
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="raise")
    return df


def bootstrap_ci(
    values: Iterable[float],
    rng: np.random.Generator,
    n_boot: int = BOOTSTRAP_SAMPLES,
) -> Tuple[float, float, float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    sem = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    if arr.size == 1:
        return (mean, std, sem, mean, mean)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return (mean, std, sem, float(low), float(high))


def summarize_methods(
    seq_df: pd.DataFrame,
    out_dir: Path,
    suffix: str,
    common_sequences: Optional[List[str]] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    df = seq_df
    if common_sequences is not None:
        df = df[df["sequence"].isin(common_sequences)].copy()

    rows: List[Dict[str, object]] = []
    for method, method_df in df.groupby("method", sort=False):
        for metric in METRICS:
            mean, std, sem, ci_low, ci_high = bootstrap_ci(method_df[metric], rng)
            rows.append(
                {
                    "analysis_set": suffix,
                    "method": method,
                    "metric": metric,
                    "n_sequences": int(method_df["sequence"].nunique()),
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"method_mean_ci_{suffix}.csv", index=False)
    return summary


def exact_or_monte_carlo_sign_flip_p(diff: np.ndarray, rng: np.random.Generator) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.abs(diff) > 1e-12]
    if diff.size == 0:
        return 1.0
    observed = abs(float(np.mean(diff)))
    if diff.size <= 20:
        total = 2 ** diff.size
        count = 0
        for mask in range(total):
            signs = np.ones(diff.size)
            for idx in range(diff.size):
                if (mask >> idx) & 1:
                    signs[idx] = -1
            if abs(float(np.mean(signs * diff))) >= observed - 1e-15:
                count += 1
        return float(count / total)

    samples = BOOTSTRAP_SAMPLES
    signs = rng.choice([-1, 1], size=(samples, diff.size), replace=True)
    values = np.mean(signs * diff, axis=1)
    return float(np.mean(np.abs(values) >= observed - 1e-15))


def paired_tests(seq_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    comparisons = [
        (MAIN_METHOD, "CellDINO"),
        (MAIN_METHOD, "Trackastra"),
        ("CellDINO", "Trackastra"),
    ]
    rows: List[Dict[str, object]] = []

    for method_a, method_b in comparisons:
        df_a = seq_df[seq_df["method"] == method_a].set_index("sequence")
        df_b = seq_df[seq_df["method"] == method_b].set_index("sequence")
        shared = sorted(df_a.index.intersection(df_b.index))
        if not shared:
            continue

        for metric in METRICS:
            a = df_a.loc[shared, metric].to_numpy(dtype=float)
            b = df_b.loc[shared, metric].to_numpy(dtype=float)
            diff = a - b
            mean_diff = float(np.mean(diff))
            if diff.size == 1:
                ci_low = mean_diff
                ci_high = mean_diff
                bootstrap_p = math.nan
            else:
                boot = rng.choice(diff, size=(BOOTSTRAP_SAMPLES, diff.size), replace=True).mean(axis=1)
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                bootstrap_p = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
                bootstrap_p = min(1.0, bootstrap_p)

            if wilcoxon is not None and np.any(np.abs(diff) > 1e-12):
                try:
                    wilcoxon_p = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
            else:
                wilcoxon_p = math.nan

            sign_flip_p = exact_or_monte_carlo_sign_flip_p(diff, rng)
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_pairs": int(diff.size),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "mean_diff_a_minus_b": mean_diff,
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "bootstrap_ci_low": float(ci_low),
                    "bootstrap_ci_high": float(ci_high),
                    "bootstrap_p_two_sided": bootstrap_p,
                    "wilcoxon_p_two_sided": wilcoxon_p,
                    "sign_flip_p_two_sided": sign_flip_p,
                    "shared_sequences": ",".join(shared),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "paired_significance_tests.csv", index=False)
    return result


def format_mean_ci(row: pd.Series) -> str:
    return f"{row['mean']:.3f} [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]"


def write_paper_table(
    common_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    out_dir: Path,
) -> None:
    common = common_summary[common_summary["metric"].isin(PLOT_METRICS)].copy()
    pivot = common.pivot(index="method", columns="metric", values=["mean", "bootstrap_ci_low", "bootstrap_ci_high"])

    order = [MAIN_METHOD, "Trackastra", "CellDINO"]
    lines = [
        "# Statistical Significance Without New Runs",
        "",
        "Important: these are sequence-level statistics from existing outputs, not multi-random-seed results.",
        "",
        "## Common Sequence Set",
        "",
        "The common set contains the 9 sequences shared by Ours, CellDINO, and Trackastra: 3t3-01, hek-01, hela-01..05, and pc-01..02. RAW264 is excluded from the common-all-method table because Trackastra raw inference was not included.",
        "",
        "| Method | CHOTA | HOTA | IDF1 | MOTA |",
        "|---|---:|---:|---:|---:|",
    ]

    for method in order:
        if method not in common["method"].unique():
            continue
        parts = [method]
        for metric in PLOT_METRICS:
            metric_row = common[(common["method"] == method) & (common["metric"] == metric)].iloc[0]
            parts.append(format_mean_ci(metric_row))
        lines.append("| " + " | ".join(parts) + " |")

    lines.extend(
        [
            "",
            "## Paired Tests",
            "",
            "Two-sided Wilcoxon signed-rank p-values are reported below; sign-flip p-values are also available in `paired_significance_tests.csv`.",
            "",
            "| Comparison | Metric | n | Mean Difference | 95% CI | Wilcoxon p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for _, row in pairwise.iterrows():
        if row["metric"] not in PLOT_METRICS:
            continue
        if row["method_a"] != MAIN_METHOD:
            continue
        lines.append(
            "| "
            + f"{row['method_a']} - {row['method_b']} | {row['metric']} | {int(row['n_pairs'])} | "
            + f"{row['mean_diff_a_minus_b']:.3f} | "
            + f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}] | "
            + f"{row['wilcoxon_p_two_sided']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Paper-Ready Wording",
            "",
            "We did not perform additional random-seed training runs. Instead, to estimate statistical reliability from the completed experiments, each test sequence was treated as an independent paired sample. We report sequence-level means with non-parametric bootstrap 95% confidence intervals and use paired two-sided Wilcoxon signed-rank tests for method comparisons.",
            "",
            "Do not describe these numbers as multi-seed mean +/- standard deviation. A correct label is: sequence-level mean with 95% bootstrap confidence interval.",
        ]
    )

    (out_dir / "paper_ready_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_common_method_ci(summary: pd.DataFrame, out_dir: Path) -> None:
    methods = [MAIN_METHOD, "Trackastra", "CellDINO"]
    colors = {
        MAIN_METHOD: "#1f77b4",
        "Trackastra": "#2ca02c",
        "CellDINO": "#d62728",
    }

    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(13.2, 3.4), sharey=False)
    if len(PLOT_METRICS) == 1:
        axes = [axes]
    for ax, metric in zip(axes, PLOT_METRICS):
        rows = summary[(summary["metric"] == metric) & (summary["method"].isin(methods))].copy()
        rows["method"] = pd.Categorical(rows["method"], categories=methods, ordered=True)
        rows = rows.sort_values("method")

        x = np.arange(len(rows))
        y = rows["mean"].to_numpy()
        yerr = np.vstack(
            [
                y - rows["bootstrap_ci_low"].to_numpy(),
                rows["bootstrap_ci_high"].to_numpy() - y,
            ]
        )
        ax.bar(
            x,
            y,
            yerr=yerr,
            capsize=4,
            color=[colors[m] for m in rows["method"].astype(str)],
            width=0.68,
            edgecolor="#222222",
            linewidth=0.7,
        )
        ax.set_title(metric)
        ax.set_ylim(min(0.0, np.nanmin(rows["bootstrap_ci_low"]) - 0.05), 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(rows["method"].astype(str), rotation=35, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Existing-output sequence-level mean and 95% bootstrap CI", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "method_mean_ci_common_sequences.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "method_mean_ci_common_sequences.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_paired_differences(pairwise: pd.DataFrame, out_dir: Path) -> None:
    rows = pairwise[
        (pairwise["method_a"] == MAIN_METHOD)
        & (pairwise["method_b"].isin(["CellDINO", "Trackastra"]))
        & (pairwise["metric"].isin(PLOT_METRICS))
    ].copy()
    rows["comparison"] = rows["method_a"] + " - " + rows["method_b"]
    comp_order = [f"{MAIN_METHOD} - Trackastra", f"{MAIN_METHOD} - CellDINO"]
    rows["comparison"] = pd.Categorical(rows["comparison"], categories=comp_order, ordered=True)
    rows["metric"] = pd.Categorical(rows["metric"], categories=PLOT_METRICS, ordered=True)
    rows = rows.sort_values(["comparison", "metric"])

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y_positions = np.arange(len(rows))
    means = rows["mean_diff_a_minus_b"].to_numpy()
    low = rows["bootstrap_ci_low"].to_numpy()
    high = rows["bootstrap_ci_high"].to_numpy()
    colors = ["#2ca02c" if "Trackastra" in str(c) else "#d62728" for c in rows["comparison"]]

    ax.errorbar(
        means,
        y_positions,
        xerr=np.vstack([means - low, high - means]),
        fmt="o",
        color="#222222",
        ecolor="#444444",
        elinewidth=1.2,
        capsize=4,
        markersize=0,
    )
    ax.scatter(means, y_positions, c=colors, s=58, edgecolor="#222222", linewidth=0.6, zorder=3)
    ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--")
    labels = [
        f"{row['comparison']} / {row['metric']} (n={int(row['n_pairs'])})"
        for _, row in rows.iterrows()
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean paired difference")
    ax.set_title("Paired bootstrap 95% CI from existing outputs")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_difference_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "paired_difference_ci.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(
    seq_df: pd.DataFrame,
    common_sequences: List[str],
    out_dir: Path,
) -> None:
    counts = seq_df.groupby("method")["sequence"].nunique().to_dict()
    text = f"""# Statistical Significance / Multi-Seed Substitute

This folder was generated without new training or inference experiments.

Important limitation: the available artifacts are single-run outputs. Therefore,
the files in this folder do not constitute multi-random-seed evidence and should
not be described as multi-seed mean +/- std.

What is reported instead:

- Sequence-level mean, standard deviation, SEM, and non-parametric bootstrap 95% CI.
- Paired method comparisons on shared sequences.
- Two-sided Wilcoxon signed-rank, exact sign-flip, and bootstrap p-values.

Method sequence counts:

- Ours: {counts.get(MAIN_METHOD, 0)}
- CellDINO: {counts.get('CellDINO', 0)}
- Trackastra: {counts.get('Trackastra', 0)}

Common-all-method sequence set ({len(common_sequences)}):

{', '.join(common_sequences)}

Generated files:

- `main_metrics_summary_final.csv`: CTC metrics for the main saved predictions.
- `sequence_level_metrics.csv`: all available method/sequence metrics.
- `method_mean_ci_all_available.csv`: per-method CIs using each method's available sequences.
- `method_mean_ci_common_all_methods.csv`: per-method CIs on sequences shared by all methods.
- `paired_significance_tests.csv`: paired differences, CIs, and p-values.
- `method_mean_ci_common_sequences.png/.pdf`: paper-friendly common-sequence CI plot.
- `paired_difference_ci.png/.pdf`: paper-friendly paired-difference plot.
- `paper_ready_table.md`: compact table and suggested wording.

Recommended wording:

> We did not perform additional random-seed training runs. Instead, to estimate
> statistical reliability from the completed experiments, each test sequence was
> treated as an independent paired sample. We report sequence-level means with
> non-parametric bootstrap 95% confidence intervals and use paired two-sided
> Wilcoxon signed-rank tests for method comparisons.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    main_df = evaluate_main_outputs(OUTPUT_DIR)
    baseline_dfs = [load_baseline_summary(method, path) for method, path in BASELINE_SUMMARIES.items()]
    seq_df = pd.concat([main_df, *baseline_dfs], ignore_index=True, sort=False)
    seq_df = seq_df.sort_values(["method", "dataset", "sequence"])
    seq_df.to_csv(OUTPUT_DIR / "sequence_level_metrics.csv", index=False)

    method_sets = {
        method: set(group["sequence"].astype(str))
        for method, group in seq_df.groupby("method")
    }
    common_sequences = sorted(set.intersection(*method_sets.values()))

    all_summary = summarize_methods(seq_df, OUTPUT_DIR, "all_available")
    common_summary = summarize_methods(seq_df, OUTPUT_DIR, "common_all_methods", common_sequences)
    pairwise = paired_tests(seq_df, OUTPUT_DIR)

    plot_common_method_ci(common_summary, OUTPUT_DIR)
    plot_paired_differences(pairwise, OUTPUT_DIR)
    write_paper_table(common_summary, pairwise, OUTPUT_DIR)
    write_readme(seq_df, common_sequences, OUTPUT_DIR)

    manifest = {
        "main_root": str(MAIN_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "metrics_root": str(METRICS_ROOT),
        "baseline_summaries": {k: str(v) for k, v in BASELINE_SUMMARIES.items()},
        "metrics": METRICS,
        "plot_metrics": PLOT_METRICS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "random_seed": RANDOM_SEED,
        "notes": [
            "No new training or inference experiments were run.",
            "Main CTC metrics were evaluated from existing saved masks and res_track.txt files.",
            "These are sequence-level statistics, not multi-random-seed results.",
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote outputs to {OUTPUT_DIR}")
    print(f"All-available rows: {len(all_summary)}")
    print(f"Common-all-method sequences: {len(common_sequences)}")
    print(f"Pairwise tests: {len(pairwise)}")


if __name__ == "__main__":
    main()
