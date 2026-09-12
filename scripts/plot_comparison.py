from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.models.registry import all_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "benchmark",
    )
    return parser.parse_args()


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in payload or "efficiency" not in payload:
            continue
        rows.append(payload)
    return rows


def _val(row: dict, *keys: str, default: float = 0.0) -> float:
    cursor: object = row
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    try:
        return float(cursor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def grouped_bars(
    rows: list[dict],
    datasets: list[str],
    value_fn,
    title: str,
    ylabel: str,
    output: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    specs = all_specs()
    models = []
    for row in rows:
        key = row.get("model_key")
        if key in specs and key not in models:
            models.append(key)
    if not models:
        return

    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    palette = {"market1501": "#2563eb", "mars": "#ea580c"}
    labels = {"market1501": "Market1501", "mars": "MARS"}

    for offset, dataset in zip((-width / 2, width / 2), datasets):
        values = []
        for model in models:
            match = next(
                (
                    row
                    for row in rows
                    if row.get("model_key") == model and row.get("dataset") == dataset
                ),
                None,
            )
            values.append(value_fn(match) if match else 0.0)
        ax.bar(
            x + offset,
            values,
            width,
            label=labels[dataset],
            color=palette[dataset],
            edgecolor="white",
        )
        for xpos, value in zip(x + offset, values):
            if value <= 0:
                continue
            ax.text(
                xpos,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([specs[key].label for key in models], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def pareto(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    markers = {"market1501": "o", "mars": "s"}
    specs = all_specs()
    for row in rows:
        key = row.get("model_key")
        dataset = row.get("dataset")
        if key not in specs:
            continue
        spec = specs[key]
        ax.scatter(
            _val(row, "efficiency", "inference", "ms_per_image", "gpu_steady"),
            _val(row, "metrics", "Rank-1"),
            s=90,
            color=spec.color,
            marker=markers.get(str(dataset), "o"),
            edgecolors="black",
            linewidths=0.4,
            label=f"{spec.label} / {dataset}",
        )
    ax.set_xlabel("GPU steady latency (ms / image)")
    ax.set_ylabel("Rank-1 (%)")
    ax.set_title("Accuracy vs inference cost")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_markdown(rows: list[dict], output: Path) -> None:
    specs = all_specs()
    lines = [
        "### Accuracy",
        "",
        "| Model | Trained on | Market1501 Rank-1 | Market1501 mAP | MARS Rank-1 | MARS mAP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    keys = []
    for row in rows:
        key = row.get("model_key")
        if key in specs and key not in keys:
            keys.append(key)

    def cell(model: str, dataset: str, metric: str) -> str:
        match = next(
            (
                row
                for row in rows
                if row.get("model_key") == model and row.get("dataset") == dataset
            ),
            None,
        )
        if match is None:
            return "—"
        return f"{_val(match, 'metrics', metric):.2f}"

    for key in keys:
        spec = specs[key]
        lines.append(
            f"| {spec.label} | {spec.trained_on} | "
            f"{cell(key, 'market1501', 'Rank-1')} | "
            f"{cell(key, 'market1501', 'mAP')} | "
            f"{cell(key, 'mars', 'Rank-1')} | "
            f"{cell(key, 'mars', 'mAP')} |"
        )

    lines += [
        "",
        "### Compute",
        "",
        "| Model | Params (M) | Weights (MiB) | Allocated (MiB) | Peak (MiB) | GPU ms/img | GPU img/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    seen: set[str] = set()
    for row in rows:
        key = row.get("model_key")
        if key not in specs or key in seen:
            continue
        seen.add(key)
        spec = specs[key]
        lines.append(
            f"| {spec.label} | "
            f"{_val(row, 'efficiency', 'model', 'parameters') / 1e6:.1f} | "
            f"{_val(row, 'efficiency', 'model', 'total_mb'):.1f} | "
            f"{_val(row, 'efficiency', 'gpu_memory', 'after_load', 'allocated_mb'):.1f} | "
            f"{_val(row, 'efficiency', 'gpu_memory', 'peak', 'peak_allocated_mb'):.1f} | "
            f"{_val(row, 'efficiency', 'inference', 'ms_per_image', 'gpu_steady'):.2f} | "
            f"{_val(row, 'efficiency', 'inference', 'images_per_s', 'gpu_steady'):.1f} |"
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_dir)
    if not rows:
        raise FileNotFoundError(f"No comparison JSON files in {args.results_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    grouped_bars(
        rows,
        ["market1501", "mars"],
        lambda row: _val(row, "metrics", "Rank-1"),
        "Rank-1 on Market1501 vs MARS",
        "Rank-1 (%)",
        args.output_dir / "rank1.png",
        ylim=(0, 105),
    )
    grouped_bars(
        rows,
        ["market1501", "mars"],
        lambda row: _val(row, "metrics", "mAP"),
        "mAP on Market1501 vs MARS",
        "mAP (%)",
        args.output_dir / "map.png",
        ylim=(0, 105),
    )
    grouped_bars(
        rows,
        ["market1501", "mars"],
        lambda row: _val(
            row, "efficiency", "gpu_memory", "peak", "peak_allocated_mb"
        ),
        "Peak GPU memory",
        "Peak allocated (MiB)",
        args.output_dir / "peak_vram.png",
    )
    grouped_bars(
        rows,
        ["market1501", "mars"],
        lambda row: _val(row, "efficiency", "inference", "ms_per_image", "gpu_steady"),
        "Steady GPU latency",
        "ms / image",
        args.output_dir / "latency.png",
    )
    pareto(rows, args.output_dir / "pareto.png")
    write_markdown(rows, args.output_dir / "tables.md")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
