from __future__ import annotations

import argparse
import csv
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
        nargs="+",
        default=[
            PROJECT_ROOT / "docs" / "benchmark" / "results",
            PROJECT_ROOT / "results" / "comparison",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "benchmark",
    )
    return parser.parse_args()


def load_rows(results_dirs: list[Path]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        for path in sorted(results_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "metrics" not in payload:
                continue
            model_key = str(payload.get("model_key", ""))
            dataset = str(payload.get("dataset", ""))
            if not model_key or not dataset:
                continue
            by_key[(model_key, dataset)] = payload
    return list(by_key.values())


def _val(row: dict | None, *keys: str, default: float = 0.0) -> float:
    if row is None:
        return default
    cursor: object = row
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    try:
        return float(cursor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _ours_gpu_protocol(row: dict) -> bool:
    efficiency = row.get("efficiency")
    if not isinstance(efficiency, dict):
        return False
    return efficiency.get("source") is None


def grouped_bars(
    rows: list[dict],
    datasets: list[str],
    value_fn,
    title: str,
    ylabel: str,
    output: Path,
    ylim: tuple[float, float] | None = None,
    rotate: int = 18,
    figsize: tuple[float, float] = (12.0, 5.8),
) -> None:
    specs = all_specs()
    models: list[str] = []
    for row in rows:
        model_key = row.get("model_key")
        if model_key in specs and model_key not in models:
            models.append(model_key)

    def best_rank1(model_key: str) -> float:
        best = 0.0
        for row in rows:
            if row.get("model_key") == model_key:
                best = max(best, _val(row, "metrics", "Rank-1"))
        return best

    models.sort(key=best_rank1, reverse=True)
    if not models:
        return

    present = [
        dataset
        for dataset in datasets
        if any(row.get("dataset") == dataset for row in rows)
    ]
    if not present:
        return

    x = np.arange(len(models))
    width = 0.72 / max(len(present), 1)
    offsets = (np.arange(len(present)) - (len(present) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=figsize)
    palette = {
        "market1501": "#2563eb",
        "mars": "#ea580c",
        "msmt17": "#0f766e",
    }
    labels = {
        "market1501": "Market1501",
        "mars": "MARS",
        "msmt17": "MSMT17",
    }

    for offset, dataset in zip(offsets, present):
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
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            specs[key].label.replace("FastReID ", "").replace(" (MSMT17)", "")
            for key in models
        ],
        rotation=rotate,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def transfer_bars(rows: list[dict], output: Path) -> None:
    pairs = [
        ("sbs_r50", "msmt_sbs_r50", "SBS-R50"),
        ("agw_r50", "msmt_agw_r50", "AGW-R50"),
        ("bot_r50", "msmt_bot_r50", "BoT-R50"),
    ]
    datasets = ["market1501", "mars"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    width = 0.36
    for ax, dataset, title in zip(
        axes,
        datasets,
        ("Eval on Market1501", "Eval on MARS"),
    ):
        x = np.arange(len(pairs))
        in_domain = []
        cross = []
        for market_key, msmt_key, _label in pairs:
            in_match = next(
                (
                    row
                    for row in rows
                    if row.get("model_key") == market_key
                    and row.get("dataset") == dataset
                ),
                None,
            )
            cross_match = next(
                (
                    row
                    for row in rows
                    if row.get("model_key") == msmt_key and row.get("dataset") == dataset
                ),
                None,
            )
            in_domain.append(_val(in_match, "metrics", "Rank-1"))
            cross.append(_val(cross_match, "metrics", "Rank-1"))
        ax.bar(
            x - width / 2,
            in_domain,
            width,
            label="Trained on Market1501",
            color="#2563eb",
            edgecolor="white",
        )
        ax.bar(
            x + width / 2,
            cross,
            width,
            label="Trained on MSMT17",
            color="#ea580c",
            edgecolor="white",
        )
        for xpos, value in zip(x - width / 2, in_domain):
            ax.text(xpos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        for xpos, value in zip(x + width / 2, cross):
            ax.text(xpos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([item[2] for item in pairs])
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Rank-1 (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Same R50 recipe: in-domain Market weights vs MSMT17 transfer")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def pareto(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    markers = {"market1501": "o", "mars": "s", "msmt17": "^"}
    specs = all_specs()
    for row in rows:
        if not _ours_gpu_protocol(row):
            continue
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
    ax.set_title("Accuracy vs inference cost (our CUDA-event protocol)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_cmc(cmc_dir: Path, dataset: str, output: Path, title: str) -> None:
    specs = all_specs()
    paths = sorted(cmc_dir.glob(f"msmt_*_{dataset}.csv"))
    if not paths:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for path in paths:
        model_key = path.name.replace(f"_{dataset}.csv", "")
        spec = specs.get(model_key)
        ranks: list[int] = []
        values: list[float] = []
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ranks.append(int(row["rank"]))
                values.append(float(row["cmc"]) * 100.0)
        ax.plot(
            ranks,
            values,
            label=spec.label.replace("FastReID ", "").replace(" (MSMT17)", "")
            if spec
            else model_key,
            color=spec.color if spec else None,
            linewidth=1.6,
        )
    ax.set_xlabel("Rank")
    ax.set_ylabel("CMC (%)")
    ax.set_title(title)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _metric_cell(rows: list[dict], model: str, dataset: str, metric: str) -> str:
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
    metrics = match.get("metrics")
    if not isinstance(metrics, dict) or metric not in metrics:
        return "—"
    return f"{_val(match, 'metrics', metric):.2f}"


DATASETS = [
    ("market1501", "Market1501"),
    ("mars", "MARS"),
    ("msmt17", "MSMT17"),
]
CMC_METRICS = ("Rank-1", "Rank-5", "Rank-10", "Rank-20")
ACCURACY_METRICS = ("Rank-1", "mAP", "mINP")
PLOT_METRICS = (
    ("Rank-1", "Rank-1 (%)", "rank1"),
    ("Rank-5", "Rank-5 (%)", "rank5"),
    ("Rank-10", "Rank-10 (%)", "rank10"),
    ("Rank-20", "Rank-20 (%)", "rank20"),
    ("mAP", "mAP (%)", "map"),
    ("mINP", "mINP (%)", "minp"),
)


def _model_keys(subset: list[dict], specs: dict) -> list[str]:
    keys: list[str] = []
    for row in subset:
        key = str(row.get("model_key"))
        if key in specs and key not in keys:
            keys.append(key)
    keys.sort(
        key=lambda model_key: _val(
            next(
                (
                    row
                    for row in subset
                    if row.get("model_key") == model_key
                    and row.get("dataset") == "market1501"
                ),
                None,
            ),
            "metrics",
            "Rank-1",
        ),
        reverse=True,
    )
    return keys


def snapshot_results(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.json"):
        target = dest / path.name
        target.write_bytes(path.read_bytes())


def write_markdown(rows: list[dict], output: Path) -> None:
    specs = all_specs()
    in_domain = [
        row
        for row in rows
        if specs.get(str(row.get("model_key")))
        and specs[str(row.get("model_key"))].trained_on != "MSMT17"
    ]
    cross = [
        row
        for row in rows
        if specs.get(str(row.get("model_key")))
        and specs[str(row.get("model_key"))].trained_on == "MSMT17"
    ]

    def summary_table(title: str, subset: list[dict]) -> list[str]:
        keys = _model_keys(subset, specs)
        lines = [
            f"### {title}",
            "",
            "| Model | Trained on | Market Rank-1 | Market mAP | Market mINP | MARS Rank-1 | MARS mAP | MARS mINP | MSMT Rank-1 | MSMT mAP | MSMT mINP |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in keys:
            spec = specs[key]
            cells = [
                _metric_cell(subset, key, dataset, metric)
                for dataset, _label in DATASETS
                for metric in ACCURACY_METRICS
            ]
            lines.append(
                f"| {spec.label} | {spec.trained_on} | " + " | ".join(cells) + " |"
            )
        return lines

    def dataset_table(title: str, subset: list[dict], dataset: str) -> list[str]:
        keys = _model_keys(subset, specs)
        if not keys:
            return []
        if all(
            _metric_cell(subset, key, dataset, "Rank-1") == "—" for key in keys
        ):
            return []
        lines = [
            f"### {title}",
            "",
            "| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key in keys:
            spec = specs[key]
            cells = [
                _metric_cell(subset, key, dataset, metric)
                for metric in (*CMC_METRICS, "mAP", "mINP")
            ]
            lines.append(f"| {spec.label} | " + " | ".join(cells) + " |")
        return lines

    lines = summary_table(
        "In-domain / same-campus (Market1501 weights + foundation)",
        in_domain,
    )
    lines += [""]
    lines += summary_table(
        "Cross-domain (MSMT17 weights, no fine-tune)",
        cross,
    )
    for dataset, label in DATASETS:
        for block in (
            dataset_table(
                f"{label} CMC + mAP (Market1501-trained + foundation)",
                in_domain,
                dataset,
            ),
            dataset_table(
                f"{label} CMC + mAP (MSMT17-trained)",
                cross,
                dataset,
            ),
        ):
            if block:
                lines += [""]
                lines += block

    compute_rows: list[tuple[float, str]] = []
    seen: set[str] = set()
    for row in in_domain:
        if not _ours_gpu_protocol(row):
            continue
        if row.get("dataset") != "market1501":
            continue
        key = str(row.get("model_key"))
        if key not in specs or key in seen:
            continue
        seen.add(key)
        spec = specs[key]
        img_s = _val(row, "efficiency", "inference", "images_per_s", "gpu_steady")
        line = (
            f"| {spec.label} | "
            f"{_val(row, 'efficiency', 'model', 'parameters') / 1e6:.1f} | "
            f"{_val(row, 'efficiency', 'model', 'total_mb'):.1f} | "
            f"{_val(row, 'efficiency', 'gpu_memory', 'after_load', 'allocated_mb'):.1f} | "
            f"{_val(row, 'efficiency', 'gpu_memory', 'peak', 'peak_allocated_mb'):.1f} | "
            f"{_val(row, 'efficiency', 'inference', 'ms_per_image', 'gpu_steady'):.2f} | "
            f"{img_s:.1f} |"
        )
        compute_rows.append((img_s, line))
    compute_rows.sort(key=lambda item: item[0], reverse=True)

    lines += [
        "",
        "### Compute (CUDA-event protocol, Market1501 extract)",
        "",
        "| Model | Params (M) | Weights (MiB) | Allocated (MiB) | Peak (MiB) | GPU ms/img | GPU img/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(line for _speed, line in compute_rows)
    lines += [
        "",
        "Not in this bench (on purpose): DukeMTMC zoo, vehicle ReID,",
        "FastReID ViT (no zoo `.pth`), newer non-FastReID models",
        "(SOLIDER, CLIP-ReID, CLIMB-ReID).",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(list(args.results_dir))
    if not rows:
        raise FileNotFoundError("No comparison JSON files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    specs = all_specs()
    in_domain = [
        row
        for row in rows
        if specs.get(str(row.get("model_key")))
        and specs[str(row.get("model_key"))].trained_on != "MSMT17"
    ]
    gpu_rows = [row for row in in_domain if _ours_gpu_protocol(row)]
    cross = [
        row
        for row in rows
        if specs.get(str(row.get("model_key")))
        and specs[str(row.get("model_key"))].trained_on == "MSMT17"
    ]

    grouped_bars(
        gpu_rows,
        ["market1501", "mars", "msmt17"],
        lambda row: _val(
            row, "efficiency", "gpu_memory", "peak", "peak_allocated_mb"
        ),
        "Peak GPU memory (our extract protocol)",
        "Peak allocated (MiB)",
        args.output_dir / "peak_vram.png",
        rotate=24,
        figsize=(16.0, 6.2),
    )
    grouped_bars(
        gpu_rows,
        ["market1501", "mars", "msmt17"],
        lambda row: _val(row, "efficiency", "inference", "ms_per_image", "gpu_steady"),
        "Steady GPU latency (our CUDA-event protocol)",
        "ms / image",
        args.output_dir / "latency.png",
        rotate=24,
        figsize=(16.0, 6.2),
    )
    pareto(gpu_rows, args.output_dir / "pareto.png")

    for metric, ylabel, stem in PLOT_METRICS:
        grouped_bars(
            in_domain,
            ["market1501", "mars", "msmt17"],
            lambda row, metric=metric: _val(row, "metrics", metric),
            f"{metric} — Market1501 weights + foundation encoders",
            ylabel,
            args.output_dir / f"{stem}.png",
            ylim=(0, 105),
            rotate=24,
            figsize=(16.0, 6.2),
        )
        grouped_bars(
            cross,
            ["market1501", "mars", "msmt17"],
            lambda row, metric=metric: _val(row, "metrics", metric),
            f"{metric} — MSMT17 weights (Market/MARS transfer, MSMT17 in-domain)",
            ylabel,
            args.output_dir / f"{stem}_msmt.png",
            ylim=(0, 105),
            rotate=28,
            figsize=(15.0, 6.2),
        )

    transfer_bars(rows, args.output_dir / "transfer_rank1.png")
    plot_cmc(
        args.output_dir / "cmc",
        "market1501",
        args.output_dir / "cmc_msmt_market1501.png",
        "CMC on Market1501 (MSMT17-trained FastReID)",
    )
    plot_cmc(
        args.output_dir / "cmc",
        "mars",
        args.output_dir / "cmc_msmt_mars.png",
        "CMC on MARS (MSMT17-trained FastReID)",
    )
    snapshot_results(
        PROJECT_ROOT / "results" / "comparison",
        args.output_dir / "results",
    )
    write_markdown(rows, args.output_dir / "tables.md")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
