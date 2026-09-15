from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_CSV = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
    / "final_benchmark_summary.csv"
)

FIGURES_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "figures"
)


def load_rows() -> list[dict]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Final benchmark summary not found:\n{SUMMARY_CSV}"
        )

    with SUMMARY_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    numeric_fields = (
        "market_rank1",
        "market_map",
        "mars_rank1",
        "mars_map",
        "batch1_latency_ms",
        "throughput_images_per_second",
        "tracklet_latency_ms",
        "peak_vram_batch16_mb",
        "parameters_m",
    )

    for row in rows:
        for field in numeric_fields:
            row[field] = float(
                row[field]
            )

    return rows


def save_figure(
    filename: str,
) -> None:
    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        FIGURES_ROOT
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved                   : {path}"
    )


def horizontal_bar(
    *,
    rows: list[dict],
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
    higher_is_better: bool,
    suffix: str = "",
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: row[metric],
        reverse=not higher_is_better,
    )

    names = [
        row["model_id"]
        for row in ordered
    ]

    values = [
        row[metric]
        for row in ordered
    ]

    plt.figure(
        figsize=(11, 7)
    )

    bars = plt.barh(
        names,
        values,
    )

    plt.title(
        title,
        fontsize=15,
        pad=14,
    )

    plt.xlabel(
        xlabel
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.2f}{suffix}",
            va="center",
            fontsize=9,
        )

    save_figure(
        filename
    )


def scatter_with_labels(
    *,
    rows: list[dict],
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(
        figsize=(11, 7)
    )

    family_markers = {
        "BoT": "o",
        "AGW": "s",
        "SBS": "^",
    }

    families = sorted(
        {
            row["family"]
            for row in rows
        }
    )

    for family in families:
        family_rows = [
            row
            for row in rows
            if row["family"] == family
        ]

        x_values = [
            row[x_metric]
            for row in family_rows
        ]

        y_values = [
            row[y_metric]
            for row in family_rows
        ]

        plt.scatter(
            x_values,
            y_values,
            s=85,
            marker=family_markers.get(
                family,
                "o",
            ),
            label=family,
        )

        for row in family_rows:
            plt.annotate(
                row["model_id"],
                (
                    row[x_metric],
                    row[y_metric],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.title(
        title,
        fontsize=15,
        pad=14,
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.grid(
        alpha=0.25,
    )

    plt.legend(
        title="Family"
    )

    save_figure(
        filename
    )


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "Benchmark Figure Generator"
    )
    print("=" * 88)

    rows = load_rows()

    print()
    print(
        f"Models                  : {len(rows)}"
    )

    print(
        f"Source                  : {SUMMARY_CSV}"
    )

    print(
        f"Output                  : {FIGURES_ROOT}"
    )

    print()

    # ==========================================================
    # ACCURACY
    # ==========================================================

    horizontal_bar(
        rows=rows,
        metric="market_rank1",
        title="Market1501 Cross-Domain Rank-1",
        xlabel="Rank-1 (%)",
        filename="market1501_rank1.png",
        higher_is_better=True,
        suffix="%",
    )

    horizontal_bar(
        rows=rows,
        metric="market_map",
        title="Market1501 Cross-Domain mAP",
        xlabel="mAP (%)",
        filename="market1501_map.png",
        higher_is_better=True,
        suffix="%",
    )

    horizontal_bar(
        rows=rows,
        metric="mars_rank1",
        title="MARS Cross-Domain Rank-1",
        xlabel="Rank-1 (%)",
        filename="mars_rank1.png",
        higher_is_better=True,
        suffix="%",
    )

    horizontal_bar(
        rows=rows,
        metric="mars_map",
        title="MARS Cross-Domain mAP",
        xlabel="mAP (%)",
        filename="mars_map.png",
        higher_is_better=True,
        suffix="%",
    )

    # ==========================================================
    # EFFICIENCY
    # ==========================================================

    horizontal_bar(
        rows=rows,
        metric="batch1_latency_ms",
        title="Single-Image Inference Latency",
        xlabel="Latency (ms)",
        filename="batch1_latency.png",
        higher_is_better=False,
        suffix=" ms",
    )

    horizontal_bar(
        rows=rows,
        metric="throughput_images_per_second",
        title="Batch-16 Inference Throughput",
        xlabel="Images / second",
        filename="throughput.png",
        higher_is_better=True,
        suffix=" img/s",
    )

    horizontal_bar(
        rows=rows,
        metric="tracklet_latency_ms",
        title="8-Frame Tracklet Inference Latency",
        xlabel="Latency (ms / tracklet)",
        filename="tracklet_latency.png",
        higher_is_better=False,
        suffix=" ms",
    )

    horizontal_bar(
        rows=rows,
        metric="peak_vram_batch16_mb",
        title="Peak GPU Memory — Batch 16",
        xlabel="VRAM (MB)",
        filename="peak_vram_batch16.png",
        higher_is_better=False,
        suffix=" MB",
    )

    horizontal_bar(
        rows=rows,
        metric="parameters_m",
        title="Model Parameter Count",
        xlabel="Parameters (millions)",
        filename="parameters.png",
        higher_is_better=False,
        suffix=" M",
    )

    # ==========================================================
    # ACCURACY / EFFICIENCY TRADE-OFF
    # ==========================================================

    scatter_with_labels(
        rows=rows,
        x_metric="batch1_latency_ms",
        y_metric="market_rank1",
        title="Market1501 Rank-1 vs Inference Latency",
        xlabel="Batch-1 latency (ms)",
        ylabel="Market1501 Rank-1 (%)",
        filename="market_rank1_vs_latency.png",
    )

    scatter_with_labels(
        rows=rows,
        x_metric="throughput_images_per_second",
        y_metric="market_map",
        title="Market1501 mAP vs Throughput",
        xlabel="Throughput (images / second)",
        ylabel="Market1501 mAP (%)",
        filename="market_map_vs_throughput.png",
    )

    scatter_with_labels(
        rows=rows,
        x_metric="tracklet_latency_ms",
        y_metric="mars_rank1",
        title="MARS Rank-1 vs Tracklet Latency",
        xlabel="8-frame tracklet latency (ms)",
        ylabel="MARS Rank-1 (%)",
        filename="mars_rank1_vs_tracklet_latency.png",
    )

    scatter_with_labels(
        rows=rows,
        x_metric="peak_vram_batch16_mb",
        y_metric="market_rank1",
        title="Market1501 Rank-1 vs Peak VRAM",
        xlabel="Peak VRAM — batch 16 (MB)",
        ylabel="Market1501 Rank-1 (%)",
        filename="market_rank1_vs_vram.png",
    )

    print()
    print("=" * 88)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 88)

    print(
        "Figures generated        : 13"
    )

    print(
        f"Directory                : {FIGURES_ROOT}"
    )


if __name__ == "__main__":
    main()