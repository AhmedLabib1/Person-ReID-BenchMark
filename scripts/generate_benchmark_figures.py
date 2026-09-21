from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)

FIGURE_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "figures"
)

SUMMARY_CSV = (
    RESULT_ROOT
    / "final_benchmark_summary.csv"
)


EXPECTED_MODELS = 13


def load_rows() -> list[dict]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            "Final benchmark summary not found:\n"
            f"{SUMMARY_CSV}"
        )

    with SUMMARY_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    if len(rows) != EXPECTED_MODELS:
        raise RuntimeError(
            "Unexpected model count: "
            f"{len(rows)} "
            f"(expected {EXPECTED_MODELS})"
        )

    numeric_fields = (
        "parameters_m",
        "checkpoint_mb",

        "msmt_rank1",
        "msmt_rank5",
        "msmt_rank10",
        "msmt_map",
        "msmt_minp",

        "market_rank1",
        "market_rank5",
        "market_rank10",
        "market_map",
        "market_minp",

        "mars_rank1",
        "mars_rank5",
        "mars_rank10",
        "mars_map",
        "mars_minp",

        "batch1_latency_ms",
        "batch16_latency_ms",
        "throughput_images_per_second",
        "tracklet_latency_ms",
        "baseline_vram_mb",
        "peak_vram_batch1_mb",
        "peak_vram_batch16_mb",
    )

    integer_fields = (
        "input_height",
        "input_width",
        "embedding_dim",
    )

    for row in rows:
        for field in numeric_fields:
            row[field] = float(
                row[field]
            )

        for field in integer_fields:
            row[field] = int(
                row[field]
            )

    return rows


def save_bar_chart(
    *,
    rows: list[dict],
    field: str,
    title: str,
    ylabel: str,
    filename: str,
    higher_is_better: bool,
) -> Path:
    ordered = sorted(
        rows,
        key=lambda row: row[field],
        reverse=higher_is_better,
    )

    labels = [
        row["model_id"]
        for row in ordered
    ]

    values = [
        row[field]
        for row in ordered
    ]

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    positions = np.arange(
        len(labels)
    )

    bars = ax.bar(
        positions,
        values,
    )

    ax.set_title(
        title,
        fontsize=15,
        pad=14,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    path = (
        FIGURE_ROOT
        / filename
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


def save_scatter_chart(
    *,
    rows: list[dict],
    x_field: str,
    y_field: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    x_values = [
        row[x_field]
        for row in rows
    ]

    y_values = [
        row[y_field]
        for row in rows
    ]

    ax.scatter(
        x_values,
        y_values,
        s=70,
    )

    for row in rows:
        ax.annotate(
            row["model_id"],
            (
                row[x_field],
                row[y_field],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(
        title,
        fontsize=15,
        pad=14,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    path = (
        FIGURE_ROOT
        / filename
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


def save_cross_domain_rank1(
    rows: list[dict],
) -> Path:
    model_ids = [
        row["model_id"]
        for row in rows
    ]

    msmt = np.array(
        [
            row["msmt_rank1"]
            for row in rows
        ]
    )

    market = np.array(
        [
            row["market_rank1"]
            for row in rows
        ]
    )

    mars = np.array(
        [
            row["mars_rank1"]
            for row in rows
        ]
    )

    positions = np.arange(
        len(model_ids)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    ax.bar(
        positions - width,
        msmt,
        width,
        label="MSMT17_V2",
    )

    ax.bar(
        positions,
        market,
        width,
        label="Market1501",
    )

    ax.bar(
        positions + width,
        mars,
        width,
        label="MARS",
    )

    ax.set_title(
        "DukeMTMC-Trained FastReID Cross-Domain Rank-1",
        fontsize=15,
        pad=14,
    )

    ax.set_ylabel(
        "Rank-1 (%)"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        model_ids,
        rotation=45,
        ha="right",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURE_ROOT
        / "rank1_cross_domain.png"
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


def main() -> None:
    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows()

    print(
        "=" * 100
    )

    print(
        "SHAWAF DUKEMTMC - "
        "BENCHMARK FIGURE GENERATOR"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Models                  : "
        f"{len(rows)}"
    )

    print(
        f"Source CSV              : "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Figure root             : "
        f"{FIGURE_ROOT}"
    )

    print()

    generated: list[Path] = []

    # ============================================================
    # ACCURACY
    # ============================================================

    generated.append(
        save_bar_chart(
            rows=rows,
            field="msmt_rank1",
            title=(
                "MSMT17_V2 Cross-Domain Rank-1"
            ),
            ylabel="Rank-1 (%)",
            filename="msmt17_rank1.png",
            higher_is_better=True,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="msmt_map",
            title=(
                "MSMT17_V2 Cross-Domain mAP"
            ),
            ylabel="mAP (%)",
            filename="msmt17_map.png",
            higher_is_better=True,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="market_rank1",
            title=(
                "Market1501 Cross-Domain Rank-1"
            ),
            ylabel="Rank-1 (%)",
            filename="market1501_rank1.png",
            higher_is_better=True,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="market_map",
            title=(
                "Market1501 Cross-Domain mAP"
            ),
            ylabel="mAP (%)",
            filename="market1501_map.png",
            higher_is_better=True,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="mars_rank1",
            title=(
                "MARS Cross-Domain Tracklet Rank-1"
            ),
            ylabel="Rank-1 (%)",
            filename="mars_rank1.png",
            higher_is_better=True,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="mars_map",
            title=(
                "MARS Cross-Domain Tracklet mAP"
            ),
            ylabel="mAP (%)",
            filename="mars_map.png",
            higher_is_better=True,
        )
    )

    # ============================================================
    # EFFICIENCY
    # ============================================================

    generated.append(
        save_bar_chart(
            rows=rows,
            field="batch1_latency_ms",
            title=(
                "Batch-1 Inference Latency "
                "(Tesla T4)"
            ),
            ylabel="Latency (ms)",
            filename="batch1_latency.png",
            higher_is_better=False,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="tracklet_latency_ms",
            title=(
                "8-Frame Tracklet Latency "
                "(Tesla T4)"
            ),
            ylabel="Latency (ms)",
            filename="tracklet_latency.png",
            higher_is_better=False,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="peak_vram_batch16_mb",
            title=(
                "Peak VRAM - Batch 16 "
                "(Tesla T4)"
            ),
            ylabel="VRAM (MB)",
            filename="peak_vram_batch16.png",
            higher_is_better=False,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="throughput_images_per_second",
            title=(
                "Batch-16 Throughput "
                "(Tesla T4)"
            ),
            ylabel="Images / second",
            filename="throughput.png",
            higher_is_better=True,
        )
    )

    generated.append(
        save_bar_chart(
            rows=rows,
            field="parameters_m",
            title=(
                "FastReID Model Parameters"
            ),
            ylabel="Parameters (Millions)",
            filename="parameters.png",
            higher_is_better=False,
        )
    )

    # ============================================================
    # ACCURACY / EFFICIENCY TRADE-OFFS
    # ============================================================

    generated.append(
        save_scatter_chart(
            rows=rows,
            x_field="batch1_latency_ms",
            y_field="market_rank1",
            title=(
                "Market1501 Rank-1 vs "
                "Batch-1 Latency"
            ),
            xlabel="Batch-1 latency (ms)",
            ylabel="Market1501 Rank-1 (%)",
            filename=(
                "market_rank1_vs_latency.png"
            ),
        )
    )

    generated.append(
        save_scatter_chart(
            rows=rows,
            x_field="peak_vram_batch16_mb",
            y_field="market_rank1",
            title=(
                "Market1501 Rank-1 vs "
                "Peak VRAM"
            ),
            xlabel="Peak VRAM Batch-16 (MB)",
            ylabel="Market1501 Rank-1 (%)",
            filename=(
                "market_rank1_vs_vram.png"
            ),
        )
    )

    generated.append(
        save_scatter_chart(
            rows=rows,
            x_field=(
                "throughput_images_per_second"
            ),
            y_field="market_map",
            title=(
                "Market1501 mAP vs Throughput"
            ),
            xlabel="Throughput (images/s)",
            ylabel="Market1501 mAP (%)",
            filename=(
                "market_map_vs_throughput.png"
            ),
        )
    )

    generated.append(
        save_scatter_chart(
            rows=rows,
            x_field="tracklet_latency_ms",
            y_field="mars_rank1",
            title=(
                "MARS Rank-1 vs "
                "8-Frame Tracklet Latency"
            ),
            xlabel="Tracklet latency (ms)",
            ylabel="MARS Rank-1 (%)",
            filename=(
                "mars_rank1_vs_tracklet_latency.png"
            ),
        )
    )

    generated.append(
        save_scatter_chart(
            rows=rows,
            x_field="peak_vram_batch16_mb",
            y_field="mars_rank1",
            title=(
                "MARS Rank-1 vs Peak VRAM"
            ),
            xlabel="Peak VRAM Batch-16 (MB)",
            ylabel="MARS Rank-1 (%)",
            filename=(
                "mars_rank1_vs_vram.png"
            ),
        )
    )

    generated.append(
        save_scatter_chart(
            rows=rows,
            x_field="batch1_latency_ms",
            y_field="msmt_rank1",
            title=(
                "MSMT17_V2 Rank-1 vs "
                "Batch-1 Latency"
            ),
            xlabel="Batch-1 latency (ms)",
            ylabel="MSMT17_V2 Rank-1 (%)",
            filename=(
                "msmt_rank1_vs_latency.png"
            ),
        )
    )

    # ============================================================
    # CROSS-DOMAIN OVERVIEW
    # ============================================================

    generated.append(
        save_cross_domain_rank1(
            rows
        )
    )

    print(
        "=" * 100
    )

    print(
        "GENERATED FIGURES"
    )

    print(
        "=" * 100
    )

    for path in generated:
        print(
            f"  {path.name}"
        )

    print()

    print(
        "=" * 100
    )

    print(
        "FIGURE GENERATION COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"New benchmark figures   : "
        f"{len(generated)}"
    )

    print(
        f"Existing CMC figures    : "
        f"{len(list(FIGURE_ROOT.glob('*_cmc.png')))}"
    )

    print(
        f"Figure root             : "
        f"{FIGURE_ROOT}"
    )


if __name__ == "__main__":
    main()