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


NUMERIC_FIELDS = (
    # MSMT17_V2 — In-Domain
    "msmt_rank1",
    "msmt_map",

    # Market1501 — Cross-Domain
    "market_rank1",
    "market_map",

    # MARS — Cross-Domain
    "mars_rank1",
    "mars_map",

    # Efficiency
    "batch1_latency_ms",
    "batch16_latency_ms",
    "throughput_images_per_second",
    "tracklet_latency_ms",
    "peak_vram_batch1_mb",
    "peak_vram_batch16_mb",
    "parameters_m",
    "checkpoint_mb",
)


FAMILY_MARKERS = {
    "BoT": "o",
    "AGW": "s",
    "SBS": "^",
}


def load_rows() -> list[dict]:
    """
    Load the final benchmark CSV.

    Expected source:

        benchmarks/results/final_benchmark_summary.csv

    Each row represents one FastReID model and contains:

        - MSMT17_V2 in-domain accuracy
        - Market1501 cross-domain accuracy
        - MARS cross-domain accuracy
        - efficiency measurements
    """

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            "Final benchmark summary not found:\n"
            f"{SUMMARY_CSV}"
        )

    with SUMMARY_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise RuntimeError(
            "Final benchmark summary is empty."
        )

    required_fields = {
        "model_id",
        "family",
        "backbone",
        *NUMERIC_FIELDS,
    }

    missing_fields = (
        required_fields
        - set(rows[0].keys())
    )

    if missing_fields:
        raise RuntimeError(
            "Final benchmark summary is missing "
            "required columns:\n"
            + "\n".join(
                sorted(missing_fields)
            )
        )

    for row in rows:
        for field in NUMERIC_FIELDS:
            try:
                row[field] = float(
                    row[field]
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise RuntimeError(
                    "Invalid numeric value in "
                    f"{SUMMARY_CSV}\n"
                    f"Model: {row.get('model_id')}\n"
                    f"Field: {field}\n"
                    f"Value: {row.get(field)}"
                ) from exc

    return rows


def save_figure(
    filename: str,
) -> None:
    """
    Save the current Matplotlib figure.
    """

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_ROOT
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved                   : "
        f"{output_path}"
    )


def horizontal_bar(
    *,
    rows: list[dict],
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
    higher_is_better: bool,
    suffix: str,
) -> None:
    """
    Generate a horizontal bar chart.

    For accuracy / throughput:
        highest values appear first.

    For latency / VRAM / parameters:
        lowest values appear first.
    """

    ordered = sorted(
        rows,
        key=lambda row: row[
            metric
        ],
        reverse=higher_is_better,
    )

    labels = [
        row["model_id"]
        for row in ordered
    ]

    values = [
        row[metric]
        for row in ordered
    ]

    families = [
        row["family"]
        for row in ordered
    ]

    figure_height = max(
        6.5,
        len(rows) * 0.48,
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            figure_height,
        )
    )

    bars = ax.barh(
        labels,
        values,
    )

    ax.invert_yaxis()

    ax.set_title(
        title,
        fontsize=14,
        pad=14,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    max_value = max(
        values
    )

    value_offset = (
        max_value
        * 0.012
        if max_value > 0
        else 0.1
    )

    for (
        bar,
        value,
        family,
    ) in zip(
        bars,
        values,
        families,
    ):
        ax.text(
            value + value_offset,
            bar.get_y()
            + bar.get_height() / 2,
            (
                f"{value:.2f}"
                f"{suffix} "
                f"[{family}]"
            ),
            va="center",
            fontsize=8.5,
        )

    ax.set_xlim(
        left=0,
        right=(
            max_value
            + value_offset * 10
        ),
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
    """
    Accuracy / efficiency scatter plot.

    Models are grouped by FastReID family:

        BoT
        AGW
        SBS
    """

    fig, ax = plt.subplots(
        figsize=(
            11,
            7,
        )
    )

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

        marker = FAMILY_MARKERS.get(
            family,
            "o",
        )

        ax.scatter(
            x_values,
            y_values,
            s=85,
            marker=marker,
            label=family,
        )

        for row in family_rows:
            ax.annotate(
                row["model_id"],
                (
                    row[x_metric],
                    row[y_metric],
                ),
                xytext=(
                    5,
                    5,
                ),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(
        title,
        fontsize=14,
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

    ax.legend(
        title="Family"
    )

    save_figure(
        filename
    )


def rank1_domain_transfer_plot(
    rows: list[dict],
) -> None:
    """
    Visualize how every model behaves when moving from:

        MSMT17_V2
            ↓
        Market1501
            ↓
        MARS

    Important interpretation:

        MSMT17_V2:
            In-domain image ReID.

        Market1501:
            Cross-domain image ReID.

        MARS:
            Cross-domain tracklet ReID using
            uniform8_mean_raw_l2.

    Therefore the MSMT -> MARS drop is not purely
    domain shift. It also includes the change from
    single-image evaluation to tracklet evaluation.
    """

    ordered = sorted(
        rows,
        key=lambda row: row[
            "msmt_rank1"
        ],
        reverse=True,
    )

    x_values = [
        0,
        1,
        2,
    ]

    x_labels = [
        "MSMT17_V2\nIn-Domain",
        "Market1501\nCross-Domain",
        "MARS\nCross-Domain Tracklet",
    ]

    fig, ax = plt.subplots(
        figsize=(
            13,
            8,
        )
    )

    for row in ordered:
        values = [
            row["msmt_rank1"],
            row["market_rank1"],
            row["mars_rank1"],
        ]

        marker = FAMILY_MARKERS.get(
            row["family"],
            "o",
        )

        ax.plot(
            x_values,
            values,
            marker=marker,
            linewidth=1.6,
            markersize=6,
            label=row["model_id"],
        )

    ax.set_xticks(
        x_values,
        x_labels,
    )

    ax.set_ylabel(
        "Rank-1 (%)"
    )

    ax.set_title(
        "Rank-1 Across In-Domain and Cross-Domain Evaluation",
        fontsize=14,
        pad=14,
    )

    ax.set_ylim(
        bottom=0,
        top=100,
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        loc="upper left",
        fontsize=8,
    )

    save_figure(
        "rank1_domain_transfer.png"
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
        f"Models                  : "
        f"{len(rows)}"
    )

    print(
        f"Source                  : "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Output                  : "
        f"{FIGURES_ROOT}"
    )

    print()

    # ==========================================================
    # MSMT17_V2 — IN-DOMAIN ACCURACY
    # ==========================================================

    horizontal_bar(
        rows=rows,
        metric="msmt_rank1",
        title=(
            "MSMT17_V2 In-Domain Rank-1"
        ),
        xlabel="Rank-1 (%)",
        filename="msmt17_rank1.png",
        higher_is_better=True,
        suffix="%",
    )

    horizontal_bar(
        rows=rows,
        metric="msmt_map",
        title=(
            "MSMT17_V2 In-Domain mAP"
        ),
        xlabel="mAP (%)",
        filename="msmt17_map.png",
        higher_is_better=True,
        suffix="%",
    )

    # ==========================================================
    # MARKET1501 — CROSS-DOMAIN IMAGE ACCURACY
    # ==========================================================

    horizontal_bar(
        rows=rows,
        metric="market_rank1",
        title=(
            "Market1501 Cross-Domain Rank-1"
        ),
        xlabel="Rank-1 (%)",
        filename="market1501_rank1.png",
        higher_is_better=True,
        suffix="%",
    )

    horizontal_bar(
        rows=rows,
        metric="market_map",
        title=(
            "Market1501 Cross-Domain mAP"
        ),
        xlabel="mAP (%)",
        filename="market1501_map.png",
        higher_is_better=True,
        suffix="%",
    )

    # ==========================================================
    # MARS — CROSS-DOMAIN TRACKLET ACCURACY
    # ==========================================================

    horizontal_bar(
        rows=rows,
        metric="mars_rank1",
        title=(
            "MARS Cross-Domain Rank-1 "
            "(Uniform 8 Frames)"
        ),
        xlabel="Rank-1 (%)",
        filename="mars_rank1.png",
        higher_is_better=True,
        suffix="%",
    )

    horizontal_bar(
        rows=rows,
        metric="mars_map",
        title=(
            "MARS Cross-Domain mAP "
            "(Uniform 8 Frames)"
        ),
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
        title=(
            "Single-Image Inference Latency"
        ),
        xlabel="Latency (ms)",
        filename="batch1_latency.png",
        higher_is_better=False,
        suffix=" ms",
    )

    horizontal_bar(
        rows=rows,
        metric="throughput_images_per_second",
        title=(
            "Batch-16 Inference Throughput"
        ),
        xlabel="Images / second",
        filename="throughput.png",
        higher_is_better=True,
        suffix=" img/s",
    )

    horizontal_bar(
        rows=rows,
        metric="tracklet_latency_ms",
        title=(
            "8-Frame Tracklet "
            "Inference Latency"
        ),
        xlabel="Latency (ms / tracklet)",
        filename="tracklet_latency.png",
        higher_is_better=False,
        suffix=" ms",
    )

    horizontal_bar(
        rows=rows,
        metric="peak_vram_batch16_mb",
        title=(
            "Peak GPU Memory — Batch 16"
        ),
        xlabel="VRAM (MB)",
        filename="peak_vram_batch16.png",
        higher_is_better=False,
        suffix=" MB",
    )

    horizontal_bar(
        rows=rows,
        metric="parameters_m",
        title=(
            "Model Parameter Count"
        ),
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
        y_metric="msmt_rank1",
        title=(
            "MSMT17_V2 Rank-1 "
            "vs Inference Latency"
        ),
        xlabel="Batch-1 latency (ms)",
        ylabel="MSMT17_V2 Rank-1 (%)",
        filename="msmt_rank1_vs_latency.png",
    )

    scatter_with_labels(
        rows=rows,
        x_metric="batch1_latency_ms",
        y_metric="market_rank1",
        title=(
            "Market1501 Rank-1 "
            "vs Inference Latency"
        ),
        xlabel="Batch-1 latency (ms)",
        ylabel="Market1501 Rank-1 (%)",
        filename="market_rank1_vs_latency.png",
    )

    scatter_with_labels(
        rows=rows,
        x_metric="throughput_images_per_second",
        y_metric="market_map",
        title=(
            "Market1501 mAP "
            "vs Throughput"
        ),
        xlabel="Throughput (images / second)",
        ylabel="Market1501 mAP (%)",
        filename="market_map_vs_throughput.png",
    )

    scatter_with_labels(
        rows=rows,
        x_metric="tracklet_latency_ms",
        y_metric="mars_rank1",
        title=(
            "MARS Rank-1 "
            "vs Tracklet Latency"
        ),
        xlabel="8-frame tracklet latency (ms)",
        ylabel="MARS Rank-1 (%)",
        filename=(
            "mars_rank1_vs_tracklet_latency.png"
        ),
    )

    scatter_with_labels(
        rows=rows,
        x_metric="peak_vram_batch16_mb",
        y_metric="market_rank1",
        title=(
            "Market1501 Rank-1 "
            "vs Peak VRAM"
        ),
        xlabel="Peak VRAM — batch 16 (MB)",
        ylabel="Market1501 Rank-1 (%)",
        filename="market_rank1_vs_vram.png",
    )

    # ==========================================================
    # DOMAIN / PROTOCOL TRANSFER
    # ==========================================================

    rank1_domain_transfer_plot(
        rows
    )

    print()
    print("=" * 88)

    print(
        "FIGURE GENERATION COMPLETE"
    )

    print("=" * 88)

    print(
        "Figures generated        : 17"
    )

    print(
        f"Directory                : "
        f"{FIGURES_ROOT}"
    )


if __name__ == "__main__":
    main()