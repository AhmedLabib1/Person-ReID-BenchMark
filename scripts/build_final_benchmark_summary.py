from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)

ACCURACY_CSV = (
    RESULT_ROOT
    / "cross_domain_summary.csv"
)

EFFICIENCY_CSV = (
    RESULT_ROOT
    / "fastreid_efficiency.csv"
)

OUTPUT_CSV = (
    RESULT_ROOT
    / "final_benchmark_summary.csv"
)

OUTPUT_MARKDOWN = (
    RESULT_ROOT
    / "final_benchmark_summary.md"
)


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV not found:\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def to_float(
    value: str,
) -> float:
    return float(value)


def to_int(
    value: str,
) -> int:
    return int(value)


def load_accuracy() -> dict[str, dict]:
    rows = read_csv(
        ACCURACY_CSV
    )

    models: dict[
        str,
        dict,
    ] = {}

    for row in rows:
        model_id = row[
            "model_id"
        ]

        dataset = row[
            "dataset"
        ].lower()

        model = models.setdefault(
            model_id,
            {
                "model_id": model_id,
                "family": row["family"],
                "backbone": row["backbone"],
            },
        )

        if dataset == "market1501":
            prefix = "market"

        elif dataset == "mars":
            prefix = "mars"

        else:
            raise RuntimeError(
                "Unexpected dataset in "
                f"accuracy summary: {dataset}"
            )

        model[
            f"{prefix}_rank1"
        ] = to_float(
            row["rank_1"]
        )

        model[
            f"{prefix}_rank5"
        ] = to_float(
            row["rank_5"]
        )

        model[
            f"{prefix}_rank10"
        ] = to_float(
            row["rank_10"]
        )

        model[
            f"{prefix}_map"
        ] = to_float(
            row["mAP"]
        )

        model[
            f"{prefix}_minp"
        ] = to_float(
            row["mINP"]
        )

    return models


def load_efficiency() -> dict[str, dict]:
    rows = read_csv(
        EFFICIENCY_CSV
    )

    result: dict[
        str,
        dict,
    ] = {}

    for row in rows:
        model_id = row[
            "model_id"
        ]

        result[
            model_id
        ] = {
            "family": row[
                "family"
            ],

            "backbone": row[
                "backbone"
            ],

            "input_height": to_int(
                row[
                    "input_height"
                ]
            ),

            "input_width": to_int(
                row[
                    "input_width"
                ]
            ),

            "embedding_dim": to_int(
                row[
                    "embedding_dim"
                ]
            ),

            "parameters": to_int(
                row[
                    "parameters"
                ]
            ),

            "parameters_m": to_float(
                row[
                    "parameters_m"
                ]
            ),

            "checkpoint_mb": to_float(
                row[
                    "checkpoint_mb"
                ]
            ),

            "batch1_latency_ms": to_float(
                row[
                    "batch1_latency_ms"
                ]
            ),

            "throughput_batch_size": to_int(
                row[
                    "throughput_batch_size"
                ]
            ),

            "batch_latency_ms": to_float(
                row[
                    "batch_latency_ms"
                ]
            ),

            "throughput_images_per_second": to_float(
                row[
                    "throughput_images_per_second"
                ]
            ),

            "tracklet_frames": to_int(
                row[
                    "tracklet_frames"
                ]
            ),

            "tracklet_latency_ms": to_float(
                row[
                    "tracklet_latency_ms"
                ]
            ),

            "baseline_vram_mb": to_float(
                row[
                    "baseline_vram_mb"
                ]
            ),

            "peak_vram_batch1_mb": to_float(
                row[
                    "peak_vram_batch1_mb"
                ]
            ),

            "peak_vram_batch_mb": to_float(
                row[
                    "peak_vram_batch_mb"
                ]
            ),
        }

    return result


def build_final_rows() -> list[dict]:
    accuracy = load_accuracy()
    efficiency = load_efficiency()

    accuracy_models = set(
        accuracy
    )

    efficiency_models = set(
        efficiency
    )

    if (
        accuracy_models
        != efficiency_models
    ):
        missing_efficiency = (
            accuracy_models
            - efficiency_models
        )

        missing_accuracy = (
            efficiency_models
            - accuracy_models
        )

        raise RuntimeError(
            "Accuracy/Efficiency model mismatch.\n"
            f"Missing efficiency: "
            f"{sorted(missing_efficiency)}\n"
            f"Missing accuracy: "
            f"{sorted(missing_accuracy)}"
        )

    final_rows: list[
        dict
    ] = []

    for model_id in sorted(
        accuracy_models
    ):
        accuracy_row = (
            accuracy[
                model_id
            ]
        )

        efficiency_row = (
            efficiency[
                model_id
            ]
        )

        if (
            accuracy_row["family"]
            != efficiency_row["family"]
        ):
            raise RuntimeError(
                f"Family mismatch for {model_id}"
            )

        if (
            accuracy_row["backbone"]
            != efficiency_row["backbone"]
        ):
            raise RuntimeError(
                f"Backbone mismatch for {model_id}"
            )

        row = {
            "model_id": model_id,

            "family": (
                accuracy_row[
                    "family"
                ]
            ),

            "backbone": (
                accuracy_row[
                    "backbone"
                ]
            ),

            # ---------------------------------
            # Model information
            # ---------------------------------

            "input_height": (
                efficiency_row[
                    "input_height"
                ]
            ),

            "input_width": (
                efficiency_row[
                    "input_width"
                ]
            ),

            "embedding_dim": (
                efficiency_row[
                    "embedding_dim"
                ]
            ),

            "parameters_m": round(
                efficiency_row[
                    "parameters_m"
                ],
                3,
            ),

            "checkpoint_mb": round(
                efficiency_row[
                    "checkpoint_mb"
                ],
                2,
            ),

            # ---------------------------------
            # Market1501
            # ---------------------------------

            "market_rank1": (
                accuracy_row[
                    "market_rank1"
                ]
            ),

            "market_rank5": (
                accuracy_row[
                    "market_rank5"
                ]
            ),

            "market_rank10": (
                accuracy_row[
                    "market_rank10"
                ]
            ),

            "market_map": (
                accuracy_row[
                    "market_map"
                ]
            ),

            "market_minp": (
                accuracy_row[
                    "market_minp"
                ]
            ),

            # ---------------------------------
            # MARS
            # ---------------------------------

            "mars_rank1": (
                accuracy_row[
                    "mars_rank1"
                ]
            ),

            "mars_rank5": (
                accuracy_row[
                    "mars_rank5"
                ]
            ),

            "mars_rank10": (
                accuracy_row[
                    "mars_rank10"
                ]
            ),

            "mars_map": (
                accuracy_row[
                    "mars_map"
                ]
            ),

            "mars_minp": (
                accuracy_row[
                    "mars_minp"
                ]
            ),

            # ---------------------------------
            # Efficiency
            # ---------------------------------

            "batch1_latency_ms": round(
                efficiency_row[
                    "batch1_latency_ms"
                ],
                4,
            ),

            "batch16_latency_ms": round(
                efficiency_row[
                    "batch_latency_ms"
                ],
                4,
            ),

            "throughput_images_per_second": round(
                efficiency_row[
                    "throughput_images_per_second"
                ],
                2,
            ),

            "tracklet_latency_ms": round(
                efficiency_row[
                    "tracklet_latency_ms"
                ],
                4,
            ),

            "peak_vram_batch1_mb": round(
                efficiency_row[
                    "peak_vram_batch1_mb"
                ],
                2,
            ),

            "peak_vram_batch16_mb": round(
                efficiency_row[
                    "peak_vram_batch_mb"
                ],
                2,
            ),
        }

        final_rows.append(
            row
        )

    return final_rows


def write_csv(
    rows: list[dict],
) -> None:
    if not rows:
        raise RuntimeError(
            "No final benchmark rows."
        )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def markdown_table(
    rows: list[dict],
) -> str:
    lines = [
        (
            "| Model | Family | Backbone | "
            "Market R1 | Market mAP | "
            "MARS R1 | MARS mAP | "
            "Latency | Throughput | "
            "VRAM B16 | Params |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|"
            "---:|---:|---:|---:|"
        ),
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['model_id']} | "
            f"{row['family']} | "
            f"{row['backbone']} | "
            f"{row['market_rank1']:.2f}% | "
            f"{row['market_map']:.2f}% | "
            f"{row['mars_rank1']:.2f}% | "
            f"{row['mars_map']:.2f}% | "
            f"{row['batch1_latency_ms']:.2f} ms | "
            f"{row['throughput_images_per_second']:.2f} img/s | "
            f"{row['peak_vram_batch16_mb']:.2f} MB | "
            f"{row['parameters_m']:.2f} M |"
        )

    return "\n".join(
        lines
    )


def write_markdown(
    rows: list[dict],
) -> None:
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            row[
                "market_rank1"
            ],
            row[
                "mars_rank1"
            ],
        ),
        reverse=True,
    )

    content = [
        "# SHAWAF ReID Final Benchmark Summary",
        "",
        (
            "FastReID models trained on MSMT17 "
            "and evaluated cross-domain on "
            "Market1501 and MARS."
        ),
        "",
        markdown_table(
            ranked_rows
        ),
        "",
        "## Notes",
        "",
        (
            "- Market1501 uses single-image "
            "L2-normalized embeddings."
        ),
        (
            "- MARS uses 8 uniformly sampled "
            "frames, raw-feature mean pooling, "
            "then final L2 normalization."
        ),
        (
            "- Efficiency measurements were "
            "performed on the same GPU and "
            "with identical profiling settings."
        ),
        (
            "- MSMT17_V2 in-domain evaluation "
            "is pending a verified dataset copy."
        ),
        "",
    ]

    with OUTPUT_MARKDOWN.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                content
            )
        )


def print_table(
    rows: list[dict],
) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            row[
                "market_rank1"
            ],
            row[
                "mars_rank1"
            ],
        ),
        reverse=True,
    )

    print()
    print("=" * 150)

    print(
        f"{'Model':<18}"
        f"{'Family':<8}"
        f"{'Backbone':<12}"
        f"{'Market R1':>11}"
        f"{'Market mAP':>12}"
        f"{'MARS R1':>10}"
        f"{'MARS mAP':>10}"
        f"{'Latency':>11}"
        f"{'Throughput':>13}"
        f"{'VRAM B16':>11}"
        f"{'Params':>10}"
    )

    print("-" * 150)

    for row in rows:
        print(
            f"{row['model_id']:<18}"
            f"{row['family']:<8}"
            f"{row['backbone']:<12}"

            f"{row['market_rank1']:>10.2f}%"
            f"{row['market_map']:>11.2f}%"

            f"{row['mars_rank1']:>9.2f}%"
            f"{row['mars_map']:>9.2f}%"

            f"{row['batch1_latency_ms']:>9.2f}ms"

            f"{row['throughput_images_per_second']:>11.2f}"

            f"{row['peak_vram_batch16_mb']:>9.2f}MB"

            f"{row['parameters_m']:>8.2f}M"
        )

    print("=" * 150)


def main() -> None:
    print("=" * 88)

    print(
        "SHAWAF ReID - "
        "Final Accuracy + Efficiency Summary"
    )

    print("=" * 88)

    rows = build_final_rows()

    if len(rows) != 12:
        raise RuntimeError(
            f"Expected 12 models, "
            f"got {len(rows)}."
        )

    write_csv(
        rows
    )

    write_markdown(
        rows
    )

    print_table(
        rows
    )

    print()
    print("FINAL SUMMARY COMPLETE")
    print("-" * 88)

    print(
        f"Models                  : "
        f"{len(rows)}"
    )

    print(
        f"CSV                     : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Markdown                : "
        f"{OUTPUT_MARKDOWN}"
    )


if __name__ == "__main__":
    main()