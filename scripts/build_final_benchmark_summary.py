from __future__ import annotations

import csv
import platform
from pathlib import Path

import torch


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


EXPECTED_MODELS = 13
EXPECTED_DATASETS = 3


FINAL_FIELDS = (
    "model_id",
    "family",
    "backbone",

    # Model
    "input_height",
    "input_width",
    "embedding_dim",
    "parameters_m",
    "checkpoint_mb",

    # MSMT17_V2
    "msmt_rank1",
    "msmt_rank5",
    "msmt_rank10",
    "msmt_map",
    "msmt_minp",

    # Market1501
    "market_rank1",
    "market_rank5",
    "market_rank10",
    "market_map",
    "market_minp",

    # MARS
    "mars_rank1",
    "mars_rank5",
    "mars_rank10",
    "mars_map",
    "mars_minp",

    # Efficiency
    "batch1_latency_ms",
    "batch16_latency_ms",
    "throughput_images_per_second",
    "tracklet_latency_ms",
    "baseline_vram_mb",
    "peak_vram_batch1_mb",
    "peak_vram_batch16_mb",
)


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            "Required CSV not found:\n"
            f"{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def load_accuracy() -> dict[str, dict]:
    rows = read_csv(
        ACCURACY_CSV
    )

    expected_rows = (
        EXPECTED_MODELS
        * EXPECTED_DATASETS
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            "Unexpected accuracy row count: "
            f"{len(rows)} "
            f"(expected {expected_rows})"
        )

    models: dict[
        str,
        dict,
    ] = {}

    for row in rows:
        model_id = row[
            "model_id"
        ]

        dataset = (
            row[
                "dataset"
            ]
            .strip()
            .lower()
        )

        model = models.setdefault(
            model_id,
            {
                "model_id": model_id,
                "family": row[
                    "family"
                ],
                "backbone": row[
                    "backbone"
                ],
            },
        )

        if dataset == "msmt17_v2":
            prefix = "msmt"

        elif dataset == "market1501":
            prefix = "market"

        elif dataset == "mars":
            prefix = "mars"

        else:
            raise RuntimeError(
                "Unexpected dataset in "
                "accuracy CSV: "
                f"{dataset}"
            )

        model[
            f"{prefix}_rank1"
        ] = float(
            row[
                "rank_1"
            ]
        )

        model[
            f"{prefix}_rank5"
        ] = float(
            row[
                "rank_5"
            ]
        )

        model[
            f"{prefix}_rank10"
        ] = float(
            row[
                "rank_10"
            ]
        )

        model[
            f"{prefix}_map"
        ] = float(
            row[
                "mAP"
            ]
        )

        model[
            f"{prefix}_minp"
        ] = float(
            row[
                "mINP"
            ]
        )

    if len(models) != EXPECTED_MODELS:
        raise RuntimeError(
            "Unexpected number of accuracy models: "
            f"{len(models)}"
        )

    return models


def load_efficiency() -> dict[str, dict]:
    rows = read_csv(
        EFFICIENCY_CSV
    )

    if len(rows) != EXPECTED_MODELS:
        raise RuntimeError(
            "Unexpected efficiency row count: "
            f"{len(rows)} "
            f"(expected {EXPECTED_MODELS})"
        )

    models: dict[
        str,
        dict,
    ] = {}

    for row in rows:
        model_id = row[
            "model_id"
        ]

        models[
            model_id
        ] = {
            "model_id": model_id,
            "family": row[
                "family"
            ],
            "backbone": row[
                "backbone"
            ],
            "input_height": int(
                row[
                    "input_height"
                ]
            ),
            "input_width": int(
                row[
                    "input_width"
                ]
            ),
            "embedding_dim": int(
                row[
                    "embedding_dim"
                ]
            ),
            "parameters_m": float(
                row[
                    "parameters_m"
                ]
            ),
            "checkpoint_mb": float(
                row[
                    "checkpoint_mb"
                ]
            ),
            "batch1_latency_ms": float(
                row[
                    "batch1_latency_ms"
                ]
            ),
            "batch16_latency_ms": float(
                row[
                    "batch_latency_ms"
                ]
            ),
            "throughput_images_per_second": float(
                row[
                    "throughput_images_per_second"
                ]
            ),
            "tracklet_latency_ms": float(
                row[
                    "tracklet_latency_ms"
                ]
            ),
            "baseline_vram_mb": float(
                row[
                    "baseline_vram_mb"
                ]
            ),
            "peak_vram_batch1_mb": float(
                row[
                    "peak_vram_batch1_mb"
                ]
            ),
            "peak_vram_batch16_mb": float(
                row[
                    "peak_vram_batch_mb"
                ]
            ),
        }

    return models


def build_final_rows() -> list[dict]:
    accuracy = load_accuracy()
    efficiency = load_efficiency()

    accuracy_ids = set(
        accuracy
    )

    efficiency_ids = set(
        efficiency
    )

    if accuracy_ids != efficiency_ids:
        raise RuntimeError(
            "Accuracy and efficiency model IDs "
            "do not match.\n"
            f"Accuracy only: "
            f"{sorted(accuracy_ids - efficiency_ids)}\n"
            f"Efficiency only: "
            f"{sorted(efficiency_ids - accuracy_ids)}"
        )

    final_rows: list[dict] = []

    for model_id in accuracy:
        accuracy_row = accuracy[
            model_id
        ]

        efficiency_row = efficiency[
            model_id
        ]

        if (
            accuracy_row[
                "family"
            ]
            != efficiency_row[
                "family"
            ]
        ):
            raise RuntimeError(
                "Family mismatch for "
                f"{model_id}"
            )

        if (
            accuracy_row[
                "backbone"
            ]
            != efficiency_row[
                "backbone"
            ]
        ):
            raise RuntimeError(
                "Backbone mismatch for "
                f"{model_id}"
            )

        row = {
            "model_id": model_id,
            "family": accuracy_row[
                "family"
            ],
            "backbone": accuracy_row[
                "backbone"
            ],

            "input_height": efficiency_row[
                "input_height"
            ],
            "input_width": efficiency_row[
                "input_width"
            ],
            "embedding_dim": efficiency_row[
                "embedding_dim"
            ],

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

            "msmt_rank1": round(
                accuracy_row[
                    "msmt_rank1"
                ],
                2,
            ),
            "msmt_rank5": round(
                accuracy_row[
                    "msmt_rank5"
                ],
                2,
            ),
            "msmt_rank10": round(
                accuracy_row[
                    "msmt_rank10"
                ],
                2,
            ),
            "msmt_map": round(
                accuracy_row[
                    "msmt_map"
                ],
                2,
            ),
            "msmt_minp": round(
                accuracy_row[
                    "msmt_minp"
                ],
                2,
            ),

            "market_rank1": round(
                accuracy_row[
                    "market_rank1"
                ],
                2,
            ),
            "market_rank5": round(
                accuracy_row[
                    "market_rank5"
                ],
                2,
            ),
            "market_rank10": round(
                accuracy_row[
                    "market_rank10"
                ],
                2,
            ),
            "market_map": round(
                accuracy_row[
                    "market_map"
                ],
                2,
            ),
            "market_minp": round(
                accuracy_row[
                    "market_minp"
                ],
                2,
            ),

            "mars_rank1": round(
                accuracy_row[
                    "mars_rank1"
                ],
                2,
            ),
            "mars_rank5": round(
                accuracy_row[
                    "mars_rank5"
                ],
                2,
            ),
            "mars_rank10": round(
                accuracy_row[
                    "mars_rank10"
                ],
                2,
            ),
            "mars_map": round(
                accuracy_row[
                    "mars_map"
                ],
                2,
            ),
            "mars_minp": round(
                accuracy_row[
                    "mars_minp"
                ],
                2,
            ),

            "batch1_latency_ms": round(
                efficiency_row[
                    "batch1_latency_ms"
                ],
                3,
            ),

            "batch16_latency_ms": round(
                efficiency_row[
                    "batch16_latency_ms"
                ],
                3,
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
                3,
            ),

            "baseline_vram_mb": round(
                efficiency_row[
                    "baseline_vram_mb"
                ],
                2,
            ),

            "peak_vram_batch1_mb": round(
                efficiency_row[
                    "peak_vram_batch1_mb"
                ],
                2,
            ),

            "peak_vram_batch16_mb": round(
                efficiency_row[
                    "peak_vram_batch16_mb"
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
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FINAL_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_markdown(
    rows: list[dict],
) -> None:
    lines: list[str] = []

    lines.append(
        "# SHAWAF FastReID DukeMTMC Final Benchmark"
    )

    lines.append("")

    lines.append(
        "Source training dataset: DukeMTMC"
    )

    lines.append("")

    lines.append(
        "Evaluation datasets: MSMT17_V2, "
        "Market1501 and MARS."
    )

    lines.append("")

    lines.append(
        "All three evaluation datasets are "
        "cross-domain relative to the DukeMTMC-trained "
        "checkpoints."
    )

    lines.append("")

    lines.append(
        "MSMT17_V2 note: one unreadable query image "
        "and one unreadable gallery image were skipped "
        "during this benchmark."
    )

    lines.append("")

    lines.append(
        "## Efficiency Hardware"
    )

    lines.append("")

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "Unavailable"
    )

    lines.append(
        f"- GPU: {gpu_name}"
    )

    if torch.cuda.is_available():
        properties = (
            torch.cuda.get_device_properties(0)
        )

        vram_gb = (
            properties.total_memory
            / (1024 ** 3)
        )

        lines.append(
            f"- GPU VRAM: {vram_gb:.2f} GB"
        )

    lines.append(
        f"- Python: {platform.python_version()}"
    )

    lines.append(
        f"- PyTorch: {torch.__version__}"
    )

    lines.append(
        f"- CUDA runtime: {torch.version.cuda}"
    )

    lines.append(
        f"- OS: {platform.platform()}"
    )

    lines.append("")

    lines.append(
        "Efficiency protocol: 10 warm-up iterations, "
        "50 measurement iterations, batch size 16, "
        "8-frame tracklet."
    )

    lines.append("")

    lines.append(
        "## Final Results"
    )

    lines.append("")

    lines.append(
        "| Model | Family | Backbone | "
        "MSMT R1 | MSMT mAP | "
        "Market R1 | Market mAP | "
        "MARS R1 | MARS mAP | "
        "B1 ms | Throughput | "
        "Tracklet ms | Peak VRAM B16 | Params |"
    )

    lines.append(
        "|---|---|---|"
        "---:|---:|"
        "---:|---:|"
        "---:|---:|"
        "---:|---:|"
        "---:|---:|---:|"
    )

    for row in rows:
        lines.append(
            "| "
            f"{row['model_id']} | "
            f"{row['family']} | "
            f"{row['backbone']} | "
            f"{row['msmt_rank1']:.2f}% | "
            f"{row['msmt_map']:.2f}% | "
            f"{row['market_rank1']:.2f}% | "
            f"{row['market_map']:.2f}% | "
            f"{row['mars_rank1']:.2f}% | "
            f"{row['mars_map']:.2f}% | "
            f"{row['batch1_latency_ms']:.3f} | "
            f"{row['throughput_images_per_second']:.2f} | "
            f"{row['tracklet_latency_ms']:.3f} | "
            f"{row['peak_vram_batch16_mb']:.2f} MB | "
            f"{row['parameters_m']:.3f}M |"
        )

    lines.append("")

    lines.append(
        "## Notes"
    )

    lines.append("")

    lines.append(
        "- MSMT17_V2 and Market1501 use "
        "`single_image_l2`."
    )

    lines.append(
        "- MARS uses `uniform8_mean_raw_l2`."
    )

    lines.append(
        "- MARS uses exactly 8 uniformly sampled frames "
        "per tracklet."
    )

    lines.append(
        "- Efficiency results are directly comparable "
        "only with results measured on the same "
        "Tesla T4 hardware cohort."
    )

    lines.append("")

    with OUTPUT_MARKDOWN.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                lines
            )
        )


def main() -> None:
    print(
        "=" * 96
    )

    print(
        "SHAWAF DUKEMTMC - "
        "FINAL BENCHMARK SUMMARY"
    )

    print(
        "=" * 96
    )

    rows = build_final_rows()

    if len(rows) != EXPECTED_MODELS:
        raise RuntimeError(
            "Unexpected final model count: "
            f"{len(rows)}"
        )

    write_csv(
        rows
    )

    write_markdown(
        rows
    )

    print()
    print(
        f"Models                  : "
        f"{len(rows)}"
    )

    print(
        f"Accuracy evaluations    : "
        f"{len(rows) * EXPECTED_DATASETS}"
    )

    print(
        f"Efficiency profiles     : "
        f"{len(rows)}"
    )

    print()

    print(
        f"CSV                     : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Markdown                : "
        f"{OUTPUT_MARKDOWN}"
    )

    print()

    print(
        "=" * 96
    )

    print(
        "FINAL SUMMARY COMPLETE"
    )

    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()