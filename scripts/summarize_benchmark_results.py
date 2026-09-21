from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_dukemtmc_models.yaml"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)

OUTPUT_CSV = (
    RESULT_ROOT
    / "cross_domain_summary.csv"
)

OUTPUT_MARKDOWN = (
    RESULT_ROOT
    / "cross_domain_summary.md"
)


DATASETS = (
    "msmt17",
    "market1501",
    "mars",
)


PROTOCOLS = {
    "msmt17": "single_image_l2",
    "market1501": "single_image_l2",
    "mars": "uniform8_mean_raw_l2",
}


DISPLAY_NAMES = {
    "msmt17": "MSMT17_V2",
    "market1501": "Market1501",
    "mars": "MARS",
}


EVALUATION_TYPES = {
    "msmt17": "cross-domain",
    "market1501": "cross-domain",
    "mars": "cross-domain tracklet",
}


CSV_FIELDS = (
    "model_id",
    "family",
    "backbone",
    "source_dataset",
    "dataset",
    "evaluation_type",
    "protocol",
    "rank_1",
    "rank_5",
    "rank_10",
    "mAP",
    "mINP",
    "query_samples",
    "gallery_samples",
    "valid_queries",
    "skipped_queries",
    "embedding_dimension",
)


def load_registry() -> tuple[str, list[dict]]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Model registry not found:\n"
            f"{REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(file)

    models = registry.get(
        "models",
        [],
    )

    if not models:
        raise RuntimeError(
            "Model registry contains no models."
        )

    source_dataset = registry.get(
        "source_dataset",
        "DukeMTMC",
    )

    return source_dataset, models


def result_path(
    model_id: str,
    dataset: str,
) -> Path:
    return (
        RESULT_ROOT
        / model_id
        / dataset
        / PROTOCOLS[dataset]
        / "metrics.json"
    )


def load_result(
    model_id: str,
    dataset: str,
) -> dict:
    path = result_path(
        model_id=model_id,
        dataset=dataset,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Benchmark result is missing:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        result = json.load(file)

    expected_protocol = PROTOCOLS[
        dataset
    ]

    if (
        result.get("model_id")
        != model_id
    ):
        raise RuntimeError(
            "Model ID mismatch in:\n"
            f"{path}"
        )

    if (
        result.get("protocol")
        != expected_protocol
    ):
        raise RuntimeError(
            "Protocol mismatch in:\n"
            f"{path}"
        )

    return result


def build_rows() -> list[dict]:
    source_dataset, models = (
        load_registry()
    )

    rows: list[dict] = []

    for model in models:
        model_id = model[
            "id"
        ]

        family = model[
            "family"
        ]

        backbone = model[
            "backbone"
        ]

        for dataset in DATASETS:
            result = load_result(
                model_id=model_id,
                dataset=dataset,
            )

            row = {
                "model_id": model_id,
                "family": family,
                "backbone": backbone,
                "source_dataset": (
                    source_dataset
                ),
                "dataset": (
                    DISPLAY_NAMES[
                        dataset
                    ]
                ),
                "evaluation_type": (
                    EVALUATION_TYPES[
                        dataset
                    ]
                ),
                "protocol": (
                    PROTOCOLS[
                        dataset
                    ]
                ),
                "rank_1": (
                    float(
                        result[
                            "rank_1"
                        ]
                    )
                    * 100.0
                ),
                "rank_5": (
                    float(
                        result[
                            "rank_5"
                        ]
                    )
                    * 100.0
                ),
                "rank_10": (
                    float(
                        result[
                            "rank_10"
                        ]
                    )
                    * 100.0
                ),
                "mAP": (
                    float(
                        result[
                            "mAP"
                        ]
                    )
                    * 100.0
                ),
                "mINP": (
                    float(
                        result[
                            "mINP"
                        ]
                    )
                    * 100.0
                ),
                "query_samples": (
                    int(
                        result[
                            "query_samples"
                        ]
                    )
                ),
                "gallery_samples": (
                    int(
                        result[
                            "gallery_samples"
                        ]
                    )
                ),
                "valid_queries": (
                    int(
                        result[
                            "valid_queries"
                        ]
                    )
                ),
                "skipped_queries": (
                    int(
                        result[
                            "skipped_queries"
                        ]
                    )
                ),
                "embedding_dimension": (
                    int(
                        result[
                            "embedding_dimension"
                        ]
                    )
                ),
            }

            rows.append(
                row
            )

    return rows


def write_csv(
    rows: list[dict],
) -> None:
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def percent(
    value: float,
) -> str:
    return f"{value:.2f}%"


def write_markdown(
    rows: list[dict],
) -> None:
    lines: list[str] = []

    lines.append(
        "# SHAWAF ReID DukeMTMC Cross-Domain Benchmark"
    )

    lines.append("")

    lines.append(
        "All FastReID checkpoints in this benchmark "
        "were trained on DukeMTMC."
    )

    lines.append("")

    lines.append(
        "MSMT17_V2, Market1501 and MARS are evaluated "
        "cross-domain without target-dataset fine-tuning."
    )

    lines.append("")

    lines.append(
        "MSMT17_V2 contains a minor documented dataset "
        "deviation in this run: one unreadable query image "
        "and one unreadable gallery image were skipped."
    )

    lines.append("")

    for dataset_key in DATASETS:
        dataset_name = (
            DISPLAY_NAMES[
                dataset_key
            ]
        )

        dataset_rows = [
            row
            for row in rows
            if row[
                "dataset"
            ]
            == dataset_name
        ]

        dataset_rows.sort(
            key=lambda row: (
                row[
                    "rank_1"
                ]
            ),
            reverse=True,
        )

        lines.append(
            f"## {dataset_name}"
        )

        lines.append("")

        lines.append(
            "| Model | Family | Backbone | "
            "Rank-1 | Rank-5 | Rank-10 | "
            "mAP | mINP |"
        )

        lines.append(
            "|---|---|---|---:|---:|---:|---:|---:|"
        )

        for row in dataset_rows:
            lines.append(
                "| "
                f"{row['model_id']} | "
                f"{row['family']} | "
                f"{row['backbone']} | "
                f"{percent(row['rank_1'])} | "
                f"{percent(row['rank_5'])} | "
                f"{percent(row['rank_10'])} | "
                f"{percent(row['mAP'])} | "
                f"{percent(row['mINP'])} |"
            )

        lines.append("")

    lines.append(
        "## Protocols"
    )

    lines.append("")

    lines.append(
        "- MSMT17_V2: single-image embedding -> "
        "L2 normalization (cross-domain)."
    )

    lines.append(
        "- Market1501: single-image embedding -> "
        "L2 normalization (cross-domain)."
    )

    lines.append(
        "- MARS: uniform 8-frame sampling -> "
        "raw-feature mean pooling -> final L2 "
        "normalization (cross-domain tracklet)."
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
        "=" * 88
    )

    print(
        "SHAWAF DUKEMTMC CROSS-DOMAIN "
        "BENCHMARK SUMMARY"
    )

    print(
        "=" * 88
    )

    source_dataset, models = (
        load_registry()
    )

    rows = build_rows()

    expected_rows = (
        len(models)
        * len(DATASETS)
    )

    if (
        len(rows)
        != expected_rows
    ):
        raise RuntimeError(
            "Unexpected number of "
            f"benchmark rows: {len(rows)} "
            f"(expected {expected_rows})"
        )

    write_csv(
        rows
    )

    write_markdown(
        rows
    )

    print()
    print(
        f"Source dataset          : "
        f"{source_dataset}"
    )

    print(
        f"Models                  : "
        f"{len(models)}"
    )

    print(
        f"Datasets                : "
        f"{len(DATASETS)}"
    )

    print(
        f"Accuracy rows           : "
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
        "=" * 88
    )

    print(
        "SUMMARY GENERATION COMPLETE"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()