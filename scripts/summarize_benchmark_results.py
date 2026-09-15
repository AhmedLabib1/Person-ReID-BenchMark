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
    / "fastreid_msmt17_models.yaml"
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
    "market1501",
    "mars",
)


PROTOCOLS = {
    "market1501": "single_image_l2",
    "mars": "uniform8_mean_raw_l2",
}


DATASET_DISPLAY_NAMES = {
    "market1501": "Market1501",
    "mars": "MARS",
}


def load_registry() -> list[dict]:
    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(file)

    return registry["models"]


def result_path(
    model_id: str,
    dataset: str,
) -> Path:
    protocol = PROTOCOLS[
        dataset
    ]

    return (
        RESULT_ROOT
        / model_id
        / dataset
        / protocol
        / "metrics.json"
    )


def load_result(
    model_id: str,
    dataset: str,
) -> dict:
    path = result_path(
        model_id,
        dataset,
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

    if result["model_id"] != model_id:
        raise RuntimeError(
            f"Model mismatch in:\n{path}"
        )

    if result["dataset"] != dataset:
        raise RuntimeError(
            f"Dataset mismatch in:\n{path}"
        )

    expected_protocol = (
        PROTOCOLS[
            dataset
        ]
    )

    if (
        result["protocol"]
        != expected_protocol
    ):
        raise RuntimeError(
            "Protocol mismatch in:\n"
            f"{path}"
        )

    return result


def percentage(
    value: float | None,
) -> float | None:
    if value is None:
        return None

    return round(
        float(value) * 100.0,
        4,
    )


def build_rows(
    models: list[dict],
) -> list[dict]:
    rows: list[dict] = []

    for model in models:
        model_id = model["id"]

        for dataset in DATASETS:
            result = load_result(
                model_id,
                dataset,
            )

            row = {
                "model_id": model_id,
                "family": model["family"],
                "backbone": model["backbone"],

                "dataset": (
                    DATASET_DISPLAY_NAMES[
                        dataset
                    ]
                ),

                "protocol": (
                    result["protocol"]
                ),

                "query_samples": (
                    result["query_samples"]
                ),

                "gallery_samples": (
                    result[
                        "gallery_samples"
                    ]
                ),

                "valid_queries": (
                    result[
                        "valid_queries"
                    ]
                ),

                "skipped_queries": (
                    result[
                        "skipped_queries"
                    ]
                ),

                "embedding_dim": (
                    result[
                        "embedding_dimension"
                    ]
                ),

                "rank_1": percentage(
                    result["rank_1"]
                ),

                "rank_5": percentage(
                    result["rank_5"]
                ),

                "rank_10": percentage(
                    result["rank_10"]
                ),

                "mAP": percentage(
                    result["mAP"]
                ),

                "mINP": percentage(
                    result["mINP"]
                ),

                "evaluation_seconds": round(
                    float(
                        result[
                            "evaluation_seconds"
                        ]
                    ),
                    4,
                ),
            }

            rows.append(
                row
            )

    return rows


def write_csv(
    rows: list[dict],
) -> None:
    if not rows:
        raise RuntimeError(
            "No benchmark rows to save."
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def markdown_table(
    rows: list[dict],
) -> str:
    lines = [
        "| Model | Family | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['model_id']} | "
            f"{row['family']} | "
            f"{row['backbone']} | "
            f"{row['rank_1']:.2f}% | "
            f"{row['rank_5']:.2f}% | "
            f"{row['rank_10']:.2f}% | "
            f"{row['mAP']:.2f}% | "
            f"{row['mINP']:.2f}% |"
        )

    return "\n".join(
        lines
    )


def write_markdown(
    rows: list[dict],
) -> None:
    sections = [
        "# SHAWAF ReID Cross-Domain Benchmark",
        "",
        (
            "All FastReID checkpoints were trained "
            "on MSMT17 and evaluated without "
            "fine-tuning on Market1501 and MARS."
        ),
        "",
    ]

    for dataset in (
        "Market1501",
        "MARS",
    ):
        dataset_rows = [
            row
            for row in rows
            if row["dataset"] == dataset
        ]

        dataset_rows.sort(
            key=lambda row: (
                row["rank_1"],
                row["mAP"],
            ),
            reverse=True,
        )

        sections.extend(
            [
                f"## {dataset}",
                "",
                markdown_table(
                    dataset_rows
                ),
                "",
            ]
        )

    sections.extend(
        [
            "## Protocols",
            "",
            (
                "- Market1501: single-image "
                "embedding followed by L2 normalization."
            ),
            (
                "- MARS: 8 uniformly sampled frames "
                "per tracklet, raw-frame mean pooling, "
                "then final L2 normalization."
            ),
            "",
            (
                "MSMT17_V2 in-domain evaluation is "
                "excluded until a verified copy of "
                "the dataset is available."
            ),
            "",
        ]
    )

    with OUTPUT_MARKDOWN.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                sections
            )
        )


def print_dataset_table(
    rows: list[dict],
    dataset: str,
) -> None:
    dataset_rows = [
        row
        for row in rows
        if row["dataset"] == dataset
    ]

    dataset_rows.sort(
        key=lambda row: (
            row["rank_1"],
            row["mAP"],
        ),
        reverse=True,
    )

    print()
    print("=" * 110)
    print(
        f"{dataset.upper()} RESULTS"
    )
    print("=" * 110)

    print(
        f"{'Model':<18}"
        f"{'Family':<8}"
        f"{'Backbone':<12}"
        f"{'R1':>9}"
        f"{'R5':>9}"
        f"{'R10':>9}"
        f"{'mAP':>9}"
        f"{'mINP':>9}"
    )

    print("-" * 110)

    for row in dataset_rows:
        print(
            f"{row['model_id']:<18}"
            f"{row['family']:<8}"
            f"{row['backbone']:<12}"
            f"{row['rank_1']:>8.2f}%"
            f"{row['rank_5']:>8.2f}%"
            f"{row['rank_10']:>8.2f}%"
            f"{row['mAP']:>8.2f}%"
            f"{row['mINP']:>8.2f}%"
        )


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "Cross-Domain Results Summary"
    )
    print("=" * 88)

    models = load_registry()

    print()
    print(
        f"Models                  : "
        f"{len(models)}"
    )

    print(
        f"Datasets                : "
        f"{len(DATASETS)}"
    )

    expected_results = (
        len(models)
        * len(DATASETS)
    )

    print(
        f"Expected results        : "
        f"{expected_results}"
    )

    rows = build_rows(
        models
    )

    print(
        f"Loaded results          : "
        f"{len(rows)}"
    )

    if len(rows) != expected_results:
        raise RuntimeError(
            "Benchmark result count "
            "is incomplete."
        )

    write_csv(
        rows
    )

    write_markdown(
        rows
    )

    print_dataset_table(
        rows,
        "Market1501",
    )

    print_dataset_table(
        rows,
        "MARS",
    )

    print()
    print("=" * 88)
    print("SUMMARY COMPLETE")
    print("=" * 88)

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
        "All 24 cross-domain "
        "results were loaded successfully."
    )


if __name__ == "__main__":
    main()