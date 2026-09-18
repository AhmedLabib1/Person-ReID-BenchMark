from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

FIGURES_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "figures"
)

CMC_RESULT_ROOT = (
    RESULT_ROOT
    / "cmc"
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
    "msmt17": "In-Domain",
    "market1501": "Cross-Domain",
    "mars": "Cross-Domain Tracklet",
}


def load_model_ids() -> list[str]:
    """
    Load all FastReID benchmark model IDs
    from the central model registry.
    """

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Model registry not found:\n"
            f"{REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(
            file
        )

    models = registry.get(
        "models",
        []
    )

    if not models:
        raise RuntimeError(
            "Model registry contains no models."
        )

    model_ids = [
        model["id"]
        for model in models
    ]

    return model_ids


def metrics_path(
    *,
    model_id: str,
    dataset: str,
) -> Path:
    """
    Return the existing metrics.json path.

    Example:

        benchmarks/results/
            sbs_r50/
                msmt17/
                    single_image_l2/
                        metrics.json
    """

    protocol = (
        PROTOCOLS[
            dataset
        ]
    )

    return (
        RESULT_ROOT
        / model_id
        / dataset
        / protocol
        / "metrics.json"
    )


def load_metrics(
    *,
    model_id: str,
    dataset: str,
) -> dict:
    """
    Load one benchmark metrics.json file
    and validate its identity/protocol.
    """

    path = metrics_path(
        model_id=model_id,
        dataset=dataset,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Benchmark metrics file not found:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(
            file
        )

    if (
        metrics.get(
            "model_id"
        )
        != model_id
    ):
        raise RuntimeError(
            "Model mismatch in metrics file:\n"
            f"{path}"
        )

    if (
        metrics.get(
            "dataset"
        )
        != dataset
    ):
        raise RuntimeError(
            "Dataset mismatch in metrics file:\n"
            f"{path}"
        )

    expected_protocol = (
        PROTOCOLS[
            dataset
        ]
    )

    if (
        metrics.get(
            "protocol"
        )
        != expected_protocol
    ):
        raise RuntimeError(
            "Protocol mismatch in metrics file:\n"
            f"{path}\n"
            f"Expected: {expected_protocol}\n"
            f"Found   : "
            f"{metrics.get('protocol')}"
        )

    if "cmc" not in metrics:
        raise RuntimeError(
            "CMC data is missing from:\n"
            f"{path}"
        )

    return metrics


def validate_cmc(
    *,
    model_id: str,
    dataset: str,
    metrics: dict,
) -> np.ndarray:
    """
    Validate the CMC array stored during
    the original benchmark evaluation.

    We also verify Rank-1, Rank-5 and Rank-10
    against the scalar values in metrics.json.

    This protects the reporting stage from
    accidentally plotting inconsistent data.
    """

    cmc = np.asarray(
        metrics["cmc"],
        dtype=np.float64,
    )

    if cmc.ndim != 1:
        raise RuntimeError(
            "CMC must be a 1D array.\n"
            f"Model   : {model_id}\n"
            f"Dataset : {dataset}"
        )

    if len(cmc) < 10:
        raise RuntimeError(
            "CMC must contain at least "
            "10 ranks.\n"
            f"Model   : {model_id}\n"
            f"Dataset : {dataset}\n"
            f"Length  : {len(cmc)}"
        )

    if not np.all(
        np.isfinite(
            cmc
        )
    ):
        raise RuntimeError(
            "CMC contains non-finite values.\n"
            f"Model   : {model_id}\n"
            f"Dataset : {dataset}"
        )

    if np.any(
        cmc < 0.0
    ) or np.any(
        cmc > 1.0
    ):
        raise RuntimeError(
            "CMC values must be between "
            "0 and 1.\n"
            f"Model   : {model_id}\n"
            f"Dataset : {dataset}"
        )

    # --------------------------------------------------
    # A valid CMC should never decrease as rank grows.
    # --------------------------------------------------

    if np.any(
        np.diff(
            cmc
        )
        < -1e-12
    ):
        raise RuntimeError(
            "CMC curve is not monotonic.\n"
            f"Model   : {model_id}\n"
            f"Dataset : {dataset}"
        )

    checks = (
        (
            1,
            "rank_1",
        ),
        (
            5,
            "rank_5",
        ),
        (
            10,
            "rank_10",
        ),
    )

    for (
        rank,
        field,
    ) in checks:

        stored_value = float(
            metrics[
                field
            ]
        )

        cmc_value = float(
            cmc[
                rank - 1
            ]
        )

        if not np.isclose(
            stored_value,
            cmc_value,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "CMC/scalar metric mismatch.\n"
                f"Model   : {model_id}\n"
                f"Dataset : {dataset}\n"
                f"Metric  : {field}\n"
                f"Scalar  : {stored_value}\n"
                f"CMC     : {cmc_value}"
            )

    return cmc


def save_cmc_csv(
    *,
    model_id: str,
    dataset: str,
    cmc: np.ndarray,
) -> Path:
    """
    Save one model/dataset CMC curve as CSV.

    Output:

        benchmarks/results/cmc/
            <model_id>/
                <dataset>.csv
    """

    output_dir = (
        CMC_RESULT_ROOT
        / model_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / f"{dataset}.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "rank",
                "cmc",
                "cmc_percent",
            ]
        )

        for (
            rank,
            value,
        ) in enumerate(
            cmc,
            start=1,
        ):
            writer.writerow(
                [
                    rank,
                    float(
                        value
                    ),
                    float(
                        value
                        * 100.0
                    ),
                ]
            )

    return path


def plot_dataset_cmc(
    *,
    dataset: str,
    curves: dict[
        str,
        np.ndarray,
    ],
) -> Path:
    """
    Plot all 12 models on one CMC figure
    for a single dataset.
    """

    if not curves:
        raise RuntimeError(
            f"No CMC curves for {dataset}."
        )

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            8,
        )
    )

    # --------------------------------------------------
    # Plot models ordered by Rank-1.
    #
    # This also makes the legend easier to interpret.
    # --------------------------------------------------

    ordered_curves = sorted(
        curves.items(),
        key=lambda item: (
            item[1][0]
        ),
        reverse=True,
    )

    for (
        model_id,
        cmc,
    ) in ordered_curves:

        ranks = np.arange(
            1,
            len(cmc) + 1,
        )

        ax.plot(
            ranks,
            cmc * 100.0,
            linewidth=1.8,
            label=model_id,
        )

    dataset_name = (
        DISPLAY_NAMES[
            dataset
        ]
    )

    evaluation_type = (
        EVALUATION_TYPES[
            dataset
        ]
    )

    ax.set_title(
        (
            f"{dataset_name} "
            f"{evaluation_type} CMC Curves"
        ),
        fontsize=15,
        pad=14,
    )

    ax.set_xlabel(
        "Rank"
    )

    ax.set_ylabel(
        "Matching Rate (%)"
    )

    max_rank = max(
        len(cmc)
        for cmc in curves.values()
    )

    ax.set_xlim(
        1,
        max_rank,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xticks(
        [
            1,
            5,
            10,
            20,
            30,
            40,
            50,
        ]
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
        title="Model",
    )

    fig.tight_layout()

    path = (
        FIGURES_ROOT
        / f"{dataset}_cmc.png"
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


def print_rank_summary(
    *,
    dataset: str,
    curves: dict[
        str,
        np.ndarray,
    ],
) -> None:
    """
    Print a compact CMC summary for sanity checking.
    """

    ordered = sorted(
        curves.items(),
        key=lambda item: (
            item[1][0]
        ),
        reverse=True,
    )

    print()
    print(
        f"{DISPLAY_NAMES[dataset]} "
        f"CMC SUMMARY"
    )

    print("-" * 88)

    print(
        f"{'Model':<20}"
        f"{'Rank-1':>12}"
        f"{'Rank-5':>12}"
        f"{'Rank-10':>12}"
        f"{'Rank-50':>12}"
    )

    print("-" * 88)

    for (
        model_id,
        cmc,
    ) in ordered:

        rank_50_index = min(
            49,
            len(cmc) - 1,
        )

        print(
            f"{model_id:<20}"
            f"{cmc[0] * 100:>11.2f}%"
            f"{cmc[4] * 100:>11.2f}%"
            f"{cmc[9] * 100:>11.2f}%"
            f"{cmc[rank_50_index] * 100:>11.2f}%"
        )


def main() -> None:
    model_ids = (
        load_model_ids()
    )

    print("=" * 100)

    print(
        "SHAWAF ReID - "
        "Final CMC Curve Generator"
    )

    print("=" * 100)

    print()

    print(
        f"Models                  : "
        f"{len(model_ids)}"
    )

    print(
        f"Datasets                : "
        f"{len(DATASETS)}"
    )

    print(
        f"Expected CMC results    : "
        f"{len(model_ids) * len(DATASETS)}"
    )

    print(
        "Source                  : "
        "existing metrics.json files"
    )

    print(
        "Model inference         : "
        "not required"
    )

    print(
        "Embedding evaluation    : "
        "not repeated"
    )

    all_curves: dict[
        str,
        dict[
            str,
            np.ndarray,
        ],
    ] = {
        dataset: {}
        for dataset in DATASETS
    }

    total_jobs = (
        len(model_ids)
        * len(DATASETS)
    )

    job_number = 0

    for dataset in DATASETS:
        print()
        print("=" * 100)

        print(
            f"{DISPLAY_NAMES[dataset]} "
            f"({EVALUATION_TYPES[dataset]})"
        )

        print("=" * 100)

        for model_id in model_ids:
            job_number += 1

            print(
                f"[{job_number:02d}/{total_jobs:02d}] "
                f"{model_id:<18} "
                f"{dataset:<12}",
                end="",
            )

            metrics = load_metrics(
                model_id=model_id,
                dataset=dataset,
            )

            cmc = validate_cmc(
                model_id=model_id,
                dataset=dataset,
                metrics=metrics,
            )

            all_curves[
                dataset
            ][
                model_id
            ] = cmc

            output_csv = save_cmc_csv(
                model_id=model_id,
                dataset=dataset,
                cmc=cmc,
            )

            print(
                "  OK"
            )

            if not output_csv.exists():
                raise RuntimeError(
                    "CMC CSV was not created:\n"
                    f"{output_csv}"
                )

    print()
    print("=" * 100)
    print("GENERATING CMC FIGURES")
    print("=" * 100)

    generated_figures: list[
        Path
    ] = []

    for dataset in DATASETS:
        print_rank_summary(
            dataset=dataset,
            curves=all_curves[
                dataset
            ],
        )

        figure_path = (
            plot_dataset_cmc(
                dataset=dataset,
                curves=all_curves[
                    dataset
                ],
            )
        )

        generated_figures.append(
            figure_path
        )

        print()

        print(
            f"Saved figure            : "
            f"{figure_path}"
        )

    print()
    print("=" * 100)
    print("CMC GENERATION COMPLETE")
    print("=" * 100)

    print(
        f"Models                  : "
        f"{len(model_ids)}"
    )

    print(
        f"Datasets                : "
        f"{len(DATASETS)}"
    )

    print(
        f"CMC CSV files           : "
        f"{total_jobs}"
    )

    print(
        f"CMC figures             : "
        f"{len(generated_figures)}"
    )

    print(
        f"CSV root                : "
        f"{CMC_RESULT_ROOT}"
    )

    print(
        f"Figure root             : "
        f"{FIGURES_ROOT}"
    )

    print()

    print(
        "Generated figures:"
    )

    for path in generated_figures:
        print(
            f"  - {path.name}"
        )


if __name__ == "__main__":
    main()