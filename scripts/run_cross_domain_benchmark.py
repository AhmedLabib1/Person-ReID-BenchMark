from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_msmt17_models.yaml"
)

BUILD_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "build_embedding_cache.py"
)

EVALUATE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "evaluate_embedding_cache.py"
)

EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "embeddings"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)


DEFAULT_DATASETS = (
    "msmt17",
    "market1501",
    "mars",
)


def load_model_ids() -> list[str]:
    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(file)

    return [
        model["id"]
        for model in registry["models"]
    ]


def protocol_for_dataset(
    dataset: str,
) -> str:
    if dataset in {
        "msmt17",
        "market1501",
    }:
        return "single_image_l2"

    if dataset == "mars":
        return "uniform8_mean_raw_l2"

    raise ValueError(
        f"Unsupported dataset: {dataset}"
    )


def evaluation_type(
    dataset: str,
) -> str:
    if dataset == "msmt17":
        return "in-domain"

    return "cross-domain"


def cache_complete(
    model_id: str,
    dataset: str,
) -> bool:
    root = (
        EMBEDDING_ROOT
        / model_id
        / dataset
        / protocol_for_dataset(
            dataset
        )
    )

    query_path = (
        root
        / "query.npz"
    )

    gallery_path = (
        root
        / "gallery.npz"
    )

    return (
        query_path.exists()
        and gallery_path.exists()
    )


def result_exists(
    model_id: str,
    dataset: str,
) -> bool:
    path = (
        RESULT_ROOT
        / model_id
        / dataset
        / protocol_for_dataset(
            dataset
        )
        / "metrics.json"
    )

    return path.exists()


def run_command(
    command: list[str],
) -> None:
    print()
    print(
        "$ "
        + " ".join(
            command
        )
    )
    print()

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def build_cache(
    *,
    model_id: str,
    dataset: str,
    force: bool,
) -> None:
    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--model",
        model_id,
        "--dataset",
        dataset,
    ]

    if force:
        command.append(
            "--force"
        )

    run_command(
        command
    )


def evaluate(
    *,
    model_id: str,
    dataset: str,
) -> None:
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--model",
        model_id,
        "--dataset",
        dataset,
    ]

    run_command(
        command
    )


def format_seconds(
    seconds: float,
) -> str:
    total = int(
        round(
            seconds
        )
    )

    hours, remainder = divmod(
        total,
        3600,
    )

    minutes, secs = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours}h "
            f"{minutes}m "
            f"{secs}s"
        )

    if minutes:
        return (
            f"{minutes}m "
            f"{secs}s"
        )

    return f"{secs}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SHAWAF FastReID "
            "in-domain + cross-domain benchmark."
        )
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Optional model IDs. "
            "If omitted, all registry models "
            "are benchmarked."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=[
            "msmt17",
            "market1501",
            "mars",
        ],
        default=list(
            DEFAULT_DATASETS
        ),
    )

    parser.add_argument(
        "--force-cache",
        action="store_true",
        help=(
            "Rebuild query/gallery caches "
            "even if they already exist."
        ),
    )

    parser.add_argument(
        "--force-evaluation",
        action="store_true",
        help=(
            "Re-run evaluation even if "
            "metrics.json already exists."
        ),
    )

    args = parser.parse_args()

    registry_models = (
        load_model_ids()
    )

    if args.models:
        unknown = [
            model
            for model in args.models
            if model not in registry_models
        ]

        if unknown:
            raise ValueError(
                "Unknown model IDs: "
                + ", ".join(
                    unknown
                )
            )

        model_ids = (
            args.models
        )

    else:
        model_ids = (
            registry_models
        )

    datasets = tuple(
        args.datasets
    )

    total_jobs = (
        len(model_ids)
        * len(datasets)
    )

    print("=" * 88)

    print(
        "SHAWAF ReID - "
        "In-Domain + Cross-Domain "
        "Benchmark Runner"
    )

    print("=" * 88)

    print(
        f"Models                  : "
        f"{len(model_ids)}"
    )

    print(
        f"Datasets                : "
        f"{', '.join(datasets)}"
    )

    print(
        f"Model-dataset jobs      : "
        f"{total_jobs}"
    )

    completed_jobs = 0
    skipped_caches = 0
    skipped_results = 0

    benchmark_start = (
        time.perf_counter()
    )

    for model_id in model_ids:
        for dataset in datasets:
            job_number = (
                completed_jobs
                + 1
            )

            print()
            print("=" * 88)

            print(
                f"JOB "
                f"{job_number}/"
                f"{total_jobs}"
            )

            print("=" * 88)

            print(
                f"Model                   : "
                f"{model_id}"
            )

            print(
                f"Dataset                 : "
                f"{dataset}"
            )

            print(
                f"Evaluation type         : "
                f"{evaluation_type(dataset)}"
            )

            print(
                f"Protocol                : "
                f"{protocol_for_dataset(dataset)}"
            )

            job_start = (
                time.perf_counter()
            )

            # ==========================================
            # Embedding cache
            # ==========================================

            has_cache = (
                cache_complete(
                    model_id,
                    dataset,
                )
            )

            if (
                has_cache
                and not args.force_cache
            ):
                print(
                    "Embedding cache         : "
                    "EXISTS -> SKIP"
                )

                skipped_caches += 1

            else:
                print(
                    "Embedding cache         : "
                    "BUILD"
                )

                build_cache(
                    model_id=model_id,
                    dataset=dataset,
                    force=(
                        args.force_cache
                    ),
                )

                if not cache_complete(
                    model_id,
                    dataset,
                ):
                    raise RuntimeError(
                        "Embedding cache build "
                        "finished but query/gallery "
                        "files are incomplete."
                    )

            # ==========================================
            # Evaluation
            # ==========================================

            has_result = (
                result_exists(
                    model_id,
                    dataset,
                )
            )

            if (
                has_result
                and not args.force_evaluation
            ):
                print(
                    "Evaluation result       : "
                    "EXISTS -> SKIP"
                )

                skipped_results += 1

            else:
                print(
                    "Evaluation result       : "
                    "RUN"
                )

                evaluate(
                    model_id=model_id,
                    dataset=dataset,
                )

                if not result_exists(
                    model_id,
                    dataset,
                ):
                    raise RuntimeError(
                        "Evaluation finished but "
                        "metrics.json was not created."
                    )

            completed_jobs += 1

            job_elapsed = (
                time.perf_counter()
                - job_start
            )

            print(
                f"Job elapsed             : "
                f"{format_seconds(job_elapsed)}"
            )

            print(
                f"Progress                : "
                f"{completed_jobs}/"
                f"{total_jobs}"
            )

    total_elapsed = (
        time.perf_counter()
        - benchmark_start
    )

    print()
    print("=" * 88)
    print("BENCHMARK COMPLETE")
    print("=" * 88)

    print(
        f"Jobs processed          : "
        f"{completed_jobs}/"
        f"{total_jobs}"
    )

    print(
        f"Caches skipped          : "
        f"{skipped_caches}"
    )

    print(
        f"Results skipped         : "
        f"{skipped_results}"
    )

    print(
        f"Total elapsed           : "
        f"{format_seconds(total_elapsed)}"
    )

    print(
        f"Embeddings root         : "
        f"{EMBEDDING_ROOT}"
    )

    print(
        f"Results root            : "
        f"{RESULT_ROOT}"
    )


if __name__ == "__main__":
    main()