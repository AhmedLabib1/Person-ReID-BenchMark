from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_msmt17_models.yaml"
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
    "market1501",
    "mars",
)


PROTOCOLS = {
    "market1501": "single_image_l2",
    "mars": "uniform8_mean_raw_l2",
}


DISPLAY_NAMES = {
    "market1501": "Market1501",
    "mars": "MARS",
}


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


def cache_path(
    model_id: str,
    dataset: str,
    split: str,
) -> Path:
    return (
        EMBEDDING_ROOT
        / model_id
        / dataset
        / PROTOCOLS[dataset]
        / f"{split}.npz"
    )


def find_array(
    data: np.lib.npyio.NpzFile,
    names: tuple[str, ...],
) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(
                data[name]
            )

    raise KeyError(
        "Could not find any of these keys:\n"
        f"{names}\n"
        f"Available keys: {data.files}"
    )


def load_cache(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found:\n{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        embeddings = find_array(
            data,
            (
                "embeddings",
                "features",
            ),
        ).astype(
            np.float32,
            copy=False,
        )

        person_ids = find_array(
            data,
            (
                "person_ids",
                "pids",
                "person_id",
            ),
        ).astype(
            np.int64,
            copy=False,
        )

        camera_ids = find_array(
            data,
            (
                "camera_ids",
                "camids",
                "camera_id",
            ),
        ).astype(
            np.int64,
            copy=False,
        )

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Embeddings must be 2D: {path}"
        )

    count = embeddings.shape[0]

    if (
        len(person_ids) != count
        or len(camera_ids) != count
    ):
        raise RuntimeError(
            f"Metadata length mismatch: {path}"
        )

    return (
        embeddings,
        person_ids,
        camera_ids,
    )


def compute_cmc(
    *,
    query_embeddings: np.ndarray,
    query_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int,
    query_batch_size: int,
    device: str,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    """
    Standard ReID CMC evaluation.

    For every query:
      1. Rank gallery by cosine similarity.
      2. Remove same-PID + same-camera samples.
      3. Skip query if no valid positive remains.
      4. Accumulate CMC up to max_rank.

    Embeddings are already L2 normalized,
    so dot product == cosine similarity.
    """

    gallery_tensor = torch.from_numpy(
        gallery_embeddings
    ).to(
        device=device,
        dtype=torch.float32,
    )

    cmc_sum = np.zeros(
        max_rank,
        dtype=np.float64,
    )

    valid_queries = 0
    skipped_queries = 0

    total_queries = len(
        query_embeddings
    )

    for start in range(
        0,
        total_queries,
        query_batch_size,
    ):
        end = min(
            start + query_batch_size,
            total_queries,
        )

        query_tensor = torch.from_numpy(
            query_embeddings[start:end]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            similarities = (
                query_tensor
                @ gallery_tensor.T
            )

            rankings = torch.argsort(
                similarities,
                dim=1,
                descending=True,
            )

        rankings = rankings.cpu().numpy()

        for local_index, ranked_indices in enumerate(
            rankings
        ):
            query_index = (
                start
                + local_index
            )

            q_pid = query_pids[
                query_index
            ]

            q_camid = query_camids[
                query_index
            ]

            ranked_pids = gallery_pids[
                ranked_indices
            ]

            ranked_camids = gallery_camids[
                ranked_indices
            ]

            # Standard ReID exclusion:
            # same person + same camera.
            remove = (
                (ranked_pids == q_pid)
                & (ranked_camids == q_camid)
            )

            keep = ~remove

            matches = (
                ranked_pids[keep]
                == q_pid
            ).astype(
                np.int32
            )

            if not np.any(matches):
                skipped_queries += 1
                continue

            valid_queries += 1

            cmc = np.cumsum(
                matches
            )

            cmc[cmc > 1] = 1

            if len(cmc) >= max_rank:
                cmc_rank = cmc[
                    :max_rank
                ]

            else:
                cmc_rank = np.empty(
                    max_rank,
                    dtype=np.float64,
                )

                cmc_rank[
                    :len(cmc)
                ] = cmc

                cmc_rank[
                    len(cmc):
                ] = cmc[-1]

            cmc_sum += cmc_rank

    if valid_queries == 0:
        raise RuntimeError(
            "No valid queries were found."
        )

    final_cmc = (
        cmc_sum
        / valid_queries
    )

    return (
        final_cmc,
        valid_queries,
        skipped_queries,
    )


def save_cmc_csv(
    *,
    model_id: str,
    dataset: str,
    cmc: np.ndarray,
) -> Path:
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

        for rank, value in enumerate(
            cmc,
            start=1,
        ):
            writer.writerow(
                [
                    rank,
                    float(value),
                    float(value * 100.0),
                ]
            )

    return path


def plot_dataset_cmc(
    *,
    dataset: str,
    curves: dict[str, np.ndarray],
) -> Path:
    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(12, 8)
    )

    for model_id, cmc in curves.items():
        ranks = np.arange(
            1,
            len(cmc) + 1,
        )

        plt.plot(
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

    plt.title(
        f"{dataset_name} Cross-Domain CMC Curves",
        fontsize=15,
        pad=14,
    )

    plt.xlabel(
        "Rank"
    )

    plt.ylabel(
        "Matching rate (%)"
    )

    plt.xlim(
        1,
        max(
            len(cmc)
            for cmc in curves.values()
        ),
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    path = (
        FIGURES_ROOT
        / f"{dataset}_cmc.png"
    )

    plt.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    return path


def main() -> None:
    max_rank = 50
    query_batch_size = 128

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_ids = (
        load_model_ids()
    )

    print("=" * 100)
    print(
        "SHAWAF ReID - "
        "Full CMC Curve Generator"
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
        f"Maximum rank            : "
        f"{max_rank}"
    )

    print(
        f"Query batch size        : "
        f"{query_batch_size}"
    )

    print(
        f"Device                  : "
        f"{device}"
    )

    all_curves: dict[
        str,
        dict[str, np.ndarray],
    ] = {
        dataset: {}
        for dataset in DATASETS
    }

    total_jobs = (
        len(model_ids)
        * len(DATASETS)
    )

    job = 0

    for model_id in model_ids:
        for dataset in DATASETS:
            job += 1

            print()
            print("=" * 100)

            print(
                f"[{job:02d}/{total_jobs:02d}] "
                f"{model_id} / {dataset}"
            )

            print("=" * 100)

            query_path = cache_path(
                model_id,
                dataset,
                "query",
            )

            gallery_path = cache_path(
                model_id,
                dataset,
                "gallery",
            )

            (
                query_embeddings,
                query_pids,
                query_camids,
            ) = load_cache(
                query_path
            )

            (
                gallery_embeddings,
                gallery_pids,
                gallery_camids,
            ) = load_cache(
                gallery_path
            )

            print(
                f"Query embeddings        : "
                f"{query_embeddings.shape}"
            )

            print(
                f"Gallery embeddings      : "
                f"{gallery_embeddings.shape}"
            )

            (
                cmc,
                valid_queries,
                skipped_queries,
            ) = compute_cmc(
                query_embeddings=query_embeddings,
                query_pids=query_pids,
                query_camids=query_camids,
                gallery_embeddings=gallery_embeddings,
                gallery_pids=gallery_pids,
                gallery_camids=gallery_camids,
                max_rank=max_rank,
                query_batch_size=query_batch_size,
                device=device,
            )

            output_csv = save_cmc_csv(
                model_id=model_id,
                dataset=dataset,
                cmc=cmc,
            )

            all_curves[
                dataset
            ][
                model_id
            ] = cmc

            print(
                f"Valid queries           : "
                f"{valid_queries:,}"
            )

            print(
                f"Skipped queries         : "
                f"{skipped_queries:,}"
            )

            print(
                f"Rank-1                  : "
                f"{cmc[0] * 100:.2f}%"
            )

            print(
                f"Rank-5                  : "
                f"{cmc[4] * 100:.2f}%"
            )

            print(
                f"Rank-10                 : "
                f"{cmc[9] * 100:.2f}%"
            )

            print(
                f"Rank-20                 : "
                f"{cmc[19] * 100:.2f}%"
            )

            print(
                f"Rank-50                 : "
                f"{cmc[49] * 100:.2f}%"
            )

            print(
                f"Saved                   : "
                f"{output_csv}"
            )

    print()
    print("=" * 100)
    print("GENERATING CMC FIGURES")
    print("=" * 100)

    for dataset in DATASETS:
        path = plot_dataset_cmc(
            dataset=dataset,
            curves=all_curves[
                dataset
            ],
        )

        print(
            f"Saved                   : "
            f"{path}"
        )

    print()
    print("=" * 100)
    print("CMC GENERATION COMPLETE")
    print("=" * 100)

    print(
        f"CMC result root         : "
        f"{CMC_RESULT_ROOT}"
    )

    print(
        f"Figures root            : "
        f"{FIGURES_ROOT}"
    )


if __name__ == "__main__":
    main()