
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

FASTREID_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "fast-reid"
)

for path in [
    PROJECT_ROOT,
    FASTREID_ROOT,
]:
    path_text = str(path)

    if path_text not in sys.path:
        sys.path.insert(
            0,
            path_text,
        )


# ============================================================
# Imports after project paths are configured
# ============================================================

from reid.benchmark.embedding_cache import (
    CachedEmbeddings,
)

from reid.evaluation.rank import (
    ReIDMetricAccumulator,
)


# ============================================================
# Paths
# ============================================================

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


# ============================================================
# Protocols
# ============================================================

def default_protocol(
    dataset: str,
) -> str:

    if dataset in {
        "market1501",
        "msmt17",
    }:
        return (
            "single_image_l2"
        )

    if dataset == "mars":
        return (
            "uniform8_mean_raw_l2"
        )

    raise ValueError(
        f"Unsupported dataset: {dataset}"
    )


# ============================================================
# Cache paths
# ============================================================

def get_cache_paths(
    *,
    model_id: str,
    dataset: str,
    protocol: str,
) -> tuple[Path, Path]:

    root = (
        EMBEDDING_ROOT
        / model_id
        / dataset
        / protocol
    )

    return (
        root / "query.npz",
        root / "gallery.npz",
    )


# ============================================================
# Cache pair validation
# ============================================================

def validate_cache_pair(
    query: CachedEmbeddings,
    gallery: CachedEmbeddings,
) -> None:

    if (
        query.model_id
        != gallery.model_id
    ):
        raise RuntimeError(
            "Query/gallery model mismatch."
        )

    if (
        query.dataset_name
        != gallery.dataset_name
    ):
        raise RuntimeError(
            "Query/gallery dataset mismatch."
        )

    if (
        query.embedding_dim
        != gallery.embedding_dim
    ):
        raise RuntimeError(
            "Query/gallery embedding "
            "dimension mismatch."
        )

    if query.split_name != "query":
        raise RuntimeError(
            "Unexpected query split name."
        )

    if gallery.split_name != "gallery":
        raise RuntimeError(
            "Unexpected gallery split name."
        )

    if (
        query.unit_type
        != gallery.unit_type
    ):
        raise RuntimeError(
            "Query/gallery unit type mismatch."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cached SHAWAF ReID embeddings."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model ID.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "market1501",
            "msmt17",
            "mars",
        ],
    )

    parser.add_argument(
        "--protocol",
        default=None,
        help=(
            "Optional protocol directory. "
            "If omitted, use dataset default."
        ),
    )

    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-rank",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

    if args.query_batch_size <= 0:
        raise ValueError(
            "--query-batch-size must be > 0."
        )

    if args.max_rank <= 0:
        raise ValueError(
            "--max-rank must be > 0."
        )

    protocol = (
        args.protocol
        if args.protocol is not None
        else default_protocol(
            args.dataset
        )
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device == "auto":

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    # --------------------------------------------------------
    # Cache paths
    # --------------------------------------------------------

    (
        query_path,
        gallery_path,
    ) = get_cache_paths(
        model_id=args.model,
        dataset=args.dataset,
        protocol=protocol,
    )

    print("=" * 90)
    print(
        "SHAWAF DUKE - "
        "CACHED EMBEDDING EVALUATION"
    )
    print("=" * 90)

    print()
    print(
        "Model       :",
        args.model,
    )

    print(
        "Dataset     :",
        args.dataset,
    )

    print(
        "Protocol    :",
        protocol,
    )

    print(
        "Device      :",
        device,
    )

    print(
        "Query batch :",
        args.query_batch_size,
    )

    print(
        "Max rank    :",
        args.max_rank,
    )

    # ========================================================
    # Load caches
    # ========================================================

    print()
    print("Loading caches...")

    query_cache = (
        CachedEmbeddings.load(
            query_path
        )
    )

    gallery_cache = (
        CachedEmbeddings.load(
            gallery_path
        )
    )

    validate_cache_pair(
        query_cache,
        gallery_cache,
    )

    print(
        "Query shape :",
        query_cache.embeddings.shape,
    )

    print(
        "Gallery     :",
        gallery_cache.embeddings.shape,
    )

    print(
        "Dimension   :",
        query_cache.embedding_dim,
    )

    print(
        "Unit type   :",
        query_cache.unit_type,
    )

    # ========================================================
    # Move gallery embeddings to compute device once
    # ========================================================

    gallery_embeddings = (
        torch.from_numpy(
            gallery_cache.embeddings
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    accumulator = (
        ReIDMetricAccumulator(
            max_rank=args.max_rank
        )
    )

    num_queries = (
        query_cache.num_samples
    )

    num_batches = (
        num_queries
        + args.query_batch_size
        - 1
    ) // args.query_batch_size

    # ========================================================
    # Evaluation
    # ========================================================

    print()
    print("=" * 90)
    print("EVALUATION")
    print("=" * 90)

    print(
        "Similarity   : cosine / dot product"
    )

    print(
        "Embeddings   : already L2 normalized"
    )

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = (
        time.perf_counter()
    )

    progress = tqdm(
        range(
            0,
            num_queries,
            args.query_batch_size,
        ),
        total=num_batches,
        desc="Ranking queries",
        unit="batch",
    )

    for start in progress:

        end = min(
            start
            + args.query_batch_size,
            num_queries,
        )

        query_embeddings = (
            torch.from_numpy(
                query_cache.embeddings[
                    start:end
                ]
            )
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        # ----------------------------------------------------
        # Embeddings are already L2-normalized:
        #
        # cosine_similarity(q, g)
        #
        # =
        #
        # dot_product(q, g)
        # ----------------------------------------------------

        similarities = (
            query_embeddings
            @ gallery_embeddings.T
        )

        rankings = torch.argsort(
            similarities,
            dim=1,
            descending=True,
        )

        rankings = (
            rankings
            .cpu()
            .numpy()
        )

        for local_index in range(
            end - start
        ):

            query_index = (
                start
                + local_index
            )

            accumulator.update(
                ranked_gallery_indices=(
                    rankings[
                        local_index
                    ]
                ),

                query_pid=int(
                    query_cache.person_ids[
                        query_index
                    ]
                ),

                query_camera_id=int(
                    query_cache.camera_ids[
                        query_index
                    ]
                ),

                gallery_pids=(
                    gallery_cache.person_ids
                ),

                gallery_camera_ids=(
                    gallery_cache.camera_ids
                ),
            )

        progress.set_postfix(
            processed=end,
            valid=(
                accumulator.valid_queries
            ),
            skipped=(
                accumulator.skipped_queries
            ),
        )

        del query_embeddings
        del similarities
        del rankings

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    metrics = (
        accumulator.compute()
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)

    print(
        f"Rank-1   : "
        f"{metrics.rank(1) * 100:.2f}%"
    )

    if args.max_rank >= 5:

        print(
            f"Rank-5   : "
            f"{metrics.rank(5) * 100:.2f}%"
        )

    if args.max_rank >= 10:

        print(
            f"Rank-10  : "
            f"{metrics.rank(10) * 100:.2f}%"
        )

    print(
        f"mAP      : "
        f"{metrics.mean_ap * 100:.2f}%"
    )

    print(
        f"mINP     : "
        f"{metrics.mean_inp * 100:.2f}%"
    )

    print()
    print(
        "Total queries   :",
        f"{metrics.total_queries:,}",
    )

    print(
        "Valid queries   :",
        f"{metrics.valid_queries:,}",
    )

    print(
        "Skipped queries :",
        f"{metrics.skipped_queries:,}",
    )

    print(
        "Evaluation time :",
        f"{elapsed:.2f} s",
    )

    # ========================================================
    # Save metrics
    # ========================================================

    output_dir = (
        RESULT_ROOT
        / args.model
        / args.dataset
        / protocol
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "metrics.json"
    )

    result_data = {
        "model_id": (
            args.model
        ),

        "dataset": (
            args.dataset
        ),

        "dataset_name": (
            query_cache.dataset_name
        ),

        "protocol": (
            protocol
        ),

        "similarity": (
            "cosine_dot_product_l2_normalized"
        ),

        "embedding_dimension": (
            query_cache.embedding_dim
        ),

        "query_samples": (
            query_cache.num_samples
        ),

        "gallery_samples": (
            gallery_cache.num_samples
        ),

        "valid_queries": (
            metrics.valid_queries
        ),

        "skipped_queries": (
            metrics.skipped_queries
        ),

        "rank_1": (
            metrics.rank(1)
        ),

        "rank_5": (
            metrics.rank(5)
            if args.max_rank >= 5
            else None
        ),

        "rank_10": (
            metrics.rank(10)
            if args.max_rank >= 10
            else None
        ),

        "mAP": (
            metrics.mean_ap
        ),

        "mINP": (
            metrics.mean_inp
        ),

        "max_rank": (
            args.max_rank
        ),

        "cmc": [
            float(value)
            for value
            in metrics.cmc
        ],

        "evaluation_device": (
            device
        ),

        "evaluation_seconds": (
            elapsed
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_data,
            file,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_path,
    )

    print()
    print("=" * 90)
    print(
        "ReID evaluation completed successfully."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
