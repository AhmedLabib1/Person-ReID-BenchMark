from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from reid.benchmark.embedding_cache import (
    CachedEmbeddings,
    extract_image_cache,
    extract_tracklet_cache,
)
from reid.data.market1501 import (
    load_market1501,
)
from reid.data.mars import (
    load_mars,
)
from reid.embeddings.fastreid_extractor import (
    FastReIDFeatureExtractor,
)
from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SMOKE_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "embeddings"
    / "_smoke"
)

MODEL_ID = "sbs_r50_ibn"


def validate_round_trip(
    original: CachedEmbeddings,
    loaded: CachedEmbeddings,
) -> None:
    if (
        original.model_id
        != loaded.model_id
    ):
        raise RuntimeError(
            "Model ID changed after "
            "cache round trip."
        )

    if (
        original.dataset_name
        != loaded.dataset_name
    ):
        raise RuntimeError(
            "Dataset name changed after "
            "cache round trip."
        )

    if (
        original.split_name
        != loaded.split_name
    ):
        raise RuntimeError(
            "Split name changed after "
            "cache round trip."
        )

    if (
        original.sample_ids
        != loaded.sample_ids
    ):
        raise RuntimeError(
            "Sample IDs changed after "
            "cache round trip."
        )

    if not np.array_equal(
        original.person_ids,
        loaded.person_ids,
    ):
        raise RuntimeError(
            "Person IDs changed after "
            "cache round trip."
        )

    if not np.array_equal(
        original.camera_ids,
        loaded.camera_ids,
    ):
        raise RuntimeError(
            "Camera IDs changed after "
            "cache round trip."
        )

    if not np.array_equal(
        original.embeddings,
        loaded.embeddings,
    ):
        raise RuntimeError(
            "Embeddings changed after "
            "cache round trip."
        )


def print_cache(
    title: str,
    cache: CachedEmbeddings,
    path: Path,
) -> None:
    norms = np.linalg.norm(
        cache.embeddings,
        axis=1,
    )

    file_size_mb = (
        path.stat().st_size
        / 1024**2
    )

    print()
    print(title)
    print("-" * 88)

    print(
        f"Model                   : "
        f"{cache.model_id}"
    )

    print(
        f"Dataset                 : "
        f"{cache.dataset_name}"
    )

    print(
        f"Split                   : "
        f"{cache.split_name}"
    )

    print(
        f"Unit type               : "
        f"{cache.unit_type}"
    )

    print(
        f"Samples                 : "
        f"{cache.num_samples}"
    )

    print(
        f"Embedding dimension     : "
        f"{cache.embedding_dim}"
    )

    print(
        f"Embedding shape         : "
        f"{cache.embeddings.shape}"
    )

    print(
        f"Min norm                : "
        f"{norms.min():.6f}"
    )

    print(
        f"Max norm                : "
        f"{norms.max():.6f}"
    )

    print(
        f"Cache size              : "
        f"{file_size_mb:.2f} MB"
    )

    print(
        f"Cache path              : "
        f"{path}"
    )


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "Embedding Cache Smoke Test"
    )
    print("=" * 88)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("MODEL")
    print("-" * 88)

    adapter = FastReIDAdapter(
        model_id=MODEL_ID,
        device=device,
    )

    extractor = (
        FastReIDFeatureExtractor(
            adapter=adapter,
        )
    )

    print(
        f"Model                   : "
        f"{MODEL_ID}"
    )

    print(
        f"Device                  : "
        f"{device}"
    )

    print(
        f"Input size              : "
        f"{extractor.input_size}"
    )

    # ==================================================
    # Market1501 smoke cache
    # ==================================================

    market = load_market1501()

    market_samples = (
        market.query[:16]
    )

    market_cache = (
        extract_image_cache(
            model_id=MODEL_ID,
            dataset_name=market.name,
            split_name="query",
            samples=market_samples,
            extractor=extractor,
            batch_size=8,
        )
    )

    market_path = (
        SMOKE_ROOT
        / (
            f"{MODEL_ID}"
            "_market1501"
            "_query_16.npz"
        )
    )

    market_cache.save(
        market_path
    )

    loaded_market = (
        CachedEmbeddings.load(
            market_path
        )
    )

    validate_round_trip(
        market_cache,
        loaded_market,
    )

    print_cache(
        "MARKET1501 CACHE",
        loaded_market,
        market_path,
    )

    print(
        "Round trip              : PASS"
    )

    # ==================================================
    # MARS smoke cache
    # ==================================================

    mars = load_mars()

    mars_samples = (
        mars.query[:4]
    )

    mars_cache = (
        extract_tracklet_cache(
            model_id=MODEL_ID,
            dataset_name=mars.name,
            split_name="query",
            samples=mars_samples,
            extractor=extractor,
            num_frames=8,
            frame_batch_size=8,
            tracklet_chunk_size=4,
        )
    )

    mars_path = (
        SMOKE_ROOT
        / (
            f"{MODEL_ID}"
            "_mars"
            "_query_4.npz"
        )
    )

    mars_cache.save(
        mars_path
    )

    loaded_mars = (
        CachedEmbeddings.load(
            mars_path
        )
    )

    validate_round_trip(
        mars_cache,
        loaded_mars,
    )

    print_cache(
        "MARS CACHE",
        loaded_mars,
        mars_path,
    )

    print(
        "Round trip              : PASS"
    )

    # ==================================================
    # Metadata check
    # ==================================================

    print()
    print("METADATA CHECK")
    print("-" * 88)

    print(
        f"Market first PID        : "
        f"{loaded_market.person_ids[0]}"
    )

    print(
        f"Market first camera     : "
        f"{loaded_market.camera_ids[0]}"
    )

    print(
        f"Market first sample     : "
        f"{loaded_market.sample_ids[0]}"
    )

    print()

    print(
        f"MARS first PID          : "
        f"{loaded_mars.person_ids[0]}"
    )

    print(
        f"MARS first camera       : "
        f"{loaded_mars.camera_ids[0]}"
    )

    print(
        f"MARS first tracklet     : "
        f"{loaded_mars.sample_ids[0]}"
    )

    print()
    print("=" * 88)
    print(
        "Embedding cache smoke test: SUCCESS"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()