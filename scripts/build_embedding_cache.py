
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
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


from reid.benchmark.embedding_cache import (
    CachedEmbeddings,
    extract_tracklet_cache,
)

from reid.data.contracts import (
    ImageSample,
    TrackletSample,
)

from reid.data.market1501 import (
    load_market1501,
)

from reid.data.mars import (
    load_mars,
)

from reid.data.msmt17 import (
    load_msmt17,
)

from reid.embeddings.fastreid_extractor import (
    FastReIDFeatureExtractor,
)

from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "embeddings"
)


# ============================================================
# Image split
# ============================================================

def build_image_split(
    *,
    model_id: str,
    dataset_name: str,
    split_name: str,
    samples: Sequence[ImageSample],
    extractor: FastReIDFeatureExtractor,
    batch_size: int,
    chunk_size: int,
) -> CachedEmbeddings:

    if not samples:
        raise ValueError(
            f"{dataset_name} {split_name} "
            "is empty."
        )

    embedding_chunks: list[
        np.ndarray
    ] = []

    total_chunks = (
        len(samples)
        + chunk_size
        - 1
    ) // chunk_size

    progress = tqdm(
        range(
            0,
            len(samples),
            chunk_size,
        ),
        total=total_chunks,
        desc=f"{dataset_name} {split_name}",
        unit="chunk",
    )

    for start in progress:

        end = min(
            start + chunk_size,
            len(samples),
        )

        chunk = samples[
            start:end
        ]

        image_paths = tuple(
            sample.image_path
            for sample in chunk
        )

        embeddings = (
            extractor.extract_paths(
                image_paths,
                batch_size=batch_size,
                normalize=True,
            )
        )

        embedding_chunks.append(
            embeddings
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        progress.set_postfix(
            processed=end,
            total=len(samples),
        )

    embeddings = np.concatenate(
        embedding_chunks,
        axis=0,
    )

    person_ids = np.asarray(
        [
            sample.person_id
            for sample in samples
        ],
        dtype=np.int64,
    )

    camera_ids = np.asarray(
        [
            sample.camera_id
            for sample in samples
        ],
        dtype=np.int64,
    )

    sample_ids = tuple(
        sample.sample_id
        for sample in samples
    )

    cache = CachedEmbeddings(
        model_id=model_id,
        dataset_name=dataset_name,
        split_name=split_name,
        unit_type="image",
        embeddings=embeddings,
        person_ids=person_ids,
        camera_ids=camera_ids,
        sample_ids=sample_ids,
    )

    cache.validate()

    return cache


# ============================================================
# Tracklet split
# ============================================================

def build_tracklet_split(
    *,
    model_id: str,
    dataset_name: str,
    split_name: str,
    samples: Sequence[TrackletSample],
    extractor: FastReIDFeatureExtractor,
    num_frames: int,
    frame_batch_size: int,
    outer_chunk_size: int,
) -> CachedEmbeddings:

    if not samples:
        raise ValueError(
            f"{dataset_name} {split_name} "
            "is empty."
        )

    embedding_chunks: list[
        np.ndarray
    ] = []

    total_chunks = (
        len(samples)
        + outer_chunk_size
        - 1
    ) // outer_chunk_size

    progress = tqdm(
        range(
            0,
            len(samples),
            outer_chunk_size,
        ),
        total=total_chunks,
        desc=f"{dataset_name} {split_name}",
        unit="chunk",
    )

    for start in progress:

        end = min(
            start + outer_chunk_size,
            len(samples),
        )

        chunk = samples[
            start:end
        ]

        chunk_cache = (
            extract_tracklet_cache(
                model_id=model_id,
                dataset_name=dataset_name,
                split_name=split_name,
                samples=chunk,
                extractor=extractor,
                num_frames=num_frames,
                frame_batch_size=(
                    frame_batch_size
                ),
                tracklet_chunk_size=(
                    outer_chunk_size
                ),
            )
        )

        embedding_chunks.append(
            chunk_cache.embeddings
        )

        progress.set_postfix(
            processed=end,
            total=len(samples),
        )

    embeddings = np.concatenate(
        embedding_chunks,
        axis=0,
    )

    person_ids = np.asarray(
        [
            sample.person_id
            for sample in samples
        ],
        dtype=np.int64,
    )

    camera_ids = np.asarray(
        [
            sample.camera_id
            for sample in samples
        ],
        dtype=np.int64,
    )

    sample_ids = tuple(
        sample.tracklet_id
        for sample in samples
    )

    cache = CachedEmbeddings(
        model_id=model_id,
        dataset_name=dataset_name,
        split_name=split_name,
        unit_type="tracklet",
        embeddings=embeddings,
        person_ids=person_ids,
        camera_ids=camera_ids,
        sample_ids=sample_ids,
    )

    cache.validate()

    return cache


# ============================================================
# Dataset helpers
# ============================================================

def get_dataset(
    dataset_key: str,
):

    if dataset_key == "market1501":
        return load_market1501()

    if dataset_key == "msmt17":
        return load_msmt17()

    if dataset_key == "mars":
        return load_mars()

    raise ValueError(
        f"Unsupported dataset: {dataset_key}"
    )


def get_protocol_name(
    dataset_key: str,
    num_frames: int,
) -> str:

    if dataset_key == "mars":
        return (
            f"uniform{num_frames}"
            "_mean_raw_l2"
        )

    return "single_image_l2"


def cache_path(
    *,
    model_id: str,
    dataset_key: str,
    protocol_name: str,
    split_name: str,
) -> Path:

    return (
        EMBEDDING_ROOT
        / model_id
        / dataset_key
        / protocol_name
        / f"{split_name}.npz"
    )


def save_split(
    *,
    cache: CachedEmbeddings,
    path: Path,
) -> None:

    cache.save(
        path
    )

    size_mb = (
        path.stat().st_size
        / 1024**2
    )

    print()
    print(
        "Saved                   :",
        path,
    )

    print(
        "Samples                 :",
        f"{cache.num_samples:,}",
    )

    print(
        "Embedding shape         :",
        cache.embeddings.shape,
    )

    print(
        "File size               :",
        f"{size_mb:.2f} MB",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build SHAWAF ReID embedding caches."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
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
        "--split",
        choices=[
            "query",
            "gallery",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--image-chunk-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--tracklet-chunk-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    protocol_name = (
        get_protocol_name(
            args.dataset,
            args.num_frames,
        )
    )

    print("=" * 90)
    print("SHAWAF DUKE - EMBEDDING CACHE BUILDER")
    print("=" * 90)

    print()
    print("Model      :", args.model)
    print("Dataset    :", args.dataset)
    print("Split      :", args.split)
    print("Protocol   :", protocol_name)
    print("Device     :", device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    adapter = FastReIDAdapter(
        model_id=args.model,
        device=device,
    )

    extractor = (
        FastReIDFeatureExtractor(
            adapter=adapter
        )
    )

    print(
        "Family     :",
        adapter.family,
    )

    print(
        "Backbone   :",
        adapter.backbone,
    )

    print(
        "Input size :",
        extractor.input_size,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Loading dataset...")

    dataset = get_dataset(
        args.dataset
    )

    print(
        "Dataset    :",
        dataset.name,
    )

    print(
        "Query      :",
        f"{dataset.num_query:,}",
    )

    print(
        "Gallery    :",
        f"{dataset.num_gallery:,}",
    )

    if args.split == "both":

        split_names = (
            "query",
            "gallery",
        )

    else:

        split_names = (
            args.split,
        )

    total_start = (
        time.perf_counter()
    )

    for split_name in split_names:

        samples = getattr(
            dataset,
            split_name,
        )

        output_path = cache_path(
            model_id=args.model,
            dataset_key=args.dataset,
            protocol_name=protocol_name,
            split_name=split_name,
        )

        print()
        print("=" * 90)
        print(
            f"BUILDING {split_name.upper()}"
        )
        print("=" * 90)

        if (
            output_path.exists()
            and not args.force
        ):
            print(
                "Cache already exists:"
            )
            print(
                output_path
            )
            print(
                "Use --force to rebuild."
            )
            continue

        start_time = (
            time.perf_counter()
        )

        if args.dataset == "mars":

            cache = (
                build_tracklet_split(
                    model_id=args.model,
                    dataset_name=dataset.name,
                    split_name=split_name,
                    samples=samples,
                    extractor=extractor,
                    num_frames=args.num_frames,
                    frame_batch_size=(
                        args.frame_batch_size
                    ),
                    outer_chunk_size=(
                        args.tracklet_chunk_size
                    ),
                )
            )

        else:

            cache = (
                build_image_split(
                    model_id=args.model,
                    dataset_name=dataset.name,
                    split_name=split_name,
                    samples=samples,
                    extractor=extractor,
                    batch_size=args.batch_size,
                    chunk_size=(
                        args.image_chunk_size
                    ),
                )
            )

        save_split(
            cache=cache,
            path=output_path,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            "Elapsed                 :",
            f"{elapsed:.2f} seconds",
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 90)
    print("CACHE BUILD COMPLETE")
    print("=" * 90)

    print(
        "Total elapsed:",
        f"{total_elapsed:.2f} seconds",
    )


if __name__ == "__main__":
    main()
