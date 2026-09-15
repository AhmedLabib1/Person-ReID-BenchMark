from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

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


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "embeddings"
)


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
    """
    Extract one image-based dataset split.

    Images are processed in chunks so that large datasets
    such as MSMT17 do not require all intermediate tensors
    to remain in memory at once.
    """

    if not samples:
        raise ValueError(
            f"{dataset_name} {split_name} is empty."
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

        embeddings_np = (
            embeddings
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        embedding_chunks.append(
            embeddings_np
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
    """
    Extract a complete MARS split with visible progress.

    Each outer chunk is independently converted into
    tracklet embeddings.

    Protocol:

        tracklet
            -> uniform N frames
            -> raw frame embeddings
            -> mean pooling
            -> L2 normalization
    """

    if not samples:
        raise ValueError(
            f"{dataset_name} {split_name} is empty."
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
        f"Saved                   : "
        f"{path}"
    )

    print(
        f"Samples                 : "
        f"{cache.num_samples:,}"
    )

    print(
        f"Embedding shape         : "
        f"{cache.embeddings.shape}"
    )

    print(
        f"File size               : "
        f"{size_mb:.2f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build SHAWAF ReID embedding caches."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model ID from the FastReID "
            "model registry."
        ),
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
        help=(
            "GPU image batch size for "
            "Market1501/MSMT17."
        ),
    )

    parser.add_argument(
        "--image-chunk-size",
        type=int,
        default=1024,
        help=(
            "Number of image samples processed "
            "before storing an intermediate "
            "embedding chunk in RAM."
        ),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help=(
            "Frames sampled per MARS tracklet."
        ),
    )

    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=16,
        help=(
            "GPU batch size for MARS frames."
        ),
    )

    parser.add_argument(
        "--tracklet-chunk-size",
        type=int,
        default=64,
        help=(
            "MARS tracklets processed per "
            "outer chunk."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing embedding caches."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0"
        )

    if args.image_chunk_size <= 0:
        raise ValueError(
            "--image-chunk-size must be > 0"
        )

    if args.num_frames <= 0:
        raise ValueError(
            "--num-frames must be > 0"
        )

    if args.frame_batch_size <= 0:
        raise ValueError(
            "--frame-batch-size must be > 0"
        )

    if args.tracklet_chunk_size <= 0:
        raise ValueError(
            "--tracklet-chunk-size must be > 0"
        )

    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "Full Embedding Cache Builder"
    )
    print("=" * 88)

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

    print()
    print("CONFIGURATION")
    print("-" * 88)

    print(
        f"Model                   : "
        f"{args.model}"
    )

    print(
        f"Dataset                 : "
        f"{args.dataset}"
    )

    print(
        f"Split                   : "
        f"{args.split}"
    )

    print(
        f"Protocol                : "
        f"{protocol_name}"
    )

    print(
        f"Device                  : "
        f"{device}"
    )

    if args.dataset == "mars":
        print(
            f"Frames / tracklet       : "
            f"{args.num_frames}"
        )

        print(
            f"Frame batch size        : "
            f"{args.frame_batch_size}"
        )

    else:
        print(
            f"Image batch size        : "
            f"{args.batch_size}"
        )

    print()
    print("LOADING MODEL")
    print("-" * 88)

    adapter = FastReIDAdapter(
        model_id=args.model,
        device=device,
    )

    extractor = (
        FastReIDFeatureExtractor(
            adapter=adapter,
        )
    )

    print(
        f"Family                  : "
        f"{adapter.family}"
    )

    print(
        f"Backbone                : "
        f"{adapter.backbone}"
    )

    print(
        f"Input size              : "
        f"{extractor.input_size}"
    )

    print()
    print("LOADING DATASET")
    print("-" * 88)

    dataset = get_dataset(
        args.dataset
    )

    print(
        f"Dataset name            : "
        f"{dataset.name}"
    )

    print(
        f"Query samples           : "
        f"{dataset.num_query:,}"
    )

    print(
        f"Gallery samples         : "
        f"{dataset.num_gallery:,}"
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

    total_start = time.perf_counter()

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
        print("=" * 88)
        print(
            f"BUILDING {split_name.upper()}"
        )
        print("=" * 88)

        if (
            output_path.exists()
            and not args.force
        ):
            print(
                "Cache already exists."
            )

            print(
                f"Path                    : "
                f"{output_path}"
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
                    batch_size=(
                        args.batch_size
                    ),
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
            f"Elapsed                 : "
            f"{elapsed:.2f} seconds"
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 88)
    print("CACHE BUILD COMPLETE")
    print("=" * 88)

    print(
        f"Total elapsed           : "
        f"{total_elapsed:.2f} seconds"
    )

    print(
        f"Root                    : "
        f"{EMBEDDING_ROOT}"
    )


if __name__ == "__main__":
    main()