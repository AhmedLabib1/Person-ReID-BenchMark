from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reid.data.contracts import (
    ImageSample,
    TrackletSample,
)
from reid.data.sampling import (
    uniform_sample_frames,
)
from reid.embeddings.fastreid_extractor import (
    FastReIDFeatureExtractor,
)


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedEmbeddings:
    """
    One cached ReID embedding split.

    Example:

        model:
            sbs_r50_ibn

        dataset:
            Market1501

        split:
            query

        embeddings:
            [3368, 2048]

        person_ids:
            [3368]

        camera_ids:
            [3368]

        sample_ids:
            3368 strings
    """

    model_id: str
    dataset_name: str
    split_name: str
    unit_type: str

    embeddings: np.ndarray
    person_ids: np.ndarray
    camera_ids: np.ndarray

    sample_ids: tuple[str, ...]

    def validate(self) -> None:
        """
        Validate shape, metadata alignment,
        numerical validity and L2 normalization.
        """

        if self.embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape "
                "[N, D]."
            )

        num_samples = (
            self.embeddings.shape[0]
        )

        if num_samples == 0:
            raise ValueError(
                "Embedding cache cannot be empty."
            )

        if (
            len(self.person_ids)
            != num_samples
        ):
            raise ValueError(
                "person_ids length does not "
                "match embeddings."
            )

        if (
            len(self.camera_ids)
            != num_samples
        ):
            raise ValueError(
                "camera_ids length does not "
                "match embeddings."
            )

        if (
            len(self.sample_ids)
            != num_samples
        ):
            raise ValueError(
                "sample_ids length does not "
                "match embeddings."
            )

        if not np.isfinite(
            self.embeddings
        ).all():
            raise ValueError(
                "Embeddings contain NaN or Inf."
            )

        if not np.issubdtype(
            self.person_ids.dtype,
            np.integer,
        ):
            raise ValueError(
                "person_ids must be integers."
            )

        if not np.issubdtype(
            self.camera_ids.dtype,
            np.integer,
        ):
            raise ValueError(
                "camera_ids must be integers."
            )

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                "Cached embeddings are not "
                "L2 normalized."
            )

    @property
    def num_samples(self) -> int:
        return int(
            self.embeddings.shape[0]
        )

    @property
    def embedding_dim(self) -> int:
        return int(
            self.embeddings.shape[1]
        )

    def save(
        self,
        path: Path,
    ) -> None:
        """
        Save cache as a NumPy NPZ file.

        No pickle is required.
        """

        self.validate()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            path,
            schema_version=np.asarray(
                CACHE_SCHEMA_VERSION,
                dtype=np.int64,
            ),
            model_id=np.asarray(
                self.model_id
            ),
            dataset_name=np.asarray(
                self.dataset_name
            ),
            split_name=np.asarray(
                self.split_name
            ),
            unit_type=np.asarray(
                self.unit_type
            ),
            embeddings=self.embeddings.astype(
                np.float32,
                copy=False,
            ),
            person_ids=self.person_ids.astype(
                np.int64,
                copy=False,
            ),
            camera_ids=self.camera_ids.astype(
                np.int64,
                copy=False,
            ),
            sample_ids=np.asarray(
                self.sample_ids,
                dtype=np.str_,
            ),
        )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "CachedEmbeddings":
        """
        Load and validate an embedding cache.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                "Embedding cache not found:\n"
                f"{path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            schema_version = int(
                data[
                    "schema_version"
                ].item()
            )

            if (
                schema_version
                != CACHE_SCHEMA_VERSION
            ):
                raise RuntimeError(
                    "Unsupported cache schema "
                    f"version: {schema_version}"
                )

            cache = cls(
                model_id=str(
                    data[
                        "model_id"
                    ].item()
                ),
                dataset_name=str(
                    data[
                        "dataset_name"
                    ].item()
                ),
                split_name=str(
                    data[
                        "split_name"
                    ].item()
                ),
                unit_type=str(
                    data[
                        "unit_type"
                    ].item()
                ),
                embeddings=np.asarray(
                    data[
                        "embeddings"
                    ],
                    dtype=np.float32,
                ),
                person_ids=np.asarray(
                    data[
                        "person_ids"
                    ],
                    dtype=np.int64,
                ),
                camera_ids=np.asarray(
                    data[
                        "camera_ids"
                    ],
                    dtype=np.int64,
                ),
                sample_ids=tuple(
                    str(value)
                    for value in data[
                        "sample_ids"
                    ].tolist()
                ),
            )

        cache.validate()

        return cache


def extract_image_cache(
    *,
    model_id: str,
    dataset_name: str,
    split_name: str,
    samples: Sequence[ImageSample],
    extractor: FastReIDFeatureExtractor,
    batch_size: int = 32,
) -> CachedEmbeddings:
    """
    Extract and cache one image-based split.

    Used by:
        - Market1501
        - MSMT17
    """

    if not samples:
        raise ValueError(
            "Image sample list cannot be empty."
        )

    image_paths = tuple(
        sample.image_path
        for sample in samples
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
        embeddings=embeddings_np,
        person_ids=person_ids,
        camera_ids=camera_ids,
        sample_ids=sample_ids,
    )

    cache.validate()

    return cache


def extract_tracklet_cache(
    *,
    model_id: str,
    dataset_name: str,
    split_name: str,
    samples: Sequence[TrackletSample],
    extractor: FastReIDFeatureExtractor,
    num_frames: int = 8,
    frame_batch_size: int = 32,
    tracklet_chunk_size: int = 64,
) -> CachedEmbeddings:
    """
    Efficiently extract one MARS split.

    Instead of doing:

        one GPU forward per tracklet

    we process tracklets in chunks.

    For each chunk:

        tracklets
            ↓
        sample N frames each
            ↓
        flatten all frame paths
            ↓
        batched FastReID inference
            ↓
        [tracklets, frames, D]
            ↓
        mean pooling
            ↓
        L2 normalization
    """

    if not samples:
        raise ValueError(
            "Tracklet sample list cannot be empty."
        )

    if num_frames <= 0:
        raise ValueError(
            "num_frames must be greater than 0."
        )

    if frame_batch_size <= 0:
        raise ValueError(
            "frame_batch_size must be "
            "greater than 0."
        )

    if tracklet_chunk_size <= 0:
        raise ValueError(
            "tracklet_chunk_size must be "
            "greater than 0."
        )

    embedding_chunks: list[
        torch.Tensor
    ] = []

    for start in range(
        0,
        len(samples),
        tracklet_chunk_size,
    ):
        end = min(
            start + tracklet_chunk_size,
            len(samples),
        )

        chunk = samples[
            start:end
        ]

        flattened_paths: list[
            Path
        ] = []

        for sample in chunk:
            sampled_frames = (
                uniform_sample_frames(
                    frame_paths=(
                        sample.frame_paths
                    ),
                    num_samples=num_frames,
                )
            )

            flattened_paths.extend(
                sampled_frames
            )

        frame_embeddings = (
            extractor.extract_paths(
                flattened_paths,
                batch_size=frame_batch_size,
                normalize=False,
            )
        )

        expected_frames = (
            len(chunk)
            * num_frames
        )

        if (
            frame_embeddings.shape[0]
            != expected_frames
        ):
            raise RuntimeError(
                "Unexpected number of MARS "
                "frame embeddings."
            )

        embedding_dim = (
            frame_embeddings.shape[1]
        )

        # [T * F, D]
        #
        # ->
        #
        # [T, F, D]
        frame_embeddings = (
            frame_embeddings.reshape(
                len(chunk),
                num_frames,
                embedding_dim,
            )
        )

        # Mean pool raw frame features.
        tracklet_embeddings = (
            frame_embeddings.mean(
                dim=1,
            )
        )

        # Final tracklet normalization.
        tracklet_embeddings = F.normalize(
            tracklet_embeddings,
            p=2,
            dim=1,
        )

        embedding_chunks.append(
            tracklet_embeddings
        )

    embeddings = torch.cat(
        embedding_chunks,
        dim=0,
    )

    embeddings_np = (
        embeddings
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
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
        embeddings=embeddings_np,
        person_ids=person_ids,
        camera_ids=camera_ids,
        sample_ids=sample_ids,
    )

    cache.validate()

    return cache