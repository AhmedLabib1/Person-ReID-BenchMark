
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reid.data.contracts import (
    ImageSample,
    TrackletSample,
)
from reid.data.sampling import (
    DEFAULT_TRACKLET_FRAMES,
    uniform_sample_frames,
)
from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


class FastReIDFeatureExtractor:
    """
    Unified feature extraction interface for FastReID models.

    Responsibilities:
        1. Read person images from disk.
        2. Apply FastReID inference preprocessing.
        3. Extract image embeddings.
        4. Extract MARS frame embeddings.
        5. Aggregate frame embeddings into tracklet embeddings.

    Preprocessing:

        OpenCV image
            -> BGR to RGB
            -> resize to model SIZE_TEST
            -> float32
            -> CHW tensor

    Important:
        Pixel values remain in the 0..255 range.

        FastReID performs pixel normalization internally
        using MODEL.PIXEL_MEAN and MODEL.PIXEL_STD.
    """

    def __init__(
        self,
        adapter: FastReIDAdapter,
    ) -> None:

        self.adapter = adapter

    # ========================================================
    # Input size
    # ========================================================

    @property
    def input_size(
        self,
    ) -> tuple[int, int]:
        """
        Return:

            (height, width)
        """

        height, width = (
            self.adapter.input_size
        )

        return (
            int(height),
            int(width),
        )

    # ========================================================
    # Preprocessing
    # ========================================================

    def preprocess_path(
        self,
        image_path: Path,
    ) -> torch.Tensor:
        """
        Read and preprocess one image.

        Returns:
            CPU Tensor [3, H, W]

        dtype:
            float32

        Pixel range:
            0..255
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                "Image does not exist:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # OpenCV loads BGR
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise RuntimeError(
                "OpenCV failed to read image:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = (
            self.input_size
        )

        # OpenCV uses:
        #
        #   (width, height)
        #
        image = cv2.resize(
            image,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        # ----------------------------------------------------
        # HWC uint8
        #      ->
        # CHW float32
        # ----------------------------------------------------

        image = np.ascontiguousarray(
            image.transpose(
                2,
                0,
                1,
            ),
            dtype=np.float32,
        )

        tensor = torch.from_numpy(
            image
        )

        expected_shape = (
            3,
            height,
            width,
        )

        if tuple(
            tensor.shape
        ) != expected_shape:

            raise RuntimeError(
                "Unexpected preprocessed "
                "image shape:\n"
                f"Got:      {tuple(tensor.shape)}\n"
                f"Expected: {expected_shape}"
            )

        if tensor.dtype != torch.float32:

            raise RuntimeError(
                "Expected float32 tensor."
            )

        return tensor

    # ========================================================
    # Generic path extraction
    # ========================================================

    def extract_paths(
        self,
        image_paths: Sequence[Path],
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract embeddings for arbitrary image paths.

        Args:
            image_paths:
                Image paths.

            batch_size:
                Number of images per forward pass.

            normalize:
                If True, row-wise final L2 normalization
                is applied after extraction.

        Returns:
            CPU Tensor [N, D]
        """

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than 0."
            )

        if not image_paths:

            raise ValueError(
                "image_paths cannot be empty."
            )

        paths = tuple(
            Path(path)
            for path in image_paths
        )

        feature_batches: list[
            torch.Tensor
        ] = []

        for start in range(
            0,
            len(paths),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(paths),
            )

            batch_paths = paths[
                start:end
            ]

            # ------------------------------------------------
            # [B, 3, H, W]
            # ------------------------------------------------

            images = torch.stack(
                [
                    self.preprocess_path(
                        path
                    )
                    for path
                    in batch_paths
                ],
                dim=0,
            )

            # ------------------------------------------------
            # RAW FastReID features
            # ------------------------------------------------

            features = (
                self.adapter.encode(
                    images
                )
            )

            if features.ndim != 2:

                raise RuntimeError(
                    "Expected FastReID output "
                    "shape [B, D], got "
                    f"{tuple(features.shape)}"
                )

            if (
                features.shape[0]
                != len(batch_paths)
            ):

                raise RuntimeError(
                    "FastReID returned an "
                    "unexpected batch size."
                )

            # Move caches/results to CPU.
            features = (
                features
                .detach()
                .float()
                .cpu()
            )

            feature_batches.append(
                features
            )

        embeddings = torch.cat(
            feature_batches,
            dim=0,
        )

        # ----------------------------------------------------
        # Image protocol:
        #
        # raw feature
        #     ->
        # final L2
        # ----------------------------------------------------

        if normalize:

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
            )

        return embeddings

    # ========================================================
    # Image-based ReID
    # ========================================================

    def extract_image(
        self,
        sample: ImageSample,
    ) -> torch.Tensor:
        """
        Image protocol:

            image
              ->
            FastReID
              ->
            raw embedding
              ->
            final L2

        Returns:
            Tensor [D]
        """

        embeddings = (
            self.extract_paths(
                [
                    sample.image_path
                ],
                batch_size=1,
                normalize=True,
            )
        )

        return embeddings[0]

    # ========================================================
    # Tracklet-based ReID
    # ========================================================

    def extract_tracklet(
        self,
        sample: TrackletSample,
        *,
        num_frames: int = DEFAULT_TRACKLET_FRAMES,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Frozen SHAWAF MARS protocol:

            complete tracklet
                  ↓
            deterministic uniform-N
                  ↓
            raw frame embeddings
                  ↓
            arithmetic mean
                  ↓
            final L2
                  ↓
            one tracklet embedding

        IMPORTANT:
            Individual frame embeddings are NOT
            L2-normalized before mean pooling.

        Returns:
            Tensor [D]
        """

        if num_frames <= 0:

            raise ValueError(
                "num_frames must be greater than 0."
            )

        sampled_frames = (
            uniform_sample_frames(
                frame_paths=(
                    sample.frame_paths
                ),
                num_samples=num_frames,
            )
        )

        if batch_size is None:
            batch_size = num_frames

        # ----------------------------------------------------
        # RAW frame embeddings.
        #
        # DO NOT normalize per frame here.
        # ----------------------------------------------------

        frame_embeddings = (
            self.extract_paths(
                sampled_frames,
                batch_size=batch_size,
                normalize=False,
            )
        )

        # ----------------------------------------------------
        # [N, D]
        #   ->
        # mean
        #   ->
        # [1, D]
        # ----------------------------------------------------

        tracklet_embedding = (
            frame_embeddings.mean(
                dim=0,
                keepdim=True,
            )
        )

        # ----------------------------------------------------
        # Final tracklet L2 only
        # ----------------------------------------------------

        tracklet_embedding = F.normalize(
            tracklet_embedding,
            p=2,
            dim=1,
        )

        return tracklet_embedding[0]
