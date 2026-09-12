from __future__ import annotations

from pathlib import Path

import numpy as np

from reid.data.tracklet import Tracklet


def sample_crop_paths(
    tracklet: Tracklet,
    num_frames: int = 8,
) -> list[Path]:
    """
    Uniformly sample up to ``num_frames`` crop paths from a tracklet.

    If the tracklet has fewer crops than ``num_frames``, every crop is used.
    """

    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    crop_paths = tracklet.crop_paths
    if not crop_paths:
        return []

    if len(crop_paths) <= num_frames:
        return list(crop_paths)

    indices = np.linspace(
        0,
        len(crop_paths) - 1,
        num=num_frames,
        dtype=np.int64,
    )
    unique_indices = np.unique(indices)
    return [crop_paths[int(index)] for index in unique_indices]
