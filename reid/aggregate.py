from __future__ import annotations

import numpy as np


def mean_pool_l2(embeddings: np.ndarray) -> np.ndarray:
    """
    Aggregate frame embeddings into one L2-normalized tracklet embedding.

    embeddings:
        Array of shape (num_frames, dim).
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape (num_frames, dim), "
            f"got {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must contain at least one frame")

    pooled = embeddings.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm == 0:
        return pooled.astype(np.float32)

    return (pooled / norm).astype(np.float32)
