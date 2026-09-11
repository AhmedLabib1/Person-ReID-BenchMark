from __future__ import annotations

from typing import Any

import numpy as np


def cosine_distance(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
) -> np.ndarray:
    """
    Cosine distance matrix of shape (num_query, num_gallery).

    Features are L2-normalized before the dot product.
    """

    query = _l2_normalize(query_features)
    gallery = _l2_normalize(gallery_features)
    similarity = query @ gallery.T
    return (1.0 - similarity).astype(np.float32)


def evaluate_rank(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int = 50,
) -> dict[str, float]:
    """
    Market1501-style CMC / mAP.

    Gallery samples that share both person ID and camera ID with the
    query are discarded. Rank-1 / Rank-5 / Rank-10 and mAP are returned
    as percentages.
    """

    cmc, all_ap, all_inp = _eval_market1501(
        distmat=distmat,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        max_rank=max_rank,
    )

    return {
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "Rank-10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "mAP": float(np.mean(all_ap) * 100.0),
        "mINP": float(np.mean(all_inp) * 100.0),
        "num_valid_queries": float(len(all_ap)),
    }


def camera_id_to_int(camera_id: str) -> int:
    """Convert SHAWAF camera labels such as CAM_01 into integers."""

    if camera_id.upper().startswith("CAM_"):
        return int(camera_id.split("_", 1)[1])
    return int(camera_id)


def format_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        "ReID evaluation",
        f"  Rank-1  : {metrics['Rank-1']:.2f}",
        f"  Rank-5  : {metrics['Rank-5']:.2f}",
        f"  Rank-10 : {metrics['Rank-10']:.2f}",
        f"  mAP     : {metrics['mAP']:.2f}",
        f"  mINP    : {metrics['mINP']:.2f}",
        f"  valid Q : {int(metrics['num_valid_queries'])}",
    ]
    return "\n".join(lines)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return features / norms


def _eval_market1501(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int,
) -> tuple[np.ndarray, list[float], list[float]]:
    num_query, num_gallery = distmat.shape
    if num_gallery < max_rank:
        max_rank = num_gallery

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(
        np.int32
    )

    all_cmc: list[np.ndarray] = []
    all_ap: list[float] = []
    all_inp: list[float] = []
    num_valid_q = 0.0

    for q_idx in range(num_query):
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]
        order = indices[q_idx]
        remove = (gallery_pids[order] == q_pid) & (
            gallery_camids[order] == q_camid
        )
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)[0]
        max_pos_idx = int(np.max(pos_idx))
        all_inp.append(float(cmc[max_pos_idx] / (max_pos_idx + 1.0)))

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = float(raw_cmc.sum())
        tmp_cmc = raw_cmc.cumsum().astype(np.float64)
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        all_ap.append(float(tmp_cmc.sum() / num_rel))

    if num_valid_q == 0:
        raise RuntimeError(
            "All query identities are missing from the gallery."
        )

    all_cmc_arr = np.asarray(all_cmc, dtype=np.float32)
    cmc = all_cmc_arr.sum(axis=0) / num_valid_q
    return cmc, all_ap, all_inp
