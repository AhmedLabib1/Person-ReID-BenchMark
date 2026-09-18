from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from reid.aggregate import mean_pool_l2
from reid.data.tracklet import Tracklet
from reid.metrics import camera_id_to_int, evaluate_rank_from_features
from reid.profiling import (
    gpu_memory_snapshot,
    merge_extract_stats,
    model_footprint,
    reset_peak_gpu_memory,
)
from reid.sampling import sample_crop_paths


def embed_tracklets(
    tracklets: list[Tracklet],
    encoder: Any,
    num_frames: int,
    batch_size: int,
    split_name: str,
    warmup_batches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    pids: list[int] = []
    camids: list[int] = []

    crop_paths: list[Path] = []
    owners: list[int] = []

    for tracklet_index, tracklet in enumerate(tracklets):
        sampled = sample_crop_paths(tracklet, num_frames=num_frames)
        for path in sampled:
            if not Path(path).is_file():
                continue
            crop_paths.append(path)
            owners.append(tracklet_index)

    features = encoder.extract(
        crop_paths,
        batch_size=batch_size,
        show_progress=True,
        warmup_batches=warmup_batches,
    )
    if features.shape[0] != len(crop_paths):
        raise RuntimeError(
            f"{split_name}: extracted {features.shape[0]} features "
            f"for {len(crop_paths)} sampled crops."
        )

    grouped: dict[int, list[np.ndarray]] = {
        index: [] for index in range(len(tracklets))
    }
    for owner, feature in zip(owners, features):
        grouped[owner].append(feature)

    skipped = 0
    for index, tracklet in enumerate(
        tqdm(tracklets, desc=f"Aggregate {split_name}", unit="trk")
    ):
        frame_features = grouped[index]
        if not frame_features or tracklet.person_id is None:
            skipped += 1
            continue
        embeddings.append(mean_pool_l2(np.stack(frame_features, axis=0)))
        pids.append(int(tracklet.person_id))
        camids.append(camera_id_to_int(tracklet.camera_id))

    if skipped:
        print(f"{split_name}: skipped {skipped} tracklets without embeddings")
    if not embeddings:
        raise RuntimeError(f"{split_name}: no tracklet embeddings were produced")

    return (
        np.stack(embeddings, axis=0),
        np.asarray(pids, dtype=np.int64),
        np.asarray(camids, dtype=np.int64),
    )


def evaluate_encoder(
    encoder: Any,
    query: list[Tracklet],
    gallery: list[Tracklet],
    num_frames: int,
    batch_size: int,
    warmup_batches: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    after_load = gpu_memory_snapshot(encoder.device)
    reset_peak_gpu_memory(encoder.device)

    query_feat, query_pids, query_camids = embed_tracklets(
        query, encoder, num_frames, batch_size, "query", warmup_batches
    )
    query_stats = encoder.last_extract_stats
    gallery_feat, gallery_pids, gallery_camids = embed_tracklets(
        gallery, encoder, num_frames, batch_size, "gallery", warmup_batches
    )
    gallery_stats = encoder.last_extract_stats
    peak = gpu_memory_snapshot(encoder.device)

    infer_stats = merge_extract_stats([query_stats, gallery_stats])
    efficiency: dict[str, Any] = {
        "batch_size": batch_size,
        "num_frames": num_frames,
        "warmup_batches": warmup_batches,
        "model": model_footprint(encoder.model),
        "gpu_memory": {
            "device": after_load.get("device", str(encoder.device)),
            "total_mb": after_load.get("total_mb", 0.0),
            "after_load": {
                "allocated_mb": after_load.get("allocated_mb", 0.0),
                "reserved_mb": after_load.get("reserved_mb", 0.0),
            },
            "peak": {
                "peak_allocated_mb": peak.get("peak_allocated_mb", 0.0),
                "peak_reserved_mb": peak.get("peak_reserved_mb", 0.0),
                "allocated_mb": peak.get("allocated_mb", 0.0),
                "reserved_mb": peak.get("reserved_mb", 0.0),
            },
        },
        "inference": infer_stats.as_dict(
            num_tracklets=int(query_feat.shape[0] + gallery_feat.shape[0])
        ),
        "splits": {
            "query": query_stats.as_dict(num_tracklets=int(query_feat.shape[0])),
            "gallery": gallery_stats.as_dict(
                num_tracklets=int(gallery_feat.shape[0])
            ),
        },
    }

    rank_start = time.perf_counter()
    metrics = evaluate_rank_from_features(
        query_features=query_feat,
        gallery_features=gallery_feat,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
    )
    efficiency["ranking_s"] = round(time.perf_counter() - rank_start, 4)
    return metrics, efficiency
