from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.aggregate import mean_pool_l2
from reid.data import MarsDataset
from reid.data.tracklet import Tracklet
from reid.metrics import (
    camera_id_to_int,
    cosine_distance,
    evaluate_rank,
    format_metrics,
)
from reid.models.fastreid import FastReIDEncoder
from reid.profiling import (
    format_efficiency,
    gpu_memory_snapshot,
    merge_extract_stats,
    model_footprint,
    reset_peak_gpu_memory,
)
from reid.sampling import sample_crop_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained FastReID encoder on MARS tracklets."
    )
    parser.add_argument(
        "--mars-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "MARS",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=PROJECT_ROOT / "fast-reid" / "configs" / "Market1501" / "sbs_R50.yml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "market_sbs_R50.pth",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "fastreid_mars.json",
    )
    return parser.parse_args()


def embed_tracklets(
    tracklets: list[Tracklet],
    encoder: FastReIDEncoder,
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
            f"for {len(crop_paths)} sampled crops. "
            "Some crop files could not be read."
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
        if not frame_features:
            skipped += 1
            continue
        if tracklet.person_id is None:
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


def main() -> None:
    args = parse_args()

    if not args.mars_root.exists():
        raise FileNotFoundError(
            f"MARS root not found: {args.mars_root}\n"
            "Expected datasets/MARS with bbox_train, bbox_test, and info."
        )
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print(f"Loading MARS from {args.mars_root}")
    splits = MarsDataset(args.mars_root).load(include_train=False)
    print(
        f"Query tracklets   : {len(splits.query)}\n"
        f"Gallery tracklets : {len(splits.gallery)}"
    )

    encoder = FastReIDEncoder(
        config_file=args.config_file,
        weights=args.weights,
        device=args.device,
    )
    after_load = gpu_memory_snapshot(encoder.device)
    reset_peak_gpu_memory(encoder.device)

    query_feat, query_pids, query_camids = embed_tracklets(
        tracklets=splits.query,
        encoder=encoder,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        split_name="query",
        warmup_batches=args.warmup_batches,
    )
    query_stats = encoder.last_extract_stats
    gallery_feat, gallery_pids, gallery_camids = embed_tracklets(
        tracklets=splits.gallery,
        encoder=encoder,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        split_name="gallery",
        warmup_batches=args.warmup_batches,
    )
    gallery_stats = encoder.last_extract_stats
    peak = gpu_memory_snapshot(encoder.device)

    infer_stats = merge_extract_stats([query_stats, gallery_stats])
    efficiency = {
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "warmup_batches": args.warmup_batches,
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
    distmat = cosine_distance(query_feat, gallery_feat)
    metrics = evaluate_rank(
        distmat=distmat,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
    )
    efficiency["ranking_s"] = round(time.perf_counter() - rank_start, 4)
    print()
    print(format_metrics(metrics))
    print()
    print(format_efficiency(efficiency))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "fastreid-sbs-r50",
        "weights": str(args.weights),
        "config_file": str(args.config_file),
        "mars_root": str(args.mars_root),
        "num_frames": args.num_frames,
        "batch_size": args.batch_size,
        "device": args.device,
        "query_tracklets": int(query_feat.shape[0]),
        "gallery_tracklets": int(gallery_feat.shape[0]),
        "metrics": metrics,
        "efficiency": efficiency,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
