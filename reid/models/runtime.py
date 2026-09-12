from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from tqdm import tqdm

from reid.profiling import ExtractStats, synchronize as sync_device


class ImageEncoder(Protocol):
    device: torch.device
    last_extract_stats: ExtractStats

    def preprocess(self, path: Path) -> torch.Tensor | None:
        ...

    def forward_gpu(self, batch: torch.Tensor) -> torch.Tensor:
        ...


def extract_images(
    encoder: ImageEncoder,
    crop_paths: list[Path],
    batch_size: int = 32,
    show_progress: bool = False,
    warmup_batches: int = 2,
    desc: str = "extract",
) -> np.ndarray:
    """Shared batched extract with wall / GPU timing."""

    stats = ExtractStats(warmup_batches=max(warmup_batches, 0))
    encoder.last_extract_stats = stats

    if not crop_paths:
        return np.zeros((0, 0), dtype=np.float32)

    features: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    iterator: list[Path] | tqdm = crop_paths
    if show_progress:
        iterator = tqdm(crop_paths, desc=desc, unit="img")

    sync_device(encoder.device)
    wall_start = time.perf_counter()

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        count_steady = stats.num_batches >= stats.warmup_batches
        images_in_batch = len(batch)
        stacked = torch.stack(batch, dim=0)
        chunk, forward_s = _timed_forward(encoder, stacked)
        stats.num_batches += 1
        stats.num_images += images_in_batch
        stats.gpu_forward_s += forward_s
        stats.batch_forward_s.append(forward_s)
        if count_steady:
            stats.timed_images += images_in_batch
        features.append(chunk)
        batch = []

    for path in iterator:
        load_start = time.perf_counter()
        image = encoder.preprocess(Path(path))
        stats.preprocess_s += time.perf_counter() - load_start
        if image is None:
            raise FileNotFoundError(f"Could not read person crop: {path}")
        batch.append(image)
        if len(batch) >= batch_size:
            flush()

    flush()
    sync_device(encoder.device)
    stats.wall_s = time.perf_counter() - wall_start

    if not features:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(features, axis=0)


def _timed_forward(
    encoder: ImageEncoder,
    batch: torch.Tensor,
) -> tuple[np.ndarray, float]:
    gpu_batch = batch.to(encoder.device, non_blocking=True)
    if encoder.device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = encoder.forward_gpu(gpu_batch)
        end.record()
        end.synchronize()
        forward_s = start.elapsed_time(end) / 1000.0
    else:
        cpu_start = time.perf_counter()
        outputs = encoder.forward_gpu(gpu_batch)
        forward_s = time.perf_counter() - cpu_start
    return outputs.detach().cpu().numpy(), forward_s
