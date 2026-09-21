from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


MB = 1024 ** 2


@dataclass(frozen=True)
class FastReIDEfficiencyProfile:
    model_id: str
    family: str
    backbone: str

    input_height: int
    input_width: int
    embedding_dim: int

    parameters: int
    parameters_m: float
    checkpoint_mb: float

    batch1_latency_ms: float

    throughput_batch_size: int
    batch_latency_ms: float
    throughput_images_per_second: float

    tracklet_frames: int
    tracklet_latency_ms: float

    baseline_vram_mb: float
    peak_vram_batch1_mb: float
    peak_vram_batch_mb: float


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def synchronize(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def create_input(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create synthetic FastReID input.

    Pixel scale matches real inference input:
        0..255
    """

    return (
        torch.rand(
            batch_size,
            3,
            height,
            width,
            device=device,
            dtype=torch.float32,
        )
        * 255.0
    )


def warmup(
    *,
    adapter: FastReIDAdapter,
    images: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            adapter.encode(
                images
            )

    synchronize(
        device
    )


def measure_forward_latency(
    *,
    adapter: FastReIDAdapter,
    images: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> float:
    """
    Return average model forward latency
    in milliseconds per batch.
    """

    synchronize(
        device
    )

    if device.type == "cuda":
        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        with torch.inference_mode():
            start_event.record()

            for _ in range(iterations):
                adapter.encode(
                    images
                )

            end_event.record()

        torch.cuda.synchronize()

        total_ms = (
            start_event.elapsed_time(
                end_event
            )
        )

        return (
            total_ms
            / iterations
        )

    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(iterations):
            adapter.encode(
                images
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        elapsed
        * 1000.0
        / iterations
    )


def measure_tracklet_latency(
    *,
    adapter: FastReIDAdapter,
    images: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> float:
    """
    Measure one complete synthetic tracklet:

        N frames
            ↓
        FastReID forward
            ↓
        mean pooling
            ↓
        final L2 normalization

    Returns:
        milliseconds / tracklet
    """

    synchronize(
        device
    )

    if device.type == "cuda":
        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        with torch.inference_mode():
            start_event.record()

            for _ in range(iterations):
                features = adapter.encode(
                    images
                )

                tracklet = features.mean(
                    dim=0,
                    keepdim=True,
                )

                F.normalize(
                    tracklet,
                    p=2,
                    dim=1,
                )

            end_event.record()

        torch.cuda.synchronize()

        total_ms = (
            start_event.elapsed_time(
                end_event
            )
        )

        return (
            total_ms
            / iterations
        )

    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(iterations):
            features = adapter.encode(
                images
            )

            tracklet = features.mean(
                dim=0,
                keepdim=True,
            )

            F.normalize(
                tracklet,
                p=2,
                dim=1,
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        elapsed
        * 1000.0
        / iterations
    )


def measure_peak_vram(
    *,
    adapter: FastReIDAdapter,
    images: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Peak allocated CUDA memory while the
    model and input batch are resident.
    """

    if device.type != "cuda":
        return 0.0

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    synchronize(
        device
    )

    with torch.inference_mode():
        adapter.encode(
            images
        )

    synchronize(
        device
    )

    return (
        torch.cuda.max_memory_allocated()
        / MB
    )


def profile_fastreid_model(
    *,
    adapter: FastReIDAdapter,
    warmup_iterations: int = 10,
    measurement_iterations: int = 50,
    throughput_batch_size: int = 16,
    tracklet_frames: int = 8,
) -> FastReIDEfficiencyProfile:

    if warmup_iterations < 0:
        raise ValueError(
            "warmup_iterations must be >= 0."
        )

    if measurement_iterations <= 0:
        raise ValueError(
            "measurement_iterations must be > 0."
        )

    if throughput_batch_size <= 0:
        raise ValueError(
            "throughput_batch_size must be > 0."
        )

    if tracklet_frames <= 0:
        raise ValueError(
            "tracklet_frames must be > 0."
        )

    device = torch.device(
        adapter.device
    )

    height, width = (
        adapter.input_size
    )

    height = int(
        height
    )

    width = int(
        width
    )

    batch1 = create_input(
        batch_size=1,
        height=height,
        width=width,
        device=device,
    )

    throughput_batch = create_input(
        batch_size=throughput_batch_size,
        height=height,
        width=width,
        device=device,
    )

    tracklet_batch = create_input(
        batch_size=tracklet_frames,
        height=height,
        width=width,
        device=device,
    )

    # Determine embedding dimension.
    with torch.inference_mode():
        output = adapter.encode(
            batch1
        )

    if output.ndim != 2:
        raise RuntimeError(
            "Unexpected FastReID output "
            f"shape: {tuple(output.shape)}"
        )

    embedding_dim = int(
        output.shape[1]
    )

    del output

    # Warm-up.
    warmup(
        adapter=adapter,
        images=throughput_batch,
        iterations=warmup_iterations,
        device=device,
    )

    # Batch-1 latency.
    batch1_latency_ms = (
        measure_forward_latency(
            adapter=adapter,
            images=batch1,
            iterations=measurement_iterations,
            device=device,
        )
    )

    # Batch throughput latency.
    batch_latency_ms = (
        measure_forward_latency(
            adapter=adapter,
            images=throughput_batch,
            iterations=measurement_iterations,
            device=device,
        )
    )

    throughput_images_per_second = (
        throughput_batch_size
        / (
            batch_latency_ms
            / 1000.0
        )
    )

    # 8-frame tracklet latency.
    tracklet_latency_ms = (
        measure_tracklet_latency(
            adapter=adapter,
            images=tracklet_batch,
            iterations=measurement_iterations,
            device=device,
        )
    )

    # VRAM.
    if device.type == "cuda":
        baseline_vram_mb = (
            torch.cuda.memory_allocated()
            / MB
        )
    else:
        baseline_vram_mb = 0.0

    peak_vram_batch1_mb = (
        measure_peak_vram(
            adapter=adapter,
            images=batch1,
            device=device,
        )
    )

    peak_vram_batch_mb = (
        measure_peak_vram(
            adapter=adapter,
            images=throughput_batch,
            device=device,
        )
    )

    # Static model information.
    parameters = count_parameters(
        adapter.model
    )

    parameters_m = (
        parameters
        / 1_000_000
    )

    checkpoint_mb = (
        adapter.spec.checkpoint_path.stat().st_size
        / MB
    )

    return FastReIDEfficiencyProfile(
        model_id=adapter.model_id,
        family=adapter.family,
        backbone=adapter.backbone,

        input_height=height,
        input_width=width,
        embedding_dim=embedding_dim,

        parameters=parameters,
        parameters_m=parameters_m,
        checkpoint_mb=checkpoint_mb,

        batch1_latency_ms=batch1_latency_ms,

        throughput_batch_size=(
            throughput_batch_size
        ),

        batch_latency_ms=batch_latency_ms,

        throughput_images_per_second=(
            throughput_images_per_second
        ),

        tracklet_frames=tracklet_frames,

        tracklet_latency_ms=(
            tracklet_latency_ms
        ),

        baseline_vram_mb=(
            baseline_vram_mb
        ),

        peak_vram_batch1_mb=(
            peak_vram_batch1_mb
        ),

        peak_vram_batch_mb=(
            peak_vram_batch_mb
        ),
    )