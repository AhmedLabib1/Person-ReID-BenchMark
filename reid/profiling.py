from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


def bytes_to_mb(num_bytes: int | float) -> float:
    return round(float(num_bytes) / (1024.0 ** 2), 2)


def model_footprint(model: nn.Module) -> dict[str, float | int]:
    """Parameter / buffer size of a module (weights on whatever device they sit)."""

    param_bytes = 0
    param_count = 0
    for parameter in model.parameters():
        param_count += int(parameter.numel())
        param_bytes += int(parameter.numel() * parameter.element_size())

    buffer_bytes = 0
    for buffer in model.buffers():
        buffer_bytes += int(buffer.numel() * buffer.element_size())

    return {
        "parameters": param_count,
        "weights_mb": bytes_to_mb(param_bytes),
        "buffers_mb": bytes_to_mb(buffer_bytes),
        "total_mb": bytes_to_mb(param_bytes + buffer_bytes),
    }


def gpu_memory_snapshot(device: torch.device) -> dict[str, Any]:
    """Current and peak CUDA allocator stats in MiB."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "available": False,
            "device": str(device),
        }

    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "device": torch.cuda.get_device_name(index),
        "device_index": int(index),
        "total_mb": bytes_to_mb(props.total_memory),
        "allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(index)),
        "reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(index)),
        "peak_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(index)),
    }


def reset_peak_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass
class ExtractStats:
    """Timing for one encoder.extract() call."""

    num_images: int = 0
    num_batches: int = 0
    timed_images: int = 0
    warmup_batches: int = 0
    wall_s: float = 0.0
    preprocess_s: float = 0.0
    gpu_forward_s: float = 0.0
    batch_forward_s: list[float] = field(default_factory=list)

    def as_dict(self, num_tracklets: int = 0) -> dict[str, Any]:
        timed = self._timed_batches()
        gpu_steady_s = float(sum(timed))
        timed_images = self.timed_images or self._timed_images()
        return {
            "num_images": self.num_images,
            "num_batches": self.num_batches,
            "timed_images": timed_images,
            "num_tracklets": num_tracklets,
            "warmup_batches": self.warmup_batches,
            "wall_s": round(self.wall_s, 4),
            "preprocess_s": round(self.preprocess_s, 4),
            "gpu_forward_s": round(self.gpu_forward_s, 4),
            "gpu_forward_steady_s": round(gpu_steady_s, 4),
            "ms_per_image": {
                "wall": _ms_per(self.wall_s, self.num_images),
                "gpu": _ms_per(self.gpu_forward_s, self.num_images),
                "gpu_steady": _ms_per(gpu_steady_s, timed_images),
            },
            "ms_per_tracklet": {
                "wall": _ms_per(self.wall_s, num_tracklets),
                "gpu": _ms_per(self.gpu_forward_s, num_tracklets),
            },
            "images_per_s": {
                "wall": _per_s(self.num_images, self.wall_s),
                "gpu": _per_s(self.num_images, self.gpu_forward_s),
                "gpu_steady": _per_s(timed_images, gpu_steady_s),
            },
            "tracklets_per_s": {
                "wall": _per_s(num_tracklets, self.wall_s),
                "gpu": _per_s(num_tracklets, self.gpu_forward_s),
            },
        }

    def _timed_batches(self) -> list[float]:
        if self.warmup_batches <= 0:
            return list(self.batch_forward_s)
        return self.batch_forward_s[self.warmup_batches :]

    def _timed_images(self) -> int:
        if self.num_batches == 0:
            return 0
        timed_batches = max(self.num_batches - self.warmup_batches, 0)
        if timed_batches == 0:
            return 0
        mean_batch = self.num_images / self.num_batches
        return int(round(mean_batch * timed_batches))


def merge_extract_stats(parts: list[ExtractStats]) -> ExtractStats:
    """Combine query + gallery extracts.

    Warmup batches from each part are dropped from ``batch_forward_s`` so
    ``gpu_forward_steady_s`` is steady-state only. ``gpu_forward_s`` still
    includes warmup.
    """

    merged = ExtractStats(warmup_batches=0)
    for part in parts:
        merged.num_images += part.num_images
        merged.num_batches += part.num_batches
        merged.wall_s += part.wall_s
        merged.preprocess_s += part.preprocess_s
        merged.gpu_forward_s += part.gpu_forward_s
        merged.timed_images += part.timed_images or part._timed_images()
        merged.batch_forward_s.extend(part._timed_batches())
    return merged


def format_efficiency(report: dict[str, Any]) -> str:
    memory = report.get("gpu_memory", {})
    inference = report.get("inference", {})
    model = report.get("model", {})
    ms_image = inference.get("ms_per_image", {})
    ms_track = inference.get("ms_per_tracklet", {})
    ips = inference.get("images_per_s", {})
    after = memory.get("after_load", {})
    peak = memory.get("peak", {})

    lines = [
        "Efficiency",
        f"  device           : {memory.get('device', 'cpu')}",
        f"  VRAM total       : {memory.get('total_mb', 0):.2f} MiB",
        f"  model weights    : {model.get('total_mb', 0):.2f} MiB "
        f"({model.get('parameters', 0):,} params)",
        f"  GPU allocated    : {after.get('allocated_mb', 0):.2f} MiB "
        f"(reserved {after.get('reserved_mb', 0):.2f} MiB)",
        f"  GPU peak         : {peak.get('peak_allocated_mb', 0):.2f} MiB "
        f"(reserved {peak.get('peak_reserved_mb', 0):.2f} MiB)",
        f"  wall time        : {inference.get('wall_s', 0):.2f} s "
        f"({inference.get('num_images', 0)} images, "
        f"{inference.get('num_tracklets', 0)} tracklets)",
        f"  preprocess       : {inference.get('preprocess_s', 0):.2f} s",
        f"  GPU forward      : {inference.get('gpu_forward_s', 0):.2f} s",
        f"  ms / image       : wall {ms_image.get('wall', 0):.2f} | "
        f"GPU {ms_image.get('gpu', 0):.2f} | "
        f"GPU steady {ms_image.get('gpu_steady', 0):.2f}",
        f"  ms / tracklet    : wall {ms_track.get('wall', 0):.2f} | "
        f"GPU {ms_track.get('gpu', 0):.2f}",
        f"  images / s       : wall {ips.get('wall', 0):.2f} | "
        f"GPU {ips.get('gpu', 0):.2f} | "
        f"GPU steady {ips.get('gpu_steady', 0):.2f}",
    ]
    return "\n".join(lines)


def _ms_per(seconds: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return round((seconds * 1000.0) / count, 4)


def _per_s(count: int, seconds: float) -> float:
    if seconds <= 0 or count <= 0:
        return 0.0
    return round(count / seconds, 4)
