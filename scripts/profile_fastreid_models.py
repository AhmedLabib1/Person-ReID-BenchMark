from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)
from reid.profiling.fastreid_profiler import (
    profile_fastreid_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_msmt17_models.yaml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
    / "fastreid_efficiency.csv"
)


def load_model_ids() -> list[str]:
    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(
            file
        )

    return [
        model["id"]
        for model in registry["models"]
    ]


def cleanup() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_results(
    rows: list[dict],
) -> None:
    if not rows:
        return

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile FastReID models for "
            "SHAWAF efficiency benchmarking."
        )
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--tracklet-frames",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    registry_models = (
        load_model_ids()
    )

    if args.models:
        unknown = [
            model
            for model in args.models
            if model not in registry_models
        ]

        if unknown:
            raise ValueError(
                "Unknown models: "
                + ", ".join(
                    unknown
                )
            )

        model_ids = (
            args.models
        )

    else:
        model_ids = (
            registry_models
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 100)
    print(
        "SHAWAF ReID - "
        "FastReID Efficiency Benchmark"
    )
    print("=" * 100)

    print()
    print(
        f"Models                  : "
        f"{len(model_ids)}"
    )

    print(
        f"Device                  : "
        f"{device}"
    )

    if device == "cuda":
        print(
            f"GPU                     : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Warm-up iterations      : "
        f"{args.warmup}"
    )

    print(
        f"Measurement iterations  : "
        f"{args.iterations}"
    )

    print(
        f"Throughput batch        : "
        f"{args.batch_size}"
    )

    print(
        f"Tracklet frames         : "
        f"{args.tracklet_frames}"
    )

    rows: list[dict] = []

    for index, model_id in enumerate(
        model_ids,
        start=1,
    ):
        print()
        print("=" * 100)
        print(
            f"[{index:02d}/{len(model_ids):02d}] "
            f"{model_id}"
        )
        print("=" * 100)

        cleanup()

        adapter = FastReIDAdapter(
            model_id=model_id,
            device=device,
        )

        profile = (
            profile_fastreid_model(
                adapter=adapter,
                warmup_iterations=(
                    args.warmup
                ),
                measurement_iterations=(
                    args.iterations
                ),
                throughput_batch_size=(
                    args.batch_size
                ),
                tracklet_frames=(
                    args.tracklet_frames
                ),
            )
        )

        row = asdict(
            profile
        )

        # Round display/storage values.
        for key in (
            "parameters_m",
            "checkpoint_mb",
            "batch1_latency_ms",
            "batch_latency_ms",
            "throughput_images_per_second",
            "tracklet_latency_ms",
            "baseline_vram_mb",
            "peak_vram_batch1_mb",
            "peak_vram_batch_mb",
        ):
            row[key] = round(
                float(row[key]),
                4,
            )

        rows.append(
            row
        )

        # Save after every model so an interruption
        # does not lose completed measurements.
        write_results(
            rows
        )

        print(
            f"Family                  : "
            f"{profile.family}"
        )

        print(
            f"Backbone                : "
            f"{profile.backbone}"
        )

        print(
            f"Input                   : "
            f"{profile.input_height} x "
            f"{profile.input_width}"
        )

        print(
            f"Parameters              : "
            f"{profile.parameters_m:.2f} M"
        )

        print(
            f"Checkpoint              : "
            f"{profile.checkpoint_mb:.2f} MB"
        )

        print()
        print("LATENCY / THROUGHPUT")
        print("-" * 100)

        print(
            f"Batch-1 latency         : "
            f"{profile.batch1_latency_ms:.3f} ms"
        )

        print(
            f"Batch-{args.batch_size} latency        : "
            f"{profile.batch_latency_ms:.3f} ms"
        )

        print(
            f"Throughput              : "
            f"{profile.throughput_images_per_second:.2f} images/s"
        )

        print(
            f"{args.tracklet_frames}-frame tracklet latency : "
            f"{profile.tracklet_latency_ms:.3f} ms"
        )

        print()
        print("GPU MEMORY")
        print("-" * 100)

        print(
            f"Baseline VRAM           : "
            f"{profile.baseline_vram_mb:.2f} MB"
        )

        print(
            f"Peak VRAM batch 1       : "
            f"{profile.peak_vram_batch1_mb:.2f} MB"
        )

        print(
            f"Peak VRAM batch {args.batch_size:<2}      : "
            f"{profile.peak_vram_batch_mb:.2f} MB"
        )

        del adapter

        cleanup()

    print()
    print("=" * 100)
    print("EFFICIENCY BENCHMARK COMPLETE")
    print("=" * 100)

    print(
        f"Models profiled         : "
        f"{len(rows)}"
    )

    print(
        f"CSV                     : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()