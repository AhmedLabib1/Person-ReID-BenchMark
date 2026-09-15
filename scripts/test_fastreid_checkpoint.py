from __future__ import annotations

import argparse

import torch

from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test loading an official FastReID "
            "MSMT17 checkpoint."
        )
    )

    parser.add_argument(
        "--model",
        default="sbs_r50_ibn",
        help=(
            "Model ID from "
            "fastreid_msmt17_models.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print(
        "SHAWAF ReID - "
        "FastReID Checkpoint Test"
    )
    print("=" * 72)

    print()
    print(
        f"Loading model : {args.model}"
    )

    adapter = FastReIDAdapter(
        model_id=args.model,
        device=args.device,
    )

    print()
    print("MODEL")
    print("-" * 72)

    print(
        f"ID             : "
        f"{adapter.model_id}"
    )

    print(
        f"Family         : "
        f"{adapter.family}"
    )

    print(
        f"Backbone       : "
        f"{adapter.backbone}"
    )

    print(
        f"Source dataset : "
        f"{adapter.spec.source_dataset}"
    )

    print(
        f"Input size     : "
        f"{adapter.input_size}"
    )

    print(
        f"Checkpoint     : "
        f"{adapter.spec.checkpoint_path.name}"
    )

    print()
    print("OFFICIAL MSMT17 REFERENCE")
    print("-" * 72)

    print(
        f"Rank-1 : "
        f"{adapter.spec.official_rank1:.1f}%"
    )

    print(
        f"mAP    : "
        f"{adapter.spec.official_map:.1f}%"
    )

    print(
        f"mINP   : "
        f"{adapter.spec.official_minp:.1f}%"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    embeddings = adapter.dummy_encode(
        batch_size=1,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )
    else:
        peak_memory_mb = 0.0

    print()
    print("FORWARD TEST")
    print("-" * 72)

    print(
        f"Output shape   : "
        f"{tuple(embeddings.shape)}"
    )

    print(
        f"Output dtype   : "
        f"{embeddings.dtype}"
    )

    print(
        f"Output device  : "
        f"{embeddings.device}"
    )

    print(
        f"Feature norm   : "
        f"{embeddings[0].norm().item():.4f}"
    )

    print(
        f"Peak VRAM      : "
        f"{peak_memory_mb:.2f} MB"
    )

    print()
    print("=" * 72)
    print(
        "MSMT17 checkpoint loaded "
        "and inference completed successfully."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()