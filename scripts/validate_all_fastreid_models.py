
from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import torch
import yaml


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FASTREID_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "fast-reid"
)

for path in [
    PROJECT_ROOT,
    FASTREID_ROOT,
]:
    path_string = str(path)

    if path_string not in sys.path:
        sys.path.insert(
            0,
            path_string,
        )


from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_dukemtmc_models.yaml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
    / "fastreid_model_compatibility.csv"
)


MB = 1024 ** 2


# ============================================================
# Utilities
# ============================================================

def load_registry() -> dict:
    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def cleanup_gpu() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_embedding(
    embedding: torch.Tensor,
    model_id: str,
) -> None:

    if not isinstance(
        embedding,
        torch.Tensor,
    ):
        raise TypeError(
            f"{model_id}: output is not a Tensor."
        )

    if embedding.ndim != 2:
        raise RuntimeError(
            f"{model_id}: expected [B, D], "
            f"got {tuple(embedding.shape)}"
        )

    if embedding.shape[0] != 1:
        raise RuntimeError(
            f"{model_id}: expected batch size 1, "
            f"got {embedding.shape[0]}"
        )

    if embedding.shape[1] <= 0:
        raise RuntimeError(
            f"{model_id}: invalid embedding dimension."
        )

    if torch.isnan(
        embedding
    ).any():
        raise RuntimeError(
            f"{model_id}: NaN detected."
        )

    if torch.isinf(
        embedding
    ).any():
        raise RuntimeError(
            f"{model_id}: Inf detected."
        )


# ============================================================
# Main validation
# ============================================================

def main() -> None:

    print("=" * 110)
    print(
        "SHAWAF ReID - "
        "DUKE FASTREID MODEL COMPATIBILITY VALIDATION"
    )
    print("=" * 110)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this validation."
        )

    registry = load_registry()
    models = registry["models"]

    print()
    print(
        "Source dataset :",
        registry["source_dataset"],
    )
    print(
        "Models         :",
        len(models),
    )
    print(
        "GPU            :",
        torch.cuda.get_device_name(0),
    )
    print(
        "PyTorch        :",
        torch.__version__,
    )
    print(
        "CUDA           :",
        torch.version.cuda,
    )
    print()

    if len(models) != 13:
        raise RuntimeError(
            f"Expected 13 Duke models, "
            f"found {len(models)}."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []

    successful = []
    failed = []

    # ========================================================
    # Validate models sequentially
    # ========================================================

    for index, model_spec in enumerate(
        models,
        start=1,
    ):

        model_id = model_spec["id"]

        print("-" * 110)
        print(
            f"[{index:02d}/{len(models):02d}] "
            f"{model_id}"
        )
        print("-" * 110)

        adapter = None
        embedding = None

        try:
            cleanup_gpu()

            torch.cuda.reset_peak_memory_stats()

            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            adapter = FastReIDAdapter(
                model_id=model_id,
                device="cuda",
            )

            height, width = (
                adapter.input_size
            )

            # ------------------------------------------------
            # One raw embedding
            # ------------------------------------------------

            embedding = adapter.dummy_encode(
                batch_size=1,
            )

            torch.cuda.synchronize()

            validate_embedding(
                embedding,
                model_id,
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            embedding_dim = int(
                embedding.shape[1]
            )

            parameters = count_parameters(
                adapter.model
            )

            parameters_m = (
                parameters / 1_000_000
            )

            checkpoint_mb = (
                adapter.spec.checkpoint_path
                .stat()
                .st_size
                / MB
            )

            peak_vram_mb = (
                torch.cuda.max_memory_allocated()
                / MB
            )

            raw_l2_norm = float(
                embedding[
                    0
                ]
                .float()
                .norm(
                    p=2
                )
                .item()
            )

            row = {
                "model_id": model_id,
                "family": adapter.family,
                "backbone": adapter.backbone,
                "source_dataset": (
                    adapter.source_dataset
                ),
                "input_height": height,
                "input_width": width,
                "embedding_dim": embedding_dim,
                "parameters": parameters,
                "parameters_m": round(
                    parameters_m,
                    4,
                ),
                "checkpoint_mb": round(
                    checkpoint_mb,
                    4,
                ),
                "raw_embedding_l2": round(
                    raw_l2_norm,
                    6,
                ),
                "peak_vram_validation_mb": round(
                    peak_vram_mb,
                    4,
                ),
                "official_duke_rank1": (
                    adapter.spec.official_rank1
                ),
                "official_duke_map": (
                    adapter.spec.official_map
                ),
                "official_duke_minp": (
                    adapter.spec.official_minp
                ),
                "status": "PASS",
                "error": "",
            }

            rows.append(
                row
            )

            successful.append(
                model_id
            )

            print(
                f"Family          : "
                f"{adapter.family}"
            )

            print(
                f"Backbone        : "
                f"{adapter.backbone}"
            )

            print(
                f"Input           : "
                f"{height} x {width}"
            )

            print(
                f"Embedding       : "
                f"{tuple(embedding.shape)}"
            )

            print(
                f"Raw L2 norm     : "
                f"{raw_l2_norm:.4f}"
            )

            print(
                f"Parameters      : "
                f"{parameters_m:.2f} M"
            )

            print(
                f"Checkpoint      : "
                f"{checkpoint_mb:.2f} MB"
            )

            print(
                f"Peak VRAM       : "
                f"{peak_vram_mb:.2f} MB"
            )

            print(
                "NaN / Inf       : False / False"
            )

            print(
                "Status          : PASS"
            )

        except Exception as exc:

            failed.append(
                (
                    model_id,
                    str(exc),
                )
            )

            rows.append(
                {
                    "model_id": model_id,
                    "family": model_spec[
                        "family"
                    ],
                    "backbone": model_spec[
                        "backbone"
                    ],
                    "source_dataset": registry[
                        "source_dataset"
                    ],
                    "input_height": "",
                    "input_width": "",
                    "embedding_dim": "",
                    "parameters": "",
                    "parameters_m": "",
                    "checkpoint_mb": "",
                    "raw_embedding_l2": "",
                    "peak_vram_validation_mb": "",
                    "official_duke_rank1": (
                        model_spec[
                            "official_dukemtmc"
                        ][
                            "rank1"
                        ]
                    ),
                    "official_duke_map": (
                        model_spec[
                            "official_dukemtmc"
                        ][
                            "map"
                        ]
                    ),
                    "official_duke_minp": (
                        model_spec[
                            "official_dukemtmc"
                        ][
                            "minp"
                        ]
                    ),
                    "status": "FAILED",
                    "error": str(exc),
                }
            )

            print(
                "Status          : FAILED"
            )
            print(
                "Error           :",
                exc,
            )

        finally:

            if embedding is not None:
                del embedding

            if adapter is not None:
                del adapter

            cleanup_gpu()

    # ========================================================
    # Save compatibility CSV
    # ========================================================

    fieldnames = [
        "model_id",
        "family",
        "backbone",
        "source_dataset",
        "input_height",
        "input_width",
        "embedding_dim",
        "parameters",
        "parameters_m",
        "checkpoint_mb",
        "raw_embedding_l2",
        "peak_vram_validation_mb",
        "official_duke_rank1",
        "official_duke_map",
        "official_duke_minp",
        "status",
        "error",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 110)
    print("VALIDATION SUMMARY")
    print("=" * 110)

    print(
        f"Passed : "
        f"{len(successful)}/{len(models)}"
    )

    print(
        f"Failed : "
        f"{len(failed)}/{len(models)}"
    )

    print(
        f"CSV    : "
        f"{OUTPUT_PATH}"
    )

    if failed:
        print()
        print("FAILED MODELS:")

        for model_id, error in failed:
            print(
                f"  - {model_id}"
            )
            print(
                f"    {error}"
            )

        raise SystemExit(1)

    print()
    print(
        "ALL DUKE FASTREID MODELS: PASS"
    )


if __name__ == "__main__":
    main()
