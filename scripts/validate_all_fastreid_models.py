from __future__ import annotations

import csv
import gc
from pathlib import Path

import torch
import yaml

from reid.models.fastreid_adapter import (
    FastReIDAdapter,
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
    / "fastreid_model_compatibility.csv"
)


def load_model_ids() -> list[str]:
    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(file)

    return [
        model["id"]
        for model in registry["models"]
    ]


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


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "FastReID 12-Model Compatibility Validation"
    )
    print("=" * 88)

    model_ids = load_model_ids()

    print()
    print(f"Models : {len(model_ids)}")
    print(
        f"Device : "
        f"{torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "Device : CPU"
    )
    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[dict] = []

    for index, model_id in enumerate(
        model_ids,
        start=1,
    ):
        print("-" * 88)
        print(
            f"[{index:02d}/{len(model_ids):02d}] "
            f"{model_id}"
        )

        adapter = None

        try:
            cleanup_gpu()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            adapter = FastReIDAdapter(
                model_id=model_id,
                device="cuda"
                if torch.cuda.is_available()
                else "cpu",
            )

            height, width = adapter.input_size

            embeddings = adapter.dummy_encode(
                batch_size=1,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

                peak_vram_mb = (
                    torch.cuda.max_memory_allocated()
                    / 1024**2
                )
            else:
                peak_vram_mb = 0.0

            embedding_dim = int(
                embeddings.shape[1]
            )

            parameters = count_parameters(
                adapter.model
            )

            params_m = (
                parameters / 1_000_000
            )

            checkpoint_size_mb = (
                adapter.spec.checkpoint_path.stat().st_size
                / 1024**2
            )

            result = {
                "model_id": adapter.model_id,
                "family": adapter.family,
                "backbone": adapter.backbone,
                "input_height": height,
                "input_width": width,
                "embedding_dim": embedding_dim,
                "parameters": parameters,
                "parameters_m": round(
                    params_m,
                    3,
                ),
                "checkpoint_mb": round(
                    checkpoint_size_mb,
                    2,
                ),
                "peak_vram_mb": round(
                    peak_vram_mb,
                    2,
                ),
                "status": "PASS",
                "error": "",
            }

            results.append(result)

            print(
                f"Family         : "
                f"{adapter.family}"
            )

            print(
                f"Backbone       : "
                f"{adapter.backbone}"
            )

            print(
                f"Input          : "
                f"{height} x {width}"
            )

            print(
                f"Embedding      : "
                f"{embedding_dim}"
            )

            print(
                f"Parameters     : "
                f"{params_m:.2f} M"
            )

            print(
                f"Checkpoint     : "
                f"{checkpoint_size_mb:.2f} MB"
            )

            print(
                f"Peak VRAM      : "
                f"{peak_vram_mb:.2f} MB"
            )

            print("Status         : PASS")

            del embeddings

        except Exception as exc:
            results.append(
                {
                    "model_id": model_id,
                    "family": "",
                    "backbone": "",
                    "input_height": "",
                    "input_width": "",
                    "embedding_dim": "",
                    "parameters": "",
                    "parameters_m": "",
                    "checkpoint_mb": "",
                    "peak_vram_mb": "",
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

            print("Status         : FAIL")
            print(f"Error          : {exc}")

        finally:
            if adapter is not None:
                del adapter

            cleanup_gpu()

    fieldnames = [
        "model_id",
        "family",
        "backbone",
        "input_height",
        "input_width",
        "embedding_dim",
        "parameters",
        "parameters_m",
        "checkpoint_mb",
        "peak_vram_mb",
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
        writer.writerows(results)

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = len(results) - passed

    print()
    print("=" * 88)
    print("VALIDATION SUMMARY")
    print("=" * 88)

    print(
        f"Passed : {passed}/{len(results)}"
    )

    print(
        f"Failed : {failed}/{len(results)}"
    )

    print(
        f"Report : {OUTPUT_PATH}"
    )

    if failed:
        print()
        print("Failed models:")

        for result in results:
            if result["status"] == "FAIL":
                print(
                    f"  - {result['model_id']}"
                )

        raise SystemExit(1)

    print()
    print(
        "All FastReID MSMT17 models are "
        "compatible and ready for benchmarking."
    )


if __name__ == "__main__":
    main()