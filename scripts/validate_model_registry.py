from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_msmt17_models.yaml"
)


def main() -> None:
    print("=" * 78)
    print("SHAWAF ReID - FastReID Model Registry Validation")
    print("=" * 78)

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry file not found:\n{REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(file)

    fastreid_root = (
        PROJECT_ROOT
        / registry["fastreid_root"]
    )

    checkpoint_root = (
        PROJECT_ROOT
        / registry["checkpoint_root"]
    )

    models = registry["models"]

    print()
    print(f"Registry       : {REGISTRY_PATH}")
    print(f"FastReID root : {fastreid_root}")
    print(f"Checkpoint dir: {checkpoint_root}")
    print(f"Source dataset: {registry['source_dataset']}")
    print(f"Models        : {len(models)}")
    print()

    if len(models) != 12:
        raise RuntimeError(
            f"Expected 12 models, found {len(models)}."
        )

    if not fastreid_root.exists():
        raise FileNotFoundError(
            f"FastReID root does not exist:\n"
            f"{fastreid_root}"
        )

    model_ids = [
        model["id"]
        for model in models
    ]

    duplicate_ids = [
        model_id
        for model_id, count
        in Counter(model_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        raise RuntimeError(
            "Duplicate model IDs found: "
            + ", ".join(duplicate_ids)
        )

    families = Counter(
        model["family"]
        for model in models
    )

    expected_families = {
        "BoT": 4,
        "AGW": 4,
        "SBS": 4,
    }

    if dict(families) != expected_families:
        raise RuntimeError(
            "Unexpected family distribution.\n"
            f"Expected: {expected_families}\n"
            f"Found   : {dict(families)}"
        )

    missing_configs: list[Path] = []

    print(
        f"{'#':<4}"
        f"{'MODEL':<18}"
        f"{'FAMILY':<8}"
        f"{'BACKBONE':<12}"
        f"{'CONFIG':<10}"
        f"{'CHECKPOINT':<12}"
    )

    print("-" * 78)

    for index, model in enumerate(
        models,
        start=1,
    ):
        config_path = (
            fastreid_root
            / model["config"]
        )

        checkpoint_path = (
            checkpoint_root
            / model["checkpoint"]
        )

        config_status = (
            "OK"
            if config_path.exists()
            else "MISSING"
        )

        checkpoint_status = (
            "FOUND"
            if checkpoint_path.exists()
            else "NOT YET"
        )

        if not config_path.exists():
            missing_configs.append(
                config_path
            )

        print(
            f"{index:<4}"
            f"{model['id']:<18}"
            f"{model['family']:<8}"
            f"{model['backbone']:<12}"
            f"{config_status:<10}"
            f"{checkpoint_status:<12}"
        )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        f"Models     : {len(models)}/12"
    )

    print(
        f"BoT        : {families['BoT']}/4"
    )

    print(
        f"AGW        : {families['AGW']}/4"
    )

    print(
        f"SBS        : {families['SBS']}/4"
    )

    if missing_configs:
        print()
        print("Missing FastReID configs:")

        for path in missing_configs:
            print(f"  - {path}")

        raise RuntimeError(
            "Registry validation failed."
        )

    print()
    print("All FastReID config files exist.")
    print("Model registry validation: SUCCESS")


if __name__ == "__main__":
    main()