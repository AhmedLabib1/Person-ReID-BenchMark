from __future__ import annotations

from pathlib import Path

import requests
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_msmt17_models.yaml"
)


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry not found:\n{REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def validate_checkpoint(
    checkpoint_path: Path,
) -> None:
    """
    Verify that the downloaded file can be read by PyTorch.

    These checkpoints come from the official FastReID
    GitHub release, so weights_only=False is explicitly
    used for compatibility with older checkpoint formats.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Invalid checkpoint:\n"
            f"{checkpoint_path}\n"
            f"Reason: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format for:\n"
            f"{checkpoint_path}"
        )


def download_checkpoint(
    model_id: str,
    url: str,
    destination: Path,
) -> None:
    if destination.exists():
        print(
            f"[FOUND] {model_id}: "
            f"{destination.name}"
        )

        validate_checkpoint(destination)

        print(
            f"[VALID] {model_id}"
        )

        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(destination) + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    print()
    print(
        f"[DOWNLOAD] {model_id}"
    )
    print(
        f"File: {destination.name}"
    )

    with requests.get(
        url,
        stream=True,
        timeout=60,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()

        total_size = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        chunk_size = 1024 * 1024

        with temporary_path.open("wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=model_id,
            ) as progress:
                for chunk in response.iter_content(
                    chunk_size=chunk_size,
                ):
                    if not chunk:
                        continue

                    file.write(chunk)

                    progress.update(
                        len(chunk)
                    )

    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Downloaded file is empty: "
            f"{model_id}"
        )

    temporary_path.replace(
        destination
    )

    try:
        validate_checkpoint(
            destination
        )

    except Exception:
        destination.unlink(
            missing_ok=True
        )

        raise

    print(
        f"[VALID] {model_id}"
    )


def main() -> None:
    print("=" * 78)
    print(
        "SHAWAF ReID - FastReID MSMT17 "
        "Checkpoint Downloader"
    )
    print("=" * 78)

    registry = load_registry()

    checkpoint_root = (
        PROJECT_ROOT
        / registry["checkpoint_root"]
    )

    models = registry["models"]

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Source dataset : "
        f"{registry['source_dataset']}"
    )

    print(
        f"Models         : "
        f"{len(models)}"
    )

    print(
        f"Destination    : "
        f"{checkpoint_root}"
    )

    successful: list[str] = []
    failed: list[tuple[str, str]] = []

    for index, model in enumerate(
        models,
        start=1,
    ):
        model_id = model["id"]

        checkpoint_path = (
            checkpoint_root
            / model["checkpoint"]
        )

        print()
        print("-" * 78)

        print(
            f"[{index:02d}/{len(models):02d}] "
            f"{model_id}"
        )

        try:
            download_checkpoint(
                model_id=model_id,
                url=model["download_url"],
                destination=checkpoint_path,
            )

            successful.append(
                model_id
            )

        except Exception as exc:
            failed.append(
                (
                    model_id,
                    str(exc),
                )
            )

            print(
                f"[FAILED] {model_id}"
            )

            print(exc)

    print()
    print("=" * 78)
    print("DOWNLOAD SUMMARY")
    print("=" * 78)

    print(
        f"Successful : "
        f"{len(successful)}/{len(models)}"
    )

    print(
        f"Failed     : "
        f"{len(failed)}"
    )

    if failed:
        print()
        print("Failed models:")

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
        "All FastReID MSMT17 checkpoints "
        "downloaded and validated."
    )

    print(
        "Checkpoint download: SUCCESS"
    )


if __name__ == "__main__":
    main()