from __future__ import annotations

from collections import Counter

from reid.data.market1501 import (
    load_market1501,
)


EXPECTED = {
    "train_images": 12936,
    "query_images": 3368,
    "gallery_images": 15913,

    "train_ids": 751,
    "query_ids": 750,
    "gallery_ids": 751,

    "cameras": 6,
}


def unique_pids(samples) -> set[int]:
    return {
        sample.person_id
        for sample in samples
    }


def unique_cameras(samples) -> set[int]:
    return {
        sample.camera_id
        for sample in samples
    }


def validate_value(
    name: str,
    actual: int,
    expected: int,
) -> None:
    status = (
        "PASS"
        if actual == expected
        else "FAIL"
    )

    print(
        f"{name:<22}"
        f"{actual:>8,}   "
        f"expected={expected:<8,} "
        f"{status}"
    )

    if actual != expected:
        raise RuntimeError(
            f"{name} validation failed: "
            f"expected {expected}, "
            f"got {actual}"
        )


def main() -> None:
    print("=" * 80)
    print(
        "SHAWAF ReID - "
        "Market1501 Loader Validation"
    )
    print("=" * 80)

    dataset = load_market1501()

    train_pids = unique_pids(
        dataset.train
    )

    query_pids = unique_pids(
        dataset.query
    )

    gallery_pids = unique_pids(
        dataset.gallery
    )

    all_cameras = (
        unique_cameras(dataset.train)
        | unique_cameras(dataset.query)
        | unique_cameras(dataset.gallery)
    )

    print()
    print("DATASET")
    print("-" * 80)

    print(
        f"Name                  : "
        f"{dataset.name}"
    )

    print(
        f"Unit type             : "
        f"{dataset.unit_type.value}"
    )

    print()
    print("IMAGE COUNTS")
    print("-" * 80)

    validate_value(
        "Train images",
        dataset.num_train,
        EXPECTED["train_images"],
    )

    validate_value(
        "Query images",
        dataset.num_query,
        EXPECTED["query_images"],
    )

    validate_value(
        "Gallery images",
        dataset.num_gallery,
        EXPECTED["gallery_images"],
    )

    print()
    print("IDENTITY COUNTS")
    print("-" * 80)

    validate_value(
        "Train IDs",
        len(train_pids),
        EXPECTED["train_ids"],
    )

    validate_value(
        "Query IDs",
        len(query_pids),
        EXPECTED["query_ids"],
    )

    validate_value(
        "Gallery IDs",
        len(gallery_pids),
        EXPECTED["gallery_ids"],
    )

    print()
    print("CAMERAS")
    print("-" * 80)

    validate_value(
        "Cameras",
        len(all_cameras),
        EXPECTED["cameras"],
    )

    print(
        f"Camera IDs            : "
        f"{sorted(all_cameras)}"
    )

    print()
    print("SPECIAL IDS")
    print("-" * 80)

    gallery_pid_counts = Counter(
        sample.person_id
        for sample in dataset.gallery
    )

    print(
        f"PID -1 present        : "
        f"{-1 in gallery_pid_counts}"
    )

    print(
        f"PID 0 present         : "
        f"{0 in gallery_pid_counts}"
    )

    if -1 in gallery_pid_counts:
        raise RuntimeError(
            "Junk PID -1 should not be "
            "present in loaded gallery."
        )

    print()
    print("EXAMPLE QUERY")
    print("-" * 80)

    query_sample = dataset.query[0]

    print(
        f"Sample ID             : "
        f"{query_sample.sample_id}"
    )

    print(
        f"PID                   : "
        f"{query_sample.person_id}"
    )

    print(
        f"Camera                : "
        f"{query_sample.camera_id}"
    )

    print(
        f"Image                 : "
        f"{query_sample.image_path}"
    )

    print()
    print("EXAMPLE GALLERY")
    print("-" * 80)

    gallery_sample = dataset.gallery[0]

    print(
        f"Sample ID             : "
        f"{gallery_sample.sample_id}"
    )

    print(
        f"PID                   : "
        f"{gallery_sample.person_id}"
    )

    print(
        f"Camera                : "
        f"{gallery_sample.camera_id}"
    )

    print(
        f"Image                 : "
        f"{gallery_sample.image_path}"
    )

    print()
    print("=" * 80)
    print(
        "Market1501 loader validation: "
        "SUCCESS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()