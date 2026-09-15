from __future__ import annotations

from reid.data.msmt17 import (
    load_msmt17,
)


EXPECTED = {
    "train_images": 32621,
    "query_images": 11659,
    "gallery_images": 82161,

    "train_ids": 1041,
    "query_ids": 3060,
    "gallery_ids": 3060,

    "cameras": 15,
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
        "MSMT17 Loader Validation"
    )
    print("=" * 80)

    dataset = load_msmt17()

    train_pids = unique_pids(
        dataset.train
    )

    query_pids = unique_pids(
        dataset.query
    )

    gallery_pids = unique_pids(
        dataset.gallery
    )

    train_cameras = unique_cameras(
        dataset.train
    )

    query_cameras = unique_cameras(
        dataset.query
    )

    gallery_cameras = unique_cameras(
        dataset.gallery
    )

    all_cameras = (
        train_cameras
        | query_cameras
        | gallery_cameras
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
    print("QUERY / GALLERY ID CONSISTENCY")
    print("-" * 80)

    missing_in_gallery = (
        query_pids
        - gallery_pids
    )

    gallery_only = (
        gallery_pids
        - query_pids
    )

    print(
        f"Query IDs missing "
        f"in gallery          : "
        f"{len(missing_in_gallery)}"
    )

    print(
        f"Gallery-only IDs      : "
        f"{len(gallery_only)}"
    )

    if missing_in_gallery:
        raise RuntimeError(
            "Some MSMT17 query identities "
            "do not exist in the gallery."
        )

    print()
    print("EXAMPLE TRAIN")
    print("-" * 80)

    train_sample = dataset.train[0]

    print(
        f"Sample ID             : "
        f"{train_sample.sample_id}"
    )

    print(
        f"PID                   : "
        f"{train_sample.person_id}"
    )

    print(
        f"Camera                : "
        f"{train_sample.camera_id}"
    )

    print(
        f"Image                 : "
        f"{train_sample.image_path}"
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
        "MSMT17 loader validation: "
        "SUCCESS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()