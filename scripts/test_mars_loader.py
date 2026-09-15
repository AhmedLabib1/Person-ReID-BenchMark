from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from reid.data.mars import (
    MARS_ROOT,
    load_mars,
)


EXPECTED = {
    "train_tracklets": 8298,
    "query_tracklets": 1980,
    "gallery_tracklets": 9330,

    "train_ids": 625,
    "query_ids": 626,
    "gallery_ids": 621,

    "train_name_count": 509914,
    "test_name_count": 681089,

    "cameras": 6,

    # Standard MARS evaluation:
    #
    # 1980 total query tracklets
    # 1840 have at least one valid positive
    # after same-PID + same-camera removal.
    "valid_queries": 1840,
    "skipped_queries": 140,
}


def count_lines(
    path: Path,
) -> int:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return sum(
            1
            for line in file
            if line.strip()
        )


def unique_pids(
    samples,
) -> set[int]:
    return {
        sample.person_id
        for sample in samples
    }


def unique_cameras(
    samples,
) -> set[int]:
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
        f"{name:<26}"
        f"{actual:>10,}   "
        f"expected={expected:<10,} "
        f"{status}"
    )

    if actual != expected:
        raise RuntimeError(
            f"{name} validation failed: "
            f"expected {expected}, "
            f"got {actual}"
        )


def validate_sample_paths(
    samples,
    split_name: str,
) -> None:
    if not samples:
        raise RuntimeError(
            f"{split_name} is empty."
        )

    sample_indices = {
        0,
        len(samples) // 2,
        len(samples) - 1,
    }

    for index in sorted(
        sample_indices
    ):
        sample = samples[index]

        if not sample.frame_paths:
            raise RuntimeError(
                f"{split_name} tracklet "
                "contains no frames."
            )

        paths_to_check = {
            sample.frame_paths[0],
            sample.frame_paths[
                len(sample.frame_paths) // 2
            ],
            sample.frame_paths[-1],
        }

        for path in paths_to_check:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing MARS frame:\n"
                    f"{path}"
                )


def analyze_query_validity(
    query_samples,
    gallery_samples,
) -> tuple[int, int, int, int]:
    """
    Analyze queries according to the standard
    person ReID evaluation protocol.

    For one query:

        1. Find gallery tracklets with same PID.
        2. Remove same-PID + same-camera matches.
        3. If no positive remains, the query is skipped.

    Returns:
        valid_queries
        skipped_queries
        missing_pid_queries
        same_camera_only_queries
    """

    gallery_by_pid = defaultdict(
        list
    )

    for gallery in gallery_samples:
        gallery_by_pid[
            gallery.person_id
        ].append(
            gallery
        )

    valid_queries = 0
    skipped_queries = 0

    missing_pid_queries = 0
    same_camera_only_queries = 0

    for query in query_samples:
        same_pid_gallery = (
            gallery_by_pid.get(
                query.person_id,
                [],
            )
        )

        # PID does not appear in gallery at all.
        if not same_pid_gallery:
            missing_pid_queries += 1
            skipped_queries += 1
            continue

        # Standard ReID protocol:
        #
        # same PID + same camera
        # is not a valid positive match.
        valid_positives = [
            gallery
            for gallery in same_pid_gallery
            if (
                gallery.camera_id
                != query.camera_id
            )
        ]

        if not valid_positives:
            same_camera_only_queries += 1
            skipped_queries += 1
            continue

        valid_queries += 1

    return (
        valid_queries,
        skipped_queries,
        missing_pid_queries,
        same_camera_only_queries,
    )


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "MARS Loader Validation"
    )
    print("=" * 88)

    train_name_path = (
        MARS_ROOT
        / "info"
        / "train_name.txt"
    )

    test_name_path = (
        MARS_ROOT
        / "info"
        / "test_name.txt"
    )

    train_name_count = count_lines(
        train_name_path
    )

    test_name_count = count_lines(
        test_name_path
    )

    print()
    print("RAW FRAME METADATA")
    print("-" * 88)

    validate_value(
        "Train frame names",
        train_name_count,
        EXPECTED["train_name_count"],
    )

    validate_value(
        "Test frame names",
        test_name_count,
        EXPECTED["test_name_count"],
    )

    dataset = load_mars()

    print()
    print("DATASET")
    print("-" * 88)

    print(
        f"Name                      : "
        f"{dataset.name}"
    )

    print(
        f"Unit type                 : "
        f"{dataset.unit_type.value}"
    )

    print()
    print("TRACKLET COUNTS")
    print("-" * 88)

    validate_value(
        "Train tracklets",
        dataset.num_train,
        EXPECTED["train_tracklets"],
    )

    validate_value(
        "Query tracklets",
        dataset.num_query,
        EXPECTED["query_tracklets"],
    )

    validate_value(
        "Gallery tracklets",
        dataset.num_gallery,
        EXPECTED["gallery_tracklets"],
    )

    train_pids = unique_pids(
        dataset.train
    )

    query_pids = unique_pids(
        dataset.query
    )

    gallery_pids = unique_pids(
        dataset.gallery
    )

    print()
    print("IDENTITIES")
    print("-" * 88)

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
    print("JUNK IDENTITIES")
    print("-" * 88)

    print(
        f"PID -1 in train           : "
        f"{-1 in train_pids}"
    )

    print(
        f"PID -1 in query           : "
        f"{-1 in query_pids}"
    )

    print(
        f"PID -1 in gallery         : "
        f"{-1 in gallery_pids}"
    )

    if (
        -1 in train_pids
        or -1 in query_pids
        or -1 in gallery_pids
    ):
        raise RuntimeError(
            "MARS junk PID -1 was not "
            "filtered correctly."
        )

    all_cameras = (
        unique_cameras(
            dataset.train
        )
        | unique_cameras(
            dataset.query
        )
        | unique_cameras(
            dataset.gallery
        )
    )

    print()
    print("CAMERAS")
    print("-" * 88)

    validate_value(
        "Cameras",
        len(all_cameras),
        EXPECTED["cameras"],
    )

    print(
        f"Camera IDs                : "
        f"{sorted(all_cameras)}"
    )

    all_tracklets = (
        dataset.train
        + dataset.query
        + dataset.gallery
    )

    tracklet_lengths = [
        sample.num_frames
        for sample in all_tracklets
    ]

    average_frames = (
        sum(tracklet_lengths)
        / len(tracklet_lengths)
    )

    print()
    print("TRACKLET LENGTHS")
    print("-" * 88)

    print(
        f"Minimum frames            : "
        f"{min(tracklet_lengths):,}"
    )

    print(
        f"Maximum frames            : "
        f"{max(tracklet_lengths):,}"
    )

    print(
        f"Average frames            : "
        f"{average_frames:.2f}"
    )

    print()
    print("QUERY / GALLERY ID COVERAGE")
    print("-" * 88)

    query_ids_missing_from_gallery = (
        query_pids
        - gallery_pids
    )

    gallery_only_ids = (
        gallery_pids
        - query_pids
    )

    print(
        f"Query IDs not in gallery  : "
        f"{len(query_ids_missing_from_gallery)}"
    )

    print(
        f"Gallery-only IDs          : "
        f"{len(gallery_only_ids)}"
    )

    print()
    print("EVALUATION QUERY VALIDITY")
    print("-" * 88)

    (
        valid_queries,
        skipped_queries,
        missing_pid_queries,
        same_camera_only_queries,
    ) = analyze_query_validity(
        dataset.query,
        dataset.gallery,
    )

    validate_value(
        "Valid queries",
        valid_queries,
        EXPECTED["valid_queries"],
    )

    validate_value(
        "Skipped queries",
        skipped_queries,
        EXPECTED["skipped_queries"],
    )

    print()
    print(
        f"No PID in gallery         : "
        f"{missing_pid_queries:,}"
    )

    print(
        f"Same-camera positives only: "
        f"{same_camera_only_queries:,}"
    )

    print()
    print(
        "NOTE: skipped queries are expected "
        "in the standard MARS protocol."
    )

    print(
        "They must not contribute to "
        "CMC, mAP, or mINP."
    )

    print()
    print("FRAME PATH VALIDATION")
    print("-" * 88)

    validate_sample_paths(
        dataset.train,
        "train",
    )

    validate_sample_paths(
        dataset.query,
        "query",
    )

    validate_sample_paths(
        dataset.gallery,
        "gallery",
    )

    print(
        "Sampled frame paths      : PASS"
    )

    print()
    print("EXAMPLE QUERY TRACKLET")
    print("-" * 88)

    query = dataset.query[0]

    print(
        f"Tracklet ID               : "
        f"{query.tracklet_id}"
    )

    print(
        f"PID                       : "
        f"{query.person_id}"
    )

    print(
        f"Camera                    : "
        f"{query.camera_id}"
    )

    print(
        f"Frames                    : "
        f"{query.num_frames}"
    )

    print(
        f"First frame               : "
        f"{query.frame_paths[0]}"
    )

    print(
        f"Last frame                : "
        f"{query.frame_paths[-1]}"
    )

    print()
    print("=" * 88)
    print(
        "MARS loader validation: SUCCESS"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()