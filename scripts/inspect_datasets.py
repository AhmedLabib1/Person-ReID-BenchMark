
from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from reid.data.market1501 import (
    load_market1501,
)

from reid.data.msmt17 import (
    KNOWN_BAD_MSMT17_FILES,
    load_msmt17,
)

from reid.data.mars import (
    load_mars,
)

from reid.data.sampling import (
    DEFAULT_TRACKLET_FRAMES,
    uniform_sample_frames,
)


# ============================================================
# Frozen expected counts
# ============================================================

EXPECTED = {
    "Market1501": {
        "train": 12_936,
        "query": 3_368,
        "gallery": 15_913,
    },

    # Modified because the source archive contains:
    #
    # 1 corrupt train image
    # 1 corrupt query image
    # 1 corrupt gallery image
    #
    "MSMT17_V2": {
        "train": 32_620,
        "query": 11_658,
        "gallery": 82_160,
    },

    "MARS": {
        "train": 8_298,
        "query": 1_980,
        "gallery": 9_330,
    },
}


# ============================================================
# Generic helpers
# ============================================================

def unique_pids(samples) -> int:
    return len(
        {
            sample.person_id
            for sample in samples
        }
    )


def camera_range(samples) -> tuple[int, int]:
    cameras = [
        sample.camera_id
        for sample in samples
    ]

    if not cameras:
        raise RuntimeError(
            "Cannot compute camera range "
            "from an empty split."
        )

    return (
        min(cameras),
        max(cameras),
    )


def assert_counts(
    *,
    dataset_name: str,
    train_count: int,
    query_count: int,
    gallery_count: int,
) -> None:

    expected = EXPECTED[
        dataset_name
    ]

    assert (
        train_count
        == expected["train"]
    ), (
        f"{dataset_name} train mismatch: "
        f"{train_count:,} != "
        f"{expected['train']:,}"
    )

    assert (
        query_count
        == expected["query"]
    ), (
        f"{dataset_name} query mismatch: "
        f"{query_count:,} != "
        f"{expected['query']:,}"
    )

    assert (
        gallery_count
        == expected["gallery"]
    ), (
        f"{dataset_name} gallery mismatch: "
        f"{gallery_count:,} != "
        f"{expected['gallery']:,}"
    )


def assert_image_paths_exist(
    samples,
    *,
    split_name: str,
) -> None:

    missing = []

    for sample in samples:
        if not sample.image_path.exists():
            missing.append(
                sample.image_path
            )

    if missing:

        print()
        print(
            f"Missing {split_name} images:"
        )

        for path in missing[:20]:
            print(
                "  ",
                path,
            )

        raise RuntimeError(
            f"{len(missing):,} missing "
            f"{split_name} images."
        )


def assert_tracklet_paths_exist(
    tracklets,
    *,
    split_name: str,
) -> None:

    missing = []

    for tracklet in tracklets:

        if not tracklet.frame_paths:
            raise RuntimeError(
                "Tracklet has zero frames:\n"
                f"{tracklet.tracklet_id}"
            )

        for path in tracklet.frame_paths:

            if not path.exists():
                missing.append(
                    path
                )

    if missing:

        print()
        print(
            f"Missing MARS {split_name} frames:"
        )

        for path in missing[:20]:
            print(
                "  ",
                path,
            )

        raise RuntimeError(
            f"{len(missing):,} missing "
            f"MARS {split_name} frames."
        )


# ============================================================
# Market1501
# ============================================================

def inspect_market1501() -> None:

    print()
    print("=" * 90)
    print("MARKET1501")
    print("=" * 90)

    dataset = load_market1501()

    print(
        "Train   :",
        f"{dataset.num_train:,}",
    )

    print(
        "Query   :",
        f"{dataset.num_query:,}",
    )

    print(
        "Gallery :",
        f"{dataset.num_gallery:,}",
    )

    print()
    print(
        "Train PIDs   :",
        unique_pids(dataset.train),
    )

    print(
        "Query PIDs   :",
        unique_pids(dataset.query),
    )

    print(
        "Gallery PIDs :",
        unique_pids(dataset.gallery),
    )

    print()
    print(
        "Query cameras   :",
        camera_range(dataset.query),
    )

    print(
        "Gallery cameras :",
        camera_range(dataset.gallery),
    )

    assert_counts(
        dataset_name="Market1501",
        train_count=dataset.num_train,
        query_count=dataset.num_query,
        gallery_count=dataset.num_gallery,
    )

    # Internal Market cameras must be 0..5.
    assert camera_range(
        dataset.query
    ) == (0, 5)

    assert camera_range(
        dataset.gallery
    ) == (0, 5)

    print()
    print(
        "Checking Market query/gallery files..."
    )

    assert_image_paths_exist(
        dataset.query,
        split_name="Market query",
    )

    assert_image_paths_exist(
        dataset.gallery,
        split_name="Market gallery",
    )

    print(
        "Market1501: PASS"
    )


# ============================================================
# MSMT17_V2
# ============================================================

def inspect_msmt17() -> None:

    print()
    print("=" * 90)
    print("MSMT17_V2 - MODIFIED PROTOCOL")
    print("=" * 90)

    print(
        "Known corrupt exclusions:",
        len(
            KNOWN_BAD_MSMT17_FILES
        ),
    )

    for path in sorted(
        KNOWN_BAD_MSMT17_FILES
    ):
        print(
            "  -",
            path,
        )

    dataset = load_msmt17()

    print()
    print(
        "Train   :",
        f"{dataset.num_train:,}",
    )

    print(
        "Query   :",
        f"{dataset.num_query:,}",
    )

    print(
        "Gallery :",
        f"{dataset.num_gallery:,}",
    )

    print()
    print(
        "Train PIDs   :",
        unique_pids(dataset.train),
    )

    print(
        "Query PIDs   :",
        unique_pids(dataset.query),
    )

    print(
        "Gallery PIDs :",
        unique_pids(dataset.gallery),
    )

    print()
    print(
        "Query cameras   :",
        camera_range(dataset.query),
    )

    print(
        "Gallery cameras :",
        camera_range(dataset.gallery),
    )

    assert_counts(
        dataset_name="MSMT17_V2",
        train_count=dataset.num_train,
        query_count=dataset.num_query,
        gallery_count=dataset.num_gallery,
    )

    # Internal MSMT cameras must be 0..14.
    assert camera_range(
        dataset.query
    ) == (0, 14)

    assert camera_range(
        dataset.gallery
    ) == (0, 14)

    print()
    print(
        "Checking MSMT query/gallery files..."
    )

    assert_image_paths_exist(
        dataset.query,
        split_name="MSMT query",
    )

    assert_image_paths_exist(
        dataset.gallery,
        split_name="MSMT gallery",
    )

    print(
        "MSMT17_V2 modified protocol: PASS"
    )


# ============================================================
# MARS
# ============================================================

def inspect_mars() -> None:

    print()
    print("=" * 90)
    print("MARS")
    print("=" * 90)

    dataset = load_mars()

    print(
        "Train tracklets   :",
        f"{dataset.num_train:,}",
    )

    print(
        "Query tracklets   :",
        f"{dataset.num_query:,}",
    )

    print(
        "Gallery tracklets :",
        f"{dataset.num_gallery:,}",
    )

    print()
    print(
        "Train PIDs   :",
        unique_pids(dataset.train),
    )

    print(
        "Query PIDs   :",
        unique_pids(dataset.query),
    )

    print(
        "Gallery PIDs :",
        unique_pids(dataset.gallery),
    )

    print()
    print(
        "Query cameras   :",
        camera_range(dataset.query),
    )

    print(
        "Gallery cameras :",
        camera_range(dataset.gallery),
    )

    assert_counts(
        dataset_name="MARS",
        train_count=dataset.num_train,
        query_count=dataset.num_query,
        gallery_count=dataset.num_gallery,
    )

    # Internal MARS cameras must be 0..5.
    assert camera_range(
        dataset.query
    ) == (0, 5)

    assert camera_range(
        dataset.gallery
    ) == (0, 5)

    # --------------------------------------------------------
    # Tracklet length statistics
    # --------------------------------------------------------

    all_eval_tracklets = (
        dataset.query
        + dataset.gallery
    )

    lengths = [
        tracklet.num_frames
        for tracklet
        in all_eval_tracklets
    ]

    print()
    print(
        "Eval tracklets:",
        f"{len(lengths):,}",
    )

    print(
        "Shortest tracklet:",
        min(lengths),
        "frames",
    )

    print(
        "Longest tracklet :",
        max(lengths),
        "frames",
    )

    print(
        "Average frames   :",
        f"{sum(lengths) / len(lengths):.2f}",
    )

    # --------------------------------------------------------
    # Validate uniform-8 contract for every eval tracklet
    # --------------------------------------------------------

    print()
    print(
        "Checking uniform-8 sampling..."
    )

    for tracklet in all_eval_tracklets:

        sampled = uniform_sample_frames(
            tracklet.frame_paths,
            num_samples=(
                DEFAULT_TRACKLET_FRAMES
            ),
        )

        if (
            len(sampled)
            != DEFAULT_TRACKLET_FRAMES
        ):
            raise RuntimeError(
                "MARS uniform sampling did "
                "not return exactly 8 frames."
            )

    print(
        "Uniform-8 sampling: PASS"
    )

    # --------------------------------------------------------
    # Strict frame existence
    # --------------------------------------------------------

    print()
    print(
        "Checking MARS query frames..."
    )

    assert_tracklet_paths_exist(
        dataset.query,
        split_name="query",
    )

    print(
        "Checking MARS gallery frames..."
    )

    assert_tracklet_paths_exist(
        dataset.gallery,
        split_name="gallery",
    )

    # --------------------------------------------------------
    # Example tracklet
    # --------------------------------------------------------

    example = dataset.query[0]

    sampled = uniform_sample_frames(
        example.frame_paths,
        num_samples=8,
    )

    print()
    print(
        "Example query tracklet:"
    )

    print(
        "Tracklet ID :",
        example.tracklet_id,
    )

    print(
        "PID         :",
        example.person_id,
    )

    print(
        "Camera      :",
        example.camera_id,
    )

    print(
        "Frames      :",
        example.num_frames,
    )

    print(
        "Sampled     :",
        len(sampled),
    )

    print()
    print(
        "MARS: PASS"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 90)
    print("SHAWAF DUKE BENCHMARK - DATASET VALIDATION")
    print("=" * 90)

    inspect_market1501()

    inspect_msmt17()

    inspect_mars()

    print()
    print("=" * 90)
    print("ALL DATASET LOADERS: PASS")
    print("=" * 90)

    print()
    print(
        "Frozen evaluation matrix:"
    )

    print(
        "  Market1501 : "
        "3,368 query / "
        "15,913 gallery"
    )

    print(
        "  MSMT17_V2  : "
        "11,658 query / "
        "82,160 gallery "
        "(modified)"
    )

    print(
        "  MARS       : "
        "1,980 query / "
        "9,330 gallery tracklets"
    )


if __name__ == "__main__":
    main()
