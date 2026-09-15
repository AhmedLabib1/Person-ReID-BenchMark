from __future__ import annotations

import numpy as np

from reid.evaluation.rank import (
    evaluate_rankings,
)


def assert_close(
    name: str,
    actual: float,
    expected: float,
    tolerance: float = 1e-8,
) -> None:
    difference = abs(
        actual - expected
    )

    status = (
        "PASS"
        if difference <= tolerance
        else "FAIL"
    )

    print(
        f"{name:<20}"
        f"{actual:.8f}   "
        f"expected={expected:.8f}   "
        f"{status}"
    )

    if difference > tolerance:
        raise RuntimeError(
            f"{name} failed: "
            f"expected {expected}, "
            f"got {actual}"
        )


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "CMC / mAP / mINP Validation"
    )
    print("=" * 88)

    # --------------------------------------------------
    # Gallery
    # --------------------------------------------------
    #
    # index:
    #
    # 0 -> PID 1, cam 0
    # 1 -> PID 1, cam 1
    # 2 -> negative
    # 3 -> PID 1, cam 2
    #
    # 4 -> PID 2, cam 0
    # 5 -> negative
    # 6 -> PID 2, cam 1
    # 7 -> negative
    # 8 -> negative
    # 9 -> PID 2, cam 2
    #
    # 10,11 -> negatives
    # --------------------------------------------------

    gallery_pids = np.array(
        [
            1,
            1,
            9,
            1,
            2,
            8,
            2,
            7,
            6,
            2,
            5,
            4,
        ],
        dtype=np.int64,
    )

    gallery_camera_ids = np.array(
        [
            0,
            1,
            0,
            2,
            0,
            1,
            1,
            2,
            3,
            2,
            4,
            5,
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------
    #
    # Query 0:
    #
    # PID 1, camera 0
    #
    # Gallery index 0 must be removed because:
    #
    # same PID + same camera
    #
    # Remaining positive ranks:
    #
    # Rank 1
    # Rank 3
    #
    # AP:
    #
    # (1/1 + 2/3) / 2
    # = 0.83333333
    #
    # INP:
    #
    # 2 / 3
    # = 0.66666667
    # --------------------------------------------------
    #
    # Query 1:
    #
    # PID 2, camera 0
    #
    # Gallery index 4 must be removed.
    #
    # Remaining positive ranks:
    #
    # Rank 2
    # Rank 5
    #
    # AP:
    #
    # (1/2 + 2/5) / 2
    # = 0.45
    #
    # INP:
    #
    # 2 / 5
    # = 0.4
    # --------------------------------------------------
    #
    # Query 2:
    #
    # PID 3
    #
    # No gallery positive.
    #
    # Must be SKIPPED.
    # --------------------------------------------------

    query_pids = np.array(
        [
            1,
            2,
            3,
        ],
        dtype=np.int64,
    )

    query_camera_ids = np.array(
        [
            0,
            0,
            0,
        ],
        dtype=np.int64,
    )

    rankings = np.array(
        [
            # Query PID 1
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            ],

            # Query PID 2
            [
                5,
                6,
                7,
                8,
                9,
                4,
                0,
                1,
                2,
                3,
                10,
                11,
            ],

            # Query PID 3
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            ],
        ],
        dtype=np.int64,
    )

    results = evaluate_rankings(
        rankings=rankings,
        query_pids=query_pids,
        query_camera_ids=(
            query_camera_ids
        ),
        gallery_pids=gallery_pids,
        gallery_camera_ids=(
            gallery_camera_ids
        ),
        max_rank=10,
    )

    expected_ap_query_1 = (
        (
            1.0 / 1.0
            + 2.0 / 3.0
        )
        / 2.0
    )

    expected_ap_query_2 = (
        (
            1.0 / 2.0
            + 2.0 / 5.0
        )
        / 2.0
    )

    expected_map = (
        (
            expected_ap_query_1
            + expected_ap_query_2
        )
        / 2.0
    )

    expected_inp_query_1 = (
        2.0 / 3.0
    )

    expected_inp_query_2 = (
        2.0 / 5.0
    )

    expected_minp = (
        (
            expected_inp_query_1
            + expected_inp_query_2
        )
        / 2.0
    )

    print()
    print("QUERY ACCOUNTING")
    print("-" * 88)

    print(
        f"Total queries           : "
        f"{results.total_queries}"
    )

    print(
        f"Valid queries           : "
        f"{results.valid_queries}"
    )

    print(
        f"Skipped queries         : "
        f"{results.skipped_queries}"
    )

    if results.total_queries != 3:
        raise RuntimeError(
            "Unexpected total query count."
        )

    if results.valid_queries != 2:
        raise RuntimeError(
            "Expected exactly 2 valid queries."
        )

    if results.skipped_queries != 1:
        raise RuntimeError(
            "Expected exactly 1 skipped query."
        )

    print(
        "Query accounting       : PASS"
    )

    print()
    print("CMC")
    print("-" * 88)

    assert_close(
        "Rank-1",
        results.rank(1),
        0.5,
    )

    assert_close(
        "Rank-5",
        results.rank(5),
        1.0,
    )

    assert_close(
        "Rank-10",
        results.rank(10),
        1.0,
    )

    print()
    print("MEAN METRICS")
    print("-" * 88)

    assert_close(
        "mAP",
        results.mean_ap,
        expected_map,
    )

    assert_close(
        "mINP",
        results.mean_inp,
        expected_minp,
    )

    print()
    print("EXPECTED DETAILS")
    print("-" * 88)

    print(
        f"Query 1 AP             : "
        f"{expected_ap_query_1:.8f}"
    )

    print(
        f"Query 2 AP             : "
        f"{expected_ap_query_2:.8f}"
    )

    print(
        f"Expected mAP           : "
        f"{expected_map:.8f}"
    )

    print(
        f"Expected mINP          : "
        f"{expected_minp:.8f}"
    )

    print()
    print("=" * 88)
    print(
        "ReID metric validation: SUCCESS"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()