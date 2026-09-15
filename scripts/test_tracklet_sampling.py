from __future__ import annotations

from reid.data.mars import load_mars
from reid.data.sampling import (
    DEFAULT_TRACKLET_FRAMES,
    uniform_sample_frames,
    uniform_sample_indices,
)


def main() -> None:
    print("=" * 80)
    print(
        "SHAWAF ReID - "
        "Tracklet Sampling Validation"
    )
    print("=" * 80)

    print()
    print("SYNTHETIC 40-FRAME TEST")
    print("-" * 80)

    indices = uniform_sample_indices(
        num_available_frames=40,
        num_samples=8,
    )

    human_positions = tuple(
        index + 1
        for index in indices
    )

    expected = (
        1,
        7,
        12,
        18,
        23,
        29,
        34,
        40,
    )

    print(
        f"Zero-based indices      : "
        f"{indices}"
    )

    print(
        f"Human frame positions  : "
        f"{human_positions}"
    )

    print(
        f"Expected               : "
        f"{expected}"
    )

    if human_positions != expected:
        raise RuntimeError(
            "40-frame uniform sampling "
            "does not match expected output."
        )

    print(
        "40-frame sampling      : PASS"
    )

    print()
    print("SHORT TRACKLET TEST")
    print("-" * 80)

    short_indices = uniform_sample_indices(
        num_available_frames=2,
        num_samples=8,
    )

    print(
        f"2 frames -> 8 samples  : "
        f"{short_indices}"
    )

    if len(short_indices) != 8:
        raise RuntimeError(
            "Short-tracklet sampling did not "
            "return exactly 8 frames."
        )

    print(
        "Short sampling         : PASS"
    )

    print()
    print("REAL MARS TRACKLET")
    print("-" * 80)

    dataset = load_mars()

    query = dataset.query[0]

    sampled_frames = uniform_sample_frames(
        frame_paths=query.frame_paths,
        num_samples=DEFAULT_TRACKLET_FRAMES,
    )

    print(
        f"Tracklet ID            : "
        f"{query.tracklet_id}"
    )

    print(
        f"PID                    : "
        f"{query.person_id}"
    )

    print(
        f"Camera                 : "
        f"{query.camera_id}"
    )

    print(
        f"Original frames        : "
        f"{query.num_frames}"
    )

    print(
        f"Sampled frames         : "
        f"{len(sampled_frames)}"
    )

    print()

    for index, path in enumerate(
        sampled_frames,
        start=1,
    ):
        print(
            f"{index:02d} -> {path.name}"
        )

    if len(sampled_frames) != 8:
        raise RuntimeError(
            "MARS tracklet did not produce "
            "exactly 8 sampled frames."
        )

    print()
    print("=" * 80)
    print(
        "Tracklet sampling validation: SUCCESS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()