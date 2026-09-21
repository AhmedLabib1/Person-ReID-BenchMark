
from __future__ import annotations

from itertools import product
from pathlib import Path
from urllib.request import urlretrieve

import deeplake
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "mars"
)


# ============================================================
# Deep Lake datasets
# ============================================================

TRAIN_DATASET = "hub://activeloop/mars-train"
TEST_DATASET = "hub://activeloop/mars-test"

EXPECTED_TRAIN_IMAGES = 509_914
EXPECTED_TEST_IMAGES = 681_089


# ============================================================
# Official MARS metadata
# ============================================================

INFO_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "liangzheng06/MARS-evaluation/master/info"
)

INFO_FILES = [
    "train_name.txt",
    "test_name.txt",
    "tracks_train_info.mat",
    "tracks_test_info.mat",
    "query_IDX.mat",
]


# ============================================================
# Deep Lake tensor names
# ============================================================

IMAGE_TENSOR = "images"
PID_TENSOR = "pedestrian_ids"
CAMERA_TENSOR = "camera_nos"
TRACK_TENSOR = "track_nos"
FRAME_TENSOR = "frame_nos"


REQUIRED_TENSORS = {
    IMAGE_TENSOR,
    PID_TENSOR,
    CAMERA_TENSOR,
    TRACK_TENSOR,
    FRAME_TENSOR,
}


# ============================================================
# General utilities
# ============================================================

def get_numpy(value) -> np.ndarray:
    if hasattr(
        value,
        "numpy",
    ):
        value = value.numpy()

    return np.asarray(value)


def scalar_int(value) -> int:
    value = get_numpy(
        value
    )

    value = np.asarray(
        value
    ).reshape(-1)

    if value.size != 1:
        raise RuntimeError(
            "Expected scalar metadata value, "
            f"got shape {value.shape}"
        )

    return int(
        value[0]
    )


def count_images(
    folder: Path,
) -> int:
    if not folder.exists():
        return 0

    suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    return sum(
        1
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in suffixes
        )
    )


def read_official_names(
    path: Path,
) -> tuple[str, ...]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        names = tuple(
            line.strip()
            for line in file
            if line.strip()
        )

    if not names:
        raise RuntimeError(
            f"No names found in:\n{path}"
        )

    return names


# ============================================================
# Official filename construction
# ============================================================

def pid_token(
    pedestrian_id: int,
) -> str:
    """
    MARS person-id naming:

        1    -> 0001
        0    -> 0000
        -1   -> 00-1
    """

    if pedestrian_id == -1:
        return "00-1"

    if pedestrian_id < -1:
        raise RuntimeError(
            "Unexpected MARS pedestrian ID: "
            f"{pedestrian_id}"
        )

    return f"{pedestrian_id:04d}"


def build_filename(
    *,
    pedestrian_id: int,
    camera_no: int,
    track_no: int,
    frame_no: int,
) -> str:
    """
    Official MARS filename convention:

        xxxxCxTxxxxFxxx.jpg

    Example:

        0001C1T0001F001.jpg
    """

    if camera_no < 0:
        raise ValueError(
            f"Invalid camera number: {camera_no}"
        )

    if track_no < 0:
        raise ValueError(
            f"Invalid track number: {track_no}"
        )

    if frame_no < 0:
        raise ValueError(
            f"Invalid frame number: {frame_no}"
        )

    return (
        f"{pid_token(pedestrian_id)}"
        f"C{camera_no}"
        f"T{track_no:04d}"
        f"F{frame_no:03d}"
        f".jpg"
    )


# ============================================================
# Official metadata download
# ============================================================

def download_info() -> None:
    info_dir = (
        OUTPUT_ROOT
        / "info"
    )

    info_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("MARS OFFICIAL METADATA")
    print("=" * 80)

    for filename in INFO_FILES:
        output_file = (
            info_dir
            / filename
        )

        if (
            output_file.exists()
            and output_file.stat().st_size > 0
        ):
            print(
                f"[SKIP] {filename}"
            )
            continue

        url = (
            f"{INFO_BASE_URL}/"
            f"{filename}"
        )

        print(
            f"[DOWNLOAD] {filename}"
        )

        urlretrieve(
            url,
            output_file,
        )


# ============================================================
# Deep Lake checks
# ============================================================

def validate_tensors(
    ds,
) -> None:
    available = set(
        ds.tensors.keys()
    )

    print(
        "Available tensors:",
        sorted(available),
    )

    missing = (
        REQUIRED_TENSORS
        - available
    )

    if missing:
        raise RuntimeError(
            "Missing required Deep Lake tensors:\n"
            + "\n".join(
                sorted(missing)
            )
        )


# ============================================================
# Detect metadata offsets
# ============================================================

def detect_offsets(
    ds,
    official_names: tuple[str, ...],
) -> tuple[int, int, int]:
    """
    Deep Lake may encode camera / track / frame values
    using a slightly different zero/one-based convention.

    Determine the global offsets by checking a spread of
    samples against the official MARS filename list.

    Returns:

        camera_offset
        track_offset
        frame_offset
    """

    official_set = set(
        official_names
    )

    total = len(ds)

    probe_count = min(
        2_000,
        total,
    )

    probe_indices = np.linspace(
        0,
        total - 1,
        num=probe_count,
        dtype=np.int64,
    )

    candidates = list(
        product(
            (-1, 0, 1),
            (-1, 0, 1),
            (-1, 0, 1),
        )
    )

    scores = {
        candidate: 0
        for candidate in candidates
    }

    print()
    print(
        f"Detecting metadata offsets "
        f"using {probe_count:,} samples..."
    )

    for index in probe_indices:
        sample = ds[
            int(index)
        ]

        pid = scalar_int(
            sample[
                PID_TENSOR
            ]
        )

        camera = scalar_int(
            sample[
                CAMERA_TENSOR
            ]
        )

        track = scalar_int(
            sample[
                TRACK_TENSOR
            ]
        )

        frame = scalar_int(
            sample[
                FRAME_TENSOR
            ]
        )

        for (
            camera_offset,
            track_offset,
            frame_offset,
        ) in candidates:

            try:
                filename = build_filename(
                    pedestrian_id=pid,
                    camera_no=(
                        camera
                        + camera_offset
                    ),
                    track_no=(
                        track
                        + track_offset
                    ),
                    frame_no=(
                        frame
                        + frame_offset
                    ),
                )

            except ValueError:
                continue

            if filename in official_set:
                scores[
                    (
                        camera_offset,
                        track_offset,
                        frame_offset,
                    )
                ] += 1

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_offsets, best_score = (
        ranked[0]
    )

    print()
    print(
        "Best offsets:"
    )

    print(
        "  camera:",
        best_offsets[0],
    )

    print(
        "  track :",
        best_offsets[1],
    )

    print(
        "  frame :",
        best_offsets[2],
    )

    print(
        "Matched:",
        f"{best_score:,}/{probe_count:,}",
    )

    if best_score != probe_count:
        print()
        print(
            "Top offset candidates:"
        )

        for offsets, score in ranked[:5]:
            print(
                offsets,
                score,
            )

        raise RuntimeError(
            "Could not establish a perfect "
            "Deep Lake -> official MARS "
            "filename mapping."
        )

    return best_offsets


# ============================================================
# Image conversion
# ============================================================

def to_pil_image(
    image: np.ndarray,
) -> Image.Image:

    if (
        image.ndim == 4
        and image.shape[0] == 1
    ):
        image = image[0]

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (
                image * 255.0
            )

        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8
        )

    if image.ndim == 2:
        return Image.fromarray(
            image,
            mode="L",
        )

    if image.ndim != 3:
        raise RuntimeError(
            "Unexpected image shape: "
            f"{image.shape}"
        )

    if image.shape[-1] == 1:
        image = image.squeeze(
            axis=-1
        )

        return Image.fromarray(
            image,
            mode="L",
        )

    if image.shape[-1] == 3:
        return Image.fromarray(
            image
        )

    raise RuntimeError(
        "Unexpected image channels: "
        f"{image.shape}"
    )


# ============================================================
# Export one MARS split
# ============================================================

def export_split(
    *,
    dataset_url: str,
    output_dir: Path,
    names_path: Path,
    expected_count: int,
) -> None:

    print()
    print("=" * 80)
    print(
        f"MARS EXPORT: {output_dir.name}"
    )
    print("=" * 80)

    ds = deeplake.load(
        dataset_url,
        read_only=True,
    )

    validate_tensors(
        ds
    )

    official_names = (
        read_official_names(
            names_path
        )
    )

    official_set = set(
        official_names
    )

    dataset_count = len(ds)

    official_count = len(
        official_names
    )

    print()
    print(
        "Dataset samples :",
        f"{dataset_count:,}",
    )

    print(
        "Official names  :",
        f"{official_count:,}",
    )

    print(
        "Expected        :",
        f"{expected_count:,}",
    )

    if dataset_count != expected_count:
        raise RuntimeError(
            "Unexpected Deep Lake dataset size.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {dataset_count:,}"
        )

    if official_count != expected_count:
        raise RuntimeError(
            "Unexpected official name count.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {official_count:,}"
        )

    (
        camera_offset,
        track_offset,
        frame_offset,
    ) = detect_offsets(
        ds,
        official_names,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0
    skipped = 0

    print()
    print(
        "Exporting images..."
    )

    for index in tqdm(
        range(dataset_count),
        desc=output_dir.name,
    ):
        sample = ds[
            index
        ]

        pid = scalar_int(
            sample[
                PID_TENSOR
            ]
        )

        camera = (
            scalar_int(
                sample[
                    CAMERA_TENSOR
                ]
            )
            + camera_offset
        )

        track = (
            scalar_int(
                sample[
                    TRACK_TENSOR
                ]
            )
            + track_offset
        )

        frame = (
            scalar_int(
                sample[
                    FRAME_TENSOR
                ]
            )
            + frame_offset
        )

        filename = build_filename(
            pedestrian_id=pid,
            camera_no=camera,
            track_no=track,
            frame_no=frame,
        )

        if filename not in official_set:
            raise RuntimeError(
                "Generated MARS filename is "
                "not present in the official "
                "name list:\n"
                f"Index: {index}\n"
                f"Name : {filename}"
            )

        identity_dir = (
            output_dir
            / filename[:4]
        )

        identity_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            identity_dir
            / filename
        )

        # Resume support
        if output_file.exists():
            skipped += 1
            continue

        image = get_numpy(
            sample[
                IMAGE_TENSOR
            ]
        )

        pil_image = to_pil_image(
            image
        )

        pil_image.save(
            output_file,
            quality=95,
        )

        written += 1

    print()
    print(
        "Written:",
        f"{written:,}",
    )

    print(
        "Skipped:",
        f"{skipped:,}",
    )

    print(
        "Total:",
        f"{written + skipped:,}",
    )


# ============================================================
# Verify every official filename exists
# ============================================================

def verify_split_files(
    *,
    root: Path,
    names_path: Path,
    expected_count: int,
) -> None:

    names = read_official_names(
        names_path
    )

    if len(names) != expected_count:
        raise RuntimeError(
            f"Unexpected names count in "
            f"{names_path.name}: "
            f"{len(names):,}"
        )

    missing = []

    for filename in names:
        path = (
            root
            / filename[:4]
            / filename
        )

        if not path.exists():
            missing.append(
                filename
            )

    print(
        f"{root.name}: "
        f"{len(names) - len(missing):,}"
        f"/{len(names):,} official files present"
    )

    if missing:
        print()
        print(
            "First missing files:"
        )

        for filename in missing[:20]:
            print(
                "  ",
                filename,
            )

        raise RuntimeError(
            f"{len(missing):,} official "
            f"MARS files are missing."
        )


# ============================================================
# Final verification
# ============================================================

def verify() -> None:

    print()
    print("=" * 80)
    print("MARS FINAL VERIFICATION")
    print("=" * 80)

    train_dir = (
        OUTPUT_ROOT
        / "bbox_train"
    )

    test_dir = (
        OUTPUT_ROOT
        / "bbox_test"
    )

    info_dir = (
        OUTPUT_ROOT
        / "info"
    )

    verify_split_files(
        root=train_dir,
        names_path=(
            info_dir
            / "train_name.txt"
        ),
        expected_count=(
            EXPECTED_TRAIN_IMAGES
        ),
    )

    verify_split_files(
        root=test_dir,
        names_path=(
            info_dir
            / "test_name.txt"
        ),
        expected_count=(
            EXPECTED_TEST_IMAGES
        ),
    )

    print()
    print("Metadata:")

    for filename in INFO_FILES:

        path = (
            info_dir
            / filename
        )

        status = (
            "OK"
            if (
                path.exists()
                and path.stat().st_size > 0
            )
            else "MISSING"
        )

        print(
            f"  {filename:<25} "
            f"{status}"
        )

        if status != "OK":
            raise RuntimeError(
                "Missing MARS metadata: "
                f"{filename}"
            )

    print()
    print(
        "MARS dataset: PASS"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SHAWAF - MARS DATASET SETUP")
    print("=" * 80)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Official metadata first
    # --------------------------------------------------------

    download_info()

    info_dir = (
        OUTPUT_ROOT
        / "info"
    )

    # --------------------------------------------------------
    # 2. Train
    # --------------------------------------------------------

    train_dir = (
        OUTPUT_ROOT
        / "bbox_train"
    )

    train_count = count_images(
        train_dir
    )

    print()
    print(
        "Existing train images:",
        f"{train_count:,}",
    )

    if (
        train_count
        != EXPECTED_TRAIN_IMAGES
    ):
        export_split(
            dataset_url=TRAIN_DATASET,
            output_dir=train_dir,
            names_path=(
                info_dir
                / "train_name.txt"
            ),
            expected_count=(
                EXPECTED_TRAIN_IMAGES
            ),
        )

    else:
        print(
            "[SKIP] MARS train "
            "already complete."
        )

    # --------------------------------------------------------
    # 3. Test
    # --------------------------------------------------------

    test_dir = (
        OUTPUT_ROOT
        / "bbox_test"
    )

    test_count = count_images(
        test_dir
    )

    print()
    print(
        "Existing test images:",
        f"{test_count:,}",
    )

    if (
        test_count
        != EXPECTED_TEST_IMAGES
    ):
        export_split(
            dataset_url=TEST_DATASET,
            output_dir=test_dir,
            names_path=(
                info_dir
                / "test_name.txt"
            ),
            expected_count=(
                EXPECTED_TEST_IMAGES
            ),
        )

    else:
        print(
            "[SKIP] MARS test "
            "already complete."
        )

    # --------------------------------------------------------
    # 4. Strict verification
    # --------------------------------------------------------

    verify()


if __name__ == "__main__":
    main()
