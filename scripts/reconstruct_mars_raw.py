
from __future__ import annotations

import zipfile
from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARS_ZIP = (
    PROJECT_ROOT
    / "data"
    / "_downloads"
    / "mars.zip"
)

MARS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "mars"
)

TRAIN_ROOT = (
    MARS_ROOT
    / "bbox_train"
)

TEST_ROOT = (
    MARS_ROOT
    / "bbox_test"
)

INFO_ROOT = (
    MARS_ROOT
    / "info"
)

EXPECTED_TRAIN = 509_914
EXPECTED_TEST = 681_089


def read_names(
    path: Path,
) -> set[str]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return {
            line.strip()
            for line in file
            if line.strip()
        }


def count_images(
    root: Path,
) -> int:

    if not root.exists():
        return 0

    return sum(
        1
        for path in root.rglob("*.jpg")
    )


def extract_split(
    *,
    zf: zipfile.ZipFile,
    archive_prefix: str,
    destination_root: Path,
    official_names: set[str],
) -> None:

    members = [
        info
        for info in zf.infolist()
        if (
            info.filename.startswith(
                archive_prefix
            )
            and info.filename.lower().endswith(
                ".jpg"
            )
        )
    ]

    print()
    print(
        f"{archive_prefix} -> "
        f"{destination_root}"
    )

    print(
        "Archive images:",
        f"{len(members):,}",
    )

    destination_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0
    skipped = 0

    for info in tqdm(
        members,
        desc=destination_root.name,
    ):

        filename = Path(
            info.filename
        ).name

        if filename not in official_names:
            raise RuntimeError(
                "Unexpected filename in archive:\n"
                f"{filename}"
            )

        person_folder = (
            filename[:4]
        )

        destination = (
            destination_root
            / person_folder
            / filename
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            skipped += 1
            continue

        with zf.open(
            info,
            "r",
        ) as source:

            with destination.open(
                "wb",
            ) as output:

                while True:

                    chunk = source.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    output.write(
                        chunk
                    )

        written += 1

    print(
        "Written:",
        f"{written:,}",
    )

    print(
        "Skipped:",
        f"{skipped:,}",
    )


def verify_split(
    *,
    root: Path,
    names: set[str],
    expected: int,
) -> None:

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

    image_count = count_images(
        root
    )

    print()
    print(
        root.name,
        ":",
        f"{image_count:,}",
    )

    print(
        "Expected:",
        f"{expected:,}",
    )

    print(
        "Missing official names:",
        len(missing),
    )

    if image_count != expected:
        raise RuntimeError(
            f"{root.name} count mismatch."
        )

    if missing:
        raise RuntimeError(
            f"{len(missing)} official "
            f"files missing from {root.name}."
        )


def main() -> None:

    print("=" * 90)
    print("RECONSTRUCT ORIGINAL MARS LAYOUT")
    print("=" * 90)

    if not MARS_ZIP.exists():
        raise FileNotFoundError(
            f"MARS ZIP not found:\n"
            f"{MARS_ZIP}"
        )

    train_names = read_names(
        INFO_ROOT
        / "train_name.txt"
    )

    test_names = read_names(
        INFO_ROOT
        / "test_name.txt"
    )

    if len(train_names) != EXPECTED_TRAIN:
        raise RuntimeError(
            "Unexpected train_name count."
        )

    if len(test_names) != EXPECTED_TEST:
        raise RuntimeError(
            "Unexpected test_name count."
        )

    with zipfile.ZipFile(
        MARS_ZIP,
        "r",
    ) as zf:

        # train_split contains complete
        # original training image pool.
        extract_split(
            zf=zf,
            archive_prefix=(
                "mars/train_split/"
            ),
            destination_root=TRAIN_ROOT,
            official_names=train_names,
        )

        # gallery_split contains the complete
        # original TEST image pool:
        #
        # all 12,180 raw test tracklets.
        #
        # query_IDX.mat will later split this
        # into official query/gallery tracklets.
        extract_split(
            zf=zf,
            archive_prefix=(
                "mars/gallery_split/"
            ),
            destination_root=TEST_ROOT,
            official_names=test_names,
        )

    print()
    print("=" * 90)
    print("STRICT FILE VERIFICATION")
    print("=" * 90)

    verify_split(
        root=TRAIN_ROOT,
        names=train_names,
        expected=EXPECTED_TRAIN,
    )

    verify_split(
        root=TEST_ROOT,
        names=test_names,
        expected=EXPECTED_TEST,
    )

    print()
    print(
        "MARS raw reconstruction: PASS"
    )


if __name__ == "__main__":
    main()
