from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    return [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def print_samples(
    title: str,
    files: list[Path],
    count: int = 5,
) -> None:
    print()
    print(title)
    print("-" * 80)

    for path in files[:count]:
        print(path.name)


def inspect_market() -> None:
    print()
    print("=" * 80)
    print("MARKET1501")
    print("=" * 80)

    root = (
        DATA_ROOT
        / "Market-1501-v15.09.15"
        / "Market-1501-v15.09.15"
    )

    train_dir = root / "bounding_box_train"
    query_dir = root / "query"
    gallery_dir = root / "bounding_box_test"

    train = image_files(train_dir)
    query = image_files(query_dir)
    gallery = image_files(gallery_dir)

    print(f"Root    : {root}")
    print(f"Train   : {len(train):,}")
    print(f"Query   : {len(query):,}")
    print(f"Gallery : {len(gallery):,}")

    print_samples(
        "Train sample filenames",
        train,
    )

    print_samples(
        "Query sample filenames",
        query,
    )

    print_samples(
        "Gallery sample filenames",
        gallery,
    )


def inspect_msmt17() -> None:
    print()
    print("=" * 80)
    print("MSMT17")
    print("=" * 80)

    root = (
        DATA_ROOT
        / "MSMT17"
        / "MSMT17"
    )

    train_dir = root / "train"
    query_dir = root / "query"
    gallery_dir = root / "gallery"

    train = image_files(train_dir)
    query = image_files(query_dir)
    gallery = image_files(gallery_dir)

    print(f"Root    : {root}")
    print(f"Train   : {len(train):,}")
    print(f"Query   : {len(query):,}")
    print(f"Gallery : {len(gallery):,}")

    print_samples(
        "Train sample filenames",
        train,
    )

    print_samples(
        "Query sample filenames",
        query,
    )

    print_samples(
        "Gallery sample filenames",
        gallery,
    )


def inspect_mars() -> None:
    print()
    print("=" * 80)
    print("MARS")
    print("=" * 80)

    root = DATA_ROOT / "mars"

    train_dir = root / "bbox_train"
    test_dir = root / "bbox_test"
    info_dir = root / "info"

    train_identity_dirs = [
        path
        for path in train_dir.iterdir()
        if path.is_dir()
    ] if train_dir.exists() else []

    test_identity_dirs = [
        path
        for path in test_dir.iterdir()
        if path.is_dir()
    ] if test_dir.exists() else []

    print(f"Root              : {root}")
    print(
        f"Train ID folders  : "
        f"{len(train_identity_dirs):,}"
    )
    print(
        f"Test ID folders   : "
        f"{len(test_identity_dirs):,}"
    )

    print()
    print("MARS info files")
    print("-" * 80)

    if info_dir.exists():
        info_files = sorted(
            path
            for path in info_dir.rglob("*")
            if path.is_file()
        )

        for path in info_files:
            print(path.name)
    else:
        print("INFO DIRECTORY NOT FOUND")

    print()
    print("Example train frames")
    print("-" * 80)

    for identity_dir in sorted(
        train_identity_dirs
    )[:2]:
        frames = image_files(
            identity_dir
        )

        print(
            f"{identity_dir.name}: "
            f"{len(frames)} frames"
        )

        for frame in frames[:3]:
            print(f"  {frame.name}")

    print()
    print("Example test frames")
    print("-" * 80)

    for identity_dir in sorted(
        test_identity_dirs
    )[:2]:
        frames = image_files(
            identity_dir
        )

        print(
            f"{identity_dir.name}: "
            f"{len(frames)} frames"
        )

        for frame in frames[:3]:
            print(f"  {frame.name}")


def main() -> None:
    print("=" * 80)
    print("SHAWAF ReID - Dataset Layout Inspection")
    print("=" * 80)

    inspect_market()
    inspect_msmt17()
    inspect_mars()


if __name__ == "__main__":
    main()