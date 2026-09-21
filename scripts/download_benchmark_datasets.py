
from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"

DOWNLOAD_ROOT = (
    DATA_ROOT
    / "_downloads"
)

DATA_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

DOWNLOAD_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MSMT17_V2
# ============================================================

MSMT_SHARE_URL = (
    "https://pkueducn-my.sharepoint.com/"
    ":u:/g/personal/"
    "slzhang_jdl_pkueducn_onmicrosoft_com/"
    "EcY9ee0gArNKqzsFVlPxdeEBPEAHGJS4ACtmEK3LXppSbA"
)

MSMT_ARCHIVE = (
    DOWNLOAD_ROOT
    / "MSMT17_V2.archive"
)

MSMT_CONTAINER = (
    DATA_ROOT
    / "MSMT17_V2"
)

MSMT_TARGET = (
    MSMT_CONTAINER
    / "MSMT17_V2"
)

MSMT_REQUIRED = [
    "mask_train_v2",
    "mask_test_v2",
    "list_train.txt",
    "list_val.txt",
    "list_query.txt",
    "list_gallery.txt",
]


# ============================================================
# Market1501
# ============================================================

MARKET_ZIP_NAME = (
    "Market-1501-v15.09.15.zip"
)

MARKET_CONTAINER = (
    DATA_ROOT
    / "Market-1501-v15.09.15"
)

MARKET_TARGET = (
    MARKET_CONTAINER
    / "Market-1501-v15.09.15"
)

MARKET_REQUIRED = [
    "bounding_box_train",
    "bounding_box_test",
    "query",
]


# ============================================================
# MARS
# ============================================================

MARS_ROOT = (
    DATA_ROOT
    / "mars"
)

# Try the paths you supplied first.
#
# Activeloop's dataset page currently also documents
# emnist-mars-* names, so those are fallback candidates.

MARS_TRAIN_CANDIDATES = [
    "hub://activeloop/mars-train",
    "hub://activeloop/emnist-mars-train",
]

MARS_TEST_CANDIDATES = [
    "hub://activeloop/mars-test",
    "hub://activeloop/emnist-mars-test",
]

EXPECTED_MARS_TRAIN_IMAGES = 509_914
EXPECTED_MARS_TEST_IMAGES = 681_089

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
# General utilities
# ============================================================

def has_required(
    root: Path,
    required: list[str],
) -> bool:

    return (
        root.exists()
        and all(
            (root / item).exists()
            for item in required
        )
    )


def download_stream(
    url: str,
    output_path: Path,
) -> None:
    """
    Download with simple resume support.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    existing = (
        part_path.stat().st_size
        if part_path.exists()
        else 0
    )

    headers = {}

    if existing > 0:
        headers["Range"] = (
            f"bytes={existing}-"
        )

        print(
            f"Resuming from "
            f"{existing / (1024 ** 2):.2f} MB"
        )

    with requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=60,
    ) as response:

        response.raise_for_status()

        # If server ignored Range, start again.
        if (
            existing > 0
            and response.status_code != 206
        ):
            existing = 0

        mode = (
            "ab"
            if existing > 0
            else "wb"
        )

        content_length = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        total = (
            existing + content_length
            if content_length
            else None
        )

        with part_path.open(
            mode
        ) as file:

            with tqdm(
                total=total,
                initial=existing,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    file.write(
                        chunk
                    )

                    progress.update(
                        len(chunk)
                    )

    part_path.replace(
        output_path
    )


def extract_archive(
    archive: Path,
    destination: Path,
) -> None:

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    if zipfile.is_zipfile(
        archive
    ):
        print(
            "Archive type: ZIP"
        )

        with zipfile.ZipFile(
            archive
        ) as file:
            file.extractall(
                destination
            )

        return

    if tarfile.is_tarfile(
        archive
    ):
        print(
            "Archive type: TAR"
        )

        with tarfile.open(
            archive
        ) as file:
            file.extractall(
                destination
            )

        return

    first_bytes = archive.read_bytes()[:200]

    raise RuntimeError(
        "Downloaded MSMT file is not a "
        "supported ZIP/TAR archive.\n"
        "The SharePoint link may have returned "
        "an HTML/login page instead.\n\n"
        f"First bytes:\n{first_bytes!r}"
    )


def find_dataset_root(
    search_root: Path,
    required: list[str],
) -> Path | None:

    if has_required(
        search_root,
        required,
    ):
        return search_root

    for directory in search_root.rglob(
        "*"
    ):
        if not directory.is_dir():
            continue

        if has_required(
            directory,
            required,
        ):
            return directory

    return None


# ============================================================
# Market1501
# ============================================================

def locate_market_zip() -> Path:

    candidates = [
        PROJECT_ROOT
        / MARKET_ZIP_NAME,

        PROJECT_ROOT.parent
        / MARKET_ZIP_NAME,

        Path.cwd()
        / MARKET_ZIP_NAME,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Search Lightning studio directory.
    search_roots = [
        Path.cwd(),
        PROJECT_ROOT.parent,
    ]

    for root in search_roots:

        if not root.exists():
            continue

        for candidate in root.glob(
            f"**/{MARKET_ZIP_NAME}"
        ):
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "\nCould not find:\n"
        f"  {MARKET_ZIP_NAME}\n\n"
        "Upload the Market ZIP into the "
        "Lightning Studio/notebook root, "
        "then run this script again."
    )


def setup_market1501() -> None:

    print()
    print("=" * 80)
    print("MARKET1501")
    print("=" * 80)

    if has_required(
        MARKET_TARGET,
        MARKET_REQUIRED,
    ):
        print(
            "[SKIP] Market1501 already prepared."
        )
        print(
            MARKET_TARGET
        )
        return

    archive = locate_market_zip()

    print(
        "Archive:",
        archive,
    )

    MARKET_CONTAINER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Extracting..."
    )

    with zipfile.ZipFile(
        archive
    ) as file:
        file.extractall(
            MARKET_CONTAINER
        )

    if not has_required(
        MARKET_TARGET,
        MARKET_REQUIRED,
    ):
        discovered = find_dataset_root(
            MARKET_CONTAINER,
            MARKET_REQUIRED,
        )

        if discovered is None:
            raise RuntimeError(
                "Market1501 extraction completed, "
                "but dataset root could not be found."
            )

        if discovered != MARKET_TARGET:

            if MARKET_TARGET.exists():
                shutil.rmtree(
                    MARKET_TARGET
                )

            MARKET_TARGET.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(discovered),
                str(MARKET_TARGET),
            )

    if not has_required(
        MARKET_TARGET,
        MARKET_REQUIRED,
    ):
        raise RuntimeError(
            "Market1501 validation failed."
        )

    print(
        "Market1501: PASS"
    )
    print(
        MARKET_TARGET
    )


# ============================================================
# MSMT17
# ============================================================

def setup_msmt17() -> None:

    print()
    print("=" * 80)
    print("MSMT17_V2")
    print("=" * 80)

    if has_required(
        MSMT_TARGET,
        MSMT_REQUIRED,
    ):
        print(
            "[SKIP] MSMT17_V2 already prepared."
        )
        print(
            MSMT_TARGET
        )
        return

    # SharePoint direct-download flag.
    download_url = (
        MSMT_SHARE_URL
        + "?download=1"
    )

    if not MSMT_ARCHIVE.exists():

        print(
            "Downloading MSMT17_V2..."
        )

        download_stream(
            download_url,
            MSMT_ARCHIVE,
        )

    else:
        print(
            "[SKIP] MSMT archive already downloaded."
        )

    extraction_root = (
        DATA_ROOT
        / "_msmt_extract"
    )

    if extraction_root.exists():
        shutil.rmtree(
            extraction_root
        )

    print(
        "Extracting MSMT17_V2..."
    )

    extract_archive(
        MSMT_ARCHIVE,
        extraction_root,
    )

    discovered = find_dataset_root(
        extraction_root,
        MSMT_REQUIRED,
    )

    if discovered is None:
        raise RuntimeError(
            "Could not locate the MSMT17_V2 "
            "dataset root after extraction."
        )

    MSMT_CONTAINER.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MSMT_TARGET.exists():
        shutil.rmtree(
            MSMT_TARGET
        )

    shutil.move(
        str(discovered),
        str(MSMT_TARGET),
    )

    shutil.rmtree(
        extraction_root,
        ignore_errors=True,
    )

    if not has_required(
        MSMT_TARGET,
        MSMT_REQUIRED,
    ):
        raise RuntimeError(
            "MSMT17_V2 validation failed."
        )

    print(
        "MSMT17_V2: PASS"
    )
    print(
        MSMT_TARGET
    )


# ============================================================
# MARS
# ============================================================

def decode_filename(
    value,
) -> str:

    if hasattr(
        value,
        "numpy",
    ):
        value = value.numpy()

    if isinstance(
        value,
        np.ndarray,
    ):
        if value.size == 1:
            value = value.item()

    if isinstance(
        value,
        bytes,
    ):
        value = value.decode(
            "utf-8"
        )

    return str(
        value
    )


def get_numpy(
    value,
) -> np.ndarray:

    if hasattr(
        value,
        "numpy",
    ):
        value = value.numpy()

    return np.asarray(
        value
    )


def find_tensor(
    ds,
    possible_names: list[str],
) -> str:

    available = list(
        ds.tensors.keys()
    )

    for name in possible_names:

        if name in available:
            return name

    raise RuntimeError(
        "Could not find tensor.\n"
        f"Expected one of: {possible_names}\n"
        f"Available tensors: {available}"
    )


def download_mars_info() -> None:

    info_dir = (
        MARS_ROOT
        / "info"
    )

    info_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Downloading official MARS metadata..."
    )

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


def open_deeplake_dataset(
    candidates: list[str],
):

    import deeplake

    errors = []

    for dataset_url in candidates:

        print(
            f"Trying: {dataset_url}"
        )

        try:

            ds = deeplake.load(
                dataset_url,
                read_only=True,
            )

            print(
                f"Loaded: {dataset_url}"
            )

            return (
                dataset_url,
                ds,
            )

        except Exception as exc:

            errors.append(
                (
                    dataset_url,
                    str(exc),
                )
            )

            print(
                "Failed:",
                exc,
            )

    message = "\n".join(
        f"{url}: {error}"
        for url, error in errors
    )

    raise RuntimeError(
        "Could not load any Deep Lake "
        "MARS dataset candidate:\n"
        f"{message}"
    )


def export_mars_split(
    dataset_candidates: list[str],
    output_dir: Path,
    expected_count: int,
) -> None:

    print()
    print("=" * 80)
    print(
        f"MARS EXPORT -> {output_dir.name}"
    )
    print("=" * 80)

    dataset_url, ds = (
        open_deeplake_dataset(
            dataset_candidates
        )
    )

    print()
    print(
        "Dataset:",
        dataset_url,
    )

    print(
        "Available tensors:"
    )

    for tensor_name in ds.tensors.keys():
        print(
            f"  - {tensor_name}"
        )

    image_key = find_tensor(
        ds,
        [
            "images",
            "image",
        ],
    )

    filename_key = find_tensor(
        ds,
        [
            "filenames",
            "filename",
            "names",
            "name",
        ],
    )

    total = len(
        ds
    )

    print()
    print(
        "Image tensor    :",
        image_key,
    )

    print(
        "Filename tensor :",
        filename_key,
    )

    print(
        f"Dataset samples : {total:,}"
    )

    if total != expected_count:

        raise RuntimeError(
            f"Unexpected MARS count.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {total:,}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0
    skipped = 0

    for index in tqdm(
        range(total)
    ):

        sample = ds[
            index
        ]

        filename = decode_filename(
            sample[
                filename_key
            ]
        )

        person_id = (
            filename[:4]
        )

        person_dir = (
            output_dir
            / person_id
        )

        person_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            person_dir
            / filename
        )

        # Resume support
        if output_file.exists():

            skipped += 1
            continue

        image = get_numpy(
            sample[
                image_key
            ]
        )

        if (
            image.ndim == 4
            and image.shape[0] == 1
        ):
            image = image[0]

        if image.dtype != np.uint8:

            if image.max() <= 1.0:
                image = (
                    image * 255
                )

            image = np.clip(
                image,
                0,
                255,
            ).astype(
                np.uint8
            )

        if image.ndim == 2:

            pil_image = Image.fromarray(
                image,
                mode="L",
            )

        elif image.ndim == 3:

            if image.shape[-1] == 1:

                image = image.squeeze(
                    -1
                )

                pil_image = Image.fromarray(
                    image,
                    mode="L",
                )

            else:

                pil_image = Image.fromarray(
                    image
                )

        else:

            raise RuntimeError(
                "Unexpected image shape: "
                f"{image.shape}"
            )

        # Preserve your previous MARS
        # reconstruction procedure.
        pil_image.save(
            output_file,
            quality=95,
        )

        written += 1

    print()
    print(
        f"Written : {written:,}"
    )

    print(
        f"Skipped : {skipped:,}"
    )

    print(
        f"Total   : {written + skipped:,}"
    )


def count_images(
    folder: Path,
) -> int:

    if not folder.exists():
        return 0

    valid_suffixes = {
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
            in valid_suffixes
        )
    )


def setup_mars() -> None:

    print()
    print("=" * 80)
    print("MARS")
    print("=" * 80)

    MARS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    download_mars_info()

    train_dir = (
        MARS_ROOT
        / "bbox_train"
    )

    test_dir = (
        MARS_ROOT
        / "bbox_test"
    )

    train_count = count_images(
        train_dir
    )

    if train_count != EXPECTED_MARS_TRAIN_IMAGES:

        export_mars_split(
            MARS_TRAIN_CANDIDATES,
            train_dir,
            EXPECTED_MARS_TRAIN_IMAGES,
        )

    else:

        print(
            f"[SKIP] MARS train already complete "
            f"({train_count:,})"
        )

    test_count = count_images(
        test_dir
    )

    if test_count != EXPECTED_MARS_TEST_IMAGES:

        export_mars_split(
            MARS_TEST_CANDIDATES,
            test_dir,
            EXPECTED_MARS_TEST_IMAGES,
        )

    else:

        print(
            f"[SKIP] MARS test already complete "
            f"({test_count:,})"
        )

    # Final verification
    train_count = count_images(
        train_dir
    )

    test_count = count_images(
        test_dir
    )

    print()
    print(
        f"MARS train: "
        f"{train_count:,} / "
        f"{EXPECTED_MARS_TRAIN_IMAGES:,}"
    )

    print(
        f"MARS test : "
        f"{test_count:,} / "
        f"{EXPECTED_MARS_TEST_IMAGES:,}"
    )

    for filename in INFO_FILES:

        path = (
            MARS_ROOT
            / "info"
            / filename
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing MARS metadata: "
                f"{filename}"
            )

    if (
        train_count
        != EXPECTED_MARS_TRAIN_IMAGES
    ):
        raise RuntimeError(
            "MARS train count mismatch."
        )

    if (
        test_count
        != EXPECTED_MARS_TEST_IMAGES
    ):
        raise RuntimeError(
            "MARS test count mismatch."
        )

    print(
        "MARS: PASS"
    )


# ============================================================
# Final verification
# ============================================================

def verify_final_structure() -> None:

    print()
    print("=" * 80)
    print("FINAL DATASET STRUCTURE")
    print("=" * 80)

    print()
    print(
        "Market1501 :",
        MARKET_TARGET,
    )

    print(
        "MSMT17_V2  :",
        MSMT_TARGET,
    )

    print(
        "MARS       :",
        MARS_ROOT,
    )

    market_ok = has_required(
        MARKET_TARGET,
        MARKET_REQUIRED,
    )

    msmt_ok = has_required(
        MSMT_TARGET,
        MSMT_REQUIRED,
    )

    mars_required = [
        MARS_ROOT / "bbox_train",
        MARS_ROOT / "bbox_test",
        MARS_ROOT / "info",
    ]

    mars_ok = all(
        path.exists()
        for path in mars_required
    )

    print()
    print(
        "Market1501:",
        "PASS" if market_ok else "FAILED",
    )

    print(
        "MSMT17_V2 :",
        "PASS" if msmt_ok else "FAILED",
    )

    print(
        "MARS      :",
        "PASS" if mars_ok else "FAILED",
    )

    if not (
        market_ok
        and msmt_ok
        and mars_ok
    ):
        raise RuntimeError(
            "One or more datasets failed "
            "structure validation."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    setup_market1501()

    setup_msmt17()

    setup_mars()

    verify_final_structure()

    print()
    print("=" * 80)
    print("ALL BENCHMARK DATASETS: READY")
    print("=" * 80)


if __name__ == "__main__":
    main()
