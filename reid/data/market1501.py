from __future__ import annotations

import re
from pathlib import Path

from reid.data.contracts import (
    DatasetSplit,
    ImageReIDDataset,
    ImageSample,
)
from reid.data.paths import MARKET1501_ROOT


DATASET_NAME = "Market1501"

FILENAME_PATTERN = re.compile(
    r"^([-\d]+)_c(\d)"
)


class Market1501Loader:
    """
    Loader for the standard Market1501 dataset.

    Expected structure:

    Market-1501-v15.09.15/
        bounding_box_train/
        bounding_box_test/
        query/

    Notes:
        PID = -1:
            junk images -> ignored

        PID = 0:
            background/distractor -> kept

        Camera IDs in filenames:
            1..6

        Internal camera IDs:
            0..5
    """

    def __init__(
        self,
        root: Path = MARKET1501_ROOT,
    ) -> None:
        self.root = Path(root)

        self.train_dir = (
            self.root
            / "bounding_box_train"
        )

        self.query_dir = (
            self.root
            / "query"
        )

        self.gallery_dir = (
            self.root
            / "bounding_box_test"
        )

        self._validate_structure()

    def _validate_structure(self) -> None:
        required_paths = (
            self.root,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        )

        missing = [
            path
            for path in required_paths
            if not path.exists()
        ]

        if missing:
            message = "\n".join(
                str(path)
                for path in missing
            )

            raise FileNotFoundError(
                "Market1501 required paths "
                "are missing:\n"
                f"{message}"
            )

    @staticmethod
    def _parse_filename(
        image_path: Path,
    ) -> tuple[int, int]:
        match = FILENAME_PATTERN.match(
            image_path.name
        )

        if match is None:
            raise ValueError(
                "Invalid Market1501 filename:\n"
                f"{image_path.name}"
            )

        person_id = int(
            match.group(1)
        )

        camera_id = int(
            match.group(2)
        )

        if person_id < -1:
            raise ValueError(
                f"Invalid person ID "
                f"{person_id} in "
                f"{image_path.name}"
            )

        if not 1 <= camera_id <= 6:
            raise ValueError(
                f"Invalid camera ID "
                f"{camera_id} in "
                f"{image_path.name}"
            )

        # Convert Market camera indexing:
        #
        # c1..c6
        #
        # into:
        #
        # 0..5
        camera_id -= 1

        return (
            person_id,
            camera_id,
        )

    def _load_split(
        self,
        directory: Path,
        split: DatasetSplit,
    ) -> tuple[ImageSample, ...]:

        image_paths = sorted(
            directory.glob("*.jpg")
        )

        samples: list[ImageSample] = []

        for image_path in image_paths:
            person_id, camera_id = (
                self._parse_filename(
                    image_path
                )
            )

            # Standard Market1501 protocol:
            # PID -1 is junk and must be ignored.
            if person_id == -1:
                continue

            sample = ImageSample(
                dataset=DATASET_NAME,
                split=split,
                sample_id=image_path.stem,
                person_id=person_id,
                camera_id=camera_id,
                image_path=image_path,
            )

            samples.append(
                sample
            )

        return tuple(samples)

    def load(self) -> ImageReIDDataset:
        train = self._load_split(
            directory=self.train_dir,
            split=DatasetSplit.TRAIN,
        )

        query = self._load_split(
            directory=self.query_dir,
            split=DatasetSplit.QUERY,
        )

        gallery = self._load_split(
            directory=self.gallery_dir,
            split=DatasetSplit.GALLERY,
        )

        return ImageReIDDataset(
            name=DATASET_NAME,
            train=train,
            query=query,
            gallery=gallery,
        )


def load_market1501(
    root: Path = MARKET1501_ROOT,
) -> ImageReIDDataset:
    """
    Convenience function used by the benchmark.
    """

    loader = Market1501Loader(
        root=root,
    )

    return loader.load()