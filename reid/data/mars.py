from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from reid.data.contracts import (
    DatasetSplit,
    TrackletReIDDataset,
    TrackletSample,
)
from reid.data.paths import MARS_ROOT


DATASET_NAME = "MARS"


class MarsLoader:
    """
    Loader for the standard MARS video person ReID dataset.

    Expected structure:

    mars/
        bbox_train/
        bbox_test/
        info/
            train_name.txt
            test_name.txt
            tracks_train_info.mat
            tracks_test_info.mat
            query_IDX.mat

    MARS metadata row format:

        [start_index, end_index, person_id, camera_id]

    Notes:
        - start/end indices are 1-based and inclusive.
        - PID = -1 means junk and is ignored.
        - Camera IDs in metadata are 1..6.
        - Internal camera IDs are converted to 0..5.
    """

    def __init__(
        self,
        root: Path = MARS_ROOT,
    ) -> None:
        self.root = Path(root)

        self.train_dir = (
            self.root
            / "bbox_train"
        )

        self.test_dir = (
            self.root
            / "bbox_test"
        )

        self.info_dir = (
            self.root
            / "info"
        )

        self.train_name_path = (
            self.info_dir
            / "train_name.txt"
        )

        self.test_name_path = (
            self.info_dir
            / "test_name.txt"
        )

        self.track_train_info_path = (
            self.info_dir
            / "tracks_train_info.mat"
        )

        self.track_test_info_path = (
            self.info_dir
            / "tracks_test_info.mat"
        )

        self.query_idx_path = (
            self.info_dir
            / "query_IDX.mat"
        )

        self._validate_structure()

    def _validate_structure(self) -> None:
        required_paths = (
            self.root,
            self.train_dir,
            self.test_dir,
            self.info_dir,
            self.train_name_path,
            self.test_name_path,
            self.track_train_info_path,
            self.track_test_info_path,
            self.query_idx_path,
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
                "MARS required paths "
                "are missing:\n"
                f"{message}"
            )

    @staticmethod
    def _load_names(
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
                f"No image names found in:\n"
                f"{path}"
            )

        return names

    @staticmethod
    def _load_track_info(
        path: Path,
        key: str,
    ) -> np.ndarray:
        data = loadmat(
            path
        )

        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in:\n"
                f"{path}"
            )

        track_info = np.asarray(
            data[key]
        )

        if (
            track_info.ndim != 2
            or track_info.shape[1] != 4
        ):
            raise RuntimeError(
                "Unexpected MARS track metadata "
                f"shape: {track_info.shape}"
            )

        return track_info.astype(
            np.int64,
            copy=False,
        )

    def _load_query_indices(
        self,
        num_test_tracklets: int,
    ) -> np.ndarray:
        data = loadmat(
            self.query_idx_path
        )

        if "query_IDX" not in data:
            raise KeyError(
                "Key 'query_IDX' not found in:\n"
                f"{self.query_idx_path}"
            )

        query_indices = np.asarray(
            data["query_IDX"]
        ).reshape(-1)

        query_indices = query_indices.astype(
            np.int64,
            copy=False,
        )

        # MARS stores query indices as 1-based MATLAB indices.
        query_indices = (
            query_indices - 1
        )

        if len(query_indices) != len(
            np.unique(query_indices)
        ):
            raise RuntimeError(
                "Duplicate indices found in "
                "MARS query_IDX."
            )

        if (
            query_indices.min() < 0
            or query_indices.max()
            >= num_test_tracklets
        ):
            raise RuntimeError(
                "MARS query index is outside "
                "the test-tracklet range."
            )

        return query_indices

    def _build_tracklet(
        self,
        *,
        names: tuple[str, ...],
        metadata_row: np.ndarray,
        split: DatasetSplit,
        home_dir: Path,
        source_name: str,
        source_index: int,
    ) -> TrackletSample | None:
        (
            start_index,
            end_index,
            person_id,
            camera_id,
        ) = (
            int(value)
            for value in metadata_row
        )

        # Standard MARS protocol:
        # PID -1 represents junk.
        if person_id == -1:
            return None

        if person_id < 0:
            raise RuntimeError(
                f"Unexpected MARS PID: "
                f"{person_id}"
            )

        if not 1 <= camera_id <= 6:
            raise RuntimeError(
                f"Invalid MARS camera ID: "
                f"{camera_id}"
            )

        if start_index < 1:
            raise RuntimeError(
                "MARS start index must be "
                "1-based and >= 1."
            )

        if end_index < start_index:
            raise RuntimeError(
                "MARS tracklet end index "
                "precedes start index."
            )

        if end_index > len(names):
            raise RuntimeError(
                "MARS tracklet references "
                "an image outside the name list."
            )

        # MATLAB:
        #
        # start_index ... end_index
        #
        # inclusive, 1-based.
        #
        # Python:
        #
        # start_index - 1 : end_index
        frame_names = names[
            start_index - 1:
            end_index
        ]

        if not frame_names:
            raise RuntimeError(
                "MARS tracklet contains "
                "zero frames."
            )

        identity_prefixes = {
            name[:4]
            for name in frame_names
        }

        if len(identity_prefixes) != 1:
            raise RuntimeError(
                "One MARS tracklet contains "
                "multiple identity folders."
            )

        frame_paths = tuple(
            home_dir
            / frame_name[:4]
            / frame_name
            for frame_name in frame_names
        )

        # Convert:
        #
        # camera 1..6
        #
        # into:
        #
        # camera 0..5
        internal_camera_id = (
            camera_id - 1
        )

        tracklet_id = (
            f"{source_name}_"
            f"{source_index:05d}"
        )

        return TrackletSample(
            dataset=DATASET_NAME,
            split=split,
            tracklet_id=tracklet_id,
            person_id=person_id,
            camera_id=internal_camera_id,
            frame_paths=frame_paths,
        )

    def _process_rows(
        self,
        *,
        names: tuple[str, ...],
        track_info: np.ndarray,
        source_indices: np.ndarray,
        split: DatasetSplit,
        home_dir: Path,
        source_name: str,
    ) -> tuple[TrackletSample, ...]:
        tracklets: list[TrackletSample] = []

        for row, source_index in zip(
            track_info,
            source_indices,
        ):
            tracklet = self._build_tracklet(
                names=names,
                metadata_row=row,
                split=split,
                home_dir=home_dir,
                source_name=source_name,
                source_index=int(
                    source_index
                ),
            )

            if tracklet is not None:
                tracklets.append(
                    tracklet
                )

        return tuple(tracklets)

    def load(
        self,
    ) -> TrackletReIDDataset:
        train_names = self._load_names(
            self.train_name_path
        )

        test_names = self._load_names(
            self.test_name_path
        )

        track_train = self._load_track_info(
            self.track_train_info_path,
            key="track_train_info",
        )

        track_test = self._load_track_info(
            self.track_test_info_path,
            key="track_test_info",
        )

        query_indices = (
            self._load_query_indices(
                num_test_tracklets=(
                    track_test.shape[0]
                )
            )
        )

        all_test_indices = np.arange(
            track_test.shape[0],
            dtype=np.int64,
        )

        query_mask = np.zeros(
            track_test.shape[0],
            dtype=bool,
        )

        query_mask[
            query_indices
        ] = True

        gallery_indices = (
            all_test_indices[
                ~query_mask
            ]
        )

        track_query = track_test[
            query_indices
        ]

        track_gallery = track_test[
            gallery_indices
        ]

        train_indices = np.arange(
            track_train.shape[0],
            dtype=np.int64,
        )

        train = self._process_rows(
            names=train_names,
            track_info=track_train,
            source_indices=train_indices,
            split=DatasetSplit.TRAIN,
            home_dir=self.train_dir,
            source_name="mars_train",
        )

        query = self._process_rows(
            names=test_names,
            track_info=track_query,
            source_indices=query_indices,
            split=DatasetSplit.QUERY,
            home_dir=self.test_dir,
            source_name="mars_test",
        )

        gallery = self._process_rows(
            names=test_names,
            track_info=track_gallery,
            source_indices=gallery_indices,
            split=DatasetSplit.GALLERY,
            home_dir=self.test_dir,
            source_name="mars_test",
        )

        return TrackletReIDDataset(
            name=DATASET_NAME,
            train=train,
            query=query,
            gallery=gallery,
        )


def load_mars(
    root: Path = MARS_ROOT,
) -> TrackletReIDDataset:
    loader = MarsLoader(
        root=root,
    )

    return loader.load()