from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .tracklet import Detection, Tracklet


@dataclass
class MarsSplits:
    """
    Contains the three standard MARS splits.

    train:
        Tracklets used for model training.

    query:
        Tracklets used as search queries during evaluation.

    gallery:
        Candidate tracklets searched for each query.
    """

    train: list[Tracklet]
    query: list[Tracklet]
    gallery: list[Tracklet]


class MarsDataset:
    """
    Loader for the MARS video person ReID dataset.

    The loader converts the original MARS representation:

        image names
        +
        MATLAB track metadata

    into SHAWAF Tracklet objects.
    """

    NUM_CAMERAS = 6

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

        self.bbox_train_dir = self.root / "bbox_train"
        self.bbox_test_dir = self.root / "bbox_test"
        self.info_dir = self.root / "info"

        self.train_name_path = (
            self.info_dir / "train_name.txt"
        )

        self.test_name_path = (
            self.info_dir / "test_name.txt"
        )

        self.track_train_info_path = (
            self.info_dir / "tracks_train_info.mat"
        )

        self.track_test_info_path = (
            self.info_dir / "tracks_test_info.mat"
        )

        self.query_idx_path = (
            self.info_dir / "query_IDX.mat"
        )

        self._validate_dataset_structure()

    def load(self) -> MarsSplits:
        """
        Load the standard MARS train/query/gallery splits.
        """

        train_names = self._read_names(
            self.train_name_path
        )

        test_names = self._read_names(
            self.test_name_path
        )

        train_metadata = self._load_mat_array(
            path=self.track_train_info_path,
            key="track_train_info",
        )

        test_metadata = self._load_mat_array(
            path=self.track_test_info_path,
            key="track_test_info",
        )

        query_indices = self._load_mat_array(
            path=self.query_idx_path,
            key="query_IDX",
        ).squeeze()

        # MATLAB uses 1-based indexing:
        #
        # first element = 1
        #
        # Python uses 0-based indexing:
        #
        # first element = 0
        #
        # Therefore:
        #
        # 1, 2, 3 ...
        # becomes
        # 0, 1, 2 ...
        query_indices = (
            query_indices.astype(np.int64) - 1
        )

        train_tracklets = self._build_tracklets(
            names=train_names,
            metadata=train_metadata,
            images_dir=self.bbox_train_dir,
            split_name="TRAIN",
        )

        query_metadata = test_metadata[
            query_indices
        ]

        query_tracklets = self._build_tracklets(
            names=test_names,
            metadata=query_metadata,
            images_dir=self.bbox_test_dir,
            split_name="QUERY",
        )

        # Query tracklets must not also appear in gallery.
        query_index_set = set(
            query_indices.tolist()
        )

        gallery_indices = [
            index
            for index in range(
                len(test_metadata)
            )
            if index not in query_index_set
        ]

        gallery_metadata = test_metadata[
            gallery_indices
        ]

        gallery_tracklets = self._build_tracklets(
            names=test_names,
            metadata=gallery_metadata,
            images_dir=self.bbox_test_dir,
            split_name="GALLERY",
        )

        return MarsSplits(
            train=train_tracklets,
            query=query_tracklets,
            gallery=gallery_tracklets,
        )

    def _validate_dataset_structure(
        self,
    ) -> None:
        """
        Make sure all required MARS files exist.
        """

        required_paths = [
            self.bbox_train_dir,
            self.bbox_test_dir,
            self.info_dir,
            self.train_name_path,
            self.test_name_path,
            self.track_train_info_path,
            self.track_test_info_path,
            self.query_idx_path,
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.exists()
        ]

        if missing_paths:
            formatted = "\n".join(
                f"  - {path}"
                for path in missing_paths
            )

            raise FileNotFoundError(
                "MARS dataset structure is incomplete.\n"
                "Missing paths:\n"
                f"{formatted}"
            )

    @staticmethod
    def _read_names(
        path: Path,
    ) -> list[str]:
        """
        Read image names from train_name.txt
        or test_name.txt.
        """

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        return names

    @staticmethod
    def _load_mat_array(
        path: Path,
        key: str,
    ) -> np.ndarray:
        """
        Load one array from a MATLAB .mat file.
        """

        mat_data = loadmat(path)

        if key not in mat_data:
            raise KeyError(
                f"Key '{key}' was not found "
                f"in {path}"
            )

        return np.asarray(
            mat_data[key]
        )

    def _build_tracklets(
        self,
        names: list[str],
        metadata: np.ndarray,
        images_dir: Path,
        split_name: str,
    ) -> list[Tracklet]:
        """
        Convert MARS metadata rows into SHAWAF Tracklet objects.

        Each metadata row has:

        [
            start_index,
            end_index,
            person_id,
            camera_id
        ]

        start_index/end_index refer to positions inside
        train_name.txt or test_name.txt.

        They are NOT real video frame numbers.
        """

        tracklets: list[Tracklet] = []

        for metadata_index, row in enumerate(
            metadata
        ):
            (
                start_index,
                end_index,
                person_id,
                camera_id,
            ) = [
                int(value)
                for value in row
            ]

            # MARS uses person_id = -1
            # for junk / distractor samples.
            #
            # They are ignored in standard ReID evaluation.
            if person_id == -1:
                continue

            if not (
                1
                <= camera_id
                <= self.NUM_CAMERAS
            ):
                raise ValueError(
                    f"Invalid MARS camera ID: "
                    f"{camera_id}"
                )

            # MARS metadata uses MATLAB indexing.
            #
            # Example:
            #
            # start_index = 1
            # end_index   = 3
            #
            # means:
            #
            # names[0:3]
            #
            # Python excludes the final slice position,
            # which gives elements 0, 1, 2.
            image_names = names[
                start_index - 1:end_index
            ]

            if not image_names:
                raise ValueError(
                    "Tracklet contains zero images. "
                    f"Metadata row: {row}"
                )

            self._validate_tracklet_images(
                image_names=image_names,
                expected_person_id=person_id,
                expected_camera_id=camera_id,
            )

            detections = [
                Detection(
                    crop_path=self._build_image_path(
                        images_dir=images_dir,
                        image_name=image_name,
                    )
                )
                for image_name in image_names
            ]

            tracklet_id = (
                f"MARS_{split_name}_"
                f"{metadata_index + 1:06d}"
            )

            tracklet = Tracklet(
                tracklet_id=tracklet_id,
                camera_id=(
                    f"CAM_{camera_id:02d}"
                ),
                detections=detections,
                person_id=person_id,
                source="MARS",
            )

            tracklets.append(tracklet)

        return tracklets

    @staticmethod
    def _build_image_path(
        images_dir: Path,
        image_name: str,
    ) -> Path:
        """
        MARS stores images inside a folder named
        using the first four characters of the image name.

        Example:

        image:
            0001C1T0001F001.jpg

        folder:
            0001/

        final path:
            bbox_train/0001/0001C1T0001F001.jpg
        """

        person_folder = image_name[:4]

        return (
            images_dir
            / person_folder
            / image_name
        )

    @staticmethod
    def _validate_tracklet_images(
        image_names: list[str],
        expected_person_id: int,
        expected_camera_id: int,
    ) -> None:
        """
        Check that all images inside one MARS tracklet
        belong to the same person and camera.
        """

        person_names = {
            image_name[:4]
            for image_name in image_names
        }

        if len(person_names) != 1:
            raise ValueError(
                "One MARS tracklet contains "
                "multiple person IDs."
            )

        camera_names = {
            image_name[5]
            for image_name in image_names
        }

        if len(camera_names) != 1:
            raise ValueError(
                "One MARS tracklet contains "
                "images from multiple cameras."
            )

        expected_person_name = (
            f"{expected_person_id:04d}"
        )

        actual_person_name = next(
            iter(person_names)
        )

        if (
            actual_person_name
            != expected_person_name
        ):
            raise ValueError(
                "Person ID mismatch between "
                "MARS metadata and image name. "
                f"Expected {expected_person_name}, "
                f"found {actual_person_name}."
            )

        actual_camera_id = int(
            next(iter(camera_names))
        )

        if (
            actual_camera_id
            != expected_camera_id
        ):
            raise ValueError(
                "Camera ID mismatch between "
                "MARS metadata and image name. "
                f"Expected {expected_camera_id}, "
                f"found {actual_camera_id}."
            )