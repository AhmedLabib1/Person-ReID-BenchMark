from __future__ import annotations

import re
from pathlib import Path

from .mars import MarsSplits
from .tracklet import Detection, Tracklet


_FILENAME = re.compile(r"([-\d]+)_c(\d)")


class Market1501Dataset:
    """
    Load Market1501 query / gallery images as one-frame tracklets.

    Junk images (person ID -1) are dropped. Distractors (person ID 0)
    stay in the gallery, matching the standard Market1501 protocol.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if (self.root / "query").is_dir():
            self.data_dir = self.root
        elif (self.root / "Market-1501-v15.09.15" / "query").is_dir():
            self.data_dir = self.root / "Market-1501-v15.09.15"
        else:
            raise FileNotFoundError(
                f"Market1501 query folder not found under {self.root}"
            )
        self.query_dir = self.data_dir / "query"
        self.gallery_dir = self.data_dir / "bounding_box_test"

    def load(self) -> MarsSplits:
        return MarsSplits(
            train=[],
            query=self._load_split(self.query_dir, "QUERY"),
            gallery=self._load_split(self.gallery_dir, "GALLERY"),
        )

    def _load_split(self, image_dir: Path, split_name: str) -> list[Tracklet]:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing Market1501 folder: {image_dir}")

        tracklets: list[Tracklet] = []
        for index, path in enumerate(sorted(image_dir.glob("*.jpg"))):
            match = _FILENAME.search(path.name)
            if match is None:
                continue
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            if person_id == -1:
                continue
            tracklets.append(
                Tracklet(
                    tracklet_id=f"MARKET_{split_name}_{index + 1:06d}",
                    camera_id=f"CAM_{camera_id:02d}",
                    detections=[Detection(crop_path=path)],
                    person_id=person_id,
                    source="Market1501",
                )
            )
        return tracklets
