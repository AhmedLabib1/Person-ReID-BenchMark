from __future__ import annotations

from pathlib import Path

from .mars import MarsSplits
from .tracklet import Detection, Tracklet


VERSIONS = {
    "MSMT17_V2": ("mask_train_v2", "mask_test_v2"),
    "MSMT17_V1": ("train", "test"),
}

EXPECTED_QUERY = 11659
EXPECTED_GALLERY = 82161


class MSMT17Dataset:
    """
    Load MSMT17 query / gallery images as one-frame tracklets.

    Accepts V2 (FastReID zoo layout) or V1. Ranking uses the same
    same-pid + same-camera junk rule as Market1501.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.data_dir, self.train_dir, self.test_dir, self.version = self._resolve(
            self.root
        )

    def load(self) -> MarsSplits:
        query = self._load_list(
            self.test_dir, self.data_dir / "list_query.txt", "QUERY"
        )
        gallery = self._load_list(
            self.test_dir, self.data_dir / "list_gallery.txt", "GALLERY"
        )
        if len(query) != EXPECTED_QUERY:
            print(f"MSMT17 query count {len(query)} (expected {EXPECTED_QUERY})")
        if len(gallery) != EXPECTED_GALLERY:
            print(
                f"MSMT17 gallery count {len(gallery)} "
                f"(expected {EXPECTED_GALLERY})"
            )
        return MarsSplits(train=[], query=query, gallery=gallery)

    def _resolve(self, root: Path) -> tuple[Path, Path, Path, str]:
        candidates = [root]
        if root.is_dir():
            candidates.extend(p for p in root.iterdir() if p.is_dir())
            nested = root / "MSMT17"
            if nested.is_dir():
                candidates.append(nested)
                candidates.extend(p for p in nested.iterdir() if p.is_dir())

        seen: set[Path] = set()
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            for version, (train_name, test_name) in VERSIONS.items():
                if (candidate / "list_query.txt").is_file() and (
                    candidate / test_name
                ).is_dir():
                    return (
                        candidate,
                        candidate / train_name,
                        candidate / test_name,
                        version,
                    )
                nested_dir = candidate / version
                if (nested_dir / "list_query.txt").is_file() and (
                    nested_dir / test_name
                ).is_dir():
                    return (
                        nested_dir,
                        nested_dir / train_name,
                        nested_dir / test_name,
                        version,
                    )

        raise FileNotFoundError(
            f"MSMT17 V1/V2 not found under {root}. Expected "
            "MSMT17_V2/mask_test_v2 plus list_query.txt "
            "(or MSMT17_V1/test)."
        )

    def _load_list(
        self, image_root: Path, list_path: Path, split_name: str
    ) -> list[Tracklet]:
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing MSMT17 list file: {list_path}")

        tracklets: list[Tracklet] = []
        for index, raw in enumerate(
            list_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line:
                continue
            rel_path, pid_s = line.split()
            person_id = int(pid_s)
            camera_id = int(Path(rel_path).name.split("_")[2])
            image_path = image_root / rel_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing MSMT17 image: {image_path}")
            tracklets.append(
                Tracklet(
                    tracklet_id=f"MSMT_{split_name}_{index:06d}",
                    camera_id=f"CAM_{camera_id:02d}",
                    detections=[Detection(crop_path=image_path)],
                    person_id=person_id,
                    source="MSMT17",
                )
            )
        return tracklets
