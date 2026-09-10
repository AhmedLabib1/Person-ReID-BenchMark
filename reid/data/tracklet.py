from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BBox = tuple[int, int, int, int]


@dataclass
class Detection:
    """
    Represents one person crop / observation from one frame.

    crop_path is always required.

    Other fields may not exist in benchmark datasets such as MARS,
    so they are optional.

    In real SHAWAF tracklets, Detection + Tracking should provide
    as much metadata as possible.
    """

    crop_path: Path

    frame_id: int | None = None
    timestamp: float | None = None
    bbox: BBox | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        self.crop_path = Path(self.crop_path)
        self._validate()

    def _validate(self) -> None:
        if self.frame_id is not None and self.frame_id < 0:
            raise ValueError("frame_id must be >= 0")

        if self.timestamp is not None and self.timestamp < 0:
            raise ValueError("timestamp must be >= 0")

        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError(
                    "bbox must contain exactly 4 values: "
                    "(x1, y1, x2, y2)"
                )

            x1, y1, x2, y2 = self.bbox

            if x2 <= x1:
                raise ValueError(
                    f"Invalid bbox: x2 ({x2}) "
                    f"must be greater than x1 ({x1})"
                )

            if y2 <= y1:
                raise ValueError(
                    f"Invalid bbox: y2 ({y2}) "
                    f"must be greater than y1 ({y1})"
                )

        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0.0 and 1.0"
                )

    @property
    def width(self) -> int | None:
        if self.bbox is None:
            return None

        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> int | None:
        if self.bbox is None:
            return None

        _, y1, _, y2 = self.bbox
        return y2 - y1

    @property
    def area(self) -> int | None:
        if self.width is None or self.height is None:
            return None

        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": self.confidence,
            "crop_path": str(self.crop_path),
        }


@dataclass
class Tracklet:
    """
    Represents one person observed continuously inside one camera.

    person_id:
        Ground-truth identity used only for datasets/evaluation.

        MARS:
            person_id is known.

        Real SHAWAF:
            person_id will normally be None.
    """

    tracklet_id: str
    camera_id: str
    detections: list[Detection] = field(default_factory=list)

    person_id: int | None = None

    start_frame: int | None = None
    end_frame: int | None = None

    start_time: float | None = None
    end_time: float | None = None

    source: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.tracklet_id.strip():
            raise ValueError("tracklet_id cannot be empty")

        if not self.camera_id.strip():
            raise ValueError("camera_id cannot be empty")

        if self.person_id is not None and self.person_id < 0:
            raise ValueError(
                "person_id must be >= 0 when provided"
            )

        if self.start_frame is not None:
            if self.start_frame < 0:
                raise ValueError("start_frame must be >= 0")

        if self.end_frame is not None:
            if self.end_frame < 0:
                raise ValueError("end_frame must be >= 0")

        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame < self.start_frame
        ):
            raise ValueError(
                "end_frame must be >= start_frame"
            )

        if self.start_time is not None:
            if self.start_time < 0:
                raise ValueError("start_time must be >= 0")

        if self.end_time is not None:
            if self.end_time < 0:
                raise ValueError("end_time must be >= 0")

        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError(
                "end_time must be >= start_time"
            )

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def duration(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None

        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        if (
            self.start_frame is not None
            and self.end_frame is not None
        ):
            return self.end_frame - self.start_frame + 1

        return self.num_detections

    @property
    def crop_paths(self) -> list[Path]:
        return [
            detection.crop_path
            for detection in self.detections
        ]

    def add_detection(
        self,
        detection: Detection,
    ) -> None:

        if (
            detection.frame_id is not None
            and self.start_frame is not None
            and self.end_frame is not None
        ):
            if not (
                self.start_frame
                <= detection.frame_id
                <= self.end_frame
            ):
                raise ValueError(
                    f"Detection frame {detection.frame_id} "
                    f"is outside tracklet frame range "
                    f"{self.start_frame}-{self.end_frame}"
                )

        self.detections.append(detection)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "source": self.source,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_detections": self.num_detections,
            "detections": [
                detection.to_dict()
                for detection in self.detections
            ],
        }