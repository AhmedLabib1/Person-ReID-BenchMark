from reid.data import Detection, Tracklet


def main() -> None:
    detection_1 = Detection(
        frame_id=120,
        timestamp=4.80,
        bbox=(340, 120, 470, 510),
        confidence=0.94,
        crop_path="crops/T000001/frame_000120.jpg",
    )

    detection_2 = Detection(
        frame_id=121,
        timestamp=4.84,
        bbox=(343, 121, 472, 512),
        confidence=0.95,
        crop_path="crops/T000001/frame_000121.jpg",
    )

    tracklet = Tracklet(
        tracklet_id="T000001",
        camera_id="CAM_01",
        detections=[
            detection_1,
            detection_2,
        ],
        start_frame=120,
        end_frame=245,
        start_time=4.80,
        end_time=9.80,
        source="SHAWAF",
    )

    print("Tracklet created successfully")
    print()

    print("Tracklet ID:")
    print(tracklet.tracklet_id)

    print()

    print("Camera:")
    print(tracklet.camera_id)

    print()

    print("Number of detections:")
    print(tracklet.num_detections)

    print()

    print("Duration:")
    print(
        round(tracklet.duration, 2)
        if tracklet.duration is not None
        else None
    )

    print()

    print("Frame count:")
    print(tracklet.frame_count)

    print()

    print("Tracklet dictionary:")
    print(tracklet.to_dict())


if __name__ == "__main__":
    main()