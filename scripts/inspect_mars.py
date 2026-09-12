from pathlib import Path

from reid.data import MarsDataset


def print_tracklet(
    title: str,
    tracklet,
) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Tracklet ID       : "
        f"{tracklet.tracklet_id}"
    )

    print(
        f"Person ID         : "
        f"{tracklet.person_id}"
    )

    print(
        f"Camera ID         : "
        f"{tracklet.camera_id}"
    )

    print(
        f"Source            : "
        f"{tracklet.source}"
    )

    print(
        f"Number of images  : "
        f"{tracklet.num_detections}"
    )

    print(
        f"First image       : "
        f"{tracklet.crop_paths[0]}"
    )

    print(
        f"Last image        : "
        f"{tracklet.crop_paths[-1]}"
    )


def main() -> None:
    dataset_root = Path("datasets/MARS")

    print("Loading MARS dataset...")
    print(
        f"Dataset path: "
        f"{dataset_root.resolve()}"
    )

    dataset = MarsDataset(
        root=dataset_root
    )

    include_train = any(dataset.bbox_train_dir.iterdir())
    splits = dataset.load(include_train=include_train)
    if not include_train:
        print("bbox_train is empty; skipping train split.")

    print()
    print("MARS loaded successfully")
    print()

    print(
        f"Train tracklets   : "
        f"{len(splits.train)}"
    )

    print(
        f"Query tracklets   : "
        f"{len(splits.query)}"
    )

    print(
        f"Gallery tracklets : "
        f"{len(splits.gallery)}"
    )

    if splits.train:
        print_tracklet(
            "EXAMPLE TRAIN TRACKLET",
            splits.train[0],
        )

    if splits.query:
        print_tracklet(
            "EXAMPLE QUERY TRACKLET",
            splits.query[0],
        )

    if splits.gallery:
        print_tracklet(
            "EXAMPLE GALLERY TRACKLET",
            splits.gallery[0],
        )


if __name__ == "__main__":
    main()