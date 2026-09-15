from __future__ import annotations

import collections
import collections.abc

import torch
import torch.nn.functional as F

# --------------------------------------------------
# FastReID Python 3.10 compatibility
# --------------------------------------------------
#
# The pinned FastReID commit imports:
#
#     from collections import Mapping
#
# which was removed in Python 3.10.
#
# We patch it at runtime for this diagnostic only.
# We do NOT modify third_party/fast-reid.
# --------------------------------------------------

if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping


from fastreid.config import get_cfg
from fastreid.data.data_utils import read_image
from fastreid.data.transforms import build_transforms

from reid.data.msmt17 import load_msmt17
from reid.embeddings.fastreid_extractor import (
    FastReIDFeatureExtractor,
)
from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


MODEL_ID = "sbs_r50_ibn"
NUM_IMAGES = 20


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "Current vs Official FastReID Preprocessing"
    )
    print("=" * 88)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    adapter = FastReIDAdapter(
        model_id=MODEL_ID,
        device=device,
    )

    current_extractor = (
        FastReIDFeatureExtractor(
            adapter=adapter,
        )
    )

    # --------------------------------------------------
    # Official FastReID configuration
    # --------------------------------------------------

    cfg = get_cfg()

    cfg.merge_from_file(
        str(adapter.spec.config_path)
    )

    official_transform = (
        build_transforms(
            cfg,
            is_train=False,
        )
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = load_msmt17()

    samples = dataset.query[
        :NUM_IMAGES
    ]

    print()
    print("CONFIGURATION")
    print("-" * 88)

    print(
        f"Model                   : "
        f"{MODEL_ID}"
    )

    print(
        f"Images                  : "
        f"{len(samples)}"
    )

    print(
        f"Input size              : "
        f"{current_extractor.input_size}"
    )

    print(
        f"Device                  : "
        f"{device}"
    )

    # --------------------------------------------------
    # Build both versions of the same images
    # --------------------------------------------------

    current_tensors = []
    official_tensors = []

    pixel_mean_differences = []
    pixel_max_differences = []

    for sample in samples:
        current_tensor = (
            current_extractor.preprocess_path(
                sample.image_path
            )
        )

        official_image = read_image(
            str(sample.image_path)
        )

        official_tensor = (
            official_transform(
                official_image
            )
            .float()
        )

        if (
            current_tensor.shape
            != official_tensor.shape
        ):
            raise RuntimeError(
                "Preprocessing shape mismatch:\n"
                f"Current  : {current_tensor.shape}\n"
                f"Official : {official_tensor.shape}"
            )

        difference = torch.abs(
            current_tensor
            - official_tensor
        )

        pixel_mean_differences.append(
            difference.mean().item()
        )

        pixel_max_differences.append(
            difference.max().item()
        )

        current_tensors.append(
            current_tensor
        )

        official_tensors.append(
            official_tensor
        )

    current_batch = torch.stack(
        current_tensors,
        dim=0,
    )

    official_batch = torch.stack(
        official_tensors,
        dim=0,
    )

    # --------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------

    current_features = (
        adapter.encode(
            current_batch
        )
        .detach()
        .float()
        .cpu()
    )

    official_features = (
        adapter.encode(
            official_batch
        )
        .detach()
        .float()
        .cpu()
    )

    current_features = F.normalize(
        current_features,
        p=2,
        dim=1,
    )

    official_features = F.normalize(
        official_features,
        p=2,
        dim=1,
    )

    similarities = F.cosine_similarity(
        current_features,
        official_features,
        dim=1,
    )

    embedding_differences = torch.linalg.vector_norm(
        current_features
        - official_features,
        dim=1,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()
    print("PIXEL PREPROCESSING DIFFERENCE")
    print("-" * 88)

    print(
        f"Average mean abs diff   : "
        f"{sum(pixel_mean_differences) / len(pixel_mean_differences):.6f}"
    )

    print(
        f"Largest pixel diff      : "
        f"{max(pixel_max_differences):.6f}"
    )

    print()
    print("EMBEDDING COMPARISON")
    print("-" * 88)

    print(
        f"Mean cosine similarity  : "
        f"{similarities.mean().item():.8f}"
    )

    print(
        f"Minimum cosine          : "
        f"{similarities.min().item():.8f}"
    )

    print(
        f"Maximum cosine          : "
        f"{similarities.max().item():.8f}"
    )

    print(
        f"Mean embedding L2 diff  : "
        f"{embedding_differences.mean().item():.8f}"
    )

    print(
        f"Maximum embedding diff  : "
        f"{embedding_differences.max().item():.8f}"
    )

    print()
    print("PER-IMAGE COSINE")
    print("-" * 88)

    for index, sample in enumerate(
        samples
    ):
        print(
            f"{index + 1:02d}  "
            f"{sample.sample_id:<24} "
            f"{similarities[index].item():.8f}"
        )

    print()
    print("=" * 88)

    if torch.allclose(
        current_features,
        official_features,
        atol=1e-6,
        rtol=1e-6,
    ):
        print(
            "RESULT: preprocessing produces "
            "effectively identical embeddings."
        )
    else:
        print(
            "RESULT: preprocessing changes "
            "the model embeddings."
        )

    print("=" * 88)


if __name__ == "__main__":
    main()