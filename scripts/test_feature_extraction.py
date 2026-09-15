from __future__ import annotations

import torch
import torch.nn.functional as F

from reid.data.market1501 import (
    load_market1501,
)
from reid.data.mars import (
    load_mars,
)
from reid.data.sampling import (
    uniform_sample_frames,
)
from reid.embeddings.fastreid_extractor import (
    FastReIDFeatureExtractor,
)
from reid.models.fastreid_adapter import (
    FastReIDAdapter,
)


MODEL_ID = "sbs_r50_ibn"


def main() -> None:
    print("=" * 88)
    print(
        "SHAWAF ReID - "
        "Unified Feature Extraction Test"
    )
    print("=" * 88)

    print()
    print("MODEL")
    print("-" * 88)

    adapter = FastReIDAdapter(
        model_id=MODEL_ID,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    extractor = FastReIDFeatureExtractor(
        adapter=adapter,
    )

    print(
        f"Model ID                 : "
        f"{adapter.model_id}"
    )

    print(
        f"Family                   : "
        f"{adapter.family}"
    )

    print(
        f"Backbone                 : "
        f"{adapter.backbone}"
    )

    print(
        f"Input size               : "
        f"{extractor.input_size}"
    )

    # --------------------------------------------------
    # Market1501
    # --------------------------------------------------

    print()
    print("MARKET1501 IMAGE")
    print("-" * 88)

    market = load_market1501()

    market_query = market.query[0]

    preprocessed = (
        extractor.preprocess_path(
            market_query.image_path
        )
    )

    print(
        f"Sample ID                : "
        f"{market_query.sample_id}"
    )

    print(
        f"PID                      : "
        f"{market_query.person_id}"
    )

    print(
        f"Camera                   : "
        f"{market_query.camera_id}"
    )

    print(
        f"Preprocessed shape       : "
        f"{tuple(preprocessed.shape)}"
    )

    print(
        f"Pixel minimum            : "
        f"{preprocessed.min().item():.2f}"
    )

    print(
        f"Pixel maximum            : "
        f"{preprocessed.max().item():.2f}"
    )

    image_embedding = (
        extractor.extract_image(
            market_query
        )
    )

    image_norm = torch.linalg.vector_norm(
        image_embedding
    ).item()

    print(
        f"Embedding shape          : "
        f"{tuple(image_embedding.shape)}"
    )

    print(
        f"Embedding norm           : "
        f"{image_norm:.6f}"
    )

    if image_embedding.shape != (
        2048,
    ):
        raise RuntimeError(
            "Unexpected Market1501 "
            "embedding shape."
        )

    if not torch.isclose(
        torch.tensor(image_norm),
        torch.tensor(1.0),
        atol=1e-5,
    ):
        raise RuntimeError(
            "Market1501 embedding is "
            "not L2 normalized."
        )

    print(
        "Market embedding         : PASS"
    )

    # --------------------------------------------------
    # MARS
    # --------------------------------------------------

    print()
    print("MARS TRACKLET")
    print("-" * 88)

    mars = load_mars()

    query_tracklet = mars.query[0]

    sampled_frames = (
        uniform_sample_frames(
            query_tracklet.frame_paths,
            num_samples=8,
        )
    )

    print(
        f"Tracklet ID              : "
        f"{query_tracklet.tracklet_id}"
    )

    print(
        f"PID                      : "
        f"{query_tracklet.person_id}"
    )

    print(
        f"Camera                   : "
        f"{query_tracklet.camera_id}"
    )

    print(
        f"Original frames          : "
        f"{query_tracklet.num_frames}"
    )

    print(
        f"Sampled frames           : "
        f"{len(sampled_frames)}"
    )

    frame_embeddings = (
        extractor.extract_paths(
            sampled_frames,
            batch_size=8,
            normalize=False,
        )
    )

    print(
        f"Frame embeddings         : "
        f"{tuple(frame_embeddings.shape)}"
    )

    if frame_embeddings.shape != (
        8,
        2048,
    ):
        raise RuntimeError(
            "Unexpected MARS frame "
            "embedding shape."
        )

    tracklet_embedding = (
        extractor.extract_tracklet(
            query_tracklet,
            num_frames=8,
            batch_size=8,
        )
    )

    tracklet_norm = (
        torch.linalg.vector_norm(
            tracklet_embedding
        ).item()
    )

    print(
        f"Tracklet embedding       : "
        f"{tuple(tracklet_embedding.shape)}"
    )

    print(
        f"Tracklet norm            : "
        f"{tracklet_norm:.6f}"
    )

    if tracklet_embedding.shape != (
        2048,
    ):
        raise RuntimeError(
            "Unexpected MARS tracklet "
            "embedding shape."
        )

    if not torch.isclose(
        torch.tensor(tracklet_norm),
        torch.tensor(1.0),
        atol=1e-5,
    ):
        raise RuntimeError(
            "MARS tracklet embedding "
            "is not L2 normalized."
        )

    print(
        "Tracklet embedding       : PASS"
    )

    # --------------------------------------------------
    # Determinism test
    # --------------------------------------------------

    print()
    print("DETERMINISM")
    print("-" * 88)

    tracklet_embedding_2 = (
        extractor.extract_tracklet(
            query_tracklet,
            num_frames=8,
            batch_size=8,
        )
    )

    cosine_similarity = (
        F.cosine_similarity(
            tracklet_embedding.unsqueeze(0),
            tracklet_embedding_2.unsqueeze(0),
        ).item()
    )

    max_difference = (
        torch.max(
            torch.abs(
                tracklet_embedding
                - tracklet_embedding_2
            )
        ).item()
    )

    print(
        f"Repeat cosine similarity : "
        f"{cosine_similarity:.8f}"
    )

    print(
        f"Maximum difference       : "
        f"{max_difference:.8f}"
    )

    if cosine_similarity < 0.99999:
        raise RuntimeError(
            "Feature extraction is "
            "not deterministic."
        )

    print(
        "Deterministic extraction : PASS"
    )

    print()
    print("=" * 88)
    print(
        "Unified feature extraction: SUCCESS"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()