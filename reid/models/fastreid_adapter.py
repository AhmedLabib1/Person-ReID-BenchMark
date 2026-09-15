from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark"
    / "fastreid_msmt17_models.yaml"
)


@dataclass(frozen=True)
class FastReIDModelSpec:
    model_id: str
    family: str
    backbone: str

    config_path: Path
    checkpoint_path: Path

    source_dataset: str

    official_rank1: float
    official_map: float
    official_minp: float


class CompatibleCheckpointer(Checkpointer):
    """
    Compatibility wrapper for modern PyTorch.

    FastReID was written before PyTorch changed the default
    torch.load(weights_only=...).

    The checkpoints used by this project come from the official
    FastReID release and are explicitly loaded with
    weights_only=False.
    """

    def _load_file(
        self,
        file_path: str,
    ) -> dict:
        return torch.load(
            file_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )


def load_model_spec(
    model_id: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> FastReIDModelSpec:
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found:\n"
            f"{registry_path}"
        )

    with registry_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(file)

    fastreid_root = (
        PROJECT_ROOT
        / registry["fastreid_root"]
    )

    checkpoint_root = (
        PROJECT_ROOT
        / registry["checkpoint_root"]
    )

    selected_model = None

    for model in registry["models"]:
        if model["id"] == model_id:
            selected_model = model
            break

    if selected_model is None:
        available_models = [
            model["id"]
            for model in registry["models"]
        ]

        raise ValueError(
            f"Unknown model id: {model_id}\n"
            f"Available models:\n"
            + "\n".join(
                f"  - {name}"
                for name in available_models
            )
        )

    config_path = (
        fastreid_root
        / selected_model["config"]
    )

    checkpoint_path = (
        checkpoint_root
        / selected_model["checkpoint"]
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"FastReID config not found:\n"
            f"{config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    official = selected_model["official_msmt17"]

    return FastReIDModelSpec(
        model_id=selected_model["id"],
        family=selected_model["family"],
        backbone=selected_model["backbone"],
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        source_dataset=registry["source_dataset"],
        official_rank1=float(
            official["rank1"]
        ),
        official_map=float(
            official["map"]
        ),
        official_minp=float(
            official["minp"]
        ),
    )


class FastReIDAdapter:
    """
    Unified inference interface for official FastReID models.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
    ) -> None:
        self.spec = load_model_spec(
            model_id=model_id,
        )

        if (
            device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        self.device = torch.device(device)

        self.cfg = self._build_config()

        self.model = self._build_model()

    def _build_config(self):
        cfg = get_cfg()

        cfg.merge_from_file(
            str(self.spec.config_path)
        )

        cfg.defrost()

        # We already have the complete MSMT17-trained checkpoint.
        # Do not download/load ImageNet initialization first.
        cfg.MODEL.BACKBONE.PRETRAIN = False

        cfg.MODEL.WEIGHTS = str(
            self.spec.checkpoint_path
        )

        cfg.MODEL.DEVICE = str(
            self.device
        )

        cfg.freeze()

        return cfg

    def _build_model(self):
        model = build_model(
            self.cfg
        )

        checkpointer = CompatibleCheckpointer(
            model
        )

        checkpointer.load(
            str(self.spec.checkpoint_path)
        )

        model.to(
            self.device
        )

        model.eval()

        return model

    @property
    def input_size(self) -> tuple[int, int]:
        height, width = self.cfg.INPUT.SIZE_TEST

        return int(height), int(width)

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def backbone(self) -> str:
        return self.spec.backbone

    @torch.inference_mode()
    def encode(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract ReID embeddings.

        Expected input:
            images:
                Float tensor with shape:
                [B, 3, H, W]

                Pixel range:
                [0, 255]

        Returns:
            embeddings:
                Float tensor with shape:
                [B, embedding_dim]
        """

        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape "
                "[B, 3, H, W]."
            )

        if images.shape[1] != 3:
            raise ValueError(
                "Expected exactly 3 image channels."
            )

        images = (
            images
            .to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            .clone()
        )

        embeddings = self.model(
            {
                "images": images,
            }
        )

        if not isinstance(
            embeddings,
            torch.Tensor,
        ):
            raise TypeError(
                "FastReID model did not return "
                "a Tensor during inference."
            )

        return embeddings

    @torch.inference_mode()
    def dummy_encode(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        height, width = self.input_size

        images = torch.rand(
            batch_size,
            3,
            height,
            width,
            device=self.device,
            dtype=torch.float32,
        )

        images *= 255.0

        return self.encode(
            images
        )