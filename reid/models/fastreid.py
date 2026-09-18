from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from reid.models.runtime import extract_images
from reid.profiling import ExtractStats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASTREID_ROOT = PROJECT_ROOT / "fast-reid"


def _ensure_fastreid_on_path() -> None:
    fastreid_path = str(FASTREID_ROOT)
    if fastreid_path not in sys.path:
        sys.path.insert(0, fastreid_path)


class FastReIDEncoder:
    """
    Feature extractor around a FastReID YAML config and checkpoint.

    Preprocessing matches FastReID's demo predictor:
    RGB resize to INPUT.SIZE_TEST, float32 pixels in [0, 255].
    Mean/std normalization is applied inside the model.
    """

    def __init__(
        self,
        config_file: str | Path,
        weights: str | Path,
        device: str = "cuda",
    ) -> None:
        _ensure_fastreid_on_path()

        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model

        self.device = torch.device(self._resolve_device(device))
        cfg = get_cfg()
        cfg.merge_from_file(str(config_file))
        cfg.defrost()
        cfg.MODEL.DEVICE = str(self.device)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS = str(Path(weights).resolve())
        cfg.freeze()

        self.cfg = cfg
        self.height, self.width = cfg.INPUT.SIZE_TEST
        self.model = build_model(cfg)
        self.model.eval()
        self._load_weights(Path(cfg.MODEL.WEIGHTS))
        self.model.to(self.device)
        self.last_extract_stats = ExtractStats()

    def _load_weights(self, weights: Path) -> None:
        checkpoint = torch.load(
            weights,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["model"] if (
            isinstance(checkpoint, dict) and "model" in checkpoint
        ) else checkpoint

        model_state = self.model.state_dict()
        remapped = self._remap_checkpoint(state, model_state)

        incompatible = self.model.load_state_dict(remapped, strict=False)
        skipped = [
            key for key in incompatible.unexpected_keys
            if not key.startswith("pixel_")
        ]
        missing = [
            key
            for key in incompatible.missing_keys
            if key != "heads.weight" and not key.endswith("_head.weight")
        ]
        if missing:
            raise RuntimeError(
                "FastReID checkpoint is missing required weights: "
                + ", ".join(missing)
            )
        if skipped:
            print("Unused checkpoint keys:", ", ".join(skipped))

    @staticmethod
    def _remap_checkpoint(
        state: dict,
        model_state: dict,
    ) -> dict:
        """
        FastReID zoo checkpoints predate some head refactors.

        SBS/BoT/AGW: heads.bnneck / heads.classifier.
        MGN: per-branch b*_pool (GeM + 1x1 + BN) plus b*_head.bnneck.
        Stripe heads (b21, b31, ...) reuse the parent branch pool weights.
        """

        remapped: dict = {}
        for key, value in state.items():
            new_key = key
            if new_key.endswith(".classifier.weight"):
                new_key = new_key[: -len("classifier.weight")] + "weight"
            elif ".bnneck." in new_key:
                prefix, rest = new_key.split(".bnneck.", 1)
                if f"{prefix}.bottleneck.1.weight" in model_state:
                    new_key = f"{prefix}.bottleneck.1.{rest}"
                else:
                    new_key = f"{prefix}.bottleneck.0.{rest}"
            remapped[new_key] = value

        pool_dests = {
            "b1_pool": ["b1_head"],
            "b2_pool": ["b2_head", "b21_head", "b22_head"],
            "b3_pool": ["b3_head", "b31_head", "b32_head", "b33_head"],
        }
        for key, value in state.items():
            for pool_prefix, heads in pool_dests.items():
                prefix = pool_prefix + "."
                if not key.startswith(prefix):
                    continue
                index, _, name = key[len(prefix) :].partition(".")
                for head in heads:
                    if index == "0":
                        remapped[f"{head}.pool_layer.{name}"] = value
                    elif index == "1":
                        remapped[f"{head}.bottleneck.0.{name}"] = value
                    elif index == "2":
                        remapped[f"{head}.bottleneck.1.{name}"] = value

        for key in list(remapped):
            expected = model_state.get(key)
            if expected is None or expected.shape != remapped[key].shape:
                remapped.pop(key)
        return remapped

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def preprocess(self, path: Path) -> torch.Tensor | None:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            image_rgb,
            (self.width, self.height),
            interpolation=cv2.INTER_CUBIC,
        )
        return torch.from_numpy(
            resized.astype(np.float32).transpose(2, 0, 1)
        )

    @torch.no_grad()
    def forward_gpu(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model({"images": batch})

    def extract(
        self,
        crop_paths: list[Path],
        batch_size: int = 32,
        show_progress: bool = False,
        warmup_batches: int = 2,
    ) -> np.ndarray:
        return extract_images(
            self,
            crop_paths,
            batch_size=batch_size,
            show_progress=show_progress,
            warmup_batches=warmup_batches,
            desc="FastReID extract",
        )
