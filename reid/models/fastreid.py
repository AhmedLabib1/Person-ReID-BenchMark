from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


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

    def _load_weights(self, weights: Path) -> None:
        checkpoint = torch.load(
            weights,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["model"] if (
            isinstance(checkpoint, dict) and "model" in checkpoint
        ) else checkpoint

        remapped = {}
        for key, value in state.items():
            new_key = key
            if new_key.startswith("heads.bnneck."):
                new_key = "heads.bottleneck.0." + new_key[len("heads.bnneck."):]
            elif new_key == "heads.classifier.weight":
                new_key = "heads.weight"
            remapped[new_key] = value

        # Eval builds the head with 0 classes; the zoo classifier is 751-way.
        model_state = self.model.state_dict()
        classifier = remapped.get("heads.weight")
        if classifier is not None:
            expected = model_state.get("heads.weight")
            if expected is None or expected.shape != classifier.shape:
                remapped.pop("heads.weight")

        incompatible = self.model.load_state_dict(remapped, strict=False)
        skipped = [
            key for key in incompatible.unexpected_keys
            if not key.startswith("pixel_")
        ]
        missing = [
            key for key in incompatible.missing_keys
            if key != "heads.weight"
        ]
        if missing:
            raise RuntimeError(
                "FastReID checkpoint is missing required weights: "
                + ", ".join(missing)
            )
        if skipped:
            print("Unused checkpoint keys:", ", ".join(skipped))

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def _load_image(self, path: Path) -> torch.Tensor | None:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            image_rgb,
            (self.width, self.height),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = torch.from_numpy(
            resized.astype(np.float32).transpose(2, 0, 1)
        )
        return tensor

    @torch.no_grad()
    def extract(
        self,
        crop_paths: list[Path],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Return embeddings of shape (num_valid_crops, dim).

        Crops that cannot be read are skipped. The caller should keep
        path/embedding alignment themselves if they need it; this method
        only returns features for successfully loaded images.
        """

        if not crop_paths:
            return np.zeros((0, 0), dtype=np.float32)

        features: list[np.ndarray] = []
        batch: list[torch.Tensor] = []
        iterator = crop_paths
        if show_progress:
            iterator = tqdm(crop_paths, desc="FastReID extract", unit="img")

        for path in iterator:
            image = self._load_image(Path(path))
            if image is None:
                raise FileNotFoundError(
                    f"Could not read person crop: {path}"
                )
            batch.append(image)
            if len(batch) >= batch_size:
                features.append(self._forward_batch(batch))
                batch = []

        if batch:
            features.append(self._forward_batch(batch))

        if not features:
            return np.zeros((0, 0), dtype=np.float32)

        return np.concatenate(features, axis=0)

    def extract_aligned(
        self,
        crop_paths: list[Path],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> tuple[list[Path], np.ndarray]:
        """Like extract(), but also returns the paths that were loaded."""

        valid_paths: list[Path] = []
        for path in crop_paths:
            if Path(path).is_file():
                valid_paths.append(Path(path))

        embeddings = self.extract(
            valid_paths,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        if embeddings.shape[0] != len(valid_paths):
            # Some files existed but cv2 could not decode them.
            decoded_paths: list[Path] = []
            decoded_images: list[torch.Tensor] = []
            for path in valid_paths:
                image = self._load_image(path)
                if image is None:
                    continue
                decoded_paths.append(path)
                decoded_images.append(image)
            if not decoded_images:
                return [], np.zeros((0, 0), dtype=np.float32)
            chunks = [
                self._forward_batch(decoded_images[i:i + batch_size])
                for i in range(0, len(decoded_images), batch_size)
            ]
            return decoded_paths, np.concatenate(chunks, axis=0)

        return valid_paths, embeddings

    @torch.no_grad()
    def _forward_batch(self, images: list[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(images, dim=0).to(self.device, non_blocking=True)
        outputs = self.model({"images": batch})
        return outputs.detach().cpu().numpy()
