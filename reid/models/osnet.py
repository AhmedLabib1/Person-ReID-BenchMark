from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reid.models.osnet_ain import osnet_ain_x1_0
from reid.models.runtime import extract_images
from reid.profiling import ExtractStats


class OSNetAINEncoder:
    """
    Frozen OSNet-AIN x1.0, matching Deep OC-SORT's half-val ReID path.

    Input is 256x128, ImageNet-normalized RGB in [0, 1]. Feature width is 512.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        num_classes: int = 2510,
    ) -> None:
        self.device = torch.device(
            device if not device.startswith("cuda") or torch.cuda.is_available()
            else "cpu"
        )
        self.height = 256
        self.width = 128
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.model = osnet_ain_x1_0(
            num_classes=num_classes,
            pretrained=False,
            loss="softmax",
        )
        self.model.load_state_dict(self._load_state(Path(weights)), strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.last_extract_stats = ExtractStats()

    def _load_state(self, weights: Path) -> OrderedDict:
        checkpoint = torch.load(
            weights,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint
        if isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        remapped = OrderedDict()
        for key, value in state.items():
            name = key[7:] if key.startswith("module.") else key
            remapped[name] = value
        return remapped

    def preprocess(self, path: Path) -> torch.Tensor | None:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            image_rgb,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        return torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1) / 255.0)

    @torch.no_grad()
    def forward_gpu(self, batch: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(batch.device, dtype=batch.dtype)
        std = self.std.to(batch.device, dtype=batch.dtype)
        outputs = self.model((batch - mean) / std)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return F.normalize(outputs, dim=-1)

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
            desc="osnet_ain extract",
        )
