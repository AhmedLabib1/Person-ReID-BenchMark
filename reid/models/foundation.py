from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reid.models.runtime import extract_images
from reid.profiling import ExtractStats


class TimmEncoder:
    """
    Frozen timm vision backbone (OpenCLIP / SigLIP, etc.).

    Crops are resized to the model's default square size, scaled to
    [0, 1], then normalized with the checkpoint's mean / std.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        image_size: int | None = None,
    ) -> None:
        import timm
        from timm.data import resolve_model_data_config

        self.device = torch.device(
            device if not device.startswith("cuda") or torch.cuda.is_available()
            else "cpu"
        )
        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        self.model.eval()
        self.model.to(self.device)

        data_cfg = resolve_model_data_config(self.model)
        input_size = data_cfg.get("input_size", (3, 224, 224))
        default_h = int(input_size[-2])
        default_w = int(input_size[-1])
        if image_size is not None:
            self.height = int(image_size)
            self.width = int(image_size)
        else:
            self.height = default_h
            self.width = default_w

        mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
        std = data_cfg.get("std", (0.229, 0.224, 0.225))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.last_extract_stats = ExtractStats()

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
        tensor = torch.from_numpy(
            resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        )
        return tensor

    @torch.no_grad()
    def forward_gpu(self, batch: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(batch.device, dtype=batch.dtype)
        std = self.std.to(batch.device, dtype=batch.dtype)
        normalized = (batch - mean) / std
        outputs = self.model(normalized)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if outputs.ndim > 2:
            outputs = outputs.mean(dim=tuple(range(2, outputs.ndim)))
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
            desc=f"{self.model_name} extract",
        )
