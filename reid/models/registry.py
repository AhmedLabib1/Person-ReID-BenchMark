from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASTREID_CONFIGS = PROJECT_ROOT / "fast-reid" / "configs" / "Market1501"
WEIGHTS = PROJECT_ROOT / "weights"
FASTREID_RELEASE = (
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    trained_on: str
    color: str
    backend: Literal["fastreid", "timm"]
    notes: str = ""
    weights_url: str | None = None
    weights_path: Path | None = None
    config_file: Path | None = None
    timm_name: str | None = None


def all_specs() -> dict[str, ModelSpec]:
    return {
        "sbs_r50": ModelSpec(
            key="sbs_r50",
            label="FastReID SBS-R50",
            family="FastReID",
            trained_on="Market1501",
            color="#2563eb",
            backend="fastreid",
            notes="Strong baseline bag-of-tricks + neck",
            weights_url=f"{FASTREID_RELEASE}/market_sbs_R50.pth",
            weights_path=WEIGHTS / "market_sbs_R50.pth",
            config_file=FASTREID_CONFIGS / "sbs_R50.yml",
        ),
        "bot_r50": ModelSpec(
            key="bot_r50",
            label="FastReID BoT-R50",
            family="FastReID",
            trained_on="Market1501",
            color="#7c3aed",
            backend="fastreid",
            notes="Luo et al. bag-of-tricks ResNet-50",
            weights_url=f"{FASTREID_RELEASE}/market_bot_R50.pth",
            weights_path=WEIGHTS / "market_bot_R50.pth",
            config_file=FASTREID_CONFIGS / "bagtricks_R50.yml",
        ),
        "agw_r50": ModelSpec(
            key="agw_r50",
            label="FastReID AGW-R50",
            family="FastReID",
            trained_on="Market1501",
            color="#db2777",
            backend="fastreid",
            notes="Ye et al. AGW (non-local + GeM)",
            weights_url=f"{FASTREID_RELEASE}/market_agw_R50.pth",
            weights_path=WEIGHTS / "market_agw_R50.pth",
            config_file=FASTREID_CONFIGS / "AGW_R50.yml",
        ),
        "openclip_vitb32": ModelSpec(
            key="openclip_vitb32",
            label="OpenCLIP ViT-B/32",
            family="Foundation",
            trained_on="LAION-2B",
            color="#059669",
            backend="timm",
            notes="Generic CLIP image encoder, no ReID training",
            timm_name="vit_base_patch32_clip_224.laion2b",
        ),
        "siglip_base": ModelSpec(
            key="siglip_base",
            label="SigLIP ViT-B/16",
            family="Foundation",
            trained_on="WebLI",
            color="#d97706",
            backend="timm",
            notes="Sigmoid CLIP-style encoder, no ReID training",
            timm_name="vit_base_patch16_siglip_224.webli",
        ),
    }


def build_encoder(key: str, device: str = "cuda") -> Any:
    spec = all_specs()[key]
    if spec.backend == "fastreid":
        from reid.models.fastreid import FastReIDEncoder

        if spec.config_file is None or spec.weights_path is None:
            raise ValueError(f"{key} is missing FastReID config or weights")
        if not spec.weights_path.exists():
            raise FileNotFoundError(f"Missing weights for {key}: {spec.weights_path}")
        return FastReIDEncoder(
            config_file=spec.config_file,
            weights=spec.weights_path,
            device=device,
        )

    from reid.models.foundation import TimmEncoder

    if spec.timm_name is None:
        raise ValueError(f"{key} is missing a timm model name")
    return TimmEncoder(model_name=spec.timm_name, device=device)
