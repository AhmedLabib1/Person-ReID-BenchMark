from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASTREID_ROOT = PROJECT_ROOT / "fast-reid" / "configs"
WEIGHTS = PROJECT_ROOT / "weights"
FASTREID_RELEASE = (
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1"
)

FamilyName = Literal["SBS", "BoT", "AGW", "MGN", "Foundation", "OSNet"]
DIFFMOT_RELEASE = "https://github.com/Kroery/DiffMOT/releases/download/v1.0"
# Official torchreid OSNet-AIN x1.0 trained on MSMT17 + Duke + CUHK03 (ms_d_c).
OSNET_AIN_MSDC_ID = "1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    trained_on: str
    color: str
    backend: Literal["fastreid", "timm", "osnet"]
    notes: str = ""
    weights_url: str | None = None
    weights_path: Path | None = None
    config_file: Path | None = None
    timm_name: str | None = None
    default_sweep: bool = False
    result_status: Literal["measured", "imported", "registered"] = "registered"


def _fastreid(
    *,
    key: str,
    label: str,
    family: FamilyName,
    trained_on: str,
    color: str,
    config_file: Path,
    weight_name: str,
    notes: str,
    default_sweep: bool = False,
    result_status: Literal["measured", "imported", "registered"] = "registered",
    weights_url: str | None = None,
) -> ModelSpec:
    return ModelSpec(
        key=key,
        label=label,
        family=family,
        trained_on=trained_on,
        color=color,
        backend="fastreid",
        notes=notes,
        weights_url=weights_url or f"{FASTREID_RELEASE}/{weight_name}",
        weights_path=WEIGHTS / weight_name,
        config_file=config_file,
        default_sweep=default_sweep,
        result_status=result_status,
    )


def all_specs() -> dict[str, ModelSpec]:
    market = FASTREID_ROOT / "Market1501"
    msmt = FASTREID_ROOT / "MSMT17"
    duke = FASTREID_ROOT / "DukeMTMC"

    families = [
        ("sbs", "SBS", "sbs", "Stronger BoT (circle loss, GeM, non-local)"),
        ("bot", "BoT", "bagtricks", "Luo et al. bag-of-tricks"),
        ("agw", "AGW", "AGW", "Ye et al. AGW (non-local + GeM)"),
    ]
    backbones = [
        ("r50", "R50", "R50"),
        ("r50_ibn", "R50-IBN", "R50-ibn"),
        ("s50", "S50", "S50"),
        ("r101_ibn", "R101-IBN", "R101-ibn"),
    ]
    market_colors = {
        "sbs_r50": "#2563eb",
        "sbs_r50_ibn": "#1d4ed8",
        "sbs_s50": "#1e40af",
        "sbs_r101_ibn": "#1e3a8a",
        "bot_r50": "#7c3aed",
        "bot_r50_ibn": "#6d28d9",
        "bot_s50": "#5b21b6",
        "bot_r101_ibn": "#4c1d95",
        "agw_r50": "#db2777",
        "agw_r50_ibn": "#be185d",
        "agw_s50": "#9d174d",
        "agw_r101_ibn": "#831843",
    }
    msmt_colors = {
        "msmt_sbs_r50": "#0ea5e9",
        "msmt_sbs_r50_ibn": "#0284c7",
        "msmt_sbs_s50": "#0369a1",
        "msmt_sbs_r101_ibn": "#075985",
        "msmt_bot_r50": "#a78bfa",
        "msmt_bot_r50_ibn": "#8b5cf6",
        "msmt_bot_s50": "#7c3aed",
        "msmt_bot_r101_ibn": "#6d28d9",
        "msmt_agw_r50": "#fb7185",
        "msmt_agw_r50_ibn": "#f43f5e",
        "msmt_agw_s50": "#e11d48",
        "msmt_agw_r101_ibn": "#be123c",
    }
    duke_colors = {
        "duke_sbs_r50": "#65a30d",
        "duke_sbs_r50_ibn": "#4d7c0f",
        "duke_sbs_s50": "#3f6212",
        "duke_sbs_r101_ibn": "#365314",
        "duke_bot_r50": "#ca8a04",
        "duke_bot_r50_ibn": "#a16207",
        "duke_bot_s50": "#854d0e",
        "duke_bot_r101_ibn": "#713f12",
        "duke_agw_r50": "#f97316",
        "duke_agw_r50_ibn": "#ea580c",
        "duke_agw_s50": "#c2410c",
        "duke_agw_r101_ibn": "#9a3412",
        "duke_mgn_r50_ibn": "#0f766e",
    }
    measured_market = {
        f"{family}_{backbone}"
        for family in ("sbs", "bot", "agw")
        for backbone in ("r50", "r50_ibn", "s50", "r101_ibn")
    }

    specs: dict[str, ModelSpec] = {}
    for family_key, family_name, yaml_prefix, notes in families:
        for backbone_key, backbone_label, yaml_stem in backbones:
            yaml_name = f"{yaml_prefix}_{yaml_stem}.yml"
            weight_stem = f"{family_key}_{yaml_stem}"

            market_key = f"{family_key}_{backbone_key}"
            specs[market_key] = _fastreid(
                key=market_key,
                label=f"FastReID {family_name}-{backbone_label}",
                family=family_name,
                trained_on="Market1501",
                color=market_colors[market_key],
                config_file=market / yaml_name,
                weight_name=f"market_{weight_stem}.pth",
                notes=notes,
                default_sweep=market_key in measured_market,
                result_status=(
                    "measured" if market_key in measured_market else "registered"
                ),
            )

            msmt_key = f"msmt_{family_key}_{backbone_key}"
            specs[msmt_key] = _fastreid(
                key=msmt_key,
                label=f"FastReID {family_name}-{backbone_label} (MSMT17)",
                family=family_name,
                trained_on="MSMT17",
                color=msmt_colors[msmt_key],
                config_file=msmt / yaml_name,
                weight_name=f"msmt_{weight_stem}.pth",
                notes=(
                    f"{notes}. Trained on MSMT17 V2; in-domain MSMT17 V2 plus "
                    "Market1501/MARS transfer."
                ),
                result_status="imported",
            )

            duke_key = f"duke_{family_key}_{backbone_key}"
            specs[duke_key] = _fastreid(
                key=duke_key,
                label=f"FastReID {family_name}-{backbone_label} (DukeMTMC)",
                family=family_name,
                trained_on="DukeMTMC",
                color=duke_colors[duke_key],
                config_file=duke / yaml_name,
                weight_name=f"duke_{weight_stem}.pth",
                notes=(
                    f"{notes}. Trained on DukeMTMC; eval on Market1501/MARS/MSMT17 "
                    "(cross-domain). In-domain Duke not run (dataset withdrawn)."
                ),
                result_status="imported",
            )

    specs["mgn_r50_ibn"] = _fastreid(
        key="mgn_r50_ibn",
        label="FastReID MGN-R50-IBN",
        family="MGN",
        trained_on="Market1501",
        color="#0f766e",
        config_file=market / "mgn_R50-ibn.yml",
        weight_name="market_mgn_R50-ibn.pth",
        notes="Multi-granularity network. No MSMT17 zoo checkpoint.",
        default_sweep=True,
        result_status="measured",
    )
    specs["duke_mgn_r50_ibn"] = _fastreid(
        key="duke_mgn_r50_ibn",
        label="FastReID MGN-R50-IBN (DukeMTMC)",
        family="MGN",
        trained_on="DukeMTMC",
        color=duke_colors["duke_mgn_r50_ibn"],
        config_file=duke / "mgn_R50-ibn.yml",
        weight_name="duke_mgn_R50-ibn.pth",
        notes=(
            "Multi-granularity network. Trained on DukeMTMC; eval on "
            "Market1501/MARS/MSMT17 (cross-domain)."
        ),
        result_status="imported",
    )

    specs["openclip_vitb32"] = ModelSpec(
        key="openclip_vitb32",
        label="OpenCLIP ViT-B/32",
        family="Foundation",
        trained_on="LAION-2B",
        color="#059669",
        backend="timm",
        notes="Generic CLIP image encoder, no ReID training",
        timm_name="vit_base_patch32_clip_224.laion2b",
        default_sweep=True,
        result_status="measured",
    )
    specs["siglip_base"] = ModelSpec(
        key="siglip_base",
        label="SigLIP ViT-B/16",
        family="Foundation",
        trained_on="WebLI",
        color="#d97706",
        backend="timm",
        notes="Sigmoid CLIP-style encoder, no ReID training",
        timm_name="vit_base_patch16_siglip_224.webli",
        default_sweep=True,
        result_status="measured",
    )

    market_s50 = market / "sbs_S50.yml"
    for key, label, trained_on, weight_name, color, notes in (
        (
            "mot17_sbs_s50",
            "FastReID SBS-S50 (MOT17)",
            "MOT17",
            "mot17_sbs_S50.pth",
            "#7c3aed",
            "Deep OC-SORT / BoT-SORT MOT17 test ReID. Same SBS-S50 recipe, MOT half-train.",
        ),
        (
            "mot20_sbs_s50",
            "FastReID SBS-S50 (MOT20)",
            "MOT20",
            "mot20_sbs_S50.pth",
            "#6d28d9",
            "Deep OC-SORT MOT20 test ReID (re-trained; BoT-SORT MOT20 weight was corrupt).",
        ),
        (
            "dance_sbs_s50",
            "FastReID SBS-S50 (DanceTrack)",
            "DanceTrack",
            "dance_sbs_S50.pth",
            "#db2777",
            "Deep OC-SORT DanceTrack ReID. Same SBS-S50 recipe on DanceTrack IDs.",
        ),
    ):
        specs[key] = _fastreid(
            key=key,
            label=label,
            family="SBS",
            trained_on=trained_on,
            color=color,
            config_file=market_s50,
            weight_name=weight_name,
            notes=notes,
            default_sweep=False,
            result_status="measured",
            weights_url=f"{DIFFMOT_RELEASE}/{weight_name}",
        )

    specs["osnet_ain_msdc"] = ModelSpec(
        key="osnet_ain_msdc",
        label="OSNet-AIN x1.0 (MS+D+C)",
        family="OSNet",
        trained_on="MSMT+Duke+CUHK03",
        color="#0f766e",
        backend="osnet",
        notes=(
            "Deep OC-SORT MOT17/MOT20 half-val ReID. torchreid OSNet-AIN x1.0, "
            "512-d, ImageNet norm, 256x128. Trained on MSMT17 + Duke + CUHK03."
        ),
        weights_url=f"https://drive.google.com/uc?id={OSNET_AIN_MSDC_ID}",
        weights_path=WEIGHTS / "osnet_ain_ms_d_c.pth.tar",
        default_sweep=False,
        result_status="measured",
    )
    return specs


def default_sweep_keys() -> list[str]:
    return [key for key, spec in all_specs().items() if spec.default_sweep]


def tracking_sweep_keys() -> list[str]:
    return [
        "mot17_sbs_s50",
        "mot20_sbs_s50",
        "dance_sbs_s50",
        "osnet_ain_msdc",
    ]


def leftover_specs() -> list[ModelSpec]:
    tracking = set(tracking_sweep_keys())
    return [
        spec
        for spec in all_specs().values()
        if spec.backend == "fastreid"
        and spec.result_status == "registered"
        and spec.key not in tracking
    ]


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

    if spec.backend == "osnet":
        from reid.models.osnet import OSNetAINEncoder

        if spec.weights_path is None:
            raise ValueError(f"{key} is missing OSNet weights")
        if not spec.weights_path.exists():
            raise FileNotFoundError(f"Missing weights for {key}: {spec.weights_path}")
        return OSNetAINEncoder(weights=spec.weights_path, device=device)

    from reid.models.foundation import TimmEncoder

    if spec.timm_name is None:
        raise ValueError(f"{key} is missing a timm model name")
    return TimmEncoder(model_name=spec.timm_name, device=device)
