from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.models.registry import all_specs

TEAMMATE_REF = "origin/feature/fastreid-benchmark-v2"
OUT_DIR = PROJECT_ROOT / "docs" / "benchmark" / "results"
CMC_DIR = PROJECT_ROOT / "docs" / "benchmark" / "cmc"
LOCAL_RESULTS = PROJECT_ROOT / "results" / "comparison"

MSMT_MODELS = [
    "sbs_r50",
    "sbs_r50_ibn",
    "sbs_s50",
    "sbs_r101_ibn",
    "bot_r50",
    "bot_r50_ibn",
    "bot_s50",
    "bot_r101_ibn",
    "agw_r50",
    "agw_r50_ibn",
    "agw_s50",
    "agw_r101_ibn",
]


def git_show(path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{TEAMMATE_REF}:{path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def load_efficiency() -> dict[str, dict[str, str]]:
    text = git_show("benchmarks/results/fastreid_efficiency.csv")
    reader = csv.DictReader(text.splitlines())
    return {row["model_id"]: row for row in reader}


def pct(value: float) -> float:
    return float(value) * 100.0


def rank_from_cmc(payload: dict, rank: int) -> float | None:
    cmc = payload.get("cmc")
    if not isinstance(cmc, list) or len(cmc) < rank:
        return None
    return pct(float(cmc[rank - 1]))


def convert_metrics(payload: dict, model_key: str, dataset: str, efficiency_row: dict) -> dict:
    spec = all_specs()[model_key]
    parameters = int(float(efficiency_row["parameters"]))
    checkpoint_mb = float(efficiency_row["checkpoint_mb"])
    peak_mb = float(efficiency_row["peak_vram_batch_mb"])
    batch_latency_ms = float(efficiency_row["batch_latency_ms"])
    throughput = float(efficiency_row["throughput_images_per_second"])
    batch1_ms = float(efficiency_row["batch1_latency_ms"])
    return {
        "model_key": model_key,
        "model": spec.label,
        "family": spec.family,
        "trained_on": spec.trained_on,
        "notes": spec.notes,
        "dataset": dataset,
        "source": TEAMMATE_REF,
        "protocol": payload["protocol"],
        "similarity": payload.get("similarity"),
        "num_frames": 8 if dataset == "mars" else 1,
        "batch_size": 16,
        "device": payload.get("evaluation_device", "cuda"),
        "query_tracklets": int(payload["query_samples"]),
        "gallery_tracklets": int(payload["gallery_samples"]),
        "dataset_name": payload.get("dataset_name"),
        "metrics": {
            "Rank-1": pct(payload["rank_1"]),
            "Rank-5": pct(payload["rank_5"]),
            "Rank-10": pct(payload["rank_10"]),
            "mAP": pct(payload["mAP"]),
            "mINP": pct(payload["mINP"]),
            "num_valid_queries": float(payload["valid_queries"]),
            "skipped_queries": float(payload.get("skipped_queries", 0)),
        },
        "efficiency": {
            "source": TEAMMATE_REF,
            "note": (
                "Teammate profiler at batch 16, wall-clock. "
                "Not comparable to our CUDA-event GPU ms/image."
            ),
            "batch_size": 16,
            "num_frames": 8 if dataset == "mars" else 1,
            "model": {
                "parameters": parameters,
                "total_mb": checkpoint_mb,
            },
            "gpu_memory": {
                "peak": {
                    "peak_allocated_mb": peak_mb,
                }
            },
            "inference": {
                "ms_per_image": {
                    "wall": batch1_ms,
                    "gpu_steady": batch_latency_ms / 16.0,
                },
                "images_per_s": {
                    "gpu_steady": throughput,
                },
            },
        },
    }


def copy_local_measured() -> None:
    if not LOCAL_RESULTS.exists():
        return
    for path in LOCAL_RESULTS.glob("*.json"):
        shutil.copy2(path, OUT_DIR / path.name)
        print(f"Copied local {path.name}")


def import_msmt(efficiency: dict[str, dict[str, str]]) -> None:
    for model_id in MSMT_MODELS:
        our_key = f"msmt_{model_id}"
        for dataset, protocol in (
            ("market1501", "single_image_l2"),
            ("mars", "uniform8_mean_raw_l2"),
            ("msmt17", "single_image_l2"),
        ):
            rel = (
                f"benchmarks/results/{model_id}/{dataset}/{protocol}/metrics.json"
            )
            payload = json.loads(git_show(rel))
            converted = convert_metrics(
                payload, our_key, dataset, efficiency[model_id]
            )
            rank20 = rank_from_cmc(payload, 20)
            if rank20 is not None:
                converted["metrics"]["Rank-20"] = rank20
            out = OUT_DIR / f"{our_key}_{dataset}.json"
            out.write_text(json.dumps(converted, indent=2), encoding="utf-8")
            print(f"Wrote {out.name}")

            cmc_rel = f"benchmarks/results/cmc/{model_id}/{dataset}.csv"
            cmc_text = git_show(cmc_rel)
            cmc_out = CMC_DIR / f"{our_key}_{dataset}.csv"
            cmc_out.write_text(cmc_text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CMC_DIR.mkdir(parents=True, exist_ok=True)
    copy_local_measured()
    efficiency = load_efficiency()
    import_msmt(efficiency)
    print(f"Imported teammate results into {OUT_DIR}")


if __name__ == "__main__":
    main()
