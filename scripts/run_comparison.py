from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.models.registry import all_specs, default_sweep_keys, tracking_sweep_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FastReID weights if needed, eval selected models, then plot."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Registry keys. Default is the measured in-domain sweep, not the full zoo.",
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Evaluate Deep OC-SORT appearance models (MOT/Dance SBS-S50 + OSNet-AIN).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["market1501", "mars", "msmt17"],
        help="market1501, mars, and/or msmt17.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def download_weights(keys: list[str]) -> None:
    dest_dir = PROJECT_ROOT / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    specs = all_specs()
    for key in keys:
        spec = specs[key]
        if spec.weights_url is None or spec.weights_path is None:
            continue
        if spec.weights_path.exists() and spec.weights_path.stat().st_size > 1_000_000:
            print(f"Have {spec.weights_path.name}")
            continue
        print(f"Downloading {spec.weights_url}")
        if "drive.google.com" in spec.weights_url:
            import gdown

            gdown.download(
                spec.weights_url,
                str(spec.weights_path),
                quiet=False,
                fuzzy=True,
            )
        else:
            urllib.request.urlretrieve(spec.weights_url, spec.weights_path)
        print(f"Wrote {spec.weights_path} ({spec.weights_path.stat().st_size} bytes)")


def main() -> None:
    args = parse_args()
    if args.models is None:
        args.models = tracking_sweep_keys() if args.tracking else default_sweep_keys()
    python = sys.executable
    results = PROJECT_ROOT / "results" / "comparison"
    results.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        download_weights(args.models)
        for model in args.models:
            for dataset in args.datasets:
                output = results / f"{model}_{dataset}.json"
                if args.skip_existing and output.exists():
                    print(f"Skip existing {output.name}")
                    continue
                cmd = [
                    python,
                    str(PROJECT_ROOT / "scripts" / "eval_encoder.py"),
                    "--model",
                    model,
                    "--dataset",
                    dataset,
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                    "--output",
                    str(output),
                ]
                print("\n==>", " ".join(cmd))
                subprocess.run(cmd, check=True)

    plot = [
        python,
        str(PROJECT_ROOT / "scripts" / "plot_comparison.py"),
        "--output-dir",
        str(PROJECT_ROOT / "docs" / "benchmark"),
    ]
    subprocess.run(plot, check=True)


if __name__ == "__main__":
    main()
