from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reid.data import MarsDataset
from reid.data.market1501 import Market1501Dataset
from reid.eval_loop import evaluate_encoder
from reid.metrics import format_metrics
from reid.models.registry import all_specs, build_encoder
from reid.profiling import format_efficiency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a registered pretrained encoder on Market1501 or MARS."
    )
    parser.add_argument("--model", required=True, choices=sorted(all_specs()))
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["market1501", "mars"],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root. Defaults to datasets/Market-1501-v15.09.15 or datasets/MARS.",
    )
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def default_root(dataset: str) -> Path:
    if dataset == "mars":
        return PROJECT_ROOT / "datasets" / "MARS"
    return PROJECT_ROOT / "datasets" / "Market-1501-v15.09.15"


def load_splits(dataset: str, root: Path):
    if dataset == "mars":
        return MarsDataset(root).load(include_train=False)
    return Market1501Dataset(root).load()


def main() -> None:
    args = parse_args()
    spec = all_specs()[args.model]
    data_root = args.data_root or default_root(args.dataset)
    num_frames = args.num_frames
    if num_frames is None:
        num_frames = 8 if args.dataset == "mars" else 1

    output = args.output or (
        PROJECT_ROOT / "results" / "comparison" / f"{args.model}_{args.dataset}.json"
    )

    print(f"Model   : {spec.label} ({args.model})")
    print(f"Dataset : {args.dataset} @ {data_root}")
    print(f"Frames  : {num_frames}")

    splits = load_splits(args.dataset, data_root)
    print(
        f"Query tracklets   : {len(splits.query)}\n"
        f"Gallery tracklets : {len(splits.gallery)}"
    )

    encoder = build_encoder(args.model, device=args.device)
    metrics, efficiency = evaluate_encoder(
        encoder=encoder,
        query=splits.query,
        gallery=splits.gallery,
        num_frames=num_frames,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
    )
    print()
    print(format_metrics(metrics))
    print()
    print(format_efficiency(efficiency))

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": spec.key,
        "model": spec.label,
        "family": spec.family,
        "trained_on": spec.trained_on,
        "notes": spec.notes,
        "dataset": args.dataset,
        "data_root": str(data_root),
        "num_frames": num_frames,
        "batch_size": args.batch_size,
        "device": args.device,
        "query_tracklets": len(splits.query),
        "gallery_tracklets": len(splits.gallery),
        "metrics": metrics,
        "efficiency": efficiency,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
