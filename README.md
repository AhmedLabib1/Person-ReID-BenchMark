# SHAWAF Person Re-Identification

ReID module for the **SHAWAF graduation project**.

The ReID team receives **tracklets** from the Detection + Tracking team and converts each tracklet into an appearance representation that can later be used to match the same person across different cameras.

For the Tracklet interface, see:

```text
TRACKLET_CONTRACT.md
```

---

## Current Project Structure

```text
Person-ReID-BenchMark/
├── README.md
├── TRACKLET_CONTRACT.md
├── pyproject.toml
├── .gitignore
│
├── reid/
│   ├── sampling.py
│   ├── aggregate.py
│   ├── metrics.py
│   ├── profiling.py
│   ├── eval_loop.py
│   ├── models/
│   │   ├── fastreid.py      # FastReID YAML + zoo checkpoint
│   │   ├── foundation.py    # timm CLIP / SigLIP backends
│   │   ├── runtime.py       # shared batched extract
│   │   └── registry.py      # model catalog + factory
│   └── data/
│       ├── tracklet.py
│       ├── mars.py
│       └── market1501.py
│
└── scripts/
    ├── test_tracklet.py
    ├── inspect_mars.py
    ├── eval_encoder.py
    ├── run_comparison.py
    └── plot_comparison.py
```
---

# FastReID baseline

Two eval-only benchmarks share the same pretrained **SBS ResNet-50** checkpoint trained on Market1501 (no extra training):

1. Official FastReID image ReID on Market1501 (sanity check vs the model zoo).
2. SHAWAF tracklet ReID on MARS (sample frames, extract features, mean-pool, Rank-1 / mAP).

Use a **Python 3.11 CUDA** environment. FastReID does not import on the base Python 3.13 install. On this machine the working env is `crowd-gpu` (PyTorch 2.11 + CUDA 12.8). Install extras with:

```bash
conda activate crowd-gpu
python -m pip install -e ".[fastreid]"
```

Weights (gitignored): `weights/market_sbs_R50.pth`  
https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50.pth

Zoo target on Market1501: **Rank-1 95.4% / mAP 88.2%**. Official FastReID CLI: **Rank-1 95.28% / mAP 88.48%**. SHAWAF wrapper: **95.16 / 88.46** (`results/comparison/sbs_r50_market1501.json`).

---

# Model comparison

Same protocol for every encoder, no extra training:

- **Market1501:** one crop per query/gallery image (official image ReID)
- **MARS:** 8-frame uniform sample → embed → mean-pool + L2 → cosine Rank-1 / mAP
- RTX 5070 Laptop (8 GB), batch 32, `crowd-gpu`

ReID-trained CNNs stay in the 80–95% Rank-1 range. Generic foundation encoders (OpenCLIP, SigLIP) are much weaker — they were never trained to tell people apart.

![Rank-1](docs/benchmark/rank1.png)

![mAP](docs/benchmark/map.png)

![Peak VRAM](docs/benchmark/peak_vram.png)

![GPU latency](docs/benchmark/latency.png)

![Accuracy vs latency](docs/benchmark/pareto.png)

### Accuracy

| Model | Trained on | Market1501 Rank-1 | Market1501 mAP | MARS Rank-1 | MARS mAP |
|---|---|---:|---:|---:|---:|
| FastReID SBS-R50 | Market1501 | 95.16 | 88.46 | **87.72** | **84.04** |
| FastReID AGW-R50 | Market1501 | **95.31** | **88.49** | 85.71 | 80.67 |
| FastReID BoT-R50 | Market1501 | 93.85 | 86.30 | 81.52 | 75.99 |
| SigLIP ViT-B/16 | WebLI | 20.58 | 6.51 | 38.97 | 21.49 |
| OpenCLIP ViT-B/32 | LAION-2B | 13.15 | 4.06 | 21.52 | 11.03 |

### Compute (Market1501 extract)

| Model | Params (M) | Weights (MiB) | Allocated (MiB) | Peak (MiB) | GPU ms/img | GPU img/s |
|---|---:|---:|---:|---:|---:|---:|
| FastReID BoT-R50 | 23.5 | 89.9 | 89.9 | **310** | **1.50** | **666** |
| FastReID AGW-R50 | 23.5 | 90.0 | 90.1 | 318 | 1.54 | 651 |
| FastReID SBS-R50 | 23.5 | 90.0 | 90.1 | 428 | 2.54 | 394 |
| OpenCLIP ViT-B/32 | 87.5 | 334 | 334 | 436 | 1.71 | 584 |
| SigLIP ViT-B/16 | 92.9 | 354 | 354 | 622 | 6.84 | 146 |

## How to run

Activate `crowd-gpu` first. JSON is written to `results/comparison/{model}_{dataset}.json`.

Registered `--model` keys: `sbs_r50`, `agw_r50`, `bot_r50`, `siglip_base`, `openclip_vitb32`.  
`--dataset` is `market1501` or `mars`.

### One model

```bash
conda activate crowd-gpu

python scripts/eval_encoder.py --model sbs_r50 --dataset market1501 --device cuda
python scripts/eval_encoder.py --model sbs_r50 --dataset mars --device cuda

python scripts/eval_encoder.py --model agw_r50 --dataset market1501 --device cuda
python scripts/eval_encoder.py --model agw_r50 --dataset mars --device cuda

python scripts/eval_encoder.py --model bot_r50 --dataset market1501 --device cuda
python scripts/eval_encoder.py --model bot_r50 --dataset mars --device cuda

python scripts/eval_encoder.py --model siglip_base --dataset market1501 --device cuda
python scripts/eval_encoder.py --model siglip_base --dataset mars --device cuda

python scripts/eval_encoder.py --model openclip_vitb32 --dataset market1501 --device cuda
python scripts/eval_encoder.py --model openclip_vitb32 --dataset mars --device cuda
```

### All models

Downloads missing FastReID weights, evaluates every registered model on both datasets, then regenerates `docs/benchmark/` plots:

```bash
conda activate crowd-gpu
python scripts/run_comparison.py --device cuda --batch-size 32
```

Skip JSON files that already exist:

```bash
python scripts/run_comparison.py --skip-existing --device cuda --batch-size 32
```

A subset of models or datasets:

```bash
python scripts/run_comparison.py --models sbs_r50 agw_r50 --datasets mars --device cuda
```

Plots only (no extract):

```bash
python scripts/plot_comparison.py
```

MARS only needs `bbox_test/` plus `info/` (`bbox_train/` can stay empty). SBS-R50 on MARS: **Rank-1 87.72% / mAP 84.04%**. MARS and Market1501 share the same capture site, so this is an in-scene tracklet baseline rather than a harsh domain shift.

## Official Market1501 eval

Put Market1501 under `datasets/Market-1501-v15.09.15/` (`bounding_box_train/`, `bounding_box_test/`, `query/`). From `fast-reid/`:

```bash
conda activate crowd-gpu
$env:FASTREID_DATASETS = "$PWD\..\datasets"

python tools/train_net.py --config-file ./configs/Market1501/sbs_R50.yml --eval-only `
  MODEL.WEIGHTS "$PWD\..\weights\market_sbs_R50.pth" `
  MODEL.DEVICE "cuda:0" `
  MODEL.BACKBONE.PRETRAIN False `
  DATALOADER.NUM_WORKERS 0 `
  TEST.IMS_PER_BATCH 32
```

On Windows, `DATALOADER.NUM_WORKERS 0` avoids DataLoader hangs. Cython rank eval is skipped if `make` is missing; Python CMC/mAP is used instead.

---

# MARS Dataset

We use **MARS** because it is a video Person Re-Identification dataset based on **tracklets**, which matches the type of input our SHAWAF ReID system will receive from the Detection + Tracking team.

The dataset is not uploaded to GitHub because of its large size.

---

## Required Dataset Structure

Create a local `datasets` folder with this structure:

```text
datasets/
└── MARS/
    ├── bbox_train/
    ├── bbox_test/
    └── info/
```

So the complete local project will look like:

```text
Person-ReID-BenchMark/
├── reid/
├── scripts/
├── datasets/
│   └── MARS/
│       ├── bbox_train/
│       ├── bbox_test/
│       └── info/
├── TRACKLET_CONTRACT.md
├── pyproject.toml
├── .gitignore
└── README.md
```

`datasets/` is ignored by Git and should **not** be pushed to GitHub.

---

## Download `bbox_train` and `bbox_test`

Download the MARS dataset from Google Drive:

https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0

After downloading, extract:

```text
bbox_train.zip
bbox_test.zip
```

and place them here:

```text
datasets/MARS/bbox_train/
datasets/MARS/bbox_test/
```

---

## Download `info`

The MARS metadata is available here:

https://github.com/liangzheng06/MARS-evaluation/tree/master/info

The `info` folder should contain:

```text
info/
├── query_IDX.mat
├── test_name.txt
├── tracks_test_info.mat
├── tracks_train_info.mat
└── train_name.txt
```

To download only the `info` folder using Git:

```bash
git clone --filter=blob:none --no-checkout https://github.com/liangzheng06/MARS-evaluation.git

cd MARS-evaluation

git sparse-checkout init --cone

git sparse-checkout set info

git checkout master
```

If Google Drive / SharePoint is blocked, the same bbox trees are on Kaggle (`lyqassf/marslyq`). With a Python 3.11 env:

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('lyqassf/marslyq'))"
```

Copy the resulting `bbox_train/` and `bbox_test/` into `datasets/MARS/` next to `info/`.

---

# Installation

From the project root:

```bash
python -m pip install -e .
```

This installs the SHAWAF ReID package and its required dependencies in editable mode.

---

# Current Goal

```text
MARS Dataset
      ↓
MARS Loader
      ↓
SHAWAF Tracklet Objects
      ↓
Preprocessing
      ↓
ReID Encoder
      ↓
Tracklet Embedding
      ↓
Matching / Retrieval
```