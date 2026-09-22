# SHAWAF Person Re-Identification

Frozen ReID encoders for the SHAWAF graduation project. Detection + Tracking sends **tracklets**; this module turns each tracklet into an appearance vector for cross-camera matching.

See [TRACKLET_CONTRACT.md](./TRACKLET_CONTRACT.md).

Full tables and graphs (Rank-1, mAP, mINP, transfer, compute):

- [docs/benchmark/tables.md](./docs/benchmark/tables.md)

## Setup

```bash
conda activate crowd-gpu   # Python 3.11 + CUDA. FastReID will not import on 3.13.
python -m pip install -e ".[fastreid]"
```

Put data under `datasets/` (gitignored):

```text
datasets/
├── Market-1501-v15.09.15/
├── MSMT17_V1/
└── MARS/
    ├── bbox_test/
    └── info/
```

Weights go in `weights/` (gitignored). Zoo checkpoints: https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1

Market: [Zheng et al.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) · MARS bbox: [Drive](https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0) + [info](https://github.com/liangzheng06/MARS-evaluation/tree/master/info) · MSMT V1: [Hugging Face](https://huggingface.co/datasets/xianpeijie/MSMT17_V1/resolve/main/MSMT17_V1.zip)

## Protocol

Same ranking everywhere: cosine on L2 features, same-pid + same-camera junk. No extra training.

| Eval | Protocol | Query / gallery | Tests |
|---|---|---|---|
| Market1501 | 1 crop | 3368 / 15913 | Image ReID |
| MARS | 8-frame mean-pool + L2 | 1980 / 9330 | Tracklets, **same campus** as Market |
| MSMT17 | 1 crop | ~11.7k / ~82k | Hard camera / scene shift |

**Market-trained MSMT is V1** (measured here). **MSMT-trained and Duke-trained MSMT is V2** (imported). V1/V2 share splits; V2 blurs faces.

| Train set | In-domain | Transfer | Source |
|---|---|---|---|
| Market1501 (+ CLIP / SigLIP) | Market | MARS, MSMT V1 | Measured here |
| MSMT17 | MSMT V2 | Market, MARS | Imported from `feature/fastreid-benchmark-v2` |
| DukeMTMC | — (dataset withdrawn) | Market, MARS, MSMT V2 | Imported from `feature/DukeMTMC-Benchmark` |

---

## Trained on Market1501

ReID CNNs are 93–96 Rank-1 in-domain. MARS stays high (~81–91). MSMT V1 drops to ~7–21. SigLIP is not a person-ID model but leads this set on MSMT Rank-1 (**24.40**).

![Rank-1, Market-trained](docs/benchmark/figures/rank1.png)

| Model | Market R1 | Market mAP | MARS R1 | MARS mAP | MSMT V1 R1 | MSMT V1 mAP |
|---|---:|---:|---:|---:|---:|---:|
| SBS-S50 | **96.11** | 89.67 | **90.54** | 87.03 | 19.70 | 6.80 |
| SBS-R101-IBN | 95.81 | **90.25** | 90.33 | **87.88** | 19.78 | 6.81 |
| SBS-R50-IBN | 95.55 | 89.13 | 89.18 | 86.64 | 20.79 | 7.10 |
| AGW-R101-IBN | 95.52 | 89.61 | 87.55 | 84.15 | 14.20 | 4.78 |
| AGW-R50-IBN | 95.52 | 89.09 | 87.55 | 84.47 | 14.52 | 5.14 |
| MGN-R50-IBN | 95.37 | 87.25 | 84.51 | 74.50 | 20.84 | **7.17** |
| BoT-R101-IBN | 95.34 | 89.12 | 86.79 | 83.83 | 13.29 | 4.47 |
| AGW-R50 | 95.31 | 88.49 | 85.71 | 80.67 | 10.84 | 3.61 |
| BoT-R50-IBN | 95.28 | 88.23 | 87.83 | 84.10 | 13.23 | 4.69 |
| BoT-S50 | 95.28 | 88.64 | 86.14 | 82.35 | 14.36 | 4.92 |
| AGW-S50 | 95.19 | 88.59 | 86.47 | 82.09 | 14.86 | 4.96 |
| SBS-R50 | 95.16 | 88.46 | 87.72 | 84.04 | 14.73 | 4.72 |
| BoT-R50 | 93.85 | 86.30 | 81.52 | 75.99 | 7.47 | 2.35 |
| SigLIP ViT-B/16 | 20.58 | 6.51 | 38.97 | 21.49 | **24.40** | 6.33 |
| OpenCLIP ViT-B/32 | 13.15 | 4.06 | 21.52 | 11.03 | 4.74 | 1.02 |

More graphs and mINP: [docs/benchmark/tables.md](./docs/benchmark/tables.md)

---

## Trained on MSMT17

In-domain MSMT V2 is strong (~69–83 Rank-1). Transfer to Market is ~44–62; transfer to MARS is weak (~26–29) because MARS is the Market campus, not MSMT.

![Rank-1, MSMT-trained](docs/benchmark/figures/rank1_msmt.png)

| Model | MSMT V2 R1 | MSMT V2 mAP | Market R1 | Market mAP | MARS R1 | MARS mAP |
|---|---:|---:|---:|---:|---:|---:|
| SBS-R101-IBN | 82.98 | 58.44 | **61.61** | **32.82** | **28.91** | 24.04 |
| SBS-R50-IBN | 82.22 | 56.49 | 61.55 | 32.05 | 28.80 | 24.04 |
| SBS-R50 | 81.74 | 56.36 | 59.74 | 30.48 | **28.91** | **24.19** |
| SBS-S50 | **82.54** | **59.27** | 57.10 | 29.19 | 27.50 | 23.66 |
| BoT-R101-IBN | 79.05 | 55.68 | 52.02 | 26.49 | 27.66 | 23.42 |
| BoT-R50-IBN | 76.80 | 52.59 | 50.42 | 26.22 | 27.77 | 23.48 |
| BoT-S50 | 78.57 | 55.87 | 49.64 | 24.79 | 28.04 | 23.43 |
| AGW-R101-IBN | 77.93 | 56.33 | 48.87 | 24.11 | 27.50 | 23.31 |
| AGW-R50-IBN | 77.96 | 55.71 | 47.62 | 23.93 | 27.72 | 23.28 |
| BoT-R50 | 73.03 | 48.00 | 43.68 | 20.96 | 28.37 | 23.29 |
| AGW-S50 | 69.13 | 40.27 | 43.56 | 19.82 | 26.47 | 23.00 |
| AGW-R50 | 74.97 | 51.54 | 43.53 | 21.61 | 28.42 | 23.25 |

More graphs: [docs/benchmark/tables.md](./docs/benchmark/tables.md)

---

## Trained on DukeMTMC

Transfer only — Duke images are not used. Market transfer matches the MSMT-trained band (~44–62). MARS transfer is stronger (~36–54) than MSMT→MARS.

![Rank-1, Duke-trained](docs/benchmark/figures/rank1_duke.png)

| Model | Market R1 | Market mAP | MARS R1 | MARS mAP | MSMT V2 R1 | MSMT V2 mAP |
|---|---:|---:|---:|---:|---:|---:|
| SBS-R101-IBN | **62.14** | **31.48** | 51.63 | 34.36 | 22.54 | **7.22** |
| MGN-R50-IBN | 59.65 | 26.31 | 45.82 | 26.44 | **23.64** | 6.95 |
| SBS-S50 | 59.62 | 28.94 | 52.88 | 35.28 | 21.23 | 6.63 |
| SBS-R50-IBN | 58.76 | 28.71 | **53.80** | **36.25** | 21.78 | 6.98 |
| SBS-R50 | 56.12 | 26.47 | 47.83 | 30.67 | 19.09 | 5.90 |
| BoT-R101-IBN | 51.57 | 24.24 | 42.39 | 26.79 | 13.71 | 4.22 |
| AGW-S50 | 49.38 | 22.25 | 45.54 | 28.54 | 15.93 | 4.63 |
| AGW-R101-IBN | 49.11 | 24.31 | 42.88 | 27.95 | 12.93 | 4.03 |
| BoT-S50 | 48.96 | 21.58 | 40.43 | 24.86 | 16.49 | 4.95 |
| AGW-R50-IBN | 48.81 | 23.52 | 45.16 | 28.92 | 13.48 | 4.10 |
| BoT-R50-IBN | 46.62 | 21.47 | 41.74 | 25.73 | 13.35 | 4.15 |
| AGW-R50 | 45.69 | 21.91 | 40.76 | 26.17 | 11.97 | 3.74 |
| BoT-R50 | 44.48 | 19.29 | 35.76 | 21.79 | 10.71 | 3.17 |

More graphs: [docs/benchmark/tables.md](./docs/benchmark/tables.md)

---

## Same recipe, three train sets

SBS / AGW / BoT R50. Market train stays high on Market and MARS. MSMT or Duke train → Market falls to ~44–62. Duke→MARS beats MSMT→MARS.

![Train set comparison](docs/benchmark/figures/transfer_rank1.png)

Transfer + compute: [docs/benchmark/tables.md](./docs/benchmark/tables.md)

---

## Run

```bash
conda activate crowd-gpu
python scripts/eval_encoder.py --model sbs_s50 --dataset market1501 --device cuda
python scripts/run_comparison.py --skip-existing --device cuda --batch-size 16
python scripts/plot_comparison.py
```

`--dataset` is `market1501`, `mars`, or `msmt17`. Keys: Market `sbs_r50` / `mgn_r50_ibn` / `siglip_base`; MSMT `msmt_sbs_r50`; Duke `duke_sbs_r50` / `duke_mgn_r50_ibn`.

```bash
python scripts/import_v2_results.py      # MSMT-trained JSON from teammate branch
python scripts/import_duke_results.py    # Duke-trained JSON from teammate branch
```
