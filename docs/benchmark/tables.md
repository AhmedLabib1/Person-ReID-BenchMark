### In-domain / same-campus (Market1501 weights + foundation)

| Model | Trained on | Market1501 Rank-1 | Market1501 mAP | MARS Rank-1 | MARS mAP |
|---|---|---:|---:|---:|---:|
| FastReID AGW-R50 | Market1501 | 95.31 | 88.49 | 85.71 | 80.67 |
| FastReID SBS-R50 | Market1501 | 95.16 | 88.46 | 87.72 | 84.04 |
| FastReID BoT-R50 | Market1501 | 93.85 | 86.30 | 81.52 | 75.99 |
| SigLIP ViT-B/16 | WebLI | 20.58 | 6.51 | 38.97 | 21.49 |
| OpenCLIP ViT-B/32 | LAION-2B | 13.15 | 4.06 | 21.52 | 11.03 |

### Cross-domain (MSMT17 weights, no fine-tune)

| Model | Trained on | Market1501 Rank-1 | Market1501 mAP | MARS Rank-1 | MARS mAP |
|---|---|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (MSMT17) | MSMT17 | 61.61 | 32.82 | 28.91 | 24.04 |
| FastReID SBS-R50-IBN (MSMT17) | MSMT17 | 61.55 | 32.05 | 28.80 | 24.04 |
| FastReID SBS-R50 (MSMT17) | MSMT17 | 59.74 | 30.48 | 28.91 | 24.19 |
| FastReID SBS-S50 (MSMT17) | MSMT17 | 57.10 | 29.19 | 27.50 | 23.66 |
| FastReID BoT-R101-IBN (MSMT17) | MSMT17 | 52.02 | 26.49 | 27.66 | 23.42 |
| FastReID BoT-R50-IBN (MSMT17) | MSMT17 | 50.42 | 26.22 | 27.77 | 23.48 |
| FastReID BoT-S50 (MSMT17) | MSMT17 | 49.64 | 24.79 | 28.04 | 23.43 |
| FastReID AGW-R101-IBN (MSMT17) | MSMT17 | 48.87 | 24.11 | 27.50 | 23.31 |
| FastReID AGW-R50-IBN (MSMT17) | MSMT17 | 47.62 | 23.93 | 27.72 | 23.28 |
| FastReID BoT-R50 (MSMT17) | MSMT17 | 43.68 | 20.96 | 28.37 | 23.29 |
| FastReID AGW-S50 (MSMT17) | MSMT17 | 43.56 | 19.82 | 26.47 | 23.00 |
| FastReID AGW-R50 (MSMT17) | MSMT17 | 43.53 | 21.61 | 28.42 | 23.25 |

### Compute (our CUDA-event protocol, Market1501 extract)

| Model | Params (M) | Weights (MiB) | Allocated (MiB) | Peak (MiB) | GPU ms/img | GPU img/s |
|---|---:|---:|---:|---:|---:|---:|
| FastReID AGW-R50 | 23.5 | 90.0 | 90.1 | 318.2 | 1.54 | 650.9 |
| FastReID BoT-R50 | 23.5 | 89.9 | 89.9 | 309.9 | 1.50 | 665.8 |
| OpenCLIP ViT-B/32 | 87.5 | 333.6 | 333.6 | 435.7 | 1.71 | 584.1 |
| FastReID SBS-R50 | 23.5 | 90.0 | 90.1 | 428.2 | 2.54 | 394.2 |
| SigLIP ViT-B/16 | 92.9 | 354.3 | 354.3 | 622.3 | 6.84 | 146.2 |

### FastReID zoo still left to run under this protocol

| Key | Model | Trained on | Why it is left |
|---|---|---|---|
| `sbs_r50_ibn` | FastReID SBS-R50-IBN | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `sbs_s50` | FastReID SBS-S50 | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `sbs_r101_ibn` | FastReID SBS-R101-IBN | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `bot_r50_ibn` | FastReID BoT-R50-IBN | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `bot_s50` | FastReID BoT-S50 | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `bot_r101_ibn` | FastReID BoT-R101-IBN | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `agw_r50_ibn` | FastReID AGW-R50-IBN | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `agw_s50` | FastReID AGW-S50 | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `agw_r101_ibn` | FastReID AGW-R101-IBN | Market1501 | Market1501 zoo checkpoint exists; not extracted yet on SHAWAF loaders. |
| `mgn_r50_ibn` | FastReID MGN-R50-IBN | Market1501 | Market zoo checkpoint exists; heavier multi-granularity head. |

Also not in this bench (on purpose):

- DukeMTMC zoo weights (dataset withdrawn).
- Vehicle ReID (VeRi / VehicleID / VERI-Wild).
- FastReID ViT (`bagtricks_vit.yml`) — config only, no zoo `.pth`.
- MSMT17 **in-domain** eval (dataset copy not verified).
- Newer non-FastReID models (SOLIDER, CLIP-ReID, CLIMB-ReID).
