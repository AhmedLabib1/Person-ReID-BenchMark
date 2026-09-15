# SHAWAF ReID Cross-Domain Benchmark

All FastReID checkpoints were trained on MSMT17 and evaluated without fine-tuning on Market1501 and MARS.

## Market1501

| Model | Family | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---|---|---|---:|---:|---:|---:|---:|
| sbs_r101_ibn | SBS | R101-IBN | 61.61% | 77.97% | 83.94% | 32.82% | 5.04% |
| sbs_r50_ibn | SBS | R50-IBN | 61.55% | 78.44% | 84.26% | 32.05% | 5.13% |
| sbs_r50 | SBS | R50 | 59.74% | 75.83% | 82.16% | 30.48% | 4.59% |
| sbs_s50 | SBS | S50 | 57.10% | 74.29% | 79.66% | 29.19% | 3.87% |
| bot_r101_ibn | BoT | R101-IBN | 52.02% | 69.77% | 77.29% | 26.49% | 4.63% |
| bot_r50_ibn | BoT | R50-IBN | 50.42% | 69.12% | 75.50% | 26.22% | 4.42% |
| bot_s50 | BoT | S50 | 49.64% | 68.65% | 75.45% | 24.79% | 3.38% |
| agw_r101_ibn | AGW | R101-IBN | 48.87% | 67.84% | 75.18% | 24.11% | 4.19% |
| agw_r50_ibn | AGW | R50-IBN | 47.62% | 65.86% | 72.86% | 23.93% | 3.87% |
| bot_r50 | BoT | R50 | 43.68% | 62.23% | 70.19% | 20.96% | 3.20% |
| agw_s50 | AGW | S50 | 43.56% | 63.66% | 71.26% | 19.82% | 2.28% |
| agw_r50 | AGW | R50 | 43.53% | 63.27% | 70.25% | 21.61% | 3.49% |

## MARS

| Model | Family | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---|---|---|---:|---:|---:|---:|---:|
| sbs_r50 | SBS | R50 | 28.91% | 50.71% | 57.99% | 24.19% | 14.93% |
| sbs_r101_ibn | SBS | R101-IBN | 28.91% | 50.98% | 59.29% | 24.04% | 14.88% |
| sbs_r50_ibn | SBS | R50-IBN | 28.80% | 49.29% | 57.39% | 24.04% | 14.76% |
| agw_r50 | AGW | R50 | 28.42% | 48.75% | 56.52% | 23.25% | 13.95% |
| bot_r50 | BoT | R50 | 28.37% | 49.24% | 57.07% | 23.29% | 13.90% |
| bot_s50 | BoT | S50 | 28.04% | 49.35% | 56.85% | 23.43% | 14.08% |
| bot_r50_ibn | BoT | R50-IBN | 27.77% | 49.29% | 56.96% | 23.48% | 14.29% |
| agw_r50_ibn | AGW | R50-IBN | 27.72% | 49.13% | 56.96% | 23.28% | 14.03% |
| bot_r101_ibn | BoT | R101-IBN | 27.66% | 49.51% | 57.45% | 23.42% | 14.25% |
| sbs_s50 | SBS | S50 | 27.50% | 49.46% | 58.21% | 23.66% | 14.84% |
| agw_r101_ibn | AGW | R101-IBN | 27.50% | 48.80% | 56.85% | 23.31% | 14.11% |
| agw_s50 | AGW | S50 | 26.47% | 49.78% | 57.23% | 23.00% | 13.81% |

## Protocols

- Market1501: single-image embedding followed by L2 normalization.
- MARS: 8 uniformly sampled frames per tracklet, raw-frame mean pooling, then final L2 normalization.

MSMT17_V2 in-domain evaluation is excluded until a verified copy of the dataset is available.
