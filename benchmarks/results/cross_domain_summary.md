# SHAWAF ReID DukeMTMC Cross-Domain Benchmark

All FastReID checkpoints in this benchmark were trained on DukeMTMC.

MSMT17_V2, Market1501 and MARS are evaluated cross-domain without target-dataset fine-tuning.

MSMT17_V2 contains a minor documented dataset deviation in this run: one unreadable query image and one unreadable gallery image were skipped.

## MSMT17_V2

| Model | Family | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---|---|---|---:|---:|---:|---:|---:|
| mgn_r50_ibn | MGN | R50-IBN | 23.64% | 35.07% | 40.40% | 6.95% | 0.10% |
| sbs_r101_ibn | SBS | R101-IBN | 22.54% | 33.98% | 39.73% | 7.22% | 0.18% |
| sbs_r50_ibn | SBS | R50-IBN | 21.78% | 33.22% | 39.05% | 6.98% | 0.16% |
| sbs_s50 | SBS | S50 | 21.23% | 31.97% | 37.73% | 6.63% | 0.16% |
| sbs_r50 | SBS | R50 | 19.09% | 29.36% | 34.53% | 5.90% | 0.12% |
| bot_s50 | BoT | S50 | 16.49% | 26.69% | 32.18% | 4.95% | 0.14% |
| agw_s50 | AGW | S50 | 15.93% | 25.80% | 31.13% | 4.63% | 0.13% |
| bot_r101_ibn | BoT | R101-IBN | 13.71% | 23.85% | 28.83% | 4.22% | 0.11% |
| agw_r50_ibn | AGW | R50-IBN | 13.48% | 22.47% | 27.84% | 4.10% | 0.13% |
| bot_r50_ibn | BoT | R50-IBN | 13.35% | 22.72% | 28.07% | 4.15% | 0.12% |
| agw_r101_ibn | AGW | R101-IBN | 12.93% | 21.68% | 26.60% | 4.03% | 0.13% |
| agw_r50 | AGW | R50 | 11.97% | 21.05% | 25.95% | 3.74% | 0.13% |
| bot_r50 | BoT | R50 | 10.71% | 19.04% | 23.49% | 3.17% | 0.12% |

## Market1501

| Model | Family | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---|---|---|---:|---:|---:|---:|---:|
| sbs_r101_ibn | SBS | R101-IBN | 62.14% | 77.67% | 82.78% | 31.48% | 3.38% |
| mgn_r50_ibn | MGN | R50-IBN | 59.65% | 76.25% | 82.24% | 26.31% | 1.76% |
| sbs_s50 | SBS | S50 | 59.62% | 74.44% | 79.39% | 28.94% | 2.94% |
| sbs_r50_ibn | SBS | R50-IBN | 58.76% | 74.61% | 80.58% | 28.71% | 3.24% |
| sbs_r50 | SBS | R50 | 56.12% | 73.16% | 79.10% | 26.47% | 2.51% |
| bot_r101_ibn | BoT | R101-IBN | 51.57% | 67.61% | 74.85% | 24.24% | 2.89% |
| agw_s50 | AGW | S50 | 49.38% | 66.69% | 72.57% | 22.25% | 2.24% |
| agw_r101_ibn | AGW | R101-IBN | 49.11% | 66.60% | 73.49% | 24.31% | 3.40% |
| bot_s50 | BoT | S50 | 48.96% | 65.41% | 71.88% | 21.58% | 2.17% |
| agw_r50_ibn | AGW | R50-IBN | 48.81% | 66.36% | 74.11% | 23.52% | 2.95% |
| bot_r50_ibn | BoT | R50-IBN | 46.62% | 63.87% | 70.81% | 21.47% | 2.58% |
| agw_r50 | AGW | R50 | 45.69% | 63.54% | 71.02% | 21.91% | 2.89% |
| bot_r50 | BoT | R50 | 44.48% | 61.16% | 68.17% | 19.29% | 1.96% |

## MARS

| Model | Family | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---|---|---|---:|---:|---:|---:|---:|
| sbs_r50_ibn | SBS | R50-IBN | 53.80% | 68.15% | 74.13% | 36.25% | 13.00% |
| sbs_s50 | SBS | S50 | 52.88% | 66.85% | 72.23% | 35.28% | 12.06% |
| sbs_r101_ibn | SBS | R101-IBN | 51.63% | 65.98% | 70.38% | 34.36% | 11.63% |
| sbs_r50 | SBS | R50 | 47.83% | 61.63% | 67.01% | 30.67% | 10.09% |
| mgn_r50_ibn | MGN | R50-IBN | 45.82% | 59.78% | 65.05% | 26.44% | 7.91% |
| agw_s50 | AGW | S50 | 45.54% | 59.13% | 65.22% | 28.54% | 9.42% |
| agw_r50_ibn | AGW | R50-IBN | 45.16% | 59.89% | 64.95% | 28.92% | 9.58% |
| agw_r101_ibn | AGW | R101-IBN | 42.88% | 57.17% | 62.99% | 27.95% | 9.07% |
| bot_r101_ibn | BoT | R101-IBN | 42.39% | 56.96% | 62.93% | 26.79% | 8.54% |
| bot_r50_ibn | BoT | R50-IBN | 41.74% | 56.03% | 61.63% | 25.73% | 8.14% |
| agw_r50 | AGW | R50 | 40.76% | 55.27% | 62.99% | 26.17% | 8.62% |
| bot_s50 | BoT | S50 | 40.43% | 54.84% | 60.49% | 24.86% | 8.03% |
| bot_r50 | BoT | R50 | 35.76% | 50.11% | 55.65% | 21.79% | 6.83% |

## Protocols

- MSMT17_V2: single-image embedding -> L2 normalization (cross-domain).
- Market1501: single-image embedding -> L2 normalization (cross-domain).
- MARS: uniform 8-frame sampling -> raw-feature mean pooling -> final L2 normalization (cross-domain tracklet).
