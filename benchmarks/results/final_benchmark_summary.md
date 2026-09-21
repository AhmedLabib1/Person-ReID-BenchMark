# SHAWAF FastReID DukeMTMC Final Benchmark

Source training dataset: DukeMTMC

Evaluation datasets: MSMT17_V2, Market1501 and MARS.

All three evaluation datasets are cross-domain relative to the DukeMTMC-trained checkpoints.

MSMT17_V2 note: one unreadable query image and one unreadable gallery image were skipped during this benchmark.

## Efficiency Hardware

- GPU: Tesla T4
- GPU VRAM: 14.56 GB
- Python: 3.12.11
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- OS: Linux-6.8.0-1063-aws-x86_64-with-glibc2.39

Efficiency protocol: 10 warm-up iterations, 50 measurement iterations, batch size 16, 8-frame tracklet.

## Final Results

| Model | Family | Backbone | MSMT R1 | MSMT mAP | Market R1 | Market mAP | MARS R1 | MARS mAP | B1 ms | Throughput | Tracklet ms | Peak VRAM B16 | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bot_r50 | BoT | R50 | 10.71% | 3.17% | 44.48% | 19.29% | 35.76% | 21.79% | 8.516 | 472.77 | 19.904 | 209.32 MB | 23.512M |
| bot_r50_ibn | BoT | R50-IBN | 13.35% | 4.15% | 46.62% | 21.47% | 41.74% | 25.73% | 12.528 | 447.87 | 20.139 | 209.32 MB | 23.512M |
| bot_s50 | BoT | S50 | 16.49% | 4.95% | 48.96% | 21.58% | 40.43% | 24.86% | 21.290 | 285.40 | 65.197 | 302.98 MB | 25.438M |
| bot_r101_ibn | BoT | R101-IBN | 13.71% | 4.22% | 51.57% | 24.24% | 42.39% | 26.79% | 24.641 | 293.19 | 31.983 | 282.23 MB | 42.504M |
| agw_r50 | AGW | R50 | 11.97% | 3.74% | 45.69% | 21.91% | 40.76% | 26.17% | 11.155 | 422.02 | 21.049 | 217.59 MB | 23.541M |
| agw_r50_ibn | AGW | R50-IBN | 13.48% | 4.10% | 48.81% | 23.52% | 45.16% | 28.92% | 15.637 | 401.60 | 22.005 | 218.97 MB | 23.541M |
| agw_s50 | AGW | S50 | 15.93% | 4.63% | 49.38% | 22.25% | 45.54% | 28.54% | 22.850 | 281.91 | 65.437 | 312.48 MB | 25.438M |
| agw_r101_ibn | AGW | R101-IBN | 12.93% | 4.03% | 49.11% | 24.31% | 42.88% | 27.95% | 30.414 | 264.63 | 34.965 | 291.60 MB | 42.576M |
| sbs_r50 | SBS | R50 | 19.09% | 5.90% | 56.12% | 26.47% | 47.83% | 30.67% | 11.791 | 280.24 | 32.108 | 278.66 MB | 23.541M |
| sbs_r50_ibn | SBS | R50-IBN | 21.78% | 6.98% | 58.76% | 28.71% | 53.80% | 36.25% | 14.658 | 269.81 | 32.876 | 278.66 MB | 23.541M |
| sbs_s50 | SBS | S50 | 21.23% | 6.63% | 59.62% | 28.94% | 52.88% | 35.28% | 21.742 | 189.80 | 97.190 | 414.17 MB | 25.438M |
| sbs_r101_ibn | SBS | R101-IBN | 22.54% | 7.22% | 62.14% | 31.48% | 51.63% | 34.36% | 32.329 | 173.96 | 52.168 | 351.29 MB | 42.576M |
| mgn_r50_ibn | MGN | R50-IBN | 23.64% | 6.95% | 59.65% | 26.31% | 45.82% | 26.44% | 26.018 | 148.14 | 62.805 | 475.11 MB | 68.808M |

## Notes

- MSMT17_V2 and Market1501 use `single_image_l2`.
- MARS uses `uniform8_mean_raw_l2`.
- MARS uses exactly 8 uniformly sampled frames per tracklet.
- Efficiency results are directly comparable only with results measured on the same Tesla T4 hardware cohort.
