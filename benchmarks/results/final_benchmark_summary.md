# SHAWAF ReID Final Benchmark Summary

FastReID models trained on MSMT17_V2, evaluated in-domain on MSMT17_V2 and cross-domain on Market1501 and MARS.

| Model | Family | Backbone | MSMT R1 | MSMT mAP | Market R1 | Market mAP | MARS R1 | MARS mAP | Latency | Throughput | VRAM B16 | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sbs_r101_ibn | SBS | R101-IBN | 82.98% | 58.44% | 61.61% | 32.82% | 28.91% | 24.04% | 19.49 ms | 149.51 img/s | 351.29 MB | 42.58 M |
| sbs_r50_ibn | SBS | R50-IBN | 82.22% | 56.49% | 61.55% | 32.05% | 28.80% | 24.04% | 9.07 ms | 232.88 img/s | 278.66 MB | 23.54 M |
| sbs_r50 | SBS | R50 | 81.74% | 56.36% | 59.74% | 30.48% | 28.91% | 24.19% | 6.02 ms | 244.15 img/s | 278.66 MB | 23.54 M |
| sbs_s50 | SBS | S50 | 82.54% | 59.27% | 57.10% | 29.19% | 27.50% | 23.66% | 13.35 ms | 124.39 img/s | 345.94 MB | 25.44 M |
| bot_r101_ibn | BoT | R101-IBN | 79.05% | 55.68% | 52.02% | 26.49% | 27.66% | 23.42% | 23.28 ms | 262.19 img/s | 282.23 MB | 42.50 M |
| bot_r50_ibn | BoT | R50-IBN | 76.80% | 52.59% | 50.42% | 26.22% | 27.77% | 23.48% | 11.77 ms | 402.47 img/s | 209.32 MB | 23.51 M |
| bot_s50 | BoT | S50 | 78.57% | 55.87% | 49.64% | 24.79% | 28.04% | 23.43% | 16.51 ms | 185.93 img/s | 256.75 MB | 25.44 M |
| agw_r101_ibn | AGW | R101-IBN | 77.93% | 56.33% | 48.87% | 24.11% | 27.50% | 23.31% | 20.11 ms | 231.07 img/s | 291.60 MB | 42.58 M |
| agw_r50_ibn | AGW | R50-IBN | 77.96% | 55.71% | 47.62% | 23.93% | 27.72% | 23.28% | 10.16 ms | 358.58 img/s | 218.97 MB | 23.54 M |
| bot_r50 | BoT | R50 | 73.03% | 48.00% | 43.68% | 20.96% | 28.37% | 23.29% | 9.34 ms | 428.51 img/s | 209.32 MB | 23.51 M |
| agw_s50 | AGW | S50 | 69.13% | 40.27% | 43.56% | 19.82% | 26.47% | 23.00% | 15.03 ms | 187.27 img/s | 266.25 MB | 25.44 M |
| agw_r50 | AGW | R50 | 74.97% | 51.54% | 43.53% | 21.61% | 28.42% | 23.25% | 6.40 ms | 379.51 img/s | 217.59 MB | 23.54 M |

## Notes

- MSMT17_V2 uses single-image L2-normalized embeddings (in-domain).
- Market1501 uses single-image L2-normalized embeddings (cross-domain).
- MARS uses 8 uniformly sampled frames, raw-feature mean pooling, then final L2 normalization (cross-domain).
- Efficiency measurements are unchanged and come from the same GPU/profiling protocol used before.
