### In-domain / same-campus (Market1501 weights + foundation)

| Model | Trained on | Market Rank-1 | Market mAP | Market mINP | MARS Rank-1 | MARS mAP | MARS mINP | MSMT Rank-1 | MSMT mAP | MSMT mINP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FastReID SBS-S50 | Market1501 | 96.11 | 89.67 | 68.75 | 90.54 | 87.03 | 70.41 | 19.70 | 6.80 | 0.25 |
| FastReID SBS-R101-IBN | Market1501 | 95.81 | 90.25 | 71.28 | 90.33 | 87.88 | 71.65 | 19.78 | 6.81 | 0.25 |
| FastReID SBS-R50-IBN | Market1501 | 95.55 | 89.13 | 67.26 | 89.18 | 86.64 | 70.10 | 20.79 | 7.10 | 0.28 |
| FastReID AGW-R101-IBN | Market1501 | 95.52 | 89.61 | 69.36 | 87.55 | 84.15 | 66.52 | 14.20 | 4.78 | 0.20 |
| FastReID AGW-R50-IBN | Market1501 | 95.52 | 89.09 | 67.93 | 87.55 | 84.47 | 67.57 | 14.52 | 5.14 | 0.26 |
| FastReID MGN-R50-IBN | Market1501 | 95.37 | 87.25 | 56.47 | 84.51 | 74.50 | 43.81 | 20.84 | 7.17 | 0.24 |
| FastReID BoT-R101-IBN | Market1501 | 95.34 | 89.12 | 68.23 | 86.79 | 83.83 | 65.82 | 13.29 | 4.47 | 0.17 |
| FastReID AGW-R50 | Market1501 | 95.31 | 88.49 | 65.90 | 85.71 | 80.67 | 60.50 | 10.84 | 3.61 | 0.17 |
| FastReID BoT-R50-IBN | Market1501 | 95.28 | 88.23 | 65.44 | 87.83 | 84.10 | 66.38 | 13.23 | 4.69 | 0.19 |
| FastReID BoT-S50 | Market1501 | 95.28 | 88.64 | 66.83 | 86.14 | 82.35 | 63.21 | 14.36 | 4.92 | 0.17 |
| FastReID AGW-S50 | Market1501 | 95.19 | 88.59 | 66.67 | 86.47 | 82.09 | 63.54 | 14.86 | 4.96 | 0.20 |
| FastReID SBS-R50 | Market1501 | 95.16 | 88.46 | 65.46 | 87.72 | 84.04 | 64.51 | 14.73 | 4.72 | 0.14 |
| FastReID BoT-R50 | Market1501 | 93.85 | 86.30 | 60.61 | 81.52 | 75.99 | 54.26 | 7.47 | 2.35 | 0.12 |
| SigLIP ViT-B/16 | WebLI | 20.58 | 6.51 | 0.35 | 38.97 | 21.49 | 6.29 | 24.40 | 6.33 | 0.12 |
| OpenCLIP ViT-B/32 | LAION-2B | 13.15 | 4.06 | 0.25 | 21.52 | 11.03 | 2.67 | 4.74 | 1.02 | 0.05 |

### Cross-domain (MSMT17 weights, no fine-tune)

| Model | Trained on | Market Rank-1 | Market mAP | Market mINP | MARS Rank-1 | MARS mAP | MARS mINP | MSMT Rank-1 | MSMT mAP | MSMT mINP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (MSMT17) | MSMT17 | 61.61 | 32.82 | 5.04 | 28.91 | 24.04 | 14.88 | — | — | — |
| FastReID SBS-R50-IBN (MSMT17) | MSMT17 | 61.55 | 32.05 | 5.13 | 28.80 | 24.04 | 14.76 | — | — | — |
| FastReID SBS-R50 (MSMT17) | MSMT17 | 59.74 | 30.48 | 4.59 | 28.91 | 24.19 | 14.93 | — | — | — |
| FastReID SBS-S50 (MSMT17) | MSMT17 | 57.10 | 29.19 | 3.87 | 27.50 | 23.66 | 14.84 | — | — | — |
| FastReID BoT-R101-IBN (MSMT17) | MSMT17 | 52.02 | 26.49 | 4.63 | 27.66 | 23.42 | 14.25 | — | — | — |
| FastReID BoT-R50-IBN (MSMT17) | MSMT17 | 50.42 | 26.22 | 4.42 | 27.77 | 23.48 | 14.29 | — | — | — |
| FastReID BoT-S50 (MSMT17) | MSMT17 | 49.64 | 24.79 | 3.38 | 28.04 | 23.43 | 14.08 | — | — | — |
| FastReID AGW-R101-IBN (MSMT17) | MSMT17 | 48.87 | 24.11 | 4.19 | 27.50 | 23.31 | 14.11 | — | — | — |
| FastReID AGW-R50-IBN (MSMT17) | MSMT17 | 47.62 | 23.93 | 3.87 | 27.72 | 23.28 | 14.03 | — | — | — |
| FastReID BoT-R50 (MSMT17) | MSMT17 | 43.68 | 20.96 | 3.20 | 28.37 | 23.29 | 13.90 | — | — | — |
| FastReID AGW-S50 (MSMT17) | MSMT17 | 43.56 | 19.82 | 2.28 | 26.47 | 23.00 | 13.81 | — | — | — |
| FastReID AGW-R50 (MSMT17) | MSMT17 | 43.53 | 21.61 | 3.49 | 28.42 | 23.25 | 13.95 | — | — | — |

### Market1501 CMC + mAP (Market1501-trained + foundation)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-S50 | 96.11 | 98.69 | 99.14 | 99.41 | 89.67 | 68.75 |
| FastReID SBS-R101-IBN | 95.81 | 98.75 | 99.20 | 99.55 | 90.25 | 71.28 |
| FastReID SBS-R50-IBN | 95.55 | 98.55 | 99.02 | 99.38 | 89.13 | 67.26 |
| FastReID AGW-R101-IBN | 95.52 | 98.52 | 99.29 | 99.50 | 89.61 | 69.36 |
| FastReID AGW-R50-IBN | 95.52 | 98.60 | 99.14 | 99.47 | 89.09 | 67.93 |
| FastReID MGN-R50-IBN | 95.37 | 98.49 | 99.14 | 99.55 | 87.25 | 56.47 |
| FastReID BoT-R101-IBN | 95.34 | 98.60 | 99.14 | 99.61 | 89.12 | 68.23 |
| FastReID AGW-R50 | 95.31 | 98.40 | 99.05 | 99.35 | 88.49 | 65.90 |
| FastReID BoT-R50-IBN | 95.28 | 98.69 | 99.02 | 99.35 | 88.23 | 65.44 |
| FastReID BoT-S50 | 95.28 | 98.49 | 99.17 | 99.52 | 88.64 | 66.83 |
| FastReID AGW-S50 | 95.19 | 98.57 | 99.05 | 99.44 | 88.59 | 66.67 |
| FastReID SBS-R50 | 95.16 | 98.52 | 98.96 | 99.29 | 88.46 | 65.46 |
| FastReID BoT-R50 | 93.85 | 98.13 | 98.99 | 99.41 | 86.30 | 60.61 |
| SigLIP ViT-B/16 | 20.58 | 36.02 | 43.56 | 51.37 | 6.51 | 0.35 |
| OpenCLIP ViT-B/32 | 13.15 | 27.08 | 34.86 | 43.32 | 4.06 | 0.25 |

### Market1501 CMC + mAP (MSMT17-trained)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (MSMT17) | 61.61 | 77.97 | 83.94 | — | 32.82 | 5.04 |
| FastReID SBS-R50-IBN (MSMT17) | 61.55 | 78.44 | 84.26 | — | 32.05 | 5.13 |
| FastReID SBS-R50 (MSMT17) | 59.74 | 75.83 | 82.16 | — | 30.48 | 4.59 |
| FastReID SBS-S50 (MSMT17) | 57.10 | 74.29 | 79.66 | — | 29.19 | 3.87 |
| FastReID BoT-R101-IBN (MSMT17) | 52.02 | 69.77 | 77.29 | — | 26.49 | 4.63 |
| FastReID BoT-R50-IBN (MSMT17) | 50.42 | 69.12 | 75.50 | — | 26.22 | 4.42 |
| FastReID BoT-S50 (MSMT17) | 49.64 | 68.65 | 75.45 | — | 24.79 | 3.38 |
| FastReID AGW-R101-IBN (MSMT17) | 48.87 | 67.84 | 75.18 | — | 24.11 | 4.19 |
| FastReID AGW-R50-IBN (MSMT17) | 47.62 | 65.86 | 72.86 | — | 23.93 | 3.87 |
| FastReID BoT-R50 (MSMT17) | 43.68 | 62.23 | 70.19 | — | 20.96 | 3.20 |
| FastReID AGW-S50 (MSMT17) | 43.56 | 63.66 | 71.26 | — | 19.82 | 2.28 |
| FastReID AGW-R50 (MSMT17) | 43.53 | 63.27 | 70.25 | — | 21.61 | 3.49 |

### MARS CMC + mAP (Market1501-trained + foundation)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-S50 | 90.54 | 95.65 | 97.01 | 97.34 | 87.03 | 70.41 |
| FastReID SBS-R101-IBN | 90.33 | 96.52 | 97.39 | 97.88 | 87.88 | 71.65 |
| FastReID SBS-R50-IBN | 89.18 | 96.20 | 97.23 | 97.99 | 86.64 | 70.10 |
| FastReID AGW-R101-IBN | 87.55 | 94.73 | 95.87 | 96.68 | 84.15 | 66.52 |
| FastReID AGW-R50-IBN | 87.55 | 95.05 | 96.20 | 97.23 | 84.47 | 67.57 |
| FastReID MGN-R50-IBN | 84.51 | 92.66 | 93.80 | 95.49 | 74.50 | 43.81 |
| FastReID BoT-R101-IBN | 86.79 | 94.62 | 95.76 | 96.85 | 83.83 | 65.82 |
| FastReID AGW-R50 | 85.71 | 92.23 | 93.21 | 94.40 | 80.67 | 60.50 |
| FastReID BoT-R50-IBN | 87.83 | 95.05 | 96.20 | 97.01 | 84.10 | 66.38 |
| FastReID BoT-S50 | 86.14 | 94.51 | 95.38 | 96.14 | 82.35 | 63.21 |
| FastReID AGW-S50 | 86.47 | 93.75 | 94.57 | 95.49 | 82.09 | 63.54 |
| FastReID SBS-R50 | 87.72 | 94.35 | 95.60 | 96.36 | 84.04 | 64.51 |
| FastReID BoT-R50 | 81.52 | 89.78 | 92.01 | 93.26 | 75.99 | 54.26 |
| SigLIP ViT-B/16 | 38.97 | 53.21 | 59.02 | 64.95 | 21.49 | 6.29 |
| OpenCLIP ViT-B/32 | 21.52 | 37.17 | 44.24 | 52.07 | 11.03 | 2.67 |

### MARS CMC + mAP (MSMT17-trained)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (MSMT17) | 28.91 | 50.98 | 59.29 | — | 24.04 | 14.88 |
| FastReID SBS-R50-IBN (MSMT17) | 28.80 | 49.29 | 57.39 | — | 24.04 | 14.76 |
| FastReID SBS-R50 (MSMT17) | 28.91 | 50.71 | 57.99 | — | 24.19 | 14.93 |
| FastReID SBS-S50 (MSMT17) | 27.50 | 49.46 | 58.21 | — | 23.66 | 14.84 |
| FastReID BoT-R101-IBN (MSMT17) | 27.66 | 49.51 | 57.45 | — | 23.42 | 14.25 |
| FastReID BoT-R50-IBN (MSMT17) | 27.77 | 49.29 | 56.96 | — | 23.48 | 14.29 |
| FastReID BoT-S50 (MSMT17) | 28.04 | 49.35 | 56.85 | — | 23.43 | 14.08 |
| FastReID AGW-R101-IBN (MSMT17) | 27.50 | 48.80 | 56.85 | — | 23.31 | 14.11 |
| FastReID AGW-R50-IBN (MSMT17) | 27.72 | 49.13 | 56.96 | — | 23.28 | 14.03 |
| FastReID BoT-R50 (MSMT17) | 28.37 | 49.24 | 57.07 | — | 23.29 | 13.90 |
| FastReID AGW-S50 (MSMT17) | 26.47 | 49.78 | 57.23 | — | 23.00 | 13.81 |
| FastReID AGW-R50 (MSMT17) | 28.42 | 48.75 | 56.52 | — | 23.25 | 13.95 |

### MSMT17 CMC + mAP (Market1501-trained + foundation)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-S50 | 19.70 | 30.70 | 36.15 | 41.73 | 6.80 | 0.25 |
| FastReID SBS-R101-IBN | 19.78 | 30.53 | 35.82 | 42.10 | 6.81 | 0.25 |
| FastReID SBS-R50-IBN | 20.79 | 31.28 | 36.99 | 42.87 | 7.10 | 0.28 |
| FastReID AGW-R101-IBN | 14.20 | 22.83 | 27.40 | 32.46 | 4.78 | 0.20 |
| FastReID AGW-R50-IBN | 14.52 | 23.06 | 27.39 | 32.75 | 5.14 | 0.26 |
| FastReID MGN-R50-IBN | 20.84 | 31.45 | 36.68 | 42.41 | 7.17 | 0.24 |
| FastReID BoT-R101-IBN | 13.29 | 21.51 | 26.29 | 31.87 | 4.47 | 0.17 |
| FastReID AGW-R50 | 10.84 | 17.58 | 21.37 | 25.95 | 3.61 | 0.17 |
| FastReID BoT-R50-IBN | 13.23 | 21.86 | 26.49 | 31.63 | 4.69 | 0.19 |
| FastReID BoT-S50 | 14.36 | 23.12 | 27.54 | 32.79 | 4.92 | 0.17 |
| FastReID AGW-S50 | 14.86 | 23.70 | 28.56 | 34.21 | 4.96 | 0.20 |
| FastReID SBS-R50 | 14.73 | 23.05 | 27.84 | 33.20 | 4.72 | 0.14 |
| FastReID BoT-R50 | 7.47 | 13.23 | 16.38 | 20.55 | 2.35 | 0.12 |
| SigLIP ViT-B/16 | 24.40 | 36.83 | 42.37 | 48.39 | 6.33 | 0.12 |
| OpenCLIP ViT-B/32 | 4.74 | 10.00 | 13.51 | 18.47 | 1.02 | 0.05 |

### Compute (CUDA-event protocol, Market1501 extract)

| Model | Params (M) | Weights (MiB) | Allocated (MiB) | Peak (MiB) | GPU ms/img | GPU img/s |
|---|---:|---:|---:|---:|---:|---:|
| FastReID BoT-R50 | 23.5 | 89.9 | 89.9 | 309.9 | 1.50 | 665.8 |
| FastReID AGW-R50 | 23.5 | 90.0 | 90.1 | 318.2 | 1.54 | 650.9 |
| FastReID BoT-R50-IBN | 23.5 | 89.9 | 90.0 | 309.9 | 1.58 | 632.4 |
| OpenCLIP ViT-B/32 | 87.5 | 333.6 | 333.6 | 435.7 | 1.71 | 584.1 |
| FastReID AGW-R50-IBN | 23.5 | 90.0 | 90.1 | 318.2 | 1.73 | 576.4 |
| FastReID BoT-S50 | 25.4 | 97.3 | 97.3 | 397.4 | 2.14 | 467.7 |
| FastReID AGW-S50 | 25.4 | 97.3 | 97.3 | 397.4 | 2.16 | 463.2 |
| FastReID AGW-R101-IBN | 42.6 | 162.9 | 163.2 | 391.4 | 2.39 | 417.6 |
| FastReID BoT-R101-IBN | 42.5 | 162.5 | 162.8 | 382.9 | 2.40 | 416.6 |
| FastReID SBS-R50 | 23.5 | 90.0 | 90.1 | 428.2 | 2.54 | 394.2 |
| FastReID SBS-R50-IBN | 23.5 | 90.0 | 90.1 | 428.2 | 2.92 | 342.8 |
| FastReID SBS-R101-IBN | 42.6 | 162.9 | 163.2 | 501.4 | 3.69 | 270.9 |
| FastReID MGN-R50-IBN | 68.8 | 263.0 | 268.1 | 434.1 | 4.67 | 214.2 |
| FastReID SBS-S50 | 25.4 | 97.3 | 97.4 | 547.4 | 5.69 | 175.7 |
| SigLIP ViT-B/16 | 92.9 | 354.3 | 354.3 | 622.3 | 6.84 | 146.2 |

Not in this bench (on purpose): DukeMTMC zoo, vehicle ReID,
FastReID ViT (no zoo `.pth`), newer non-FastReID models
(SOLIDER, CLIP-ReID, CLIMB-ReID).
