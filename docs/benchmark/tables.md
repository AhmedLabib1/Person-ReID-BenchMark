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

### Cross-domain (DukeMTMC weights, no fine-tune)

| Model | Trained on | Market Rank-1 | Market mAP | Market mINP | MARS Rank-1 | MARS mAP | MARS mINP | MSMT Rank-1 | MSMT mAP | MSMT mINP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (DukeMTMC) | DukeMTMC | 62.14 | 31.48 | 3.38 | 51.63 | 34.36 | 11.63 | 22.54 | 7.22 | 0.18 |
| FastReID MGN-R50-IBN (DukeMTMC) | DukeMTMC | 59.65 | 26.31 | 1.76 | 45.82 | 26.44 | 7.91 | 23.64 | 6.95 | 0.10 |
| FastReID SBS-S50 (DukeMTMC) | DukeMTMC | 59.62 | 28.94 | 2.94 | 52.88 | 35.28 | 12.06 | 21.23 | 6.63 | 0.16 |
| FastReID SBS-R50-IBN (DukeMTMC) | DukeMTMC | 58.76 | 28.71 | 3.24 | 53.80 | 36.25 | 13.00 | 21.78 | 6.98 | 0.16 |
| FastReID SBS-R50 (DukeMTMC) | DukeMTMC | 56.12 | 26.47 | 2.51 | 47.83 | 30.67 | 10.09 | 19.09 | 5.90 | 0.12 |
| FastReID BoT-R101-IBN (DukeMTMC) | DukeMTMC | 51.57 | 24.24 | 2.89 | 42.39 | 26.79 | 8.54 | 13.71 | 4.22 | 0.11 |
| FastReID AGW-S50 (DukeMTMC) | DukeMTMC | 49.38 | 22.25 | 2.24 | 45.54 | 28.54 | 9.42 | 15.93 | 4.63 | 0.13 |
| FastReID AGW-R101-IBN (DukeMTMC) | DukeMTMC | 49.11 | 24.31 | 3.40 | 42.88 | 27.95 | 9.07 | 12.93 | 4.03 | 0.13 |
| FastReID BoT-S50 (DukeMTMC) | DukeMTMC | 48.96 | 21.58 | 2.17 | 40.43 | 24.86 | 8.03 | 16.49 | 4.95 | 0.14 |
| FastReID AGW-R50-IBN (DukeMTMC) | DukeMTMC | 48.81 | 23.52 | 2.95 | 45.16 | 28.92 | 9.58 | 13.48 | 4.10 | 0.13 |
| FastReID BoT-R50-IBN (DukeMTMC) | DukeMTMC | 46.62 | 21.47 | 2.58 | 41.74 | 25.73 | 8.14 | 13.35 | 4.15 | 0.12 |
| FastReID AGW-R50 (DukeMTMC) | DukeMTMC | 45.69 | 21.91 | 2.89 | 40.76 | 26.17 | 8.62 | 11.97 | 3.74 | 0.13 |
| FastReID BoT-R50 (DukeMTMC) | DukeMTMC | 44.48 | 19.29 | 1.96 | 35.76 | 21.79 | 6.83 | 10.71 | 3.17 | 0.12 |

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

### Market1501 CMC + mAP (DukeMTMC-trained)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (DukeMTMC) | 62.14 | 77.67 | 82.78 | 87.20 | 31.48 | 3.38 |
| FastReID MGN-R50-IBN (DukeMTMC) | 59.65 | 76.25 | 82.24 | 87.56 | 26.31 | 1.76 |
| FastReID SBS-S50 (DukeMTMC) | 59.62 | 74.44 | 79.39 | 85.42 | 28.94 | 2.94 |
| FastReID SBS-R50-IBN (DukeMTMC) | 58.76 | 74.61 | 80.58 | 85.18 | 28.71 | 3.24 |
| FastReID SBS-R50 (DukeMTMC) | 56.12 | 73.16 | 79.10 | 83.61 | 26.47 | 2.51 |
| FastReID BoT-R101-IBN (DukeMTMC) | 51.57 | 67.61 | 74.85 | 80.02 | 24.24 | 2.89 |
| FastReID AGW-S50 (DukeMTMC) | 49.38 | 66.69 | 72.57 | 78.74 | 22.25 | 2.24 |
| FastReID AGW-R101-IBN (DukeMTMC) | 49.11 | 66.60 | 73.49 | 79.75 | 24.31 | 3.40 |
| FastReID BoT-S50 (DukeMTMC) | 48.96 | 65.41 | 71.88 | 78.56 | 21.58 | 2.17 |
| FastReID AGW-R50-IBN (DukeMTMC) | 48.81 | 66.36 | 74.11 | 80.14 | 23.52 | 2.95 |
| FastReID BoT-R50-IBN (DukeMTMC) | 46.62 | 63.87 | 70.81 | 77.79 | 21.47 | 2.58 |
| FastReID AGW-R50 (DukeMTMC) | 45.69 | 63.54 | 71.02 | 77.79 | 21.91 | 2.89 |
| FastReID BoT-R50 (DukeMTMC) | 44.48 | 61.16 | 68.17 | 75.12 | 19.29 | 1.96 |

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

### MARS CMC + mAP (DukeMTMC-trained)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (DukeMTMC) | 51.63 | 65.98 | 70.38 | 74.84 | 34.36 | 11.63 |
| FastReID MGN-R50-IBN (DukeMTMC) | 45.82 | 59.78 | 65.05 | 70.27 | 26.44 | 7.91 |
| FastReID SBS-S50 (DukeMTMC) | 52.88 | 66.85 | 72.23 | 76.90 | 35.28 | 12.06 |
| FastReID SBS-R50-IBN (DukeMTMC) | 53.80 | 68.15 | 74.13 | 79.08 | 36.25 | 13.00 |
| FastReID SBS-R50 (DukeMTMC) | 47.83 | 61.63 | 67.01 | 72.34 | 30.67 | 10.09 |
| FastReID BoT-R101-IBN (DukeMTMC) | 42.39 | 56.96 | 62.93 | 68.53 | 26.79 | 8.54 |
| FastReID AGW-S50 (DukeMTMC) | 45.54 | 59.13 | 65.22 | 70.60 | 28.54 | 9.42 |
| FastReID AGW-R101-IBN (DukeMTMC) | 42.88 | 57.17 | 62.99 | 69.29 | 27.95 | 9.07 |
| FastReID BoT-S50 (DukeMTMC) | 40.43 | 54.84 | 60.49 | 66.25 | 24.86 | 8.03 |
| FastReID AGW-R50-IBN (DukeMTMC) | 45.16 | 59.89 | 64.95 | 70.60 | 28.92 | 9.58 |
| FastReID BoT-R50-IBN (DukeMTMC) | 41.74 | 56.03 | 61.63 | 67.01 | 25.73 | 8.14 |
| FastReID AGW-R50 (DukeMTMC) | 40.76 | 55.27 | 62.99 | 69.57 | 26.17 | 8.62 |
| FastReID BoT-R50 (DukeMTMC) | 35.76 | 50.11 | 55.65 | 61.58 | 21.79 | 6.83 |

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

### MSMT17 CMC + mAP (DukeMTMC-trained)

| Model | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|---|---:|---:|---:|---:|---:|---:|
| FastReID SBS-R101-IBN (DukeMTMC) | 22.54 | 33.98 | 39.73 | 45.63 | 7.22 | 0.18 |
| FastReID MGN-R50-IBN (DukeMTMC) | 23.64 | 35.07 | 40.40 | 46.19 | 6.95 | 0.10 |
| FastReID SBS-S50 (DukeMTMC) | 21.23 | 31.97 | 37.73 | 43.38 | 6.63 | 0.16 |
| FastReID SBS-R50-IBN (DukeMTMC) | 21.78 | 33.22 | 39.05 | 45.08 | 6.98 | 0.16 |
| FastReID SBS-R50 (DukeMTMC) | 19.09 | 29.36 | 34.53 | 40.14 | 5.90 | 0.12 |
| FastReID BoT-R101-IBN (DukeMTMC) | 13.71 | 23.85 | 28.83 | 35.02 | 4.22 | 0.11 |
| FastReID AGW-S50 (DukeMTMC) | 15.93 | 25.80 | 31.13 | 37.23 | 4.63 | 0.13 |
| FastReID AGW-R101-IBN (DukeMTMC) | 12.93 | 21.68 | 26.60 | 32.53 | 4.03 | 0.13 |
| FastReID BoT-S50 (DukeMTMC) | 16.49 | 26.69 | 32.18 | 38.09 | 4.95 | 0.14 |
| FastReID AGW-R50-IBN (DukeMTMC) | 13.48 | 22.47 | 27.84 | 33.61 | 4.10 | 0.13 |
| FastReID BoT-R50-IBN (DukeMTMC) | 13.35 | 22.72 | 28.07 | 33.91 | 4.15 | 0.12 |
| FastReID AGW-R50 (DukeMTMC) | 11.97 | 21.05 | 25.95 | 31.63 | 3.74 | 0.13 |
| FastReID BoT-R50 (DukeMTMC) | 10.71 | 19.04 | 23.49 | 28.66 | 3.17 | 0.12 |

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

Not in this bench (on purpose): vehicle ReID, FastReID ViT
(no zoo `.pth`), newer non-FastReID models (SOLIDER, CLIP-ReID,
CLIMB-ReID). DukeMTMC images are not evaluated in-domain
(dataset withdrawn); Duke-trained zoo weights are imported as
cross-domain transfer only.
