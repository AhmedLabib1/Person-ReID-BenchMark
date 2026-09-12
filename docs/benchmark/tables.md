### Accuracy

| Model | Trained on | Market1501 Rank-1 | Market1501 mAP | MARS Rank-1 | MARS mAP |
|---|---|---:|---:|---:|---:|
| FastReID AGW-R50 | Market1501 | 95.31 | 88.49 | 85.71 | 80.67 |
| FastReID BoT-R50 | Market1501 | 93.85 | 86.30 | 81.52 | 75.99 |
| OpenCLIP ViT-B/32 | LAION-2B | 13.15 | 4.06 | 21.52 | 11.03 |
| FastReID SBS-R50 | Market1501 | 95.16 | 88.46 | 87.72 | 84.04 |
| SigLIP ViT-B/16 | WebLI | 20.58 | 6.51 | 38.97 | 21.49 |

### Compute

| Model | Params (M) | Weights (MiB) | Allocated (MiB) | Peak (MiB) | GPU ms/img | GPU img/s |
|---|---:|---:|---:|---:|---:|---:|
| FastReID AGW-R50 | 23.5 | 90.0 | 90.1 | 318.2 | 1.54 | 650.9 |
| FastReID BoT-R50 | 23.5 | 89.9 | 89.9 | 309.9 | 1.50 | 665.8 |
| OpenCLIP ViT-B/32 | 87.5 | 333.6 | 333.6 | 435.7 | 1.71 | 584.1 |
| FastReID SBS-R50 | 23.5 | 90.0 | 90.1 | 428.2 | 2.54 | 394.2 |
| SigLIP ViT-B/16 | 92.9 | 354.3 | 354.3 | 622.3 | 6.84 | 146.2 |
