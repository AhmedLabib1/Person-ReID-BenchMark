# SHAWAF ReID Benchmark Contract

## 1. Purpose

This contract defines the fixed evaluation rules for every new ReID model added to the SHAWAF benchmark.

The goal is to compare different ReID encoders fairly under the same evaluation conditions.

Models may differ in:
- Architecture
- Training method
- Training data
- Pretraining
- Native input resolution
- Embedding dimension
- Model-specific preprocessing

But the SHAWAF evaluation pipeline must remain fixed unless a separate experiment is explicitly created.

---

## 2. Evaluation Datasets

Use the same dataset versions and official query/gallery splits for every model.

| Dataset | Retrieval Unit | Role |
|---|---|---|
| MSMT17_V2 | Image | In-domain only if the model was trained on MSMT17_V2; otherwise unseen/cross-domain |
| Market1501 | Image | Cross-domain/unseen unless used during training |
| MARS | Tracklet | Cross-domain tracklet evaluation unless used during training |

Current dataset counts:

### MSMT17_V2
- Query: 11,659 images
- Gallery: 82,161 images

### Market1501
- Query: 3,368 images
- Gallery: 15,913 images

### MARS
- Query: 1,980 tracklets
- Gallery: 9,330 tracklets
- Valid queries after filtering: 1,840
- Skipped queries: 140

---

## 3. Dataset Relation Must Be Recorded

Do not automatically call MSMT17_V2 "in-domain" for every model.

Example:

### FastReID checkpoint trained on MSMT17_V2
- MSMT17_V2: In-domain
- Market1501: Cross-domain
- MARS: Cross-domain

### Model trained on unrelated large-scale data
- MSMT17_V2: Unseen / Cross-domain
- Market1501: Unseen / Cross-domain
- MARS: Unseen / Cross-domain

Each model must record its training relation to every evaluation dataset.

---

## 4. Image-Based Evaluation Protocol

Used for:
- MSMT17_V2
- Market1501

Protocol name:

`single_image_l2`

Pipeline:

```text
Image
  ↓
Official model-specific preprocessing
  ↓
ReID Encoder
  ↓
Raw Embedding
  ↓
L2 Normalization
  ↓
Cosine Similarity
  ↓
Gallery Ranking
  ↓
Standard ReID Filtering
  ↓
Rank-1 / Rank-5 / Rank-10 / mAP / mINP / CMC
```

Rules:
- Use the model's official/native preprocessing.
- Keep the model's native embedding dimension.
- L2-normalize the final image embedding.
- Use cosine similarity for retrieval.
- Use the same SHAWAF metric implementation for all models.

---

## 5. MARS Tracklet Evaluation Protocol

Protocol name:

`uniform8_mean_raw_l2`

Pipeline:

```text
Tracklet
  ↓
Uniformly sample exactly 8 frames
  ↓
Run the encoder on each frame
  ↓
8 raw frame embeddings
  ↓
Mean pooling
  ↓
Final L2 normalization
  ↓
Cosine similarity
  ↓
Gallery ranking
  ↓
Rank-1 / Rank-5 / Rank-10 / mAP / mINP / CMC
```

Important:
- Do NOT L2-normalize each frame before mean pooling.
- Mean the raw frame embeddings first.
- Apply L2 normalization only after tracklet aggregation.

---

## 6. MARS Sampling Rules

Use deterministic uniform sampling.

Example:

40 frames → sample 8:

```text
[0, 6, 11, 17, 22, 28, 33, 39]
```

If the tracklet contains fewer than 8 frames, repeated indices are allowed.

Example:

2 frames → sample 8:

```text
[0, 0, 0, 0, 1, 1, 1, 1]
```

This keeps the input cost fixed at exactly 8 frames per tracklet.

---

## 7. Retrieval and Filtering Rules

Final embeddings must satisfy:

```text
L2 norm ≈ 1
```

Similarity:

```text
cosine similarity
```

Equivalent implementation:

```text
query_embedding · gallery_embedding
```

after L2 normalization.

Filtering rule:

```text
same PID + same camera → remove from gallery
```

If a query has no valid positive after filtering:

```text
skip that query
```

---

## 8. Accuracy Metrics

Every model must report:
- Rank-1
- Rank-5
- Rank-10
- mAP
- mINP
- CMC up to Rank-50

Use the same SHAWAF evaluator for all models.

Do not compare metrics produced by different repositories unless they are only shown as external reference values.

---

## 9. Model-Specific Preprocessing

Different models may require different preprocessing.

Allowed differences include:
- Input resolution
- Resize method
- Interpolation
- RGB/BGR ordering
- Pixel range
- Mean/std normalization

Use each model's official inference preprocessing.

Common evaluation begins after the encoder output.

Record for every model:
- Input height
- Input width
- Color order
- Pixel normalization
- Interpolation
- Embedding dimension

---

## 10. Embedding Cache Contract

Use the same cache layout:

```text
benchmarks/
└── embeddings/
    └── <model_id>/
        └── <dataset>/
            └── <protocol>/
                ├── query.npz
                └── gallery.npz
```

Each NPZ file should contain:
- schema_version
- model_id
- dataset_name
- split_name
- unit_type
- embeddings
- person_ids
- camera_ids
- sample_ids

Validation:
- Embeddings shape must be `[N, D]`
- Metadata lengths must match `N`
- No NaN or Inf
- PID must be integer
- Camera ID must be integer
- Final embeddings must be L2-normalized

---

## 11. Main Benchmark Restrictions

During the first comparison of a new model, do NOT add:
- Quality filtering
- Duplicate removal
- Best-frame selection
- Attention pooling
- Weighted pooling
- Re-ranking
- Query expansion
- Test-time adaptation
- Target-dataset fine-tuning

Reason:

The first benchmark should change only the encoder/model.

Tracklet optimizations must be tested later in a separate ablation study.

---

## 12. Efficiency Benchmark Contract

Efficiency results are directly comparable only when measured on the same physical machine and GPU.

Use:
- `model.eval()`
- `torch.inference_mode()`
- Same GPU
- Same measurement method

Default profiling protocol:
- Warm-up iterations: 10
- Measurement iterations: 50
- Throughput batch size: 16
- Tracklet frames: 8

Measure:
- Batch-1 latency
- Batch-16 latency
- Batch-16 throughput
- 8-frame tracklet latency
- Baseline VRAM
- Peak VRAM batch-1
- Peak VRAM batch-16
- Parameter count
- Checkpoint size
- Embedding dimension
- Native input resolution

---

## 13. Latency Definition

Batch-1 latency measures:

```text
Preprocessed Tensor
  ↓
Model Forward
  ↓
Embedding Output
```

Do not include:
- Disk loading
- PIL/OpenCV file reading
- CPU resize
- File I/O

---

## 14. Throughput Definition

Use batch size 16.

```text
throughput = 16 / batch_latency_seconds
```

Use the same 50 measurement iterations.

---

## 15. Tracklet Latency Definition

Tracklet latency measures:

```text
8 preprocessed images
  ↓
One encoder batch
  ↓
Raw frame embeddings
  ↓
Mean pooling
  ↓
Final L2 normalization
```

The timing should include:
- Encoder forward pass
- Mean pooling
- Final L2 normalization

---

## 16. Native Input Resolution Rule

Do not force every model to use the same input resolution.

Use each model's official/native inference resolution.

Always record the input size because it affects latency and VRAM.

---

## 17. Hardware Metadata

Record for each efficiency benchmark:
- GPU model
- GPU VRAM
- Python version
- PyTorch version
- CUDA version
- Operating system

If a model is profiled on a different GPU:

```text
Accuracy → still comparable
Latency / Throughput / VRAM → not directly comparable
```

Keep those efficiency results in a separate hardware group.

---

## 18. Model Metadata Contract

Every model must have metadata similar to:

```yaml
id: isr_swin

method: ISR
architecture: Swin

checkpoint:
  type: official
  path: checkpoints/...
  source: official_repository

training:
  paradigm: self_supervised
  source_datasets:
    - ...
  pretrained_on:
    - ...
  fine_tuned_on:
    - ...

input:
  height: ...
  width: ...

embedding_dim: ...

dataset_relation:
  msmt17: unseen
  market1501: unseen
  mars: unseen
```

Checkpoint type must be recorded as one of:
- official
- reproduced
- third_party

---

## 19. Two Fairness Levels

### Level A — SHAWAF Deployment Benchmark

Required for every model.

Fixed:
- Datasets
- Query/gallery splits
- Sampling
- Post-encoder normalization
- Cosine retrieval
- Filtering
- Metrics
- MARS aggregation
- Efficiency protocol

Training data may differ.

This answers:

> Which available ReID system performs best under the SHAWAF evaluation setup?

### Level B — Strict Training-Controlled Benchmark

Optional future experiment.

Control as much as possible:
- Same source dataset
- Same source split
- No target fine-tuning
- Similar training budget

This answers:

> Which architecture/training method is stronger under similar training conditions?

---

## 20. Standard Output Structure

Each model should produce:

```text
benchmarks/
├── embeddings/
├── results/
│   ├── accuracy_summary.csv
│   ├── efficiency.csv
│   ├── final_benchmark_summary.csv
│   └── cmc/
└── figures/
```

Per-model results:

```text
benchmarks/results/<model_id>/
├── msmt17/
│   └── single_image_l2/
│       └── metrics.json
├── market1501/
│   └── single_image_l2/
│       └── metrics.json
└── mars/
    └── uniform8_mean_raw_l2/
        └── metrics.json
```

---

## 21. Standard Workflow for Every New Model

1. Read the official paper and repository.
2. Audit the training data and pretraining data.
3. Obtain the official or reproduced checkpoint.
4. Load the model successfully.
5. Extract one raw embedding.
6. Verify preprocessing and embedding shape.
7. Integrate the SHAWAF dataset loaders.
8. Build MSMT17_V2 embedding cache.
9. Build Market1501 embedding cache.
10. Build MARS `uniform8_mean_raw_l2` cache.
11. Run the SHAWAF evaluator.
12. Generate Rank / mAP / mINP / CMC.
13. Run efficiency profiling.
14. Generate summaries and figures.
15. Compare against the frozen FastReID baseline.

---

## 22. Current Frozen FastReID Reference

### SBS-R50

```text
MSMT17_V2
Rank-1: 81.74%
mAP:    56.36%

Market1501
Rank-1: 59.74%
mAP:    30.48%

MARS
Rank-1: 28.91%
mAP:    24.19%
```

Efficiency:

```text
Batch-1 latency: 6.02 ms
Throughput:      244.15 images/s
Peak VRAM B16:   278.66 MB
Parameters:      23.54M
```

These results are frozen and should not be changed while benchmarking new models.

---

## 23. Core Rule

```text
Different model-specific preprocessing is allowed.
Different embedding dimensions are allowed.
Different architectures are allowed.
Different training paradigms are allowed and must be documented.

BUT:

Dataset splits
Sampling
Post-encoder normalization
Similarity
Filtering
Metrics
MARS aggregation
Efficiency measurement protocol

must remain fixed.
```

This is the SHAWAF ReID benchmark contract for future model comparisons.
