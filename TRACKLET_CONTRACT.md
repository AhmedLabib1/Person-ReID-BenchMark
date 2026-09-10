# SHAWAF ReID — Tracklet Contract

## 1. Purpose

This document defines the interface between the **Detection + Tracking Team** and the **Re-Identification Team**.

The Detection + Tracking team generates person tracklets. The ReID team converts each person tracklet into a single appearance embedding for later cross-camera matching.

---

## 2. Pipeline Boundary

```text
Video
↓
Detection
↓
Tracking
↓
Tracklet
↓
============================
       ReID TEAM STARTS HERE
============================
↓
Tracklet Preprocessing
↓
Frame Sampling
↓
ReID Encoder
↓
Frame Embeddings
↓
Tracklet Aggregation
↓
Tracklet Appearance Embedding
```

---

## 3. Definition of a Tracklet

A tracklet represents **one detected person tracked continuously inside one camera during one continuous time interval**.

Example:

```text
Camera: CAM_01
Person: Frame 120 → Frame 245
```

This is ONE tracklet.

If the same person later appears in another camera, that appearance will initially be another tracklet:

```text
CAM_01 → Tracklet T001
CAM_03 → Tracklet T087
```

The ReID system will later determine whether:

```text
T001 == T087
```

---

## 4. Input From Detection + Tracking Team

The Detection + Tracking team must provide one record per tracklet.

```json
{
  "tracklet_id": "T000001",
  "camera_id": "CAM_01",
  "start_frame": 120,
  "end_frame": 245,
  "start_time": 4.8,
  "end_time": 9.8,
  "detections": [
    {
      "frame_id": 120,
      "timestamp": 4.8,
      "bbox": [340, 120, 470, 510],
      "confidence": 0.94,
      "crop_path": "crops/T000001/frame_000120.jpg"
    },
    {
      "frame_id": 121,
      "timestamp": 4.84,
      "bbox": [343, 121, 472, 512],
      "confidence": 0.95,
      "crop_path": "crops/T000001/frame_000121.jpg"
    }
  ]
}
```

---

## 5. Required Fields

### `tracklet_id`

Type: `string`
Example: `T000001`

Must uniquely identify the tracklet across the whole system.

### `camera_id`

Type: `string`
Example: `CAM_01`

Identifies the camera that produced the tracklet. A single tracklet must never contain detections from multiple cameras.

### `start_frame`

Type: `integer`

The first frame belonging to the tracklet.

### `end_frame`

Type: `integer`

The last frame belonging to the tracklet.

### `start_time`

Type: `float`

Time in seconds from the beginning of the video or synchronized stream.

### `end_time`

Type: `float`

Time in seconds corresponding to the last observation of the person.

### `detections`

Type: `list`

Contains the observations belonging to this tracklet.

Each detection must contain:

```text
frame_id
timestamp
bbox
confidence
crop_path
```

---

## 6. Bounding Box Format

All bounding boxes must use:

```text
[x1, y1, x2, y2]
```

Where:

```text
x1 = left
y1 = top
x2 = right
y2 = bottom
```

Example:

```text
[340, 120, 470, 510]
```

The Detection + Tracking and ReID teams must not use different bbox formats.

---

## 7. Person Crop

The ReID team expects a cropped image of the person for every available detection.

```text
crops/
└── T000001/
    ├── frame_000120.jpg
    ├── frame_000121.jpg
    ├── frame_000122.jpg
    └── ...
```

The crop should contain the complete detected person whenever possible.

The ReID team will later decide which crops should actually be used. Not every crop will necessarily enter the ReID encoder.

---

## 8. Detection Confidence

Each detection should include the detector confidence.

Example:

```text
0.94
```

This value may later help the ReID preprocessing stage remove unreliable detections.

The ReID team must not assume that detection confidence alone represents image quality.

---

## 9. Responsibilities

### Detection + Tracking Team

Responsible for:

* Person detection
* Bounding boxes
* Detection confidence
* Frame association
* Tracking IDs
* Tracklet construction
* Camera ID
* Frame IDs
* Timestamps
* Person crops

NOT responsible for:

* ReID embeddings
* Cross-camera identity matching
* ReID similarity scores
* ReID ranking

### ReID Team

Responsible for:

* Tracklet validation
* Crop quality filtering
* Frame sampling
* ReID feature extraction
* Frame embeddings
* Tracklet aggregation
* Tracklet appearance embeddings
* Query/gallery matching
* Cross-camera retrieval
* Similarity calculation
* Ranking
* ReID evaluation
* Identity-link proposals

---

## 10. ReID Output

For every valid input tracklet, the ReID system produces:

```json
{
  "tracklet_id": "T000001",
  "camera_id": "CAM_01",
  "embedding_model": "osnet",
  "embedding_dimension": 512,
  "appearance_embedding": [
    0.013,
    -0.241,
    0.093
  ],
  "total_frames": 126,
  "selected_frames": 12,
  "quality_score": 0.87
}
```

---

## 11. `appearance_embedding`

This is the most important output of the ReID pipeline.

It represents the visual appearance of the complete tracklet.

```text
Tracklet
↓
Selected Frames
↓
ReID Model
↓
Frame Embeddings
↓
Aggregation
↓
ONE Tracklet Embedding
```

Example:

```text
Frame 1 → E1
Frame 2 → E2
Frame 3 → E3
...
Frame N → EN
```

Then:

```text
T = Aggregate(E1, E2, ..., EN)
```

Where:

```text
T = Tracklet Appearance Embedding
```

---

## 12. Baseline Aggregation

The initial SHAWAF baseline will use:

```text
Mean Pooling
↓
L2 Normalization
```

Conceptually:

```text
T = mean(E1, E2, ..., EN)
T = L2Normalize(T)
```

More advanced aggregation methods may be evaluated later.

---

## 13. Initial ReID Model

The first baseline model will be:

```text
OSNet
```

The goal of the first baseline is NOT to find the final best model.

The goal is to build a complete working ReID pipeline.

Later experiments may compare:

* OSNet
* CLIP-ReID
* SigLIP-based representations

under the same evaluation protocol.

---

## 14. Cross-Camera Rule

A ReID comparison should primarily match tracklets originating from different cameras.

Example:

```text
Query:
T001
Camera: CAM_01

Gallery:
T023 → CAM_02
T091 → CAM_03
T122 → CAM_04
```

Same-camera matches may be used for specific recovery scenarios, but they must be identifiable through metadata.

---

## 15. Identity Is Not Tracking ID

Tracking ID and person identity are different concepts.

Example:

```text
CAM_01
Track ID 17
→ Tracklet T001

CAM_04
Track ID 82
→ Tracklet T092
```

These tracking IDs are unrelated.

ReID may determine:

```text
T001 == T092
```

Therefore:

```text
Tracking ID
= local identity inside the tracker

ReID Identity
= cross-camera person association
```

---

## 16. Invalid Tracklets

A tracklet may be rejected by ReID preprocessing if:

* It contains zero valid crops
* Person crops cannot be loaded
* Bounding boxes are invalid
* Tracklet metadata is corrupted
* The tracklet is too short for the configured pipeline

The ReID service must report the reason instead of silently failing.

Example:

```json
{
  "tracklet_id": "T000034",
  "status": "rejected",
  "reason": "no_valid_person_crops"
}
```

---

## 17. Versioning

The contract should contain a version.

Initial version:

```text
tracklet_contract_v1
```

Future changes must be versioned instead of silently changing the schema.

Example:

```text
tracklet_contract_v2
```

This prevents changes from the Detection + Tracking team from unexpectedly breaking the ReID pipeline.

---

## 18. First Development Target

The first complete SHAWAF ReID baseline is:

```text
MARS Tracklets
↓
Tracklet Loader
↓
Frame Sampling
↓
OSNet
↓
Frame Embeddings
↓
Mean Pooling
↓
Tracklet Embedding
↓
Cosine Similarity
↓
Query / Gallery Ranking
↓
Rank-1 + mAP
```

**Not part of the first baseline:**

* Qdrant
* Trajectory reconstruction
* Frontend integration
* Advanced models
