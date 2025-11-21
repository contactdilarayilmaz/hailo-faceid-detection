# Hailo Face ID Detection - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [API Reference](#api-reference)
7. [Database Schema](#database-schema)
8. [Performance Optimization](#performance-optimization)
9. [Usage Examples](#usage-examples)
10. [Troubleshooting](#troubleshooting)

---

# Overview

**Hailo Face ID Detection** is a real-time face recognition system built for Raspberry Pi 5 with Hailo AI accelerator. It provides persistent person identification across video frames using deep learning embeddings and an optimized tracking system.

## Key Features

* **Real-time Face Recognition**: Processes video streams at high FPS using Hailo AI accelerator
* **Persistent ID Tracking**: Maintains person identities across frames and sessions using SQLite database
* **ANN-based Fast Search**: Optional Approximate Nearest Neighbor (ANN) indexing for efficient similarity search
* **Asynchronous Database Operations**: Non-blocking DB writes to prevent FPS drops
* **Multi-stage Search Strategy**: Optimized 2-stage search for accurate ID assignment
* **Prototype-based Matching**: Uses multiple prototypes per ID for robust recognition

---

# Architecture

## System Flow

```
┌─────────────────┐
│  GStreamer      │
│  Video Stream   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Person         │
│  Detection      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  FaceIDPipeline │─────▶│  ArcFace Model   │
│  (Face Crop)    │      │  (Embedding)     │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Tracker        │
│  (ID Assignment)│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│Gallery │ │SQLite DB │
│(Active)│ │(Persist) │
└────────┘ └──────────┘
```

## Component Overview

| Component | Purpose | Key Technology |
|-----------|---------|----------------|
| **FaceIDPipeline** | Extracts face crops and generates embeddings | ArcFace MobileFaceNet (HEF) |
| **Tracker** | Assigns and maintains person IDs | Cosine similarity, ANN (Annoy) |
| **Database** | Persists embeddings across sessions | SQLite with async queue |
| **ANN Index** | Fast similarity search | Annoy (angular metric) |

---

# Installation

## Prerequisites

* Raspberry Pi 5
* Hailo AI accelerator (Hailo-8)
* Python 3.8+
* Hailo Runtime libraries

## Required Packages

```bash
pip install numpy opencv-python hailo-platform annoy
```

## Environment Setup

```bash
# Set Hailo device
export HAILO_DEVICE_IDS="0001:01:00.0"
export HAILORT_VDEVICE_GROUP_ID="1"

# Optional: Configure tracker settings
export REID_USE_ANN="1"              # Enable ANN (0 or 1)
export REID_ANN_CANDIDATES="20"      # ANN candidate count
export REID_ANN_TREES="10"           # ANN tree count
export REID_ANN_METRIC="angular"     # ANN metric type
export REID_DB_RECALL_LIMIT="200"    # Max DB entries to load
export REID_ANN_REBUILD_INTERVAL="30" # ANN rebuild interval (frames)
```

---

# Configuration

## Tracker Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | `0.58` | Minimum cosine similarity for ID match (0.0-1.0) |
| `ema_beta` | `0.12` | Exponential moving average factor for centroid update |
| `max_inactive_frames` | `900` | Frames before inactive entry is pruned |
| `max_prototypes_per_id` | `6` | Maximum prototypes stored per ID |
| `db_path` | `None` | SQLite database path (None = no persistence) |
| `use_ann` | `0` | Enable ANN indexing (0 or 1) |
| `ann_metric` | `"angular"` | ANN distance metric (`"angular"` or `"euclidean"`) |
| `ann_trees` | `10` | Number of ANN trees (more = more accurate, slower build) |
| `ann_candidates` | `20` | Number of ANN candidates to retrieve |
| `db_recall_limit` | `0` | Maximum DB entries to load at startup (0 = unlimited) |

## Similarity Threshold Guidelines

* **0.50-0.55**: More lenient, may cause false matches
* **0.55-0.60**: Balanced (recommended)
* **0.60-0.70**: Strict, may miss valid matches
* **>0.70**: Very strict, only for high-quality embeddings

---

# Core Components

## 1. Tracker Class

The `Tracker` class is the core component responsible for person ID assignment and embedding management.

### Initialization

```python
from basic_pipelines.utils import Tracker

tracker = Tracker(
    similarity_threshold=0.58,
    ema_beta=0.12,
    max_inactive_frames=300,
    max_prototypes_per_id=6,
    db_path="/path/to/faceid.sqlite",
    use_ann=1,
    ann_metric="angular",
    ann_trees=10,
    ann_candidates=20,
    db_recall_limit=200,
)
```

### Main Methods

#### `assign(detections_info, frame_idx, total_detections)`

Assigns person IDs to detections based on embedding similarity.

**Parameters:**
* `detections_info`: List of detection dicts with `embedding`, `bbox`, `quality`
* `frame_idx`: Current frame number
* `total_detections`: Total number of detections in frame

**Returns:**
* List of assigned IDs (one per detection, `None` if no match)

**Example:**
```python
embeddings = [
    {"det_index": 0, "embedding": emb1, "bbox": bbox1, "quality": 0.9},
    {"det_index": 1, "embedding": emb2, "bbox": bbox2, "quality": 0.8},
]
person_ids = tracker.assign(embeddings, frame_num, len(embeddings))
# Returns: [1, 2] or [1, None] etc.
```

### Internal Architecture

#### Database Operations

**Asynchronous Queue System**

The tracker uses a background thread for non-blocking database writes:

```python
# Internal queue structure
_db_queue = Queue(maxsize=4096)  # Thread-safe queue
_db_thread = threading.Thread(target=_db_worker, daemon=True)
```

**Operations:**
* `UPSERT`: Insert or update entry in database
* `DELETE`: Remove entry from database

**Benefits:**
* Prevents FPS drops from synchronous DB writes
* Handles high-frequency updates efficiently
* Graceful degradation if queue is full

#### Embedding Operations

**Normalization**

All embeddings are L2-normalized for cosine similarity:

```python
def _normalize(emb):
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-6 else None
```

**Similarity Calculation**

Uses dot product on normalized embeddings (equivalent to cosine similarity):

```python
def _similarity(emb, entry):
    prototypes = entry.get("prototypes") or [entry["centroid"]]
    sims = [np.dot(emb, proto) for proto in prototypes]
    return max(sims)  # Best match across all prototypes
```

#### Entry Management

**Gallery Structure**

```python
entry = {
    "id": 1,                    # Unique person ID
    "centroid": np.array(...),   # EMA-averaged embedding
    "prototypes": [...],         # List of recent embeddings
    "count": 42,                 # Total detections
    "last_seen": 1000,           # Last frame number
    "last_seen_ts": 1234567890,  # Last seen timestamp
    "bbox": (x1, y1, x2, y2),    # Last bounding box
    "quality": 0.9,              # Best quality score
    "missed_frames": 0,          # Frames since last detection
    "hits": 42,                  # Total successful matches
    "created_at": 1234560000,    # Creation timestamp
}
```

**Two-Level Storage:**

1. **`gallery`**: Active entries (pruned periodically)
2. **`db_gallery`**: Persistent buffer (never pruned, loaded from DB)

**Update Mechanism (EMA)**

```python
# Exponential Moving Average for centroid
new_centroid = (1 - ema_beta) * old_centroid + ema_beta * new_embedding
```

**Prototype Management**

* Stores up to `max_prototypes_per_id` recent embeddings
* Uses best-match across all prototypes for similarity
* Improves recognition under varying conditions

**Pruning**

Entries are removed from `gallery` when:
```python
missed_frames > max_inactive_frames
```

Pruning occurs at the end of each frame to free memory.

#### ANN Index Operations

**Approximate Nearest Neighbor (ANN) Search**

When `use_ann=1`, the tracker uses Annoy for fast similarity search.

**Index Structure:**
* Metric: Angular (cosine distance)
* Trees: Configurable (default: 10)
* Dimension: Embedding size (typically 512)

**Rebuild Strategy:**

1. **Trigger Conditions:**
   * Index is `None` (first time)
   * `_ann_dirty` flag is set (gallery changed)
   * Rebuild interval elapsed (default: 30 frames)

2. **Asynchronous Rebuild:**
   ```python
   def _build_ann_index_async(self):
       # Non-blocking rebuild in background thread
       thread = threading.Thread(target=rebuild_worker, daemon=True)
   ```

3. **Thread Safety:**
   * Uses `_ann_rebuild_lock` for atomic updates
   * Old index remains available during rebuild
   * New index replaces old atomically

**Candidate Retrieval:**

```python
def _get_ann_candidates(vector, frame_idx):
    # Get top-k candidates from ANN index
    idxs = ann_index.get_nns_by_vector(vector, k=ann_candidates)
    candidates = [ann_id_map[idx] for idx in idxs]
    return candidates
```

#### Search Operations

**2-Stage Search Strategy**

**Stage 1: Fast ANN Search**
1. Get ANN candidates from `gallery + db_gallery`
2. Linear search in candidates for best match
3. Fallback to full `gallery` search if no match

**Stage 2: Database Revive**
1. Search in `db_gallery` for inactive entries
2. Uses expanded ANN candidates (3x normal) or linear search
3. Applies lower threshold (`similarity_threshold - 0.05`)

**Why Linear Search in Candidates?**

* ANN returns approximate candidates (not exact matches)
* Linear search on small candidate list (20-60 entries) is fast
* Ensures best match is found within candidates
* O(n) on small n is acceptable vs ANN overhead

**Search Functions:**

```python
def _search_in_entries(embedding, entries, used_ids, threshold):
    """Linear search for best matching entry."""
    # Returns: (best_entry, best_similarity)

def _revive_from_db(embedding, used_ids, frame_idx):
    """Search db_gallery for inactive entries."""
    # Returns: (revived_entry, similarity) or (None, similarity)
```

---

## 2. FaceIDPipeline Class

Extracts face crops from person detections and generates embeddings using ArcFace model.

### Initialization

```python
from basic_pipelines.detection_faceid_pipeline import FaceIDPipeline

pipeline = FaceIDPipeline()
```

### Main Method

#### `get_embeddings(frame, detections)`

Extracts face crops and generates embeddings for each detection.

**Parameters:**
* `frame`: BGR numpy array (OpenCV format)
* `detections`: List of detection dicts with `bbox` and `confidence`

**Returns:**
* List of dicts with `det_index`, `embedding`, `bbox`, `quality`

**Example:**
```python
detections = [
    {"bbox": (0.1, 0.2, 0.5, 0.8), "confidence": 0.95},
    {"bbox": (0.6, 0.3, 0.9, 0.7), "confidence": 0.88},
]
embeddings = pipeline.get_embeddings(frame, detections)
```

### Face Crop Heuristic

The pipeline uses a conservative face crop strategy:

```python
# Face region within person bbox
face_height = person_height * 0.55
face_width = person_width * 0.65
center_x = person_x + person_width * 0.5
center_y = person_y + person_height * 0.25  # Slightly higher
```

**Quality Filtering:**
* Minimum face area: 4000 pixels
* Minimum dimensions: 70x70 pixels
* Quality score based on area and confidence

### ArcFace Model

**Model Details:**
* Architecture: MobileFaceNet
* Input: 112x112 RGB image (UINT8)
* Output: 512-dimensional normalized embedding
* Format: HEF (Hailo Executable Format)

**Preprocessing:**
1. Resize to 112x112
2. Convert BGR → RGB
3. Add batch dimension: (1, 112, 112, 3)

---

## 3. Detection Application

The main application (`detection_faceid.py`) integrates GStreamer, face detection, and tracking.

### Application Flow

```python
def app_callback(pad, info, user_data):
    # 1. Extract detections from GStreamer buffer
    detections = hailo.get_roi_from_buffer(buffer)
    
    # 2. Convert buffer to numpy frame
    frame = _buffer_to_numpy(pad, buffer)
    
    # 3. Extract face embeddings
    embeddings = user_data.face_pipeline.get_embeddings(frame, detections)
    
    # 4. Assign person IDs
    person_ids = user_data.tracker.assign(embeddings, frame_num, len(detections))
    
    # 5. Display results
    print(f"Frame {frame_num} | IDs: {person_ids}")
```

### FPS Calculation

```python
# Calculate FPS every second
if elapsed >= 1.0:
    current_fps = fps_frame_count / elapsed
    fps_frame_count = 0
```

---

# API Reference

## Tracker Class

### Constructor

```python
Tracker(
    similarity_threshold=0.58,
    ema_beta=0.12,
    max_inactive_frames=900,
    max_prototypes_per_id=6,
    db_path=None,
    use_ann=0,
    ann_metric="angular",
    ann_trees=10,
    ann_candidates=20,
    db_recall_limit=0,
)
```

### Public Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `assign(detections_info, frame_idx, total_detections)` | Assign IDs to detections | `List[int\|None]` |

### Internal Methods (Reference)

| Method | Purpose |
|--------|---------|
| `_normalize(emb)` | L2-normalize embedding |
| `_similarity(emb, entry)` | Calculate cosine similarity |
| `_update_entry(entry, emb, ...)` | Update entry with new detection |
| `_create_entry(emb, ...)` | Create new entry for new person |
| `_prune()` | Remove inactive entries |
| `_build_ann_index()` | Build ANN index synchronously |
| `_build_ann_index_async()` | Build ANN index asynchronously |
| `_get_ann_candidates(vector, frame_idx)` | Get ANN candidates |
| `_search_in_entries(embedding, entries, ...)` | Linear search in entries |
| `_revive_from_db(embedding, ...)` | Search db_gallery |
| `_persist_entry(entry)` | Queue DB write operation |
| `_enqueue_db_op(op, payload)` | Add operation to DB queue |
| `_db_worker()` | Background DB worker thread |

---

# Database Schema

## SQLite Table: `reid_entries`

```sql
CREATE TABLE reid_entries (
    id INTEGER PRIMARY KEY,           -- Unique person ID
    centroid BLOB,                     -- Embedding vector (binary)
    last_seen INTEGER,                 -- Last frame number
    last_seen_ts INTEGER,              -- Last seen timestamp (Unix)
    missed_frames INTEGER,            -- Frames since last detection
    total_hits INTEGER,               -- Total successful matches
    quality REAL,                     -- Best quality score
    created_at INTEGER                -- Creation timestamp (Unix)
)
```

## Data Types

| Column | Type | Description |
|--------|------|-------------|
| `id` | `INTEGER` | Auto-incrementing person ID |
| `centroid` | `BLOB` | NumPy array as bytes (`np.float32`) |
| `last_seen` | `INTEGER` | Frame number (0-based) |
| `last_seen_ts` | `INTEGER` | Unix timestamp |
| `missed_frames` | `INTEGER` | Counter for pruning |
| `total_hits` | `INTEGER` | Match count |
| `quality` | `REAL` | Float (0.0-1.0) |
| `created_at` | `INTEGER` | Unix timestamp |

## Database Operations

### UPSERT (Insert or Update)

```sql
INSERT INTO reid_entries (id, centroid, last_seen, last_seen_ts, missed_frames, total_hits, quality, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    centroid=excluded.centroid,
    last_seen=excluded.last_seen,
    last_seen_ts=excluded.last_seen_ts,
    missed_frames=excluded.missed_frames,
    total_hits=excluded.total_hits,
    quality=excluded.quality
```

### DELETE

```sql
DELETE FROM reid_entries WHERE id=?
```

### Load State

```sql
SELECT id, centroid, last_seen, last_seen_ts, missed_frames, total_hits, quality, created_at
FROM reid_entries
WHERE centroid IS NOT NULL
ORDER BY last_seen_ts DESC
LIMIT ?
```

---

# Performance Optimization

## Asynchronous Operations

### Database Writes

**Problem:** Synchronous DB writes block main thread, causing FPS drops.

**Solution:** Background thread with queue:

```python
# Main thread (non-blocking)
tracker._enqueue_db_op("UPSERT", payload)

# Background thread (async)
def _db_worker():
    while True:
        op, payload = queue.get()
        conn.execute(sql, payload)
        conn.commit()
```

**Benefits:**
* Main thread never blocks on DB
* Handles high-frequency updates
* Graceful degradation if queue full

### ANN Rebuild

**Problem:** `ann.build()` is slow (100-500ms) and blocks main thread.

**Solution:** Asynchronous rebuild:

```python
def _build_ann_index_async(self):
    # Build in background thread
    thread = threading.Thread(target=rebuild_worker, daemon=True)
    thread.start()
    
    # Old index remains available during rebuild
    # New index replaces old atomically
```

**Benefits:**
* No frame drops during rebuild
* Rebuild interval configurable (default: 30 frames)
* Thread-safe index updates

## Search Optimization

### ANN Candidate Retrieval

**Strategy:** Get top-k candidates, then linear search:

```python
# Fast approximate search
candidates = ann_index.get_nns_by_vector(embedding, k=20)

# Accurate linear search on small set
best = max(candidates, key=lambda e: similarity(embedding, e))
```

**Why not pure ANN?**
* ANN is approximate (may miss best match)
* Linear search on 20 entries is fast (<1ms)
* Ensures best match is found

### Multi-stage Search

1. **Stage 1:** ANN candidates → Linear search → Fallback to full gallery
2. **Stage 2:** Expanded ANN search in db_gallery → Linear fallback

**Benefits:**
* Fast path for active entries (Stage 1)
* Comprehensive search for inactive entries (Stage 2)
* Balanced accuracy and speed

## Memory Management

### Gallery Pruning

```python
# Remove entries inactive for > max_inactive_frames
gallery = [e for e in gallery if e["missed_frames"] <= max_inactive_frames]
```

**When:** End of each frame (after ID assignment)

**Impact:** Prevents unbounded memory growth

### Prototype Limiting

```python
# Keep only recent prototypes
if len(prototypes) > max_prototypes_per_id:
    prototypes.pop(0)  # Remove oldest
```

**Impact:** Bounded memory per ID

---

# Usage Examples

## Basic Usage

```python
from basic_pipelines.detection_faceid import user_app_callback_class, app_callback
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# Initialize
user_data = user_app_callback_class()

# Run application
app = GStreamerDetectionApp(app_callback, user_data)
app.run()
```

## Custom Tracker Configuration

```python
from basic_pipelines.utils import Tracker

tracker = Tracker(
    similarity_threshold=0.60,      # Stricter matching
    max_inactive_frames=600,        # Keep entries longer
    db_path="./custom_faceid.db",
    use_ann=1,
    ann_candidates=30,              # More candidates
    db_recall_limit=500,            # Load more from DB
)
```

## Manual ID Assignment

```python
# Prepare detections
embeddings = [
    {
        "det_index": 0,
        "embedding": np.array([...]),  # 512-dim normalized
        "bbox": (x1, y1, x2, y2),
        "quality": 0.9,
    },
]

# Assign IDs
person_ids = tracker.assign(embeddings, frame_idx=100, total_detections=1)
# Returns: [1] if match found, [None] if new person
```

## Database Inspection

```python
import sqlite3
import numpy as np

conn = sqlite3.connect("data/faceid.sqlite")
cursor = conn.cursor()

# Get all entries
rows = cursor.execute("SELECT id, centroid, last_seen_ts FROM reid_entries").fetchall()

for row_id, centroid_blob, timestamp in rows:
    centroid = np.frombuffer(centroid_blob, dtype=np.float32)
    print(f"ID {row_id}: last seen at {timestamp}, embedding norm: {np.linalg.norm(centroid)}")
```

---

# Troubleshooting

## Common Issues

### Low FPS / Camera Freezing

**Symptoms:** Terminal logs continue, but camera display freezes.

**Causes:**
1. Synchronous DB writes (if async queue not working)
2. ANN rebuild blocking main thread (if async rebuild not enabled)
3. GStreamer buffer issues

**Solutions:**
* Verify `_db_worker` thread is running
* Check `_ann_rebuild_in_progress` flag
* Increase `REID_ANN_REBUILD_INTERVAL` (e.g., 100 frames)
* Check GStreamer pipeline configuration

### Wrong ID Assignment

**Symptoms:** Same person gets different IDs, or different people get same ID.

**Causes:**
1. `similarity_threshold` too low (false matches) or too high (missed matches)
2. Gallery not pruning inactive entries correctly
3. Prototype quality issues

**Solutions:**
* Adjust `similarity_threshold` (try 0.55-0.60)
* Verify `_prune()` is called at end of frame
* Check `max_inactive_frames` value
* Review embedding quality (normalization, face crop size)

### Database Queue Full

**Symptoms:** Warning message: `"DB queue full, dropping writes."`

**Causes:**
* DB writes faster than processing
* Queue size too small (default: 4096)

**Solutions:**
* Increase queue size in `_start_db_worker()`:
  ```python
  self._db_queue = Queue(maxsize=8192)  # Double size
  ```
* Check DB file I/O performance
* Reduce write frequency (e.g., only persist on quality updates)

### ANN Index Not Building

**Symptoms:** Slow search, `_ann_index` remains `None`.

**Causes:**
1. `annoy` package not installed
2. `use_ann=0` in configuration
3. No entries in gallery

**Solutions:**
* Install: `pip install annoy`
* Set `use_ann=1` or `REID_USE_ANN="1"`
* Verify gallery has entries before first search

### Embedding Extraction Fails

**Symptoms:** `get_embeddings()` returns empty list.

**Causes:**
1. Face crop too small (< 70x70 or < 4000 pixels)
2. Invalid bounding box coordinates
3. ArcFace model not loaded

**Solutions:**
* Check `min_face_crop` threshold (default: 4000)
* Verify detection bbox format (normalized 0-1 or pixel coordinates)
* Check HEF model path: `resources/models/hailo8/arcface_mobilefacenet.hef`
* Review debug messages: `FACE_DEBUG_LIMIT=50`

## Debug Mode

Enable detailed logging:

```bash
export FACE_DEBUG_LIMIT=50  # Show first 50 debug messages
export REID_USE_ANN="1"
export REID_ANN_CANDIDATES="20"
```

Debug messages include:
* `[face-id-debug] det[N] skipped: reason (details)`
* `[tracker] DB worker error: ...`
* `[tracker] Warning: DB queue full, dropping writes.`

## Performance Tuning

### For Higher Accuracy

```python
tracker = Tracker(
    similarity_threshold=0.60,      # Stricter
    ann_candidates=50,              # More candidates
    max_prototypes_per_id=10,      # More prototypes
)
```

### For Higher Speed

```python
tracker = Tracker(
    similarity_threshold=0.55,      # More lenient
    ann_candidates=10,              # Fewer candidates
    max_prototypes_per_id=3,       # Fewer prototypes
    db_recall_limit=100,           # Load less from DB
)
```

### For Lower Memory

```python
tracker = Tracker(
    max_inactive_frames=300,       # Prune sooner
    max_prototypes_per_id=3,      # Fewer prototypes
    db_recall_limit=50,           # Load less from DB
)
```

---

# Additional Resources

## Code Structure

```
hailo-rpi5-examples/
├── basic_pipelines/
│   ├── utils.py                    # Tracker class
│   ├── detection_faceid.py         # Main application
│   └── detection_faceid_pipeline.py # FaceIDPipeline class
├── data/
│   └── faceid.sqlite              # SQLite database
├── resources/
│   └── models/
│       └── hailo8/
│           └── arcface_mobilefacenet.hef
└── DOCUMENTATION.md               # This file
```

## Key Algorithms

### Cosine Similarity

```python
# Normalized embeddings
emb1_norm = emb1 / ||emb1||
emb2_norm = emb2 / ||emb2||

# Cosine similarity = dot product
similarity = np.dot(emb1_norm, emb2_norm)
# Range: [-1, 1], typically [0, 1] for face embeddings
```

### Exponential Moving Average (EMA)

```python
# Update centroid with new embedding
centroid_new = (1 - beta) * centroid_old + beta * embedding_new

# beta = 0.12 means 12% weight for new embedding
# Higher beta = faster adaptation, more noise
# Lower beta = slower adaptation, more stable
```

### ANN Angular Distance

```python
# Angular distance = 1 - cosine similarity
angular_dist = 1 - np.dot(emb1_norm, emb2_norm)

# Annoy uses angular metric for cosine similarity search
# More trees = more accurate, slower build
# More candidates = better recall, slower search
```

---

# License

This project is part of the Hailo Raspberry Pi 5 examples repository.

---

# Changelog

## Version 1.0

* Initial release
* Tracker with SQLite persistence
* ANN-based fast search
* Asynchronous DB operations
* 2-stage search strategy
* Prototype-based matching

---

**Documentation generated for Hailo Face ID Detection Project**

*Last updated: 2025*

