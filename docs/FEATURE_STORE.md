# Feature Store Architecture

> **Pattern:** Uber Michelangelo Palette — 20,000+ features, 10 trillion computations daily. Double-write architecture eliminates training-serving skew.
>
> **Reference:** [Uber Engineering Blog — Michelangelo Palette](https://www.uber.com/blog/michelangelo-palette/)

---

## Problem: Training-Serving Skew

Uber found that **60% of production ML bugs** are caused by training-serving skew:
- Training pipeline computes features with Python 3.9 + Pandas 1.5
- Serving pipeline computes features with Python 3.11 + Pandas 2.0
- Subtle differences in `np.fft`, `librosa.stft`, or `scipy.signal` produce different feature vectors
- Model sees different distributions at inference time → degraded accuracy

---

## Solution: Double-Write Feature Store

```
Raw Accelerometer Data (16 kHz WAV)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Feature Pipeline (omni/eep/features.py)    │
│  ├─ Identical code path for train + serve   │
│  ├─ Versioned via Git commit SHA            │
│  └─ Deterministic (seeded random, no race)  │
└─────────────────────────────────────────────┘
    │
    ├──► Redis (Online Store) ──────► IEP2 Inference (real-time)
    │      TTL: 1 hour                    P95 latency: < 10 ms
    │      Key: sensor_id:timestamp
    │
    └──► TimescaleDB (Offline Store) ──► ML Training Pipeline
           Partitioned by time                Batch reads via SQL
           Retention: 90 days                 Point-in-time correctness
```

### Online Store (Redis)

**Purpose:** Serve pre-computed features for real-time inference.

**Schema:**
```
Key: feature:sensor:{sensor_id}:{window_start_ts}
Value: JSON {
  "sensor_id": "A1",
  "timestamp": "2026-05-01T18:00:00Z",
  "features_39": [0.1, 0.2, ...],
  "feature_version": "abc123",  # Git SHA of features.py
  "ttl": 3600
}
```

**Why Redis?**
- P95 latency < 1ms for in-memory reads
- Supports TTL for automatic expiration
- Pub/sub for real-time feature updates

### Offline Store (TimescaleDB)

**Purpose:** Store historical features for batch training, drift detection, and audit.

**Schema:**
```sql
CREATE TABLE feature_vectors (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT,
    feature_version TEXT,
    features_39 DOUBLE PRECISION[],
    label TEXT,  -- NULL until human confirms
    source TEXT  -- 'training', 'inference', 'synthetic'
);
SELECT create_hypertable('feature_vectors', 'time', chunk_time_interval => INTERVAL '1 day');
```

**Why TimescaleDB?**
- Native time-series partitioning (hypertables)
- SQL interface for complex analytical queries
- Compression reduces storage by 90%+ for historical data
- Time-based retention policies

---

## Training-Serving Consistency Guarantees

| Guarantee | Implementation | Verification |
|-----------|---------------|------------|
| **Same code path** | `omni/eep/features.py` imported by both train and serve | Unit tests assert identical output for identical input |
| **Same library versions** | `requirements.txt` pinned in Dockerfile | CI checks `pip freeze` against baseline |
| **Same random seeds** | `np.random.seed(42)` in feature extraction | Deterministic output verified in tests |
| **Versioned features** | Git SHA stored alongside model in MLflow | Can reproduce exact features for any model version |
| **Point-in-time correctness** | Offline store uses `AS OF` timestamp queries | Training data never sees future information |

---

## Feature Registry

```python
# omni/common/feature_registry.py
FEATURE_REGISTRY = {
    "vibration_rms": {
        "description": "Root-mean-square amplitude",
        "compute": lambda x: np.sqrt(np.mean(x**2)),
        "version": "1.0.0",
        "author": "team-platform",
    },
    "spectral_rolloff": {
        "description": "Frequency below which 85% of energy lies",
        "compute": lambda x, sr: librosa.feature.spectral_rolloff(y=x, sr=sr)[0,0],
        "version": "1.0.0",
        "author": "team-ml",
    },
    # ... 37 more features
}
```

**Benefits:**
- Feature documentation and ownership
- Discoverability — data scientists can search available features
- Reproducibility — exact computation logic stored in Git

---

## Production Roadmap

| Phase | Feature | Timeline |
|-------|---------|----------|
| **Current** | Double-write to Redis + TimescaleDB | ✅ Implemented |
| **Phase 2** | Feast integration (offline=BigQuery, online=Redis) | Post-capstone |
| **Phase 3** | Real-time feature computation (Flink/Kafka) | Post-capstone |
| **Phase 4** | Feature marketplace (cross-team sharing) | Post-capstone |

---

## Defense Talking Point

> **"How do you prevent training-serving skew?"**

"Uber found that 60% of production ML bugs are caused by training-serving skew. We implement a simplified double-write feature store pattern: features are computed by the exact same code path for both training and inference, versioned via Git SHA, and stored in both Redis for real-time serving and TimescaleDB for batch training. This guarantees that the model sees identical feature distributions in both contexts."
