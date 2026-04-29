# IEP1 (DECOMMISSIONED)

**Status:** Retired 2026-04 — kept in `archive/` for reference only.
**Replaced by:** `omni/eep/features.py` (39-d DSP feature vector extracted natively in EEP).

## Why it was removed

IEP1 was a TensorFlow-Hub YAMNet microservice that produced a 1024-d
(mean-pooled to 208-d) airborne-audio embedding. That embedding was
architecturally incompatible with the 39-d structure-borne physics
feature space used by IEP2 and the Omni orchestrator, and it wasted
~4 GB of container memory plus a 120 s cold-start.

All downstream code now consumes `extract_features(...)` from
`omni/eep/features.py` directly. No HTTP hop, no TensorFlow in the
serving path, no YAMNet weights on disk.

## Do not reactivate without ADR

Any proposal to bring a learned-embedding head back must be written up
as a new ADR covering:
1. Feature-space dimensionality (must match IEP2's trained model or
   include a retrain path).
2. Cold-start/latency budget under the p95 SLO in `MASTERPLAN.md`.
3. Memory footprint against the 512 MB / 2 GB reservations already
   allocated to IEP2 / IEP4.
