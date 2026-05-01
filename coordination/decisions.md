# Strategic Decisions Log

> **Purpose:** Capture decisions that change architecture, scope, or deployment strategy.
> **Format:** `## YYYY-MM-DD — Title (decided by: agent[, agent...])` then 3-line max body: **Context**, **Decision**, **Why-not-alternatives**.
> **Why this exists:** Future sessions (and graders, per rubric §5/§6) need to see what we considered and why we chose. No hindsight rewrites.

---

## 2026-05-02 — Cloud deployment target = Hetzner CX22 (decided by: claude-opus, user)

**Context:** Existing Render Blueprint deploys only the 4 FastAPI services. Prometheus/Grafana/MLflow/MQTT/Postgres stay local — fails rubric §11 live-observability requirement. User locked out of original Render account that owns the existing `*.onrender.com` URLs. Free-tier cold starts (30–60 s) violate rubric §8.1 ("if demo fails, grading stops").

**Decision:** Deploy full `docker-compose.yml` on Hetzner CX22 (4 GB / 2 vCPU / 40 GB, ~€4.51/mo) behind Caddy reverse proxy with Let's Encrypt TLS. Public URLs for EEP, Grafana, Prometheus, MLflow.

**Why not alternatives:**
- Fresh Render free → still no live obs, still cold-start risk
- Render paid (4 × $7) → $28/mo, still no Prom/Grafana/MLflow
- Cloud Run / Fly.io → don't naturally host stateful obs stack
- Existing Render account recovery → blocked on third party

## 2026-05-02 — On-device ML target = tiny feature-space autoencoder, NOT lift-and-shift cloud autoencoder (decided by: claude-opus, user)

**Context:** Cloud autoencoder is (1, 513, 157) STFT input at 16 kHz; ESP32-S3 captures at 3200 Hz from ADXL345 (max ODR). Direct port is infeasible (memory + sample-rate mismatch).

**Decision:** Build a tiny 39 → 8 → 39 dense autoencoder trained on 39-d DSP features (target ~12 KB int8 TFLite). C++ DSP feature extractor on-device using ESP-DSP. Publish features (~200 B JSON) instead of raw PCM (~8.5 KB).

**Why not alternatives:**
- Lift-and-shift cloud autoencoder → won't fit ESP32-S3 RAM, wrong sample rate
- On-device XGBoost classifier → tflite-converters for sklearn trees are flaky and large
- No on-device ML, just publish PCM → leaves the rubric §4.2 / "novel edge" story on the table

## 2026-05-02 — Sample-rate parity = retrain cloud at 3200 Hz, not upsample at gateway (decided by: claude-opus, user)

**Context:** ESP32 captures at 3200 Hz (Nyquist 1600 Hz). Cloud models trained on 16 kHz audio (Nyquist 8000 Hz). Feature distributions differ; deployed accuracy is currently unverified.

**Decision:** Resample training corpus 16 kHz → 3200 Hz with anti-alias + decimation. Re-extract 39-d features at 3200 Hz. Retrain XGB+RF+IF at 3200 Hz. Recalibrate threshold + golden dataset. Document accuracy delta in `MODEL_REPORT.md`.

**Why not alternatives:**
- Upsample 3200 → 16 kHz at gateway → no new info, masks the mismatch instead of fixing it; cloud and edge inference would still differ
- Switch sensor (e.g., I2S mic) → 3-day window, hardware unavailable; physics argues against airborne mic
- Accept the mismatch, document only → unverifiable accuracy claim, fails rubric §5 ("evidence required")

<!-- Append new decisions below. -->
