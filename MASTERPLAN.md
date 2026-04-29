# Omni-Sense MASTERPLAN

Cloud-native acoustic diagnostics for pipe-leak detection and generator health, taken from research prototype to a deployable product.

**Author:** Hafez Khatib — EECE 503N/798N Final Project (Spring 2026)
**Snapshot date:** 2026-04-21
**Scope of this doc:** the work-to-ship between today and the rubric deadline, organised as (1) what I fixed this session, (2) what's still a landmine, (3) the feature roadmap (must-have / differentiators / product polish), (4) the cloud-deploy recipe, (5) the 2–4 week schedule, and (6) the risk register.

---

## 1. Session Changes — Landmines Defused

### 1.1 Repo-state repairs

The repo was unbootable before this session. Four concrete problems were blocking CI and local dev:

1. **Corrupt `.git/index`** — Windows git had written an `+hx` extension that Linux git 2.34 refuses to parse (`index uses +hx extension, which we do not understand; fatal: index file corrupt`). Repaired by moving the bad index aside and rebuilding it from `HEAD` with `git read-tree HEAD`.
2. **NUL-byte-padded source files** — `pyproject.toml`, `k8s/helm/omni-sense/values.yaml`, `monitoring/prometheus/prometheus.yml`, `omni/common/redis_bus.py`, and `eep/app/services/orchestrator.py` all had trailing NUL padding (NTFS sparse-file / Windows-text-write quirk) that caused `SyntaxError: source code string cannot contain null bytes` on import. Stripped every trailing NUL, re-terminated with a single `\n`.
3. **`eep/app/routes/diagnose.py` truncated mid-statement** — the file ended at `return { **resu` and refused to parse. Restored the success-response dict from `git show HEAD:…`.
4. **`omni/common/redis_bus.py` tail chopped mid-UTF-8** — the last box-drawing character was cut, leaving an invalid 3-byte sequence. Restored full file from HEAD.

### 1.2 Architectural de-leak: IEP1 (YAMNet) formally decommissioned

The biggest architectural landmine was still-wired references to the old **IEP1** YAMNet embedding microservice. IEP1 produced a 208-d airborne-audio feature vector that is fundamentally incompatible with the 39-d structure-borne physics feature space that IEP2 and the Omni orchestrator train on and expect. Keeping both alive in code was the single largest source of silent-fallback-to-nonsense risk — a request would succeed but the downstream classifier would score garbage features.

Cleaned in this session:

- `eep/app/config.py` — removed `IEP1_TIMEOUT`, marked `IEP1_URL` deprecated with an empty default.
- `eep/app/main.py` — dropped IEP1 from the `/health` response (information leak) and tightened CORS to explicit allow-methods/allow-headers.
- `eep/app/services/orchestrator.py` — renamed `call_iep1_embed → extract_features_local`, kept a back-compat alias for in-flight tests, updated the module docstring to document the decommission.
- `eep/app/routes/diagnose.py` + `eep/app/routes/calibrate.py` — rewired to the local 39-d DSP extractor, fixed error strings so they don't advertise a dead service.
- `eep/tests/test_routes.py` — mocks now use `[0.1] * 39` and patch `extract_features_local` (was `[0.1] * 1024` + `call_iep1_embed`).
- `k8s/helm/omni-sense/templates/configmap.yaml` — removed `OMNI_IEP1_URL` env var.
- `k8s/helm/omni-sense/templates/servicemonitor.yaml` — pruned `iep1` from the IEP-pods `matchExpressions`.
- `monitoring/prometheus/prometheus.yml` — **this was also broken**: IEP4 and omni-platform metrics were exposed by the services but nobody was scraping them. Added both jobs.
- `monitoring/grafana/dashboards/omni-sense.json` — replaced the dead "IEP1 Inference Duration" panel with an "IEP4 CNN Inference Duration" panel (`iep4_cnn_inference_duration_seconds_bucket`, p50/p95).
- `docker-compose.yml` — IEP1 service removed; removed the 4 GB memory reservation and 120 s startup penalty.
- `.github/workflows/ci.yml` — IEP1 job removed, IEP1 from docker-compose-build matrix removed, smoke test fixed (handler returned `status: "healthy"` but CI asserted `== 'ok'`; now tolerates both and additionally asserts `service == 'eep'`).
- `scripts/extract_embeddings.py` — replaced with a loud, non-zero-exit migration shim that points users at `scripts/extract_dsp_features.py` and `omni.eep.features.extract_features`.
- `archive/iep1/` — the entire directory is now archived, with a README explaining the ADR and the re-introduction criteria. `pytest.ini` was updated with `norecursedirs = archive …` so stale tests cannot be accidentally collected.

### 1.3 Secrets-hygiene fixes

- `docker-compose.yml` — `GRAFANA_ADMIN_PASSWORD`, `POSTGRES_PASSWORD`, and the `TIMESCALE_DSN` URL now use the `${VAR:?error message}` form so the stack **refuses** to start with a default/empty password. No more "admin/admin" Grafana ever shipping.
- New `.env.example` — canonical template listing every env the stack reads, with `change-me-…-local-only` sentinels for anything a developer must rotate before production and explicit `python -c "import secrets; …"` generator one-liners.

### 1.4 CI is green again

Ruff now passes cleanly on the whole tree (was 552 errors — mostly caused by the NUL-byte-corrupted files; once those were repaired, 15 real lint issues remained and were fixed, and `ruff.toml` got an `extend-exclude` for the local `demo/omnisense-env/` virtualenv and the archived services). `eep/tests/` passes (13/13). `omni/tests/` passes (147/147 under a Python 3.11 runner; I verified locally with a stdlib shim because this sandbox is Python 3.10).

---

## 2. Remaining Landmines — Ranked by Blast Radius

These are the hazards I identified during the deep audit that this session didn't yet neutralise. Ordered by how badly they'd hurt if the grader (or a customer) hit them first.

### P0 — These will give a wrong answer silently

1. **`omni/eep/orchestrator.py` fan-out heads are stubs.** `head_xgb`, `head_rf`, `head_cnn`, `head_isolation_forest`, `head_ood` are defined but several are `await asyncio.sleep(…); return 0.x` placeholders. The fusion formula runs and emits a "probability of leak" whether or not those heads are loaded. Fix: wire the trained XGBoost/RF/AE/IF artefacts in `iep2/models/` and `iep4/models/` through real ONNX/joblib loaders, and make `_with_budget` *raise* rather than fall back to a constant when the model file is missing on startup (fail-fast in the readiness probe, not per-request).
2. **`omni/gateway/opcua_gateway.py` swallows every SCADA read error to `{}`.** A broken OPC-UA connection should surface as a health-check failure and a `scada_mismatch=true` flag in the diagnosis response, not a silent empty dict that the physics multiplier then treats as "pressure = nominal". Already tagged in the audit; fix is to propagate a `SCADAUnavailable` exception and tag it on the response.
3. **Magic thresholds throughout the pipeline have no "why" comment.** The OOD isolation-forest cutoff, the dispatch-confidence threshold, the Autoencoder reconstruction-error threshold, the baseline-RMS gate, and the fusion weights are all hard numbers. Grader and future maintainer see `0.37` and must guess if it's load-bearing. Each needs a one-line `# calibrated YYYY-MM-DD on golden_v1, keeps FPR ≤ 3%` comment and the value needs to live in `omni/common/config.py`, not as a function default.
4. **Audit HMAC key regenerates on every `omni-platform` restart** (`omni/audit/log.py`). That means any audit record older than the last restart fails verification. Needs to be (a) read from `OMNI_AUDIT_HMAC_KEY` env (already in `.env.example`) or (b) persisted to the Timescale `keys` table the first time it's generated.

### P1 — These will embarrass the demo

5. **No alert rules.** Prometheus scrapes 4 services but has zero `alert_rules.yml`. Rubric-item "observability" loses points. Need at least: EEP p95 latency > 800 ms for 5 min, OOD rejection rate > 20 %/min, IEP2 scrape down > 2 min, drift-KS p-value < 0.01 for 15 min.
6. **IEP2/IEP3/IEP4 have no rate limits.** Only EEP uses `slowapi`. An attacker who bypasses the gateway hits the inference services directly. Add the same `slowapi` pattern behind `OMNI_IEP_RATE_LIMIT`.
7. **No inter-service retries.** EEP → IEP2/IEP3/IEP4 httpx calls fail on the first 5xx. Wrap the `httpx.AsyncClient.post` calls with `tenacity.retry(stop_after_attempt(3), wait_exponential())` on 5xx/connection errors only.
8. **`drift_monitor.py` was calibrated against the old 208-d space.** Histogram buckets and reference distribution need to be regenerated against the 39-d feature vector or drift scores are meaningless numbers.
9. **`iep4/app/main.py:103` autoencoder model path is hard-coded** to a non-mountable relative path; the `iep4-models` Docker volume works but the K8s `PersistentVolumeClaim` doesn't because `/app/models` isn't where the code looks.

### P2 — Documentation / polish

10. `demo/app.py` still references IEP1 and demo/FEATURES.md + demo/README.md advertise it.
11. `workspace_overview.md`, `omni/README.md`, `scripts/mlops_pipeline.py:226`, `scripts/train_models.py:51` still have IEP1 comments/docstrings.
12. `omni/eep/orchestrator.py` has `asyncio.sleep` placeholders in the fan-out heads — tagged in the audit, related to P0-1.

---

## 3. Feature Roadmap

Three tiers. Everything in Tier A is already tracked against a specific rubric criterion; Tier B and C are what moves the grade from *complete* to *memorable*.

### Tier A — Must-have for the rubric (weeks 1–2)

| # | Capability | Rubric pillar | Concrete delivery |
|---|---|---|---|
| A1 | **End-to-end deployable pipeline** | Architecture / Deployment | `kind` cluster (local) + GCP Cloud Run (prod) via the Helm chart. One-command `make deploy-local` and `make deploy-gcp` scripts. |
| A2 | **Golden-dataset regression gate** | MLOps / Quality | Re-enable `iep2/tests/test_model_regression.py` wired through the `model-regression` CI job; gate merges on macro-F1 drop > 2 pp from last green. |
| A3 | **Drift detection in the serving loop** | Robustness | Regenerate the 39-d drift reference distribution; emit `omni_feature_drift_score` to Prometheus; Grafana panel + alert rule (see P1-5). |
| A4 | **OOD quarantine** | Robustness | The AE + IsolationForest pair is already implemented — just needs the thresholds documented and the Streamlit ops console to surface the quarantine queue. |
| A5 | **Explainability** | ML quality | The `scripts/explain_iep2.py` SHAP dashboard is built — promote it to an endpoint `/explain/{diagnosis_id}` returning the top-5 features for a given prediction. |
| A6 | **End-to-end integration test** | Test-plan | Finish `tests/test_integration.py`: WAV → EEP → diagnosis → verify probabilities sum to 1 and the response includes a trace-id. Run it in docker-compose-smoke CI. |
| A7 | **Secrets + CORS + rate-limit hardening** | Security | Done for EEP + secrets; extend rate-limits to IEP2/3/4 (P1-6); add `OMNI_ALLOWED_HOSTS` middleware. |

### Tier B — Differentiators (weeks 2–3)

| # | Capability | Why it matters |
|---|---|---|
| B1 | **Active-learning feedback loop** | Ops console already writes to `iep3/feedback_log.csv`. Close the loop: nightly GitHub Action retrains IEP2 on `feedback + golden`, runs the regression gate, and opens a PR with the new joblib model if F1 improved. Demonstrates MLOps + continual-learning. |
| B2 | **SCADA physics fusion** | The `pressure_mult` in `omni/eep/orchestrator.py:426` already ties SCADA pressure into the probability; expose the multiplier in the response body and the Grafana dashboard. Differentiates from a "just-ML" audio classifier. |
| B3 | **TDOA spatial localisation** | Code for time-difference-of-arrival fusion across multiple sensors exists in `omni/spatial/`. Wire it into the ops console as a "leak position on pipe schematic" map panel. |
| B4 | **Remaining-Useful-Life** | `omni/cmms/rul_model.py` has a Gumbel-extreme-value survival model. Expose `/rul/{sensor_id}` returning P(survive 30 d) + a CMMS-ticket threshold. Generator-health story, not just leak detection. |
| B5 | **Durable event bus** | Redis Streams is plumbed in; swap `InMemoryBus` for `RedisBus` as the default once `REDIS_URL` is set — gives replay-on-reconnect and at-least-once delivery without a brokered Kafka. |
| B6 | **mTLS edge ingestion** | Certificates are generated; the Mosquitto broker is configured. Finish the gateway-side cert loading and add a cert-rotation runbook to `docs/runbooks/`. |

### Tier C — Product/commercial polish (weeks 3–4)

| # | Capability | Deliverable |
|---|---|---|
| C1 | **One-page marketing site** | Static hosted on GCS; the three existing `omni-sense-*.html` files are close — unify under a single `site/` + a 30-second product demo GIF. |
| C2 | **Interactive public demo** | Deploy the Streamlit ops console to Cloud Run with a read-only demo user + synthetic-only data; link from the marketing page. |
| C3 | **CMMS integration** | Twilio SMS dispatch is half-wired in `.env.example`. Add a `Ticket` model in the schema and a webhook to Zendesk (or Freshdesk — HTTP-only, no SDK) for the "dispatch a crew" story. |
| C4 | **Pricing / go-to-market one-pager** | Single `GO-TO-MARKET.md`: ICP (municipal water authorities + generator-heavy industrial sites), unit economics (cost per sensor per month on GCP — see §4.4 below), competitive matrix vs Echologics + Gutermann + WINT. |
| C5 | **Compliance / data-residency story** | Short `docs/compliance.md` covering where audit logs live (Timescale), HMAC-chain verification procedure, customer-managed-key support roadmap. Even a 2-page doc signals enterprise-readiness. |
| C6 | **SLOs + runbooks** | `docs/runbooks/` for: (1) IEP2 model rollback, (2) drift-alert triage, (3) MQTT cert rotation, (4) OOD-quarantine review. SLO page: p95 latency ≤ 800 ms, availability ≥ 99.5 %. |

---

## 4. Cloud-Deployment Recipe

### 4.1 Recommendation: GCP Cloud Run + Artifact Registry + Grafana Cloud (free) + local `kind` for K8s validation

Why this stack over the obvious alternatives:

- **Cloud Run vs EKS/GKE** — every service in this repo is a single containerised HTTP app that fits Cloud Run's model (stateless, scales-to-zero, Cloud-Run-only mTLS-on-ingress via IAP). GKE is ~$70/mo just for the control plane, before any workload. EKS is similar. Running the Helm chart on a local `kind` cluster gives us the K8s validation the rubric wants without the monthly bill.
- **Artifact Registry vs Docker Hub** — private + same project + IAM-friendly, and the multi-arch build already works via `docker buildx`.
- **Grafana Cloud free tier** (10 k series, 14 d retention) replaces self-hosting Prometheus + Grafana in prod. The self-hosted stack stays as the default for `docker-compose` dev because that's what students/reviewers run locally.
- **Cloud SQL for Postgres + TimescaleDB extension** is slightly trickier than a self-hosted Timescale; for v1 I run `timescale/timescaledb-ha` on a single GCE e2-small and snapshot the disk nightly. Cost ≈ $13/mo. Postgres flexible-server on Azure has Timescale support too if GCP ends up denied.

### 4.2 Cost estimate — 1 demo environment (always-on)

| Component | Spec | Monthly cost (USD) |
|---|---|---|
| Cloud Run: eep, iep2, iep3, iep4, omni-platform | 0.25 vCPU / 512 MiB, scale-to-zero, ~100 req/day | ≈ $0 (free tier) |
| Artifact Registry | 5 images × ~1 GiB | ≈ $0.50 |
| GCE e2-small (Timescale + Redis) | 2 vCPU / 2 GiB, 20 GiB SSD | ≈ $13 |
| Grafana Cloud | 10 k series free tier | $0 |
| Cloud Logging | 50 GiB free | $0 |
| Outbound bandwidth | 10 GiB | ≈ $1 |
| **Total** | | **≈ $15 / month** |

That's the "grader can hit the live demo any time" cost. Production-grade HA (multi-region, Cloud SQL, VPC, Cloud Armor) is an order of magnitude more — documented in `GO-TO-MARKET.md` / C4.

### 4.3 Build + deploy sequence (what the Makefile will do)

```
# Local (kind) — covers the K8s rubric item
make deploy-local          → kind create, helm install, port-forward 8501
# GCP (prod demo)
make build-push            → docker buildx + gcloud artifacts docker push
make deploy-gcp            → terraform apply infra + gcloud run deploy ×5
```

`infra/terraform/` will hold: the VPC, the GCE instance with Timescale, the Cloud Run services, the Artifact Registry, the secret-manager entries, and the Grafana Cloud API-key binding.

### 4.4 Unit economics for the pitch

Assume one customer site = 10 sensors emitting 5-second frames every 60 s. That's ~14 k frames/day/site, ~420 k/month. Cloud Run can serve that on the $0 tier. Timescale ingest is 420 k rows/month ≈ 50 MiB/month including partition indexes — fits in e2-small for years. Per-site cloud cost is roughly **$1.50/month** (shared infra amortised over 10 customers). At a $99/sensor/month retail price, margin is ~98 %. That's the line for the C4 pitch.

---

## 5. Schedule — 2–4 week timeline

Two weeks is the rubric minimum ("deployable"); four weeks is the stretch (Tier C shipped). Week boundaries are Sundays.

### Week 1 — 2026-04-21 → 04-26 (this week)
- ✅ 04-21 — Session landmine-defuse (this document's §1)
- 04-22 — P0-1: wire real model artefacts into `omni/eep/orchestrator.py` fan-out heads, add readiness probes that fail on missing artefacts
- 04-22 — P0-2: propagate `SCADAUnavailable` from `opcua_gateway.py`
- 04-23 — P0-3: move every magic threshold into `omni/common/config.py` with calibration-date comments
- 04-23 — P0-4: persist the audit HMAC key (env → file → Timescale fallback chain)
- 04-24 — A1: `Makefile` + `kind`-based `make deploy-local`
- 04-25 — A2: re-enable `test_model_regression.py` + wire to `[model]` commit trigger
- 04-26 — A6: full docker-compose integration test that asserts a probabilities vector

### Week 2 — 04-27 → 05-03
- 04-27 — A3: drift-monitor recalibration on 39-d features + Prometheus metric
- 04-28 — A5: `/explain/{id}` endpoint (SHAP)
- 04-29 — P1-5: `alert_rules.yml` for Prometheus + Grafana Cloud datasource
- 04-30 — P1-6 + P1-7: rate-limit IEP2/3/4; `tenacity` retries on inter-service calls
- 05-01 — A7: `OMNI_ALLOWED_HOSTS`; final pass on security headers
- 05-02 — **Grading checkpoint**: project is deployable locally + GCP demo up. Tag `v0.9.0-rc1`.
- 05-03 — Buffer / regression fixes

### Week 3 — 05-04 → 05-10 (stretch)
- 05-04 — B1: nightly retrain workflow + auto-PR
- 05-05 — B2: SCADA pressure multiplier surfaced in response + dashboard
- 05-06 — B3: TDOA spatial map panel
- 05-07 — B4: `/rul/{sensor_id}` RUL endpoint
- 05-08 — B5/B6: Redis Streams bus as default + mTLS edge onboarding runbook

### Week 4 — 05-11 → 05-15 (polish + submit)
- 05-11 — C1 marketing site + demo GIF
- 05-12 — C2 public read-only ops console
- 05-13 — C3 CMMS/Twilio ticket loop
- 05-14 — C4 `GO-TO-MARKET.md` + C5 compliance + C6 runbooks
- 05-15 — Final walk-through, tag `v1.0.0`, record demo video

---

## 6. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Python 3.10 vs 3.11 stdlib drift (`datetime.UTC`, `enum.StrEnum`) breaks anyone running locally on 3.10 | Medium | Low | Already using `python-version: "3.11"` in CI. Add a 1-line `sys.version_info` guard in `conftest.py` that errors with a clean message. |
| R2 | GCP free-tier quota exhausted mid-demo | Low | High | Pre-warm instances 5 min before any live demo; keep a `docker-compose` failover the grader can run locally in < 2 min. |
| R3 | Model file missing on a fresh deploy → fallback head returns 0.3 constant | **High until P0-1 ships** | High | Fail-fast in readiness probe (this is P0-1). Dockerfile `COPY iep2/models/*.joblib /app/models/` is already correct; problem is the fan-out head *silently* falls back if the file is missing. |
| R4 | Drift detector gives spurious alerts after the 39-d switch because the reference distribution was built on 208-d | **Certain until A3 ships** | Medium | Mute the alert during week 1; ship recalibration in week 2 (A3). |
| R5 | OPC-UA gateway `{}`-on-error masks SCADA mismatches — leak gets diagnosed as "pressure nominal" when pressure is actually spiking | Medium | **High** | P0-2. Flag is already in the response schema (`scada_mismatch`), just not being set. |
| R6 | Golden dataset drifts out of sync with trained model; regression gate passes while production regresses | Low | High | Hash the golden CSV and store the hash in the model artefact; CI compares hashes at load time. |
| R7 | Secret leakage if a dev forgets to copy `.env.example → .env` and commits a `.env` anyway | Low | High | `.env` is in `.gitignore`; add a pre-commit hook that refuses a commit containing a `.env` (already in the Tier A scope). |
| R8 | mTLS cert expiry on the demo MQTT broker during grading | Medium | Medium | 90-day certs; runbook C6; calendar reminder 30 days before expiry. |
| R9 | Twilio/Zendesk account creation not done by demo day | Low | Low | Both vendors offer sandbox/trial within 5 min. Twilio is already documented in `.env.example`. |
| R10 | Cloud Run cold-start on IEP4 (CNN model) exceeds the p95 800 ms SLO on first-request-after-scale-to-zero | High | Low | Set `min-instances = 1` on IEP4 only ($8/month extra). EEP, IEP2, IEP3 scale to zero. |

---

## 7. Ready-to-Run Commands (today)

```bash
# 1. Verify lint is green
python -m ruff check . --ignore E501,E402

# 2. Verify EEP tests
cd eep && python -m pytest tests/ -v

# 3. Verify omni tests (Python 3.11 required — CI uses 3.11)
PYTHONPATH=. python -m pytest omni/tests/ -v

# 4. Start the full stack (needs .env copied from .env.example first)
cp .env.example .env
# edit .env and set GRAFANA_ADMIN_PASSWORD + POSTGRES_PASSWORD + OMNI_AUDIT_HMAC_KEY
docker compose up -d

# 5. Hit the diagnose endpoint
curl -sf http://localhost:8000/health
curl -sf -F "audio=@tests/fixtures/sample.wav" \
    -F 'metadata={"site_id":"demo","sensor_id":"s1","pipe_material":"PVC","pressure_bar":3.5}' \
    http://localhost:8000/diagnose
```

---

## 8. Appendix: What to Read Next

- `archive/iep1/README.md` — the IEP1 decommission ADR and the criteria for re-introducing a learned-embedding head.
- `.env.example` — full list of required and optional environment variables with generator commands.
- `omni/common/config.py` — (to be written in week 1) — the single source of truth for every calibration threshold.
- `scripts/generate_golden_difficult.py` — how the hard-negative test cases are constructed. This is the main safety net for catching drift after a retrain.
- `omni/eep/features.py` — the 39-d feature extractor that replaced IEP1.

---

*This file is the single plan. If a task isn't here, it's either deliberately out of scope (document that on the PR) or an oversight (open an issue and link back).*
