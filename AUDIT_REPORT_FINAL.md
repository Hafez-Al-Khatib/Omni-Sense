# Omni-Sense: Honest Pre-Submission Audit & Grading

**Auditor:** Kimi Code CLI (AI Code Review Agent)  
**Date:** 2026-05-01  
**Deadline:** ~3 days remaining  
**Scope:** Full rubric compliance check against `EECE503N_798SN_SP26` requirements

---

## Executive Summary

Omni-Sense is a **genuinely impressive engineering project** with depth that exceeds most capstones. The architecture is sophisticated (TDOA spatial fusion, SCADA cyber-physical validation, dual OOD detection, audit HMAC chains), the ML models are real (ONNX-exported XGBoost, RF, CNN, Isolation Forest, Autoencoder), and the observability instrumentation is thoughtful.

**However**, the project currently sits in a dangerous valley: **it looks production-grade in documentation but has critical operational gaps that a strict grader will penalize heavily.** The biggest risk is that the rubric treats cloud deployment and end-to-end testing as *non-negotiable*, and several "demo stoppers" remain unfixed.

### Overall Honest Grade: **72–78 / 100** (B+ to A- range)

*With 3 focused days, this can be pushed to 85–90. Without fixes, it risks landing in the 65–72 range if the demo hits any of the landmines documented below.*

---

## 1. Architecture & Service Boundaries (16/20)

### ✅ What's Working
- **5 independent services** with clear Docker boundaries: EEP, IEP2, IEP3, IEP4, Omni-Platform.
- **EEP (microservice)** does real parallel HTTP fan-out to IEP2 + IEP4 via `httpx` (`eep/app/services/orchestrator.py:187-220`).
- **Input/output contracts** are well-defined with Pydantic schemas.
- **IEP2** has substantial real modeling: Isolation Forest OOD (ONNX), XGB+RF ensemble (ONNX), SHAP explainability, drift monitor.
- **IEP4** has real 2-D CNN spectrogram classifier (PyTorch + ONNX) with autoencoder OOD.
- **IEP3** is a real SQLite-backed ticket/feedback store (business logic, not just CRUD).
- **Rate limiting** (`slowapi`) and input validation are present on EEP.

### ⚠️ What's Partially Working
- **Omni-Platform EEP (`omni/eep/orchestrator.py`) has `asyncio.sleep` stubs** for Isolation Forest and OOD heads (lines 337-344). These always return physics heuristics, never real model inference.
- XGB/RF heads in `omni/eep/orchestrator.py` silently fall back to physics stubs if ONNX models are missing (P0-1 landmine).
- IEP3 has **no Kubernetes manifests** — it exists in Docker Compose but is entirely missing from Helm.

### ❌ What's Broken / Missing
- No inter-service retries (EEP → IEP calls fail on first 5xx).
- No circuit breakers.

**Verdict:** The microservice architecture is solid. The omni-platform stubs are a risk only if the grader inspects `omni/eep/orchestrator.py` deeply.

---

## 2. Cloud Deployment (6/20) — **CRITICAL GAP**

### ✅ What's Working
- Evidence of **manual GCP Cloud Run deployment** exists in screenshots (`screenshots/cloud_console.png`, `cloud_deploy.png`, `cloud_health_browser.png`).
- EEP was deployed to `https://omni-eep-745790249979.us-central1.run.app` (now unreachable — likely scaled to zero or project paused).
- Cost estimate and deployment architecture are documented in `MASTERPLAN.md`.

### ❌ What's Broken / Missing
- **NO Terraform or Infrastructure-as-Code.** The `infra/terraform/` directory mentioned in `MASTERPLAN.md` does not exist. This means deployment is not reproducible.
- **NO CI/CD deploy stage.** GitHub Actions builds images but never pushes them (`push: false`).
- **Only EEP appears to have been deployed.** There is no evidence IEP2, IEP3, IEP4, or the observability stack were ever deployed to cloud.
- **No `Makefile`** for `make deploy-local` or `make deploy-gcp` despite being promised in `MASTERPLAN.md`.
- `.env.production` is **committed to git** (security violation, even if blocked by secret scanner).
- Self-signed TLS certificates are **committed to git** (`certs/`).

**Rubric Hard Rule:** *"Local-only demos are not accepted."*  
**Risk Level:** 🔴 **SEVERE.** If the grader hits the Cloud Run URL and it's down, or if they ask to see Terraform and you have none, this is an immediate major penalty.

**Verdict:** You have proof it *can* deploy, but not a reproducible, current, full-stack deployment.

---

## 3. Quality Assurance & Testing (13/20)

### ✅ What's Working
- **~148 tests in `omni/tests/` covering:** alerts, audit, bus, dispatch, drift, EEP heads, features, gateway, integration, MLOps, RUL, schemas, spatial, TDOA.
- **IEP4 tests** pass (6/6).
- **Physics regression in TDOA** — parametrized tests verify sub-metre accuracy.
- **Golden dataset** exists (`data/golden/` ~240 samples + `data/golden_difficult/` 15 adversarial samples).
- **CI runs per-service** (EEP, IEP2, IEP4, Omni Platform) via `.github/workflows/ci.yml`.

### ⚠️ What's Partially Working
- **`pytest.ini` overrides `pyproject.toml`** — running `pytest` from root silently skips ~32 tests across `eep/`, `iep2/`, `iep4/`, and root `tests/`.
- **Model regression gate** (`iep2/tests/test_model_regression.py`) always **SKIPS** because `iep2/models/metrics.json` is missing.
- **Golden difficult set is generated but never tested** in CI.
- **IEP3 has zero tests.** No `iep3/tests/` directory exists.
- **Smoke test only starts 2 of 11 services** (`eep` + `iep2`).

### ❌ What's Broken / Missing
- **`eep/tests/test_routes.py` is blocked** by `ModuleNotFoundError: No module named 'slowapi'` (confirmed by live run).
- **No end-to-end test calling a deployed cloud system.** The root `tests/test_integration.py` only hits `localhost:8000`.
- No coverage reporting in CI.
- No test for schema validation of golden CSVs.

**Rubric Hard Rule:** *"If the system fails during the demo, grading stops."*  
**Risk Level:** 🟡 **HIGH.** The `slowapi` import failure suggests environment fragility. The missing `metrics.json` means your regression gate is decorative.

---

## 4. MLOps & Experiment Tracking (14/20)

### ✅ What's Working
- **MLflow** is deployed in Docker Compose with ~48 tracked runs.
- **`scripts/mlops_pipeline.py`** implements real promotion/rollback logic with F1/AUC gates.
- **`scripts/train_models.py`** exports ONNX artifacts and logs to MLflow.
- **Drift detection** is mathematically sound (PSI + cosine similarity) and wired into the platform bus.
- **Retraining trigger** has intelligent cooldown (60 min) and human-review bus events.

### ⚠️ What's Partially Working
- Many MLflow runs have `end_time: null` — incomplete/aborted runs.
- No Model Registry staging tags (`Staging`, `Production`).
- MLflow backend is SQLite (not production-grade for concurrent writes).
- The retraining trigger looks for `iep2/scripts/train_models.py` — **this path does NOT exist.** The real script is at `scripts/train_models.py`. This is a **functional blocker** for automated retraining.

### ❌ What's Broken / Missing
- No scheduled/cron trigger for nightly retraining.
- No CI job that promotes Docker images based on model regression gate.
- No automated PR creation for model updates.

---

## 5. Monitoring & Observability (14/20)

### ✅ What's Working
- **Prometheus** scrapes all 4 core services + omni-platform.
- **Grafana dashboard** (`monitoring/grafana/dashboards/omni-sense.json`) has 9 panels: API latency (p50/p95), request rate, IEP4 CNN duration, IEP2 duration, OOD score, confidence distribution, OOD rejections, error rate, request volume.
- **Rich ML-specific metrics:** OOD anomaly scores, autoencoder reconstruction error, confidence histograms, SCADA mismatch counters, embedding drift gauges.
- Dashboard is provisioned as-code.

### ❌ What's Broken / Missing
- **ZERO Prometheus alert rules.** No `alertmanager.yml`, no `rules/`, no `alerts/`.
- **Omni-Platform is configured as a scrape target but has NO `/metrics` endpoint.** Streamlit (`omni/ops_console/app.py`) has zero Prometheus instrumentation. Prometheus sees this target as DOWN.
- No infrastructure metrics (Redis, Postgres, MQTT).
- No Grafana panels for IEP3 tickets/feedback despite metrics existing in code.
- No PSI drift panel in Grafana.

**Rubric Requirement:** *"At least one meaningful metric per service"* — technically satisfied.  
**Rubric Requirement:** *"Alerting-ready metrics"* — **NOT satisfied** (no alert rules).

---

## 6. Security & Robustness (12/15)

### ✅ What's Working
- Input validation and payload constraints on EEP (file size ≤5MB, format whitelist).
- Rate limiting on EEP via `slowapi`.
- CORS tightened to explicit allow-methods/allow-headers.
- Docker Compose refuses to start with empty passwords (`${VAR:?error message}`).
- mTLS certificates generated and Mosquitto broker configured.
- Health checks and readiness/liveness probes in K8s manifests.

### ❌ What's Broken / Missing
- `.env.production` committed to git.
- Self-signed TLS certs committed to git (`certs/`).
- **No rate limits on IEP2, IEP3, IEP4.** Only EEP is protected.
- No inter-service retries or circuit breakers.
- No `OMNI_ALLOWED_HOSTS` middleware.
- Helm Secret template ships with empty base64 values.

---

## 7. Documentation & Tradeoffs (15/15)

### ✅ What's Working
- **Tradeoffs** are explicitly documented in `README.md`, `Phase1.md`, and `DEFENSE_STRATEGY.md` (3+ tradeoffs with what chosen, what rejected, and evidence).
- Competitive matrix vs Echologics/Gutermann/WINT is strong.
- Cost estimate ($15/mo demo, $1.50/site) is detailed.
- `MASTERPLAN.md` is an exceptionally clear engineering document.
- Defense strategy against "solved problem" and "too many failure points" is well-articulated.

### ⚠️ What's Partially Working
- `workspace_overview.md` still references IEP1 (archived).
- Some docs claim capabilities not yet shipped (e.g., `make deploy-gcp`, Terraform).

---

## 8. Git Discipline (7/15)

### ✅ What's Working
- Commit messages are mostly meaningful (not "final commit").
- `.gitignore` is properly configured.
- Some feature branches exist (`feature/field-verification`, `feat/interventions`).

### ❌ What's Broken / Missing
- **No PR discipline visible.** Commit history shows direct pushes to `main` and merge commits without review notes.
- `.env.production` should never have been committed.
- No evidence of code review (no PR descriptions, no review comments in commits).
- If LLMs were used, prompt versions are not tracked.

---

## Detailed Score Breakdown

| Rubric Pillar | Points | Score | Notes |
|---|---|---|---|
| Architecture (EEP + 2+ IEPs, contracts, fan-out) | 20 | 16 | IEP3 missing from K8s; omni-platform stubs |
| Cloud Deployment (public, reproducible, full stack) | 20 | 6 | Manual deploy only; no IaC; URL down |
| QA & Testing (unit, integration, E2E deployed) | 20 | 13 | slowapi blocked; no cloud E2E; regression gate skips |
| MLOps (training pipeline, experiment tracking, promotion) | 20 | 14 | Broken retrain path; no Model Registry staging |
| Monitoring (Prometheus, Grafana, alerts, ML signals) | 20 | 14 | No alert rules; omni-platform scrape DOWN |
| Security (validation, rate limits, secrets, failure modes) | 15 | 12 | IEPs unprotected; .env.prod committed; certs in git |
| Documentation & Tradeoffs | 15 | 15 | Excellent |
| Git Discipline | 15 | 7 | No PRs; secrets committed |
| **TOTAL** | **145** | **97** | **~67%** |

*Note: The rubric uses competitive/dynamic grading after baseline. With 3 days of fixes, this becomes a strong ~85+ project. Without fixes, it risks being average relative to a strong cohort.*

---

## The 3-Day Triage Plan

### Day 1 (May 2) — Demo Stoppers

| Priority | Task | File(s) | Impact |
|---|---|---|---|
| 🔴 P0 | **Fix Cloud Run deployment** | Create `infra/terraform/main.tf` or at minimum a `deploy-gcp.sh` script that deploys EEP + IEP2 + IEP4 to Cloud Run using the existing Docker images. | Rubsric hard requirement |
| 🔴 P0 | **Remove `.env.production` from git** | `git rm --cached .env.production` + update `.gitignore` | Security failure |
| 🔴 P0 | **Fix `slowapi` import** | Install `slowapi` in root venv or add to CI deps | Unblocks 5 EEP route tests |
| 🔴 P0 | **Fix `pytest.ini` vs `pyproject.toml` conflict** | Delete `pytest.ini` or merge `testpaths` correctly | Unblocks ~32 skipped tests |
| 🔴 P0 | **Generate `iep2/models/metrics.json`** | Run training script or create baseline metrics file | Makes regression gate real |

### Day 2 (May 3) — Rubric Gaps

| Priority | Task | File(s) | Impact |
|---|---|---|---|
| 🟡 P1 | **Add Prometheus alert rules** | `monitoring/prometheus/alerts.yml` + AlertManager container | Rubric "alerting-ready" requirement |
| 🟡 P1 | **Add IEP3 to Helm** | `k8s/helm/omni-sense/templates/iep3/deployment.yaml` | K8s completeness |
| 🟡 P1 | **Add inter-service retries** | `eep/app/services/orchestrator.py` — wrap `httpx.post` with `tenacity` | Robustness |
| 🟡 P1 | **Fix retraining script path** | `omni/mlops/retraining_trigger.py` — change `iep2/scripts/train_models.py` → `scripts/train_models.py` | MLOps loop works |
| 🟡 P1 | **Add end-to-end cloud test** | `tests/test_integration.py` — add a test that calls the live Cloud Run URL | Rubric hard requirement |
| 🟡 P1 | **Remove certs from git** | `git rm --cached certs/*` + document generation in README | Security hygiene |

### Day 3 (May 4) — Polish & Defense Prep

| Priority | Task | File(s) | Impact |
|---|---|---|---|
| 🟢 P2 | **Add `Makefile`** | `Makefile` with `deploy-local`, `deploy-gcp`, `test`, `lint` targets | Shows engineering maturity |
| 🟢 P2 | **Fix omni-platform metrics** | Either add `/metrics` to Streamlit sidecar or remove from `prometheus.yml` | Eliminates DOWN target |
| 🟢 P2 | **Add rate limits to IEP2/3/4** | Copy `slowapi` pattern from EEP | Security depth |
| 🟢 P2 | **Document team contributions with real names** | `README.md` — replace "Reem [Lastname]" and "Maram [Lastname]" | Professionalism |
| 🟢 P2 | **Record 60-second demo video** | Deploy stack, hit `/diagnose` with a WAV, show Grafana | Backup if live demo fails |
| 🟢 P2 | **Prepare defense talking points** | Practice the 3 pillars: OOD "nuisance filter", cyber-physical fusion, industrial reliability | Prof. Ammar's concerns |

---

## Final Honest Assessment

**What makes this project special:**
- The TDOA spatial fusion, SCADA pressure multiplier, and OOD rejection are genuinely advanced for a student capstone.
- The documentation (`MASTERPLAN.md`, `DEFENSE_STRATEGY.md`) reads like a real engineering memo, not student homework.
- The Prometheus instrumentation is domain-aware (OOD scores, SCADA mismatches, reconstruction errors).

**What will hurt you if not fixed:**
1. **No reproducible cloud deployment.** A manual `gcloud run deploy` from Maram's laptop that is now down is not "deployed on AWS, Azure, GCP." You need Terraform or at minimum a script that a grader can run.
2. **No alert rules.** The rubric explicitly asks for "alerting-ready metrics." Prometheus without alerts is just a dashboard.
3. **Test fragility.** `slowapi` missing and `pytest.ini` misconfiguration mean your test suite may not pass on the grader's machine.
4. **Secrets in git.** `.env.production` and `certs/` being tracked is a red flag for any professional reviewer.

**Bottom line:** This is a top-20% *engineering* project currently masquerading as a top-5% *product* due to documentation polish. Close the deployment, testing, and observability gaps in the next 3 days and it becomes a genuine top-5% submission. Leave them open and you are gambling that the grader doesn't probe the weak spots.

**Recommended target grade after fixes: 85–90/100.**
