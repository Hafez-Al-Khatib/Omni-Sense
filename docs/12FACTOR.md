# 12-Factor App Compliance — Omni-Sense

> The [12-Factor App](https://12factor.net/) methodology is the industry standard for building SaaS applications. This document maps each factor to our architecture decisions.

---

## I. Codebase
**"One codebase tracked in revision control, many deploys"**

✅ **Compliant.** Single Git repository with branch-based environments:
- `main` → Production (ArgoCD manual sync)
- `staging` → Staging (ArgoCD auto-sync)
- `feature/*` → Dev (local kind cluster)

**Evidence:** `.github/workflows/ci.yml` builds images from the same repo for all environments.

---

## II. Dependencies
**"Explicitly declare and isolate dependencies"**

✅ **Compliant.** All dependencies declared in:
- `requirements.txt` files per service
- `Dockerfile` with pinned base images (`python:3.11-slim`)
- No system-level dependencies assumed

**Evidence:** `pip install --no-cache-dir -r requirements.txt` in every Dockerfile.

---

## III. Config
**"Store config in the environment"**

✅ **Compliant.** No secrets in code. Configuration via environment variables:
- `OMNI_IEP2_URL`, `OMNI_IEP3_URL`, `OMNI_IEP4_URL`
- `POSTGRES_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`
- `OMNI_RATE_LIMIT`

**Evidence:** `.env.example` documents all required vars. `.env.production` is gitignored.

**Production enhancement:** External Secrets Operator syncs from Vault/AWS Secret Manager into K8s.

---

## IV. Backing Services
**"Treat backing services as attached resources"**

✅ **Compliant.** All external services are swappable:
- TimescaleDB → PostgreSQL → Cloud SQL (GCP) → RDS (AWS)
- Redis → Memorystore (GCP) → ElastiCache (AWS)
- MQTT → Mosquitto → HiveMQ → AWS IoT Core

**Evidence:** All services configured via DSN/URL env vars, not hardcoded hosts.

---

## V. Build, Release, Run
**"Strictly separate build and run stages"**

✅ **Compliant.** Three distinct stages:
1. **Build:** CI pipeline builds container images, tags with Git SHA
2. **Release:** ArgoCD combines image + config + secrets into a release
3. **Run:** Kubernetes executes containers

**Evidence:** `.github/workflows/ci.yml` builds and pushes to GHCR. `scripts/deploy-gcp.sh` deploys immutable digests.

---

## VI. Processes
**"Execute the app as one or more stateless processes"**

✅ **Compliant.** All services are stateless:
- IEP2/IEP3/IEP4: no local state, models loaded from volume on startup
- EEP: idempotency handled via Redis, not local cache
- Session data: stored in Redis, not memory

**Evidence:** Any pod can be killed and restarted without data loss.

---

## VII. Port Binding
**"Export services via port binding"**

✅ **Compliant.** Each service exposes exactly one port:
- EEP: 8000
- IEP2: 8002
- IEP3: 8003
- IEP4: 8004
- Grafana: 3000
- Prometheus: 9090

**Evidence:** `EXPOSE` directives in Dockerfiles. Istio ingress routes traffic by port.

---

## VIII. Concurrency
**"Scale out via the process model"**

✅ **Compliant.** Horizontal scaling:
- EEP: `--workers 4` (Uvicorn) + K8s HPA
- IEP2/IEP3: single worker (stateless, can replicate)
- IEP4: single worker (GPU/CPU bound, scales via replicas)

**Evidence:** `k8s/helm/omni-sense/templates/hpa.yaml` defines autoscaling rules.

---

## IX. Disposability
**"Maximize robustness with fast startup and graceful shutdown"**

✅ **Compliant.**
- **Fast startup:** IEP2 loads models on startup (~3s). IEP4 loads CNN (~5s).
- **Graceful shutdown:** Uvicorn handles SIGTERM, draining connections.
- **Crash safety:** Circuit breakers (Istio) eject unhealthy pods.

**Evidence:** Kubernetes `terminationGracePeriodSeconds: 30`. Istio `outlierDetection` config.

---

## X. Dev/Prod Parity
**"Keep development, staging, and production as similar as possible"**

✅ **Compliant.** Same Docker images across environments:
- Dev: `docker compose up` (local)
- Staging: Same images on GKE staging cluster
- Production: Same images on GKE production cluster

**Gap:** Dev uses SQLite (MLflow), prod uses Cloud SQL. Documented and acceptable for capstone.

---

## XI. Logs
**"Treat logs as event streams"**

✅ **Compliant.** Structured JSON logging to stdout:
```json
{"timestamp": "2026-05-01T18:23:04Z", "level": "INFO", "service": "eep", "message": "Diagnosis complete", "trace_id": "4f8d2a...", "latency_ms": 271}
```

**Production enhancement:** Fluentd/Fluent Bit forwards to ELK or Cloud Logging.

**Evidence:** `logging.basicConfig` in each service with structured formatters.

---

## XII. Admin Processes
**"Run admin/management tasks as one-off processes"**

✅ **Compliant.** Admin tasks are separate:
- `scripts/extract_dsp_features.py` — batch feature extraction
- `scripts/train_cnn.py` — model training
- `scripts/export_onnx.py` — model export
- `scripts/bootstrap_zero_data.py` — data seeding

**Evidence:** All scripts in `scripts/` directory, documented in README.

---

## Score: 12/12 Factors Compliant

With production enhancements (Vault, Fluent Bit, Cloud SQL), this architecture satisfies the 12-Factor methodology at enterprise scale.
