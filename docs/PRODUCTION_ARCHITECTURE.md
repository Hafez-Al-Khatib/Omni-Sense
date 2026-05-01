# Production Architecture — Omni-Sense

> **Design Philosophy:** This system is architected using patterns proven at Uber (Michelangelo), Netflix (Metaflow), and Google (Vertex AI / SRE). Every decision is traceable to an industry battle-tested pattern.

---

## 1. Deployment Safety: Canary → Blue/Green → Shadow

### Industry Pattern
- **Uber Michelangelo**: Shadow testing is mandatory for 75%+ of critical models. Two shadow modes: endpoint shadowing and deployment shadow with automated drift detection.
- **Netflix**: Metaflow + Spinnaker/Kayenta ships ~300 model updates daily with automated canary analysis.
- **Google Vertex AI**: Traffic-splitting for canary, blue/green, and shadow deployments. Immutable artifacts referenced by digest, never "latest".

### Our Implementation
```
Git Tag v1.2.3 ──► CI Build ──► Container Digest sha256:abc...
                                    │
                                    ▼
                            ArgoCD Application
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                 Dev Cluster    Staging         Production
                 (auto-sync)    (auto-sync)     (manual sync)
                                    │
                                    ▼
                         Istio VirtualService
                         90% → v1.2.2 (stable)
                         10% → v1.2.3 (canary)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
              Prometheus SLO Check           Automated Rollback
              (error rate < 0.1%)            (if SLO violated)
```

**Key Decision:** We use Istio traffic splitting rather than application-level routing because it requires zero code changes and enables instant rollback — the same reason Netflix operates 700+ microservices on Istio.

---

## 2. Service Mesh: Zero-Trust Security

### Industry Pattern
- **Netflix**: Istio handles 100B requests/day across 700+ microservices. Mesh enables model-version routing, A/B testing, and circuit breakers without application code changes.
- **Uber**: Custom service mesh coordinates 4,000+ microservices across multi-region deployments.
- **LinkedIn**: Migrated to Linkerd and reduced p99 latency by 40% for ML services.

### Our Implementation
We deploy Istio sidecars alongside each service pod:

```yaml
# Istio PeerAuthentication enforces mTLS STRICT
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
spec:
  mtls:
    mode: STRICT
```

**Why Istio over Linkerd?** 
- Our use case requires **advanced traffic management** (canary, shadow, A/B) more than raw latency optimization.
- Istio's Envoy-based architecture supports WASM plugins for custom ML-specific routing logic (e.g., route by model version header).
- Linkerd's <1ms overhead advantage is irrelevant for our 200-500ms inference latency budget.

**Circuit Breaker Pattern:**
```yaml
# Istio DestinationRule with circuit breaker
outlierDetection:
  consecutive5xxErrors: 5
  interval: 30s
  baseEjectionTime: 30s
```

If IEP2 rejects 5 consecutive requests (e.g., OOD model crashed), Istio ejects the pod and traffic fails over to healthy replicas — no application code required.

---

## 3. Observability: SRE-Grade SLOs

### Industry Pattern
- **Google SRE**: SLIs (latency, error rate, throughput) → SLOs (e.g., 99.9% availability) → Error Budgets.
- **Uber**: Jaeger (open-sourced) for distributed tracing. M3 for metrics, intelligent trace sampling.
- **Netflix Atlas**: 17B metrics/day, 700B traces/day, observability costs <5% of infrastructure.

### Our SLO Definitions

| SLI | SLO | Measurement Window | Burn Rate Alert |
|-----|-----|-------------------|-----------------|
| Availability | 99.9% | 30 days | 2% budget/day → page |
| p99 Latency | < 2s | 1 hour | 14.4x burn → page |
| OOD False Positive Rate | < 5% | 7 days | Any spike → ticket |
| Model Drift Detection | < 1 hour lag | 24 hours | > 2 hours → page |

**Recording Rules (Prometheus):**
```yaml
# SLO: availability
- record: slo:eep_availability_ratio
  expr: |
    sum(rate(http_requests_total{service="eep",status!~"5.."}[5m]))
    /
    sum(rate(http_requests_total{service="eep"}[5m]))

# SLO: p99 latency
- record: slo:eep_latency_p99
  expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{service="eep"}[5m]))
```

**Why SLOs matter:** Without an error budget, every alert is a page. With SLOs, we know that 0.1% errors are acceptable — we only page when we're burning error budget faster than sustainable.

---

## 4. Distributed Tracing: End-to-End Request Flow

### Industry Pattern
- **Uber Jaeger**: Traces requests across Michelangelo's prediction pipeline. Open-sourced and became CNCF project.
- **Industry standard**: OpenTelemetry for unified instrumentation.

### Our Implementation

Every request gets a `trace_id` injected by EEP and propagated through all services:

```
Client POST /diagnose
  │ trace_id: 4f8d2a...
  ▼
EEP (extract features)
  │ trace_id: 4f8d2a...
  ├─► IEP2 (OOD + classify) ──► 45ms
  ├─► IEP4 (CNN classify) ──► 120ms
  └─► IEP3 (ticket dispatch) ──► fire-and-forget
  ▼
EEP (ensemble + respond)
  │ total: 271ms
```

**Trace Annotations:**
- `span.kind=server` on EEP entry
- `span.kind=client` on IEP2/IEP4/IEP3 calls
- Custom tags: `model_version`, `ood_score`, `ensemble_method`

This lets us answer: *"Why was this request slow?"* → IEP4 CNN inference took 120ms because the model was loaded from cold storage.

---

## 5. Feature Store: Training-Serving Consistency

### Industry Pattern
- **Uber Michelangelo (Palette)**: 20,000+ features, 10 trillion computations daily. Double-write architecture: streaming features written simultaneously to data lake (batch training) and online store (real-time inference). P95: 5ms without lookup, 10ms with lookup.
- **Airbnb (Zipline)**: Sub-10ms feature serving with point-in-time correctness and time-travel.

### Our Architecture (Feast Pattern)

We implement a **simplified feature store** using Redis as the online store and TimescaleDB as the offline store:

```
Raw Sensor Data
    │
    ▼
Feature Pipeline (39-d DSP extraction)
    │
    ├──► Redis (online store) ──► IEP2 inference (real-time)
    │
    └──► TimescaleDB (offline store) ──► ML training pipeline
```

**Training-Serving Skew Mitigation:**
- Features are computed using **exactly the same code path** for training and inference (`omni/eep/features.py`).
- Feature schema is versioned and stored alongside models in MLflow.
- Drift detection compares online feature distribution (Redis) against training distribution (TimescaleDB).

**Why this matters:** Uber found that 60% of production ML bugs are caused by training-serving skew. Our double-write pattern prevents this at the architecture level.

---

## 6. GitOps: Git as Single Source of Truth

### Industry Pattern
- **Google**: Cloud Build + GitOps. Environment configs version-controlled and promoted via PRs.
- **Industry standard**: ArgoCD/Flux with ApplicationSets. Dev uses `*` semver, staging uses release candidates, prod uses stable releases only.

### Our Implementation

```
GitHub Repo
    │
    ├──► main branch ──► ArgoCD Application ──► Production Cluster
    │                          (manual sync)
    │
    └──► staging branch ──► ArgoCD Application ──► Staging Cluster
                               (auto-sync)
```

**ArgoCD ApplicationSet:**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
spec:
  generators:
  - list:
      elements:
      - cluster: staging
        branch: staging
        autoSync: true
      - cluster: production
        branch: main
        autoSync: false  # manual promotion gate
```

**Promotion Gate:**
1. PR merged to `staging` → auto-deploy to staging cluster
2. SLO checks pass for 24h → PR to `main`
3. Manual approval → ArgoCD syncs to production
4. Canary deployment at 10% → 50% → 100%

**Immutable Artifacts:** Container images are tagged with Git commit SHA, not `latest`. This is the same pattern Google uses — you can always trace a running container back to exact source code.

---

## 7. Secrets Management: Vault Pattern

### Industry Pattern
- **Uber**: Built custom multi-cloud Secret Management Platform. 90% reduction in secrets distributed to workloads via centralized vaults.
- **Industry standard**: HashiCorp Vault with Kubernetes auth.

### Our Implementation

We use **External Secrets Operator** to sync secrets from AWS Secrets Manager / GCP Secret Manager into Kubernetes:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: omni-db-credentials
spec:
  secretStoreRef:
    kind: ClusterSecretStore
    name: gcp-secret-manager
  target:
    name: omni-db-credentials
  data:
  - secretKey: password
    remoteRef:
      key: omni-sense-db-password
      version: latest
```

**Why not env vars?** Environment variables are visible in `docker inspect`, process listings, and crash dumps. Vault-injected secrets are mounted as files that disappear when the pod dies.

---

## 8. Multi-Environment Pipeline

### Industry Pattern
- **Uber Michelangelo 2.0**: Three-plane architecture — Control Plane (K8s operators), Offline Data Plane (Spark/Ray for training), Online Data Plane (RPC services for inference).
- **Netflix Metaflow**: Same Python flow runs on laptop (dev), AWS Batch (staging), and production — zero code changes.

### Our Environments

| Environment | Cluster | Data | Auto-Sync | Purpose |
|-------------|---------|------|-----------|---------|
| **Dev** | Local kind cluster | Synthetic data | N/A | Developer testing |
| **Staging** | GKE/EKS staging | 10% real data | ArgoCD auto | Integration tests, SLO validation |
| **Production** | GKE/EKS prod | 100% real data | ArgoCD manual | Customer-facing, canary gates |

**Environment Isolation:**
- Separate Kubernetes namespaces with NetworkPolicies
- Separate service accounts and IAM roles per environment
- Separate Prometheus instances (no metric leakage between environments)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  Web UI ──► Mobile App ──► SCADA Integration ──► Hardware Sensors          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EDGE LAYER (RPi / ESP32)                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │  ADXL345    │───►│  ONNX RT    │───►│  MQTT+TLS   │                      │
│  │  3.2 kHz    │    │  39-d DSP   │    │  to Cloud   │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │ mTLS 8883
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOUD LAYER (Kubernetes + Istio)                     │
│                                                                               │
│  ┌─────────────┐    ┌─────────────────────────────────────────────┐         │
│  │  Ingress    │───►│  EEP (API Gateway)                          │         │
│  │  Gateway    │    │  ├─ Rate limiting (slowapi)                 │         │
│  │  + WAF      │    │  ├─ Idempotency cache (Redis)              │         │
│  └─────────────┘    │  ├─ Feature extraction (39-d DSP)          │         │
│                      │  └─ Fan-out to IEP2 + IEP4                 │         │
│                      └─────────────────────────────────────────────┘         │
│                                    │                    │                     │
│                      ┌─────────────┘                    └─────────────┐       │
│                      ▼                                               ▼       │
│  ┌──────────────────────────┐                            ┌──────────────────┐│
│  │  IEP2 (Classical ML)     │                            │  IEP4 (Deep CNN) ││
│  │  ├─ Isolation Forest OOD │                            │  ├─ Autoencoder  ││
│  │  ├─ XGBoost Classifier   │                            │  ├─ CNN Classifier││
│  │  └─ Calibration Manager  │                            │  └─ Spectrogram  ││
│  └──────────────────────────┘                            └──────────────────┘│
│                      │                                               │        │
│                      └───────────────────┬───────────────────────────┘        │
│                                          ▼                                   │
│                      ┌─────────────────────────────────────────────┐         │
│                      │  EEP Ensemble (weighted_avg / ood_bypass)   │         │
│                      │  ├─ SCADA fusion                            │         │
│                      │  └─ Confidence thresholding                 │         │
│                      └─────────────────────────────────────────────┘         │
│                                          │                                   │
│                      ┌───────────────────┴───────────────────┐               │
│                      ▼                                       ▼               │
│  ┌──────────────────────────┐                    ┌──────────────────────┐   │
│  │  IEP3 (Active Learning)  │                    │  IEP4 (Drift Monitor)│   │
│  │  ├─ Feedback tickets     │                    │  ├─ KL divergence    │   │
│  │  ├─ Label confirmation   │                    │  ├─ 24h rolling win  │   │
│  │  └─ CMMS integration     │                    │  └─ Retraining trigger│   │
│  └──────────────────────────┘                    └──────────────────────┘   │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  PLATFORM LAYER                                                         │ │
│  │  ├─ TimescaleDB (time-series + features)                               │ │
│  │  ├─ Redis (cache + pub/sub)                                            │ │
│  │  ├─ MLflow (model registry + experiments)                              │ │
│  │  ├─ Prometheus + Thanos (metrics + long-term storage)                  │ │
│  │  ├─ Jaeger (distributed tracing)                                       │ │
│  │  ├─ Grafana (dashboards + alerts)                                      │ │
│  │  └─ Vault (secrets management)                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Defense Talking Points

> **"Why Istio instead of Linkerd?"**

"LinkedIn reduced p99 latency by 40% with Linkerd, but our inference latency budget is 200-500ms — a <1ms sidecar overhead is irrelevant. We chose Istio because we need advanced traffic management: canary deployments, shadow mode, and A/B testing by model version. These are hard to implement in Linkerd without custom controllers."

> **"How do you prevent training-serving skew?"**

"Uber found 60% of production ML bugs are caused by training-serving skew. We implement a simplified double-write feature store pattern: features computed by the same code path (`omni/eep/features.py`) are written simultaneously to TimescaleDB for offline training and served from Redis for online inference. Feature schemas are versioned alongside models in MLflow."

> **"What happens if a model degrades in production?"**

"We have three layers of protection: (1) Istio circuit breakers eject unhealthy pods after 5 consecutive errors, (2) Prometheus SLO alerts fire when error budget burns faster than sustainable, and (3) ArgoCD can trigger automated rollback to the previous Git tag. This is the same pattern Netflix uses to ship 300 model updates daily safely."

> **"Why GitOps instead of manual kubectl apply?"**

"Google's SRE book states that manual changes are the #1 source of production incidents. GitOps makes Git the single source of truth — every deployment is a PR with code review, automated tests, and audit trail. ArgoCD continuously reconciles cluster state with Git, so any manual drift is automatically corrected."
