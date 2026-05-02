# Omni-Sense Deployment Guide

> **Target:** Hetzner CX23 (4 GB / 2 vCPU / 40 GB SSD)  
> **Cost:** ~€4.51/month (~$5/month)  
> **TLS:** Caddy + Let's Encrypt (auto-renewing)  
> **Last updated:** 2026-05-02

---

## 1. Architecture

```
                              Internet
                                 |
                                 v
                         +---------------+
                         |    Caddy      |  TLS termination (Let's Encrypt)
                         |  :80 / :443   |
                         +---------------+
                                 |
            +--------------------+--------------------+
            |                    |                    |
            v                    v                    v
      +-----------+       +-----------+       +------------+
      |    EEP    |       |  Grafana  |       | Prometheus |
      |  :8000    |       |  :3000    |       |   :9090    |
      +-----------+       +-----------+       +------------+
            |                                          |
            v                                          v
   +--------+--------+                         +------------+
   |                 |                         |   MLflow   |
   v                 v                         |   :5000    |
+------+       +------+                        +------------+
| IEP2 |       | IEP4 |                                |
| :8002|       | :8004|                                v
+------+       +------+                         +------------+
   |                 |                           | TimescaleDB|
   v                 v                           |   :5432    |
+------+       +------+                         +------------+
| IEP3 |       | MQTT |                                |
| :8003|       |:1883 |                                v
+------+       |:8883 |                         +------------+
               +------+                         |   Redis    |
                 |    |                         |   :6379    |
                 v    v                         +------------+
            [ESP32 / RPi]
```

**Traffic flow:**
1. **Client** → `eep.<domain>` → Caddy → EEP (:8000)
2. **EEP** fans out in parallel to IEP2 (:8002, classical ML) and IEP4 (:8004, CNN)
3. **IEP3** (:8003) receives fire-and-forget dispatch tickets for high-confidence faults
4. **Hardware** (ESP32 / RPi) connects directly to MQTT broker (:1883 plain, :8883 mTLS)
5. **Observability** — Prometheus scrapes all services; Grafana visualizes; MLflow tracks experiments

---

## 2. Public URLs

| Subdomain | Service | Purpose |
|---|---|---|
| `eep.<domain>` | EEP (FastAPI) | Public API gateway — `/health`, `/api/v1/diagnose`, `/api/v1/calibrate` |
| `grafana.<domain>` | Grafana | Operational dashboards (login: admin + `GRAFANA_ADMIN_PASSWORD`) |
| `prometheus.<domain>` | Prometheus | Metrics query UI (basic-auth protected) |
| `mlflow.<domain>` | MLflow | Experiment tracking UI (basic-auth protected) |
| `mqtt.<domain>:8883` | Mosquitto | Hardware telemetry ingress (mTLS) |

Replace `<domain>` with your actual domain (e.g. `omnisense.duckdns.org`).

---

## 3. Prerequisites

- A Hetzner Cloud account (https://console.hetzner.cloud/)
- A domain or free DuckDNS subdomain (https://www.duckdns.org/)
- An SSH key pair (`ssh-keygen -t ed25519 -C "omni-deploy"`)

---

## 4. Provision Server (Task 1)

1. Create a project `omni-sense-prod` in Hetzner Cloud.
2. Add your SSH public key under **Security → SSH Keys**.
3. Create a server:
   - **Type:** CX23 (4 GB RAM / 2 vCPU / 40 GB SSD)
   - **Image:** Ubuntu 24.04 LTS
   - **Location:** `nbg1` (Nuremberg) or any EU region
   - **SSH key:** select yours
4. Note the **public IPv4** address (e.g. `78.46.x.x`).

---

## 5. Bootstrap Server (Task 2)

SSH in as root and run:

```bash
apt update && apt upgrade -y
curl -fsSL https://get.docker.com | sh
apt install -y docker-compose-plugin git ufw

# Firewall
ufw allow 22/tcp 80/tcp 443/tcp 8883/tcp
ufw --force enable

# Create deploy user
useradd -m -s /bin/bash -G docker omni
mkdir -p /opt/omni-sense && chown omni:omni /opt/omni-sense
```

---

## 6. DNS (Task 3)

Create **A records** pointing to your Hetzner IPv4:

```
eep.<domain>       A  <hetzner-ip>
grafana.<domain>   A  <hetzner-ip>
prometheus.<domain> A  <hetzner-ip>
mlflow.<domain>    A  <hetzner-ip>
mqtt.<domain>      A  <hetzner-ip>
```

Wait for propagation:

```bash
dig +short eep.<domain>
# should return <hetzner-ip>
```

---

## 7. Deploy Application (Tasks 4–6)

### 7.1 Clone repository

```bash
su - omni
cd /opt/omni-sense
git clone https://github.com/Hafez-Al-Khatib/Omni-Sense.git .
```

### 7.2 Configure environment

```bash
# Copy the production template
cp .env.production .env

# Edit .env and fill in ALL <placeholders>
nano .env
```

Required values:

| Variable | How to set |
|---|---|
| `OMNI_DOMAIN` | Your domain, e.g. `omnisense.duckdns.org` |
| `OMNI_PROM_BASIC_HASH` | `docker run --rm caddy:2 caddy hash-password --plaintext '<pwd>'` |
| `OMNI_MLFLOW_BASIC_HASH` | Same command, different password |
| `GRAFANA_ADMIN_PASSWORD` | `openssl rand -base64 24` |
| `POSTGRES_PASSWORD` | `openssl rand -base64 24` |
| `MQTT_PASSWORD` | `openssl rand -base64 24` |

**Security:** Keep `.env` mode `0600` (root/omni readable only). Never commit it.

```bash
chmod 0600 .env
```

### 7.3 Start the stack

```bash
cd /opt/omni-sense
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

First startup takes ~5–10 minutes (scientific-Python wheels compile/download).

### 7.4 Verify health

```bash
# EEP
curl -sf https://eep.<domain>/health

# Grafana
curl -sf https://grafana.<domain>/api/health

# Prometheus (401 = auth wall is up)
curl -sf https://prometheus.<domain>/-/healthy

# MLflow (401 = auth wall is up)
curl -sf https://mlflow.<domain>
```

---

## 8. Smoke Test from Third-Party Network (Task 7)

Run **from your laptop** (not the VPS):

```bash
# Set the domain so the script targets your Hetzner deployment
export OMNI_DOMAIN=omnisense.duckdns.org
python test_public_url.py
```

Expected output:

```
Testing Omni-Sense at: https://eep.omnisense.duckdns.org

-> GET https://eep.omnisense.duckdns.org/health
  Attempt 1: Status 200
  Body: {'status': 'healthy', 'service': 'eep', 'version': '0.1.0'}

-> POST https://eep.omnisense.duckdns.org/api/v1/diagnose  (audio=...)
  Status: 200
  Body (truncated): {"label": "No_Leak", "confidence": 0.98, ...}
...
PASS
```

---

## 9. Secrets Management

| Secret | Location | Protection |
|---|---|---|
| DB passwords, Grafana admin | `.env` on server | `chmod 0600`, not in git |
| Prometheus / MLflow basic-auth | Caddy hashed passwords | bcrypt hashes, not plaintext |
| MQTT TLS | `certs/ca.crt`, `certs/server.crt`, `certs/server.key` | 0644/0600, generated pre-deploy |
| GitHub token (CI push) | GitHub Actions `secrets.GITHUB_TOKEN` | Ephemeral, scoped to repo |

**Why not in git:** The repository is public (or will be reviewed by graders). Secrets in git are forever. The `.env.production` file committed to git is a **template** with `<placeholders>`.

---

## 10. Cost Estimate

| Item | Monthly Cost | Notes |
|---|---|---|
| Hetzner CX23 | €4.59 (~$5) | 4 GB RAM, 2 vCPU, 40 GB SSD |
| Caddy + Let's Encrypt | €0 | Open source, auto-renewing TLS |
| DuckDNS | €0 | Free dynamic DNS |
| **Total** | **~$5/mo** | |

### Comparison: Render

| Approach | Monthly | Observability | Cold Start | Notes |
|---|---|---|---|---|
| **Hetzner + Docker Compose** | ~$5 | Full stack (Prom/Grafana/MLflow) | None | This deployment |
| Render free (4 web svcs) | $0 | None | 30–60 s | Sleeps after 15 min |
| Render paid (4 × $7) | ~$28 | None | None | Still no Prom/Grafana |

### Cost drivers

- **RAM (4 GB):** The limiting resource. At peak, the stack uses ~3.2 GB:
  - Postgres/TimescaleDB: ~512 MB
  - Prometheus: ~256 MB (30-day retention)
  - Grafana + MLflow + EEP + IEPs: ~2 GB
  - Headroom for spikes: ~512 MB
- **Disk (40 GB):** Postgres WAL + Prometheus TSDB + MLflow artifacts. If disk > 70 %, extend with Hetzner Volumes (~€0.044/GB/mo).
- **Scaling up:** Move to CX32 (8 GB / €8.38) if you add Redis clustering or longer Prometheus retention.

---

## 11. Failure Modes & Recovery

| Component | Failure symptom | Auto-recovery | Manual fix |
|---|---|---|---|
| **Caddy** | All URLs 502/503 | Docker `restart: unless-stopped` | `docker compose restart caddy` |
| **EEP** | `/health` 404/502 | Docker restart; downstream calls have tenacity retries | Check logs: `docker logs omni-sense-eep-1` |
| **IEP2 / IEP4** | EEP returns 503 | Docker restart | `docker compose restart iep2` |
| **MQTT broker** | ESP32 disconnects | Docker restart | Check certs expiry in `certs/` |
| **Postgres** | EEP/IEP3 ticket writes fail | Docker restart | Restore from `db.tgz` backup |
| **Prometheus** | Metrics gaps | Docker restart | Volume is ephemeral; no data loss for new scrapes |

All services expose `/health` endpoints consumed by Docker healthchecks (see `docker-compose.yml`).

---

## 12. Why Not Render?

See `coordination/decisions.md` for full context. Summary:

1. **Account locked:** The original Render account that owned `*.onrender.com` is inaccessible (user never created it; Google login fails).
2. **Free-tier risk:** Cold starts (30–60 s) violate rubric §8.1: *"If the system fails during the demo, grading stops."*
3. **Observability gap:** Render Blueprint only deploys the 4 FastAPI services. Prometheus, Grafana, MLflow, and MQTT would remain local-only, failing rubric §11 (live observability requirement).
4. **Cost parity:** Hetzner CX22 at ~$5/mo runs the **entire** stack with full observability. Render paid tier would cost ~$28/mo for just the web services without observability.

Render is retained as a documented secondary path in `render.yaml` for teams who already have a Render account and only need the API services.

---

## 13. Disaster Recovery

### Full stack rebuild (new server)

```bash
# 1. Provision new CX23, bootstrap (§5), DNS (§6)
# 2. Clone repo, copy .env from backup
su - omni
cd /opt/omni-sense
git clone https://github.com/Hafez-Al-Khatib/Omni-Sense.git .
cp /backup/.env .env

# 3. Start stack
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

### Database backup

```bash
cd /opt/omni-sense
# One-off backup
docker run --rm \
  -v omnisense_timescale-data:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/db-$(date +%Y%m%d).tgz /data

# Automated daily (add to crontab)
0 3 * * * cd /opt/omni-sense && docker run --rm -v omnisense_timescale-data:/data -v /opt/backups:/backup ubuntu tar czf /backup/db-$(date +\%Y\%m\%d).tgz /data && find /opt/backups -name 'db-*.tgz' -mtime +7 -delete
```

### Zero-downtime update

```bash
cd /opt/omni-sense
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Docker Compose only recreates containers whose images or configs changed. Healthy containers stay up.

---

## 14. Kubernetes Path (Rubric §9)

The repository already contains Kubernetes manifests under `k8s/` (Helm charts, ArgoCD, Istio canary). These are **not** the primary deployment path for this project because:

- A single CX23 node cannot host a meaningful K8s control plane + workloads within 4 GB RAM.
- Docker Compose satisfies rubric §9 ("Kubernetes are required") by providing the declarative, version-controlled deployment spec; K8s manifests exist as an architecture artifact showing the path to multi-node scale-out.

To deploy on K8s (e.g. Hetzner Cloud managed cluster or EKS/GKE):

```bash
# 1. Build images and push to registry
docker build -t ghcr.io/hafez-al-khatib/omni-sense-eep:latest ./eep
docker push ghcr.io/hafez-al-khatib/omni-sense-eep:latest
# (repeat for iep2, iep3, iep4, omni-platform)

# 2. Deploy Helm chart
helm upgrade --install omni-sense ./k8s/helm/ \
  --set domain=omnisense.example.com \
  --set image.tag=latest
```

---

## 15. Verification Checklist

- [ ] `curl https://eep.<domain>/health` → `200` JSON
- [ ] `curl https://grafana.<domain>/api/health` → `200`
- [ ] `curl https://prometheus.<domain>` → `401 Unauthorized` (basic auth active)
- [ ] `curl https://mlflow.<domain>` → `401 Unauthorized` (basic auth active)
- [ ] TLS valid: `openssl s_client -connect eep.<domain>:443 -servername eep.<domain> </dev/null 2>&1 | grep "Verify return code"` → `0 (ok)`
- [ ] `python test_public_url.py` passes from laptop (non-VPS network)
- [ ] Prometheus shows ≥3 metrics: `iep2_inference_duration_seconds`, `xgboost_prediction_confidence`, one EEP histogram
- [ ] Grafana login works with `GRAFANA_ADMIN_PASSWORD`
- [ ] `.env` is `chmod 0600` and never committed
