# Hetzner CX22 Deployment Walkthrough

> **Goal:** Go from zero to a publicly accessible Omni-Sense stack in ~20 minutes.
> **Cost:** ~€4.51/month (CX23: 2 vCPU, 4 GB RAM, 40 GB NVMe).
> **Time budget:** 20 min if using GHCR, 35 min if building locally.

---

## Prerequisites

| Item | What you need |
|------|---------------|
| Credit card | Hetzner requires one for verification (pre-paid works). |
| GitHub account | For GHCR images (fast path) and repo access. |
| DuckDNS account | Free dynamic DNS. Go to https://www.duckdns.org and sign in with Google/GitHub. |
| SSH key pair | Generate now if you don't have one: `ssh-keygen -t ed25519 -C "omni"` |

---

## Step 1: Create Hetzner Account & Server (5 min)

### 1.1 Sign up
1. Go to https://hetzner.com/cloud
2. Click **Sign Up** → verify email → add a project (e.g., "omni-sense").
3. Add payment method (credit card or PayPal).

### 1.2 Create the server
1. In the Hetzner Cloud Console, click **Add Server**.
2. **Location:** Choose the closest to you (e.g., `nbg1` for Germany, `hel1` for Finland).
3. **Image:** `Ubuntu 24.04` (or `Ubuntu 22.04`).
4. **Type:** `CX22` (Shared vCPU, 2 vCPU, 4 GB RAM, 40 GB NVMe).
5. **Networking:**
   - IPv4: Enable (€0.60/mo, required for public access).
   - IPv6: Optional.
6. **SSH Keys:** Paste your **public** key (`~/.ssh/id_ed25519.pub` on Linux/Mac, or the contents on Windows).
7. **Name:** `omni-sense-prod`
8. Click **Create & Buy now**.

> **Note:** The server will be ready in ~30 seconds. Copy the **IPv4 address** shown in the console.

---

## Step 2: Configure DuckDNS (2 min)

You need 5 subdomains pointing to your server IP.

1. Log in to https://www.duckdns.org
2. Create a domain (e.g., `omnisense-demo`).
3. For that domain, set the **IP** to your Hetzner server IPv4 address.
4. Click **Update IP**.

Your domains will be:
- `eep.omnisense-demo.duckdns.org`
- `grafana.omnisense-demo.duckdns.org`
- `prometheus.omnisense-demo.duckdns.org`
- `mlflow.omnisense-demo.duckdns.org`
- `mqtt.omnisense-demo.duckdns.org`

> **Note:** DuckDNS uses a wildcard `*.omnisense-demo.duckdns.org` automatically. One A record covers all subdomains.

---

## Step 3: Choose Your Deployment Mode

### Option A: GHCR (Fast — ~2 min after bootstrap)
**Best if:** Your CI has pushed images recently and you have a GitHub PAT.

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Scopes needed: `read:packages` (and `write:packages` only if you want to push from CI).
4. Copy the token — you will paste it into the bootstrap script.

> **Verify images exist:** Visit `https://github.com/YOUR_GITHUB_USERNAME/omni-sense/pkgs/container/omni-sense-eep`. If you see "Package not found", the CI hasn't pushed yet. Use **Option B** instead.

### Option B: Build on Server (Reliable — ~15 min after bootstrap)
**Best if:** No PAT, CI images are stale, or you want guaranteed latest code.

No preparation needed. The script will build all images on the CX22.

---

## Step 4: Run the Bootstrap Script (10–20 min)

### 4.1 SSH into the server

```bash
ssh root@YOUR_SERVER_IP
```

> **Windows users:** Use PowerShell, Git Bash, or Windows Terminal. If you didn't add an SSH key during server creation, Hetzner emailed you a root password.

### 4.2 Run the bootstrap

Copy-paste ONE of these commands:

**GHCR mode (fast):**
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/omni-sense/main/scripts/bootstrap-hetzner.sh)
```
The script will prompt for:
- DuckDNS domain (e.g., `omnisense-demo`)
- GitHub username
- GitHub PAT

**Build mode (reliable):**
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/omni-sense/main/scripts/bootstrap-hetzner.sh)
```
The script will prompt for:
- DuckDNS domain (e.g., `omnisense-demo`)
- Select mode `2` for BUILD

### 4.3 What the script does

1. Installs Docker + Docker Compose v2
2. Configures UFW firewall (ports 22, 80, 443, 1883, 8883)
3. Clones your repo to `/opt/omni-sense`
4. Generates strong random passwords
5. Creates Caddy bcrypt hashes for Prometheus/MLflow basic auth
6. Writes `.env.production`
7. Builds or pulls Docker images
8. Starts the stack
9. Prints all credentials and URLs

### 4.4 Save the credentials

The script prints credentials to the terminal AND saves them to `/root/omni-credentials.txt`. Copy them somewhere safe immediately.

---

## Step 5: Verify Deployment (2 min)

### 5.1 From your laptop

```bash
# Test the public API
OMNI_URL=https://eep.YOUR_DOMAIN.duckdns.org python test_public_url.py
```

Expected output:
```
[OK] GET  https://eep.omnisense-demo.duckdns.org/health  → 200
[OK] POST https://eep.omnisense-demo.duckdns.org/api/v1/diagnose  → 200
[OK] POST https://eep.omnisense-demo.duckdns.org/api/v1/diagnose  → 200
```

### 5.2 Check dashboards

Open these in your browser:

| Service | URL | Login |
|---------|-----|-------|
| API | `https://eep.YOUR_DOMAIN.duckdns.org/health` | None |
| Grafana | `https://grafana.YOUR_DOMAIN.duckdns.org` | admin / (from credentials) |
| Prometheus | `https://prometheus.YOUR_DOMAIN.duckdns.org` | admin / (from credentials) |
| MLflow | `https://mlflow.YOUR_DOMAIN.duckdns.org` | admin / (from credentials) |

### 5.3 Check container health on the server

```bash
ssh root@YOUR_SERVER_IP "cd /opt/omni-sense && docker compose ps"
```

All services should show `healthy` or `running`.

---

## Step 6: ESP32 Hardware Setup (Optional — do this AFTER cloud is live)

### 6.1 Flash the ESP32

Your ESP32 needs to:
1. Connect to your WiFi
2. Read MPU6050 (or ADXL345) accelerometer
3. Publish MQTT messages to your server

Edit `hardware/esp32/config.h`:
```cpp
#define WIFI_SSID "YourWiFi"
#define WIFI_PASSWORD "YourPassword"
#define MQTT_BROKER "mqtt.YOUR_DOMAIN.duckdns.org"
#define MQTT_PORT 1883
#define MQTT_USER "omni"
#define MQTT_PASSWORD "<from credentials file>"
```

Build and flash with PlatformIO or Arduino IDE.

### 6.2 Verify MQTT data flow

On the server:
```bash
ssh root@YOUR_SERVER_IP
mosquitto_sub -h localhost -t "sensors/+/accel" -u omni -P <MQTT_PASSWORD>
```

You should see JSON payloads arriving from the ESP32.

---

## Troubleshooting

### "Certificate not valid" / browser security warning

Let's Encrypt needs a few minutes to issue the first certificate. Wait 2–3 minutes and refresh.

If it persists after 5 minutes:
```bash
ssh root@YOUR_SERVER_IP
cd /opt/omni-sense
docker compose logs caddy | tail -30
```

Common causes:
- DuckDNS IP doesn't match server IP → update DuckDNS
- Port 80 blocked by firewall → bootstrap script opens it, but verify with `ufw status`
- Let's Encrypt rate limit (5 certs per domain per week) → use a different DuckDNS domain

### "Connection refused" on port 8000

Port 8000 is intentionally NOT exposed in production. All traffic goes through Caddy on 443. Use `https://eep.YOUR_DOMAIN.duckdns.org` not `http://YOUR_IP:8000`.

### IEP4 container keeps restarting (OOM)

The CNN model is memory-hungry. On CX23 with 4 GB, it can OOM if other services are also loading.

Fix: Add swap space:
```bash
ssh root@YOUR_SERVER_IP
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### GHCR "unauthorized" error

Your PAT is expired or missing `read:packages` scope. Generate a new one at https://github.com/settings/tokens.

### Bootstrap script fails halfway through

The script is idempotent for most steps. Just SSH in and re-run it:
```bash
ssh root@YOUR_SERVER_IP
cd /opt/omni-sense
bash scripts/bootstrap-hetzner.sh
```

It will detect the existing clone and `git pull` instead of re-cloning.

---

## Cost Breakdown

| Item | Monthly Cost |
|------|-------------|
| Hetzner CX22 | €3.79 |
| IPv4 address | €0.60 |
| DuckDNS | €0.00 |
| **Total** | **~€4.59** |

After the demo: delete the server from the Hetzner console to stop billing. You can always recreate it with the same bootstrap script.

---

## Emergency Fallback: ngrok

If Hetzner fails completely, expose your local Docker stack publicly:

```bash
# On your local machine (with Docker running)
ngrok http 8000
```

Copy the `https://XXXX.ngrok-free.app` URL and update `test_public_url.py` to use it. This satisfies the rubric's "public URL" requirement temporarily. Do this only as a last resort — Hetzner is preferred because it demonstrates real cloud deployment.

---

## Checklist for Demo Day

- [ ] Hetzner server created and running
- [ ] DuckDNS domain points to server IP
- [ ] `test_public_url.py` passes against `https://eep.YOUR_DOMAIN.duckdns.org`
- [ ] Grafana dashboard loads and shows metrics
- [ ] Prometheus targets are all UP
- [ ] (Optional) ESP32 publishes MQTT messages visible in `mosquitto_sub`
- [ ] (Optional) Screenshot of serial monitor showing ESP32 boot + sensor readings
