# Cloud Deployment Guide — Omni-Sense

> **URGENT:** Localtunnel is unreliable and has died repeatedly. You need a permanent cloud deployment for your demo.

## Option 1: GCP Cloud Run (FASTEST — Scripts Exist)

**Time: 15 minutes**

```bash
# 1. Authenticate (opens browser)
gcloud auth login

# 2. Set your project
gcloud config set project YOUR_PROJECT_ID
export GCP_PROJECT_ID=YOUR_PROJECT_ID

# 3. Deploy everything
bash scripts/deploy-gcp.sh

# 4. Get URLs
gcloud run services list --region=us-central1
```

**Pros:** Serverless, free tier, HTTPS auto, scripts ready  
**Cons:** Need GCP project, auth was expired earlier

---

## Option 2: Render (EASIEST — Blueprint Ready)

**Time: 10 minutes**

1. Go to https://render.com and sign up (free)
2. Click "New" → "Blueprint"
3. Connect your GitHub repo
4. Render reads `render.yaml` and deploys all 4 services

**Pros:** Zero config, automatic deploys, free tier  
**Cons:** Free tier sleeps after 15 min inactivity (wakes on request, ~30s cold start)

---

## Option 3: AWS App Runner (Learned in Class)

**Time: 30 minutes**

```bash
# 1. Install AWS CLI and authenticate
aws configure

# 2. Run deploy script
bash scripts/deploy-aws.sh
```

**Pros:** You learned AWS in class, App Runner is serverless  
**Cons:** New script (may need debugging), need AWS account/credits

---

## Option 4: Railway

**Time: 10 minutes**

1. Go to https://railway.app and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Railway auto-detects Dockerfiles

**Pros:** Very easy, generous free tier  
**Cons:** Need account, less control than GCP/AWS

---

## My Recommendation

**If you have a GCP project → Use Option 1 (fastest)**
**If you don't have any cloud account → Use Option 2 (Render)**
**If you have AWS credits from class → Use Option 3**

---

## Pre-Deploy Checklist

- [ ] Choose cloud provider
- [ ] Set up account + auth
- [ ] Run deployment
- [ ] Test health endpoint: `curl https://YOUR_URL/health`
- [ ] Test diagnose endpoint with WAV file
- [ ] Screenshot the working URL for your submission

---

## Post-Deploy Smoke Test

```bash
# Health check
curl -sf https://YOUR_DEPLOYED_URL/health

# Diagnose leak
curl -X POST https://YOUR_DEPLOYED_URL/api/v1/diagnose \
  -F "audio=@Processed_audio_16k/Branched_Circumferential_Crack_BR_CC_0.18_LPS_A1.wav" \
  -F "metadata={\"pipe_material\":\"PVC\",\"pressure_bar\":3.0}"
```
