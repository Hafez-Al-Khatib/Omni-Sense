# 🔊 Omni-Sense

> **Out-of-Distribution Aware Acoustic Diagnostics Platform**
> for Urban Infrastructure (Water Leakage & Generator Diagnostics)

[![CI](https://github.com/<org>/omni-sense/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/omni-sense/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Problem

Lebanon's infrastructure suffers from **40–50% water loss** in municipal networks and frequent diesel generator failures. Traditional acoustic diagnostic hardware is prohibitively expensive and fails in noisy urban environments.

## Solution

Omni-Sense is a **cloud-native, microservices-based** acoustic diagnostic platform that:

1. **Extracts features** from cheap surface microphones using **YAMNet** (transfer learning)
2. **Detects Out-of-Distribution** environments via **Isolation Forest** (epistemic safety)
3. **Classifies infrastructure health** using **XGBoost** with calibrated probabilities
4. **Monitors itself** via Prometheus + Grafana with ML-specific drift signals

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Edge App   │────▶│  EEP (API    │────▶│  IEP 1 (YAMNet)  │
│  (Smartphone │     │  Gateway)    │     │  Embedding Svc   │
│   / Sensor)  │     │  FastAPI     │     │  TensorFlow Hub  │
└──────────────┘     └──────┬───────┘     └──────────────────┘
                            │                      │
                            │              1024-d embedding
                            │                      │
                            ▼                      ▼
                     ┌──────────────────────────────────┐
                     │  IEP 2 (Diagnostic Engine)       │
                     │  Stage 1: Isolation Forest (OOD) │
                     │  Stage 2: XGBoost (Classifier)   │
                     │  ONNX Runtime                    │
                     └──────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+

### Run Locally
```bash
docker-compose up --build
```

### API Usage
```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -F "audio=@sample.wav" \
  -F 'metadata={"pipe_material": "PVC", "pressure_bar": 3.0}'
```

### Monitoring
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Project Structure

```
omni-sense/
├── eep/          # External Endpoint (API Gateway)
├── iep1/         # Internal Endpoint 1 (YAMNet Embeddings)
├── iep2/         # Internal Endpoint 2 (Diagnostic Engine)
├── scripts/      # Data synthesis & model training
├── monitoring/   # Prometheus + Grafana configs
├── web-ui/       # Demo smartphone interface
├── data/         # Datasets (gitignored)
└── tests/        # Integration & E2E tests
```

## Engineering Tradeoffs

| Tradeoff | Choice | Justification |
|---|---|---|
| Latency vs Accuracy | 5s audio buffer | Infrastructure leaks are steady-state; temporal context > speed |
| Hardware vs Cloud | Cheap sensors + cloud inference | Edge ML hardware is costly; cloud is updatable |
| E2E DL vs Pipeline | Hybrid YAMNet + IF + XGBoost | Explicit OOD failure mode; modular debugging |

## Course

**EECE503N / EECE798N** — AI Engineering Capstone, American University of Beirut

## License

MIT
