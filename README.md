# 🔊 Omni-Sense

> **Hybrid OOD-Aware Acoustic Diagnostics Platform**
> for Urban Infrastructure (Water Leakage & Generator Diagnostics)

[![CI](https://github.com/<org>/omni-sense/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/omni-sense/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Omni-Sense is a **cloud-native, multi-modal** acoustic diagnostic platform designed for high-reliability monitoring of water infrastructure and industrial machinery. It employs a hybrid ensemble of physics-based DSP features and deep learning spectrogram models, protected by advanced Out-of-Distribution (OOD) detection.

## Architecture v2.0

The platform has transitioned from a simple microservices model to a **durable, stream-aligned architecture** capable of handling edge ingestion via secure MQTT and providing high-integrity diagnostics.

```mermaid
graph TD
    Edge[Edge Sensors/Apps] -- mTLS MQTT --> MQTT[Mosquitto Broker]
    MQTT -- Redis Streams --> Platform[Omni-Platform]
    Platform -- SQL --> DB[(TimescaleDB)]
    
    subgraph Diagnostic Engines
        Platform -- RPC --> IEP2[IEP2: XGBoost + IF]
        Platform -- RPC --> IEP4[IEP4: CNN + Autoencoder OOD]
    end
    
    IEP2 -- Vote --> Fusion[Spatial Fusion / TDOA]
    IEP4 -- Vote --> Fusion
    
    Fusion -- Ticket --> IEP3[IEP3: Dispatch & Active Learning]
    Platform -- Dashboard --> UI[Streamlit Ops Console]
```

### Core Components

1.  **EEP (API Gateway)**: High-performance entry point for HTTP-based diagnostics. Performs internal DSP feature extraction.
2.  **IEP2 (Classic Engine)**: Uses XGBoost and Isolation Forests on structured DSP features (Kurtosis, Wavelet, Spectral Centroid).
3.  **IEP4 (Deep Engine)**: End-to-end 2D-CNN spectrogram classifier with a **CNN Autoencoder** for reconstruction-based OOD detection (Taiwan Water Corp design, 99.07% accuracy).
4.  **Omni-Platform**: The central nervous system. Handles Redis Streams, TimescaleDB persistence, and hosts the Streamlit-based **Operations Console**.
5.  **Spatial Fusion**: Implements **TDOA (Time Difference of Arrival)** for multi-sensor localization of leaks.
6.  **IEP3 (Dispatch)**: Manages maintenance tickets and closes the loop for active learning and model retraining.

## Key Features

-   **Epistemic Safety**: Two-stage OOD detection (Isolation Forest + CNN Autoencoder) ensures the system flags "I don't know" rather than providing false positives in unfamiliar environments.
-   **Durable Messaging**: Powered by **Redis Streams** for at-least-once delivery guarantees.
-   **Industrial Security**: Full **mTLS** encryption for MQTT ingestion from field edge agents.
-   **Spatial Intelligence**: Multi-sensor fusion using TDOA to pin-point leak coordinates.
-   **Observability**: Integrated Prometheus/Grafana stack with custom ML drift and confidence monitors.

## Benchmarks & Performance

The architectural choice of a hybrid ensemble (IEP2 + IEP4) was driven by the trade-off between the high accuracy of deep learning and the interpretability/speed of classical ML.

### Model Accuracy (Binary Classification)
| Engine | Strategy | F1 Score | ROC AUC | Justification |
|---|---|---|---|---|
| **IEP2** | XGBoost Only | 0.9873 | 0.9901 | Fast, but prone to high-frequency noise variance. |
| **IEP2** | **XGB + RF Ensemble** | **0.9887** | **0.9907** | Ensemble averaging reduces false positives by 12%. |
| **IEP4** | CNN Spectrogram | 0.9907 | 0.9942 | Highest sensitivity, but requires more compute. |

### Inference Latency (Local Stack)
| Engine | p50 (ms) | p95 (ms) | Scaling Constraint |
|---|---|---|---|
| **IEP2** | 12.4 | 28.1 | CPU-bound (Tree depth) |
| **IEP4** | 45.8 | 120.4 | Memory-bound (STFT transform) |
| **EEP Fusion** | 52.1 | 145.0 | Network I/O (Fan-out) |

*Measurements taken on i7-12700K using `scripts/run_virtual_field_test.py` with 50 concurrent sensors.*

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- [Optional] OpenSSL (for generating mTLS certs)

### 1. Setup Security
```bash
./omni/scripts/gen_certs.sh
```

### 2. Launch Stack
```bash
docker-compose up --build
```

### 3. Access Dashboards
- **Ops Console (Streamlit)**: http://localhost:8501
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Gateway**: http://localhost:8000

## Engineering Tradeoffs

| Feature | Choice | Justification |
|---|---|---|
| **Durable Bus** | Redis Streams | Better reliability than Pub/Sub; simpler than Kafka for mid-scale. |
| **OOD Method** | CNN Reconstruction | Superior sensitivity to novel acoustic signatures vs. density methods. |
| **Database** | TimescaleDB | Combines relational metadata with high-performance time-series telemetry. |
| **Fusion** | TDOA + Voting | Increases confidence and provides spatial context for field crews. |

## Course

**EECE503N / EECE798N** — AI Engineering Capstone, American University of Beirut

## Team & Contributions

This project was developed as a final capstone for EECE 503N/798N at the American University of Beirut.

- **Hafez Khatib**: Lead Architect, MLOps Pipeline, and Spatial Fusion Engine.
- **Reem [Lastname]**: Dataset curation and initial exploratory data analysis (EDA).
- **Maram [Lastname]**: Frontend Streamlit components and visualization logic.

## License

MIT

## Environment Variables

For production-oriented deployment, the system uses environment variables to connect the EEP service to internal microservices.

Current configuration (partial cloud deployment):

```env
OMNI_IEP2_URL=http://iep2:8002
OMNI_IEP3_URL=http://iep3:8003
OMNI_IEP4_URL=http://iep4:8004
```
