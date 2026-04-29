# Omni-Sense: 360° Evaluation & Defense Strategy

This document synthesizes feedback from Prof. Ammar, the Deep Code Review (Claude), and the Gemini CLI Agent to provide a roadmap for the final submission.

## 1. Executive Summary: The "Vibe Check"
*   **Gemini/Claude Consensus**: This is a top 5-10% project. The engineering depth (TDOA, RUL, PSI-Drift, mTLS) is far beyond a typical capstone.
*   **Prof. Ammar's Challenge**: He views it as a "solved problem" with "too many failure points." 
*   **The Critical Pivot**: Stop selling "Leak Detection." Start selling **"Reliability Engineering for Critical Infrastructure."**

## 2. Evaluation Matrix

| Pillar | Prof. Ammar (User) | Claude (Deep Review) | Gemini (CLI Agent) |
|---|---|---|---|
| **Novelty** | 🔴 Low (Solved problem) | 🟢 High (TDOA/OOD/RUL depth) | 🟢 High (Industrial Reliability) |
| **Architecture** | 🟡 Risk (Many failure points) | 🟢 Elite (Service-oriented/Durable) | 🟢 Elite (Production-hardened) |
| **MLOps** | N/A (Not yet seen) | 🟢 Strong (PSI Drift/Promotion) | 🟢 Strong (Regression Gates) |
| **Deployment** | N/A (Needs demo) | 🔴 CRITICAL (Localhost in prod) | 🟡 Risk (Cloud URL sync) |

## 3. The "Defense Pillars" (Addressing Prof. Ammar)

### Pillar 1: The "Nuisance" Filter (OOD)
*   **Challenge**: "Industry already solves this."
*   **Defense**: Industry solves it in *quiet* environments. In noisy cities (Beirut), commercial sensors suffer from high false-positives (generators/trucks). Omni-Sense uses **OOD Awareness** to reject these as "Unknown," a feature missing from most black-box commercial tools.

### Pillar 2: Cyber-Physical Fusion
*   **Challenge**: "Why so many services?"
*   **Defense**: To enable **multi-modal validation**. We fuse acoustic vibration with **SCADA pressure metadata**. This prevents "Acoustic Spoofing"—a known vulnerability in simpler systems.

### Pillar 3: Industrial-Grade Reliability
*   **Challenge**: "Too many failure points."
*   **Defense**: Complexity $\neq$ Fragility. We use **Redis Streams** for durability (handling 3G/LTE drops) and **mTLS** for security (standard for NERC-CIP compliance). These aren't "extra" parts; they are mandatory for **Mission-Critical Infrastructure**.

## 4. Critical "Landmines" ( Claude's "Demo Stoppers")

1.  **Cloud Config**: `.env.production` still points to `localhost`. If this isn't fixed, the cloud demo **will fail**.
2.  **Hardcoded Metrics**: `_evaluate_current_model()` in the retraining trigger returns static values. This is a "demo shortcut" that needs an honest explanation or a quick fix.
3.  **Committed Secrets**: Private keys are in the `certs/` folder. Acknowledge this as a "known trade-off for grader convenience" but note that in prod, you would use a Vault.
4.  **TDOA Validation**: The TDOA code is elite, but we need to ensure the "Wave Speed" tables are accurately mapped to Lebanon's common pipe materials (PVC/Cast Iron).

## 5. Summary Recommendation
If the **Cloud Config** and **Localhost** issues are fixed, Claude estimates a score of **87–90/100**. To push for **95+**, we must replace the "demo shortcuts" with real evidence and lean into the **Industrial Validation** framing.
