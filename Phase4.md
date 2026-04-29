# Phase 4: Enterprise Scale & Digital Twin Simulation

**Objective**: Utilize the extended one-month timeline to elevate the MVP into a fully realized, automated utility triage product.

## 1. Hard Negatives (The MIMII Dataset Integration)

A toy model only knows "Leaks" vs. "Silence." A production model must survive normal plumbing operations.

- The Upgrade: We integrated the MIMII (Malfunctioning Industrial Machine Investigation) dataset. By feeding normal water pump and valve vibrations through YAMNet, we provide our XGBoost classifier with "Hard Negatives." The AI mathematically learns the difference between the chaotic hiss of a burst pipe and the rhythmic cavitation of a municipal water pump.

## 2. SCADA Metadata Generation (The WNTR Engine)

Instead of hardcoding dummy tabular data (pressure, flow rate) for our XGBoost model, we transitioned to physical simulation.

- The Upgrade: We adopted WNTR (Water Network Tool for Resilience), an EPA-backed Python library. We simulate a leak on a digital EPANET map to generate mathematically accurate hydraulic pressure drops. We then fuse this simulated SCADA telemetry with our acoustic CSV data, creating a true Digital Twin training environment.

## 3. The Automated Dispatch Webhook (Active Learning)

We closed the loop between the AI's classification and actual business utility.

- The Upgrade: We appended an Event-Driven Webhook microservice downstream of IEP 2. When a leak is confirmed with >90% probability, it generates a prioritized maintenance ticket.

- The MLOps Flywheel: The ticket includes [Confirm Leak] and [False Alarm] buttons for field technicians. Clicking these routes the ground-truth label directly back to our MLflow registry, creating an automated Active Learning pipeline for continuous model evolution.