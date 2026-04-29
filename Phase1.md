# Phase 1: Architectural Alignment & Business Framing

**Objective**: Reposition the project from a generic "AI classification model" to a defensible, production-ready "Cyber-Physical MLOps Platform" to satisfy the rigorous EECE 503N engineering rubric.

## **1. The Core Realization: Escaping the "Demoware" Trap**

Initially, we feared being overshadowed by flashy LLM wrappers (e.g., "OS-Siri"). We realized that attempting to out-hype generative AI with our acoustic model was a losing battle.

- The Pivot: We shifted our defense strategy to highlight Systems Engineering. While LLMs suffer from unquantifiable hallucinations, we emphasize that our system operates in the physical world with strict mathematical safety boundaries (Epistemic Uncertainty Quantification).

**2. Defining the Baseline & Tradeoffs**

To satisfy Dr. Mohanna's strict rubric requirements, we formally defined the business case and the engineering constraints:

- The Non-AI Baseline: We established that the standard industry baseline—Amplitude Thresholding (>80dB)—fails catastrophically in a Lebanese urban environment due to passing trucks and generators triggering false positives.

- The Engineering Tradeoffs: We locked in three explicit tradeoffs:

- Architecture: Decoupled (YAMNet + XGBoost) vs. End-to-End CNN (saves compute, allows tabular metadata fusion).

- Bandwidth: Edge downsampling (25.6kHz to 16kHz client-side) vs. Edge Inference (saves payload size for unstable 3G/LTE networks).

- Latency: 5-second buffered context vs. Real-time streaming (prioritizes diagnostic accuracy over instantaneity).

## 3. Visual & Scope Cohesion

- The Fix: The initial pie chart included non-water anomalies (like Transformers), creating a scope disconnect. We narrowed the multi-class distribution strictly to water infrastructure (High-Pressure Leaks, Burst Mains, Pump Cavitation) to prove the pipeline's diagnostic depth without losing focus on the core business problem.