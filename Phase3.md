# Phase 3: Hardware-in-the-Loop & Adversarial Defense

**Objective**: Prove the system's robustness against real-world chaos, malicious actors, and physical deployment constraints.

## 1. The "Break My AI" Interactive Demo

To stand out during the poster session, we moved away from a passive screen presentation.

- The Strategy: We designed a Hardware-in-the-Loop demonstration. By attaching a sensor to a physical PVC pipe proxy, we invite judges to physically strike the pipe with a wrench.

- The Outcome: Instead of predicting "Leak," the system's Isolation Forest instantly calculates the Mahalanobis distance, recognizes the Out-of-Distribution (OOD) kinetic energy, and throws the 422 Safety Exception on the Grafana dashboard. This proves our MLOps failure behavior is airtight.

## 2. Defense Against Acoustic Spoofing

- The Threat: A judge asking: "What if I play a YouTube video of a leak next to the sensor?"

- The Defense: We formulated a two-tiered rebuttal:

- Physics: The piezoelectric sensor suffers an impedance mismatch with airborne audio (speakers move air, leaks vibrate solid metal).

- Sensor Fusion: IEP 2 (XGBoost) requires the acoustic prediction to match the physical SCADA metadata (pressure drops). If the system hears a massive burst but pressure remains stable, it flags a synthetic mismatch.

## 3. Edge Power Constraints

The Operational Reality: To prove enterprise readiness, we established that physical edge deployment utilizes magnetic valve clamps and operates on a "Quiet Window" schedule (waking only between 2:00 AM and 4:00 AM). This maximizes the Signal-to-Noise ratio while ensuring a 5-year battery life, demonstrating deep domain knowledge of industrial IoT.