"""Pytest configuration for omni test suite.

Relaxes per-head latency budgets for the test environment. The production
defaults (30/30/120/20/40 ms) are tuned for a Linux container with a warm
event loop; on a developer laptop or CI runner with cold-start asyncio
overhead, the heads can sporadically exceed those budgets and trigger
fallbacks, masking real logic bugs.

These overrides are applied at module-import time so they take effect
BEFORE omni.eep.orchestrator reads the env vars at its own module load.
"""
import os

# 10x production budgets — generous enough that test flakiness from event-loop
# stalls disappears, but still tight enough to catch a head that hangs.
os.environ.setdefault("HEAD_BUDGET_XGB_MS", "300")
os.environ.setdefault("HEAD_BUDGET_RF_MS",  "300")
os.environ.setdefault("HEAD_BUDGET_CNN_MS", "1200")
os.environ.setdefault("HEAD_BUDGET_IF_MS",  "200")
os.environ.setdefault("HEAD_BUDGET_OOD_MS", "400")

# Disable IEP4 HTTP call in tests — keeps head_cnn on the deterministic stub
# instead of waiting for a connection refused / DNS failure.
os.environ.setdefault("IEP4_URL", "")
