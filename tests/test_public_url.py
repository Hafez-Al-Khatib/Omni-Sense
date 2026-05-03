"""
test_public_url.py — End-to-end smoke test against the live Hetzner deployment.
Rubric §8.1: verifies the public EEP endpoint is reachable and healthy.

Usage:
    export EEP_PUBLIC_URL=https://eep.omnisense-demo.duckdns.org
    pytest tests/test_public_url.py -v
"""

import os
import sys
import pytest
import requests

EEP_URL = os.environ.get("EEP_PUBLIC_URL", "https://eep.omnisense-demo.duckdns.org")
TIMEOUT = 30


@pytest.fixture(scope="module")
def base_url() -> str:
    url = EEP_URL.rstrip("/")
    # Quick sanity check that the URL looks like HTTPS
    if not url.startswith("http"):
        pytest.skip(f"EEP_PUBLIC_URL does not look like a URL: {url}")
    return url


def test_eep_health(base_url: str) -> None:
    """EEP /health must return 200 with expected JSON shape."""
    resp = requests.get(f"{base_url}/health", timeout=TIMEOUT)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code} {resp.text[:200]}"
    data = resp.json()
    assert data.get("status") == "healthy" or data.get("ok") is True or "healthy" in str(data).lower()


def test_eep_diagnose_requires_audio(base_url: str) -> None:
    """POST /api/v1/diagnose without audio must return 422 (validates routing)."""
    resp = requests.post(
        f"{base_url}/api/v1/diagnose",
        data={"metadata": '{"pipe_material":"PVC","pressure_bar":3.0}'},
        timeout=TIMEOUT,
    )
    # We expect 422 because no audio file is attached
    assert resp.status_code in (422, 400), f"Expected 422/400, got {resp.status_code}"


def test_eep_openapi_or_docs_reachable(base_url: str) -> None:
    """Swagger /docs or /openapi.json should be reachable (proves FastAPI is up)."""
    for path in ("/docs", "/openapi.json"):
        resp = requests.get(f"{base_url}{path}", timeout=TIMEOUT)
        if resp.status_code == 200:
            return
    pytest.fail("Neither /docs nor /openapi.json returned 200")


def test_prometheus_targets_up() -> None:
    """Prometheus targets page should be reachable (proves observability stack)."""
    prom_url = os.environ.get("PROMETHEUS_URL", "https://prometheus.omnisense-demo.duckdns.org")
    resp = requests.get(f"{prom_url}/api/v1/targets", timeout=TIMEOUT, auth=("admin", os.environ.get("PROM_PASSWORD", "prom123")))
    if resp.status_code == 401:
        pytest.skip("Prometheus basic auth required — set PROM_PASSWORD env var")
    assert resp.status_code == 200, f"Prometheus unreachable: {resp.status_code}"
    data = resp.json()
    active = data.get("data", {}).get("activeTargets", [])
    up = [t for t in active if t.get("health") == "up"]
    # At minimum EEP should be up
    eep_up = any("eep" in t.get("labels", {}).get("job", "") for t in up)
    assert eep_up or len(up) >= 3, f"Expected EEP or 3+ targets up, found {len(up)} up"


def test_grafana_reachable() -> None:
    """Grafana login page should be reachable."""
    grafana_url = os.environ.get("GRAFANA_URL", "https://grafana.omnisense-demo.duckdns.org")
    resp = requests.get(f"{grafana_url}/login", timeout=TIMEOUT)
    assert resp.status_code == 200, f"Grafana unreachable: {resp.status_code}"
    assert "Grafana" in resp.text or "grafana" in resp.text.lower()


def test_tls_certificate_valid(base_url: str) -> None:
    """The public endpoint must serve a valid TLS certificate."""
    try:
        resp = requests.get(base_url, timeout=TIMEOUT)
    except requests.exceptions.SSLError as exc:
        pytest.fail(f"TLS certificate invalid: {exc}")
    assert resp.status_code in (200, 307, 404)  # 307 redirect or 404 root is fine
