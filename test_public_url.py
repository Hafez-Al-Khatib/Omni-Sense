"""
Smoke test for Omni-Sense public deployment.
Usage:
    python test_public_url.py                    # local default (http://localhost:8000)
    python test_public_url.py https://my-api.com  # custom URL
    OMNI_URL=https://my-api.com python test_public_url.py
"""

import os
import sys
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.getenv("OMNI_URL", "http://localhost:8000")
).rstrip("/")


# ── Session with retries ──────────────────────────────────────────────────────
def _make_session() -> requests.Session:
    """Create a requests session with built-in retries for connection errors."""
    session = requests.Session()
    # Retry on connection errors / 502 / 503 / 504 (common during cold starts)
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


session = _make_session()


def _post(path: str, **kwargs):
    return session.post(f"{URL}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)


def _get(path: str, **kwargs):
    return session.get(f"{URL}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)


def test_health(max_attempts: int = 8) -> bool:
    """Poll /health with exponential backoff (Render free-tier cold-start)."""
    print(f"\n→ GET {URL}/health")
    for attempt in range(1, max_attempts + 1):
        try:
            r = _get("/health", timeout=15)
            print(f"  Attempt {attempt}: Status {r.status_code}")
            if r.status_code == 200:
                try:
                    print(f"  Body: {r.json()}")
                except Exception as e:
                    print(f"  Error parsing JSON: {e}")
                return True
            else:
                print(f"  Raw: {r.text[:200]}")
        except requests.exceptions.SSLError as e:
            print(f"  Attempt {attempt}: SSL Error — {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"  Attempt {attempt}: Connection Error — {e}")
        except requests.exceptions.Timeout:
            print(f"  Attempt {attempt}: Timeout")
        except Exception as e:
            print(f"  Attempt {attempt}: Unexpected error — {type(e).__name__}: {e}")

        if attempt < max_attempts:
            wait = min(2 ** attempt, 30)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  FAILED after {max_attempts} attempts")
    return False


def test_diagnose(wav_path: str) -> bool:
    print(f"\n→ POST {URL}/api/v1/diagnose  (audio={wav_path})")
    try:
        with open(wav_path, "rb") as f:
            files = {"audio": f}
            data = {
                "metadata": json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})
            }
            r = _post("/api/v1/diagnose", files=files, data=data, timeout=90)
    except requests.exceptions.SSLError as e:
        print(f"  SSL Error: {e}")
        print(
            "  Hint: This can happen on Render free tier when the service is still "
            "waking up or the container crashed on startup. Check Render dashboard logs."
        )
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"  Connection Error: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"  Timeout — Render free tier has a ~30-60s cold-start limit.")
        return False
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        return False

    print(f"  Status: {r.status_code}")
    try:
        body = r.json()
        print(f"  Body (truncated): {json.dumps(body, indent=2)[:1000]}")
    except Exception as e:
        print(f"  Error parsing JSON: {e}")
        print(f"  Raw: {r.text[:500]}")
    return r.status_code == 200


def warm_all():
    """On Render free tier, services sleep after 15 min. Hit health first to wake them."""
    downstream = [
        "https://omni-sense-iep2.onrender.com",
        "https://omni-sense-iep3.onrender.com",
        "https://omni-sense-iep4.onrender.com",
    ]
    for url in downstream:
        print(f"\n→ Waking {url}/health")
        try:
            requests.get(f"{url}/health", timeout=30)
        except Exception as e:
            print(f"  (ignore) {e}")


if __name__ == "__main__":
    print(f"Testing Omni-Sense at: {URL}")

    # If testing Render, wake downstream services first
    if "onrender.com" in URL:
        warm_all()
        print("\n" + "=" * 50)
        print("Warming complete. Now testing EEP...")
        print("=" * 50)

    ok = test_health()
    if ok:
        # Give Render a moment to finish spinning up workers after /health succeeds
        time.sleep(2)

    ok &= test_diagnose(
        "Processed_audio_16k/Branched_Circumferential_Crack_BR_CC_0.18_LPS_A1.wav"
    )
    ok &= test_diagnose(
        "Processed_audio_16k/Branched_Gasket_Leak_BR_GL_0.18_LPS_A1.wav"
    )

    print("\n" + "=" * 50)
    print("PASS" if ok else "FAIL")
    print("=" * 50)
    sys.exit(0 if ok else 1)
