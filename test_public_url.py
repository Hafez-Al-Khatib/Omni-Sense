"""
Smoke test for Omni-Sense public deployment.
Usage:
    python test_public_url.py                    # local default (http://localhost:8000)
    python test_public_url.py https://my-api.com  # custom URL
    OMNI_URL=https://my-api.com python test_public_url.py
"""

import os, sys, json, requests

URL = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OMNI_URL", "http://localhost:8000").rstrip("/")

def _post(path: str, **kwargs):
    return requests.post(f"{URL}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)

def _get(path: str, **kwargs):
    return requests.get(f"{URL}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)

def test_health():
    print(f"\n→ GET {URL}/health")
    r = _get("/health", timeout=10)
    print(f"  Status: {r.status_code}")
    try:
        print(f"  Body: {r.json()}")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Raw: {r.text[:200]}")
    return r.status_code == 200

def test_diagnose(wav_path: str):
    print(f"\n→ POST {URL}/api/v1/diagnose  (audio={wav_path})")
    with open(wav_path, "rb") as f:
        files = {"audio": f}
        data = {"metadata": json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})}
        r = _post("/api/v1/diagnose", files=files, data=data, timeout=90)
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
    ok &= test_diagnose("Processed_audio_16k/Branched_Circumferential_Crack_BR_CC_0.18_LPS_A1.wav")
    ok &= test_diagnose("Processed_audio_16k/Branched_Gasket_Leak_BR_GL_0.18_LPS_A1.wav")

    print("\n" + "=" * 50)
    print("PASS" if ok else "FAIL")
    print("=" * 50)
    sys.exit(0 if ok else 1)
