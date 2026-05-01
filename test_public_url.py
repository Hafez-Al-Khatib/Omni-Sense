import requests, json, sys

URL = "http://localhost:8000"

def test_health():
    r = requests.get(f"{URL}/health", timeout=10)
    print(f"Health: {r.status_code} - {r.json()}")

def test_diagnose(wav_path: str):
    with open(wav_path, "rb") as f:
        files = {"audio": f}
        data = {"metadata": json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})}
        r = requests.post(f"{URL}/api/v1/diagnose", files=files, data=data, timeout=60)
    print(f"Diagnose ({wav_path}): {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2)[:800])
    except Exception as e:
        print(f"Error parsing: {e}")
        print(r.text[:500])

if __name__ == "__main__":
    test_health()
    print()
    test_diagnose("Processed_audio_16k/Branched_Circumferential_Crack_BR_CC_0.18_LPS_A1.wav")
    print()
    test_diagnose("Processed_audio_16k/Branched_Gasket_Leak_BR_GL_0.18_LPS_A1.wav")
