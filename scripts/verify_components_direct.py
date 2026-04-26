"""
Direct Component Verification (Bypassing Gateway)
=================================================
Verifies IEP4 (Deep CNN) and IEP3 (Dispatch) while the rest of the stack builds.
"""
import requests
import numpy as np
import soundfile as sf
import json
import time

def verify_iep4():
    print("[TEST] IEP4 (Deep CNN) ... ", end="", flush=True)
    # Generate sample
    sr = 16000
    t = np.linspace(0, 5, sr * 5)
    audio = np.random.normal(0, 0.1, len(t)).astype(np.float32)
    sf.write("temp_test.wav", audio, sr)
    
    try:
        with open("temp_test.wav", "rb") as f:
            resp = requests.post("http://localhost:8004/classify", files={"audio": f}, timeout=10)
        
        if resp.status_code == 200:
            print("PASS ✅")
            print(f"      Result: {resp.json().get('label')} ({resp.json().get('confidence'):.1%})")
        else:
            print(f"FAIL ❌ (Status {resp.status_code})")
            print(f"      Detail: {resp.text}")
    except Exception as e:
        print(f"ERROR ❌ ({e})")

def verify_iep3():
    print("[TEST] IEP3 (Dispatch) ... ", end="", flush=True)
    payload = {
        "label": "Leak",
        "confidence": 0.95,
        "probabilities": {"Leak": 0.95, "No_Leak": 0.05},
        "anomaly_score": 0.1,
        "pipe_material": "PVC",
        "pressure_bar": 3.0,
        "scada_mismatch": False
    }
    try:
        resp = requests.post("http://localhost:8003/api/v1/ticket", json=payload, timeout=5)
        if resp.status_code in (200, 201):
            print("PASS ✅")
            print(f"      Ticket ID: {resp.json().get('ticket_id')}")
        else:
            print(f"FAIL ❌ (Status {resp.status_code})")
    except Exception as e:
        print(f"ERROR ❌ ({e})")

if __name__ == "__main__":
    print("OMNI-SENSE COMPONENT AUDIT (DIRECT)")
    print("====================================")
    verify_iep4()
    verify_iep3()
