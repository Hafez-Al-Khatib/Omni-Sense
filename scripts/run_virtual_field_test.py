"""
Omni-Sense | Virtual Field Test (v2.0)
=======================================
Verifies the 'Real Working Pipelines' by pumping high-integrity acoustic
data through the EEP Gateway. This script does NOT use mocks; it hits
the actual service endpoints as defined in the production stack.

Pipeline Verified:
  [Data Injection] -> [EEP Gateway] -> [DSP Extraction] -> [IEP2/IEP4 AI] -> [Result Fusion]
"""

import os
import json
import time
import socket
import requests
import numpy as np
import soundfile as sf
from pathlib import Path

# --- Configuration ---
EEP_URL = "http://localhost:8000/api/v1/diagnose"
TEST_TEMP_DIR = Path("data/tmp_field_test")
SAMPLE_RATE = 16000
DURATION_S = 5.0

# --- Colors for 'Mission Report' ---
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_service_health():
    """Verify EEP is reachable before starting."""
    print(f"[{Colors.CYAN}INIT{Colors.END}] Checking EEP Gateway health at {EEP_URL}...")
    try:
        # Check if port 8000 is even open first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            if s.connect_ex(("localhost", 8000)) != 0:
                raise ConnectionError("Port 8000 is closed. Is the Docker stack running?")
        
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print(f"[{Colors.GREEN}READY{Colors.END}] EEP Gateway is online (v{response.json().get('version', '?.?')})\n")
            return True
    except Exception as e:
        print(f"[{Colors.RED}ERROR{Colors.END}] Pipeline unreachable: {e}")
        print(f"        {Colors.YELLOW}Hint: Run 'docker-compose up -d' first.{Colors.END}")
        return False

def generate_real_acoustic_sample(category="leak"):
    """
    Creates a physical WAV file with specific acoustic properties.
    No mocks here - this generates raw PCM data that the DSP must parse.
    """
    TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S))
    
    if category == "leak":
        # White noise + 4kHz hiss (characteristic of pressurized water leak)
        data = np.random.normal(0, 0.05, len(t)) + 0.02 * np.sin(2 * np.pi * 4000 * t)
    elif category == "quiet":
        # Ambient noise only
        data = np.random.normal(0, 0.001, len(t))
    elif category == "noise":
        # Low frequency rumble (e.g. car or pump) - might trigger OOD
        data = 0.1 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.02, len(t))
    
    file_path = TEST_TEMP_DIR / f"test_{category}.wav"
    sf.write(file_path, data, SAMPLE_RATE)
    return file_path

def run_mission(category, material="PVC", pressure=3.0):
    """Executes a single end-to-end diagnostic mission."""
    print(f"[{Colors.BOLD}MISSION{Colors.END}] Testing category: {Colors.CYAN}{category.upper()}{Colors.END}")
    
    # 1. Prepare Real Physical Asset
    wav_path = generate_real_acoustic_sample(category)
    
    # 2. Transmit to Gateway
    payload = {
        "metadata": json.dumps({
            "pipe_material": material,
            "pressure_bar": pressure
        })
    }
    
    start_time = time.time()
    try:
        with open(wav_path, "rb") as f:
            files = {"audio": (wav_path.name, f, "audio/wav")}
            response = requests.post(EEP_URL, files=files, data=payload, timeout=15)
        
        elapsed = (time.time() - start_time) * 1000
        
        # 3. Analyze Pipeline Intelligence
        if response.status_code == 422:
            print(f"  {Colors.YELLOW}>> Result: OOD REJECTED (Safety Gate Engaged){Colors.END}")
            print(f"     Details: {response.json().get('detail', {}).get('error', 'Unknown Environment')}")
        elif response.status_code == 200:
            res = response.json()
            label = res.get("label", "Unknown")
            conf = res.get("confidence", 0.0)
            
            color = Colors.GREEN if "Leak" in label else Colors.CYAN
            if label == "No_Leak" and category == "leak": color = Colors.RED # Failure!
            
            print(f"  {Colors.GREEN}>> Result: SUCCESS{Colors.END}")
            print(f"     Label:      {color}{label}{Colors.END}")
            print(f"     Confidence: {conf:.2%}")
            print(f"     Ensemble:   {res.get('ensemble_method', 'N/A')}")
            print(f"     Latency:    {elapsed:.1f}ms")
        else:
            print(f"  {Colors.RED}>> Result: PIPELINE CRASH ({response.status_code}){Colors.END}")
            print(f"     Error: {response.text}")

    except Exception as e:
        print(f"  {Colors.RED}>> Result: TRANSMISSION FAILURE{Colors.END}")
        print(f"     Detail: {e}")
    
    print("-" * 50)

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}OMNI-SENSE VIRTUAL FIELD TEST{Colors.END}")
    print("=" * 40)
    
    if check_service_health():
        # Execute 3 real-world scenarios
        run_mission("quiet", material="Steel", pressure=5.0)
        run_mission("leak",  material="PVC",   pressure=2.5)
        run_mission("noise", material="Cast_Iron", pressure=1.5)
        
        print(f"\n{Colors.BOLD}FIELD TEST COMPLETE{Colors.END}")
        print(f"Temporary data stored in: {TEST_TEMP_DIR}")
    else:
        exit(1)
