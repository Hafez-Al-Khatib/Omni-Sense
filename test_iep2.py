#!/usr/bin/env python3
import requests, json, sys

url = "http://iep2:8002/diagnose"
features = [0.1]*39 + [0.0, 3.0]
payload = {
    "embedding": features,
    "pipe_material": "PVC",
    "pressure_bar": 3.0
}

try:
    resp = requests.post(url, json=payload, timeout=10)
    print("STATUS:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print("ERROR:", e)
