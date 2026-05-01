import requests
r = requests.post('http://localhost:5000/api/2.0/mlflow/runs/search', json={'experiment_ids': ['1']})
data = r.json()
for run in data.get('runs', []):
    info = run['info']
    print(f"Run: {info['run_name']} | ID: {info['run_id'][:8]}")
