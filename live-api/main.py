"""Omni-Sense Live API"""
import os, json, threading, time, csv, io, uuid
from datetime import datetime, timezone

import psycopg2, paho.mqtt.client as mqtt
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

DB_DSN = os.environ.get("DATABASE_URL", "postgresql://omni:omni@timescaledb:5432/omnisense")
MQTT_HOST = os.environ.get("MQTT_HOST", "mqtt-broker")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

app = FastAPI(title="Omni-Sense Live API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_latest = {}
_history = []
_lock = threading.Lock()

def db_conn():
    return psycopg2.connect(DB_DSN)

def init_db():
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vibration_results (
                    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    sensor_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    probs JSONB,
                    features JSONB,
                    latency_ms DOUBLE PRECISION,
                    window_samples INTEGER,
                    source TEXT
                );
                SELECT create_hypertable('vibration_results', 'time',
                    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
                CREATE INDEX IF NOT EXISTS idx_vib_sensor_time ON vibration_results (sensor_id, time DESC);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    sensor_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    severity TEXT NOT NULL DEFAULT 'MEDIUM',
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    notes TEXT DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    resolved_at TIMESTAMPTZ,
                    resolution TEXT DEFAULT '',
                    false_alarm BOOLEAN DEFAULT FALSE
                );
                CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets (status, created_at DESC);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    sensor_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    false_alarm BOOLEAN NOT NULL DEFAULT FALSE,
                    correct_verdict TEXT,
                    notes TEXT DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
        conn.commit()

def on_connect(c, u, f, rc):
    c.subscribe("sensors/+/result")

def on_message(c, u, msg):
    try:
        data = json.loads(msg.payload)
        with _lock:
            _latest.clear(); _latest.update(data)
            _history.insert(0, dict(data))
            if len(_history) > 500: _history.pop()
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vibration_results (time, sensor_id, verdict, confidence, probs, features, latency_ms, window_samples, source)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                """, (data.get("sensor_id","unknown"), data.get("verdict","UNKNOWN"), data.get("confidence",0),
                      json.dumps(data.get("probs",{})), json.dumps(data.get("features",{})), data.get("latency_ms",0),
                      data.get("window_samples",0), data.get("source","vibration")))
            conn.commit()
        if data.get("verdict") != "HEALTHY":
            with db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO tickets (sensor_id, verdict, confidence, severity, status, notes)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (data.get("sensor_id","unknown"), data.get("verdict"), data.get("confidence",0),
                          "CRITICAL" if data.get("verdict")=="LEAK" else "WARNING", "OPEN",
                          f"Auto from {data.get('source','vibration')}"))
                conn.commit()
    except Exception as e:
        print(f"[mqtt] error: {e}")

def mqtt_loop():
    while True:
        try:
            c = mqtt.Client()
            c.on_connect = on_connect; c.on_message = on_message
            c.connect(MQTT_HOST, MQTT_PORT, 60)
            c.loop_forever()
        except Exception as e:
            print(f"[mqtt] conn error: {e}")
            time.sleep(5)

@app.get("/health")
def health():
    try:
        with db_conn() as conn:
            with conn.cursor() as cur: cur.execute("SELECT 1")
        db_ok = True
    except: db_ok = False
    return {"status":"ok","db_connected":db_ok,"latest_result":bool(_latest)}

@app.get("/live")
def live():
    with _lock: return _latest if _latest else {"verdict":"WAITING","message":"No data yet"}

@app.get("/history")
def history(limit:int=Query(50,ge=1,le=500)):
    with _lock: return _history[:limit]

@app.post("/tickets")
def create_ticket(body:dict):
    tid=str(uuid.uuid4())
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tickets (id,sensor_id,verdict,confidence,severity,status,notes)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (tid, body.get("sensor_id","unknown"), body.get("verdict","UNKNOWN"),
                  body.get("confidence",0), body.get("severity","MEDIUM"), "OPEN", body.get("notes","")))
        conn.commit()
    return {"ticket_id":tid,"status":"OPEN","created_at":datetime.now(timezone.utc).isoformat()}

@app.get("/tickets")
def list_tickets(status:str=None):
    with db_conn() as conn:
        with conn.cursor() as cur:
            if status:
                cur.execute("""
                    SELECT id,sensor_id,verdict,confidence,severity,status,notes,created_at,resolved_at,false_alarm
                    FROM tickets WHERE status=%s ORDER BY created_at DESC LIMIT 200
                """, (status,))
            else:
                cur.execute("""
                    SELECT id,sensor_id,verdict,confidence,severity,status,notes,created_at,resolved_at,false_alarm
                    FROM tickets ORDER BY created_at DESC LIMIT 200
                """)
            rows=cur.fetchall(); cols=[d[0] for d in cur.description]
            return [{cols[i]:(v.isoformat() if isinstance(v,datetime) else v) for i,v in enumerate(r)} for r in rows]

@app.post("/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id:str, body:dict):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE tickets SET status='RESOLVED', resolved_at=NOW(), resolution=%s, false_alarm=%s WHERE id=%s
            """, (body.get("resolution",""), body.get("false_alarm",False), ticket_id))
        conn.commit()
    return {"ticket_id":ticket_id,"status":"RESOLVED"}

@app.post("/feedback")
def feedback(body:dict):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback (sensor_id,verdict,confidence,false_alarm,correct_verdict,notes)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (body.get("sensor_id","unknown"), body.get("verdict","UNKNOWN"), body.get("confidence",0),
                  body.get("false_alarm",False), body.get("correct_verdict"), body.get("notes","")))
        conn.commit()
    return {"status":"recorded"}

@app.get("/metrics")
def metrics():
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM vibration_results"); total=cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM vibration_results WHERE verdict!='HEALTHY'"); alerts=cur.fetchone()[0]
            cur.execute("SELECT AVG(latency_ms) FROM vibration_results WHERE time>NOW()-INTERVAL'1 hour'")
            avg_lat=cur.fetchone()[0] or 0
            cur.execute("SELECT COUNT(*) FROM tickets WHERE status='OPEN'")
            open_tix=cur.fetchone()[0]
    return {"total_inferences":total,"alert_count":alerts,"avg_latency_ms":round(avg_lat,1),"open_tickets":open_tix}

@app.get("/export/csv")
def export_csv():
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT time,sensor_id,verdict,confidence,probs,features,latency_ms,window_samples,source
                FROM vibration_results ORDER BY time DESC LIMIT 10000
            """)
            rows=cur.fetchall()
    out=io.StringIO(); w=csv.writer(out)
    w.writerow(["time","sensor_id","verdict","confidence","probs_json","features_json","latency_ms","window_samples","source"])
    for r in rows: w.writerow(r)
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=omni-sense-data.csv"})

@app.on_event("startup")
def startup():
    init_db()
    threading.Thread(target=mqtt_loop, daemon=True).start()
    print("[live-api] started")
