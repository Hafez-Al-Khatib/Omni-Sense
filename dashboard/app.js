// ── Omni-Sense Operations Console v2 ──
// Polling-based with live-api backend. MQTT WebSocket as bonus.

const CFG = {
  apiUrl: (() => {
    const h = location.hostname;
    if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8080';
    const domain = h.includes('.') ? h.split('.').slice(1).join('.') : h;
    return `https://api.${domain}`;
  })(),
  mqttUrl: (() => {
    const h = location.hostname;
    if (h === 'localhost' || h === '127.0.0.1') return 'ws://localhost:9001/mqtt';
    const domain = h.includes('.') ? h.split('.').slice(1).join('.') : h;
    return `wss://mqtt.${domain}/mqtt`;
  })(),
  eepUrl: (() => {
    const h = location.hostname;
    if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8000';
    const domain = h.includes('.') ? h.split('.').slice(1).join('.') : h;
    return `https://eep.${domain}`;
  })(),
  pollInterval: 2000,
  maxHistory: 100,
};

const S = {
  client: null, wsConnected: false, wsAttempted: false,
  frames: 0, samples: [], inferHist: [], ticketHist: [],
  lastRms: 0, lastSnr: 0,
  sensorMeta: { id:'esp32-s3-01', site:'demo-site', lat:33.8938, lng:35.5018 },
  sensors: {}, // sensor_id -> { lastSeen, online, lat, lng, marker, verdict, site }
};
const SENSOR_TIMEOUT_MS = 30000; // mark offline after 30s of no data

const $  = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const fmtTime = ts => ts ? new Date(ts).toLocaleTimeString() : '--';
const fmtDate = ts => ts ? new Date(ts).toLocaleString() : '--';
const tagCls = v => v==='HEALTHY'?'ok': v==='LEAK'?'crit': v==='CRACK'?'warn':'unknown';

/* ── UI helpers ── */
function setMqttStatus(cls, text) {
  const dot = $('#mqttDot'), txt = $('#mqttText');
  if (dot) dot.className = 'status-dot ' + cls;
  if (txt) txt.textContent = text;
  const badge = $('#connModeBadge'), modeTxt = $('#connModeText'), proto = $('#connProto');
  if (badge) {
    badge.textContent = cls === 'online' ? 'ONLINE' : cls === 'connecting' ? 'CONNECTING' : 'OFFLINE';
    badge.className = 'conn-mode-badge ' + (cls === 'online' ? (S.wsConnected ? 'mode-mqtt' : 'mode-polling') : cls === 'connecting' ? 'mode-polling' : 'mode-none');
  }
  if (modeTxt) modeTxt.textContent = text;
  if (proto) proto.textContent = S.wsConnected ? 'MQTT WebSocket' : 'HTTP Polling';
}
function setBridgeStatus(cls, text) {
  const dot = $('#bridgeDot'), txt = $('#bridgeText');
  if (dot) dot.className = 'status-dot ' + cls;
  if (txt) txt.textContent = text;
}
function toast(msg, type='info') {
  const banner = $('#debugBanner'), txt = $('#debugText');
  if (!banner || !txt) return;
  banner.style.display = 'block';
  banner.style.background = type==='error'?'#fee2e2': type==='ok'?'#dcfce7':'#e0f7fa';
  banner.style.borderColor = type==='error'?'#fecaca': type==='ok'?'#bbf7d0':'#b2ebf2';
  banner.style.color = type==='error'?'#991b1b': type==='ok'?'#166534':'#0e7490';
  txt.textContent = msg;
  setTimeout(() => { banner.style.display = 'none'; }, 5000);
}

/* ── Clock ── */
setInterval(() => {
  const d = new Date();
  $('#clock').textContent = d.toISOString().slice(11,19) + ' UTC';
}, 1000);

/* ── Navigation ── */
const PAGE_TITLES = {
  ops: ['Operations Console', 'Live sensor mesh · leak detection · fleet health'],
  sensor: ['Sensor Telemetry', 'Configuration · calibration · signal quality'],
  diagnose: ['Diagnose Center', 'Manual analysis · real-time monitor · history'],
  dispatch: ['Dispatch Console', 'Alerts · tickets · maintenance scheduling'],
  mlops: ['MLOps Health', 'System metrics · model status · container fleet'],
};
$$('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const target = item.dataset.screen;
    $$('.nav-item').forEach(n => n.classList.remove('active'));
    item.classList.add('active');
    $$('.screen').forEach(s => s.classList.remove('active'));
    $('#screen-' + target).classList.add('active');
    const [t, st] = PAGE_TITLES[target] || PAGE_TITLES.ops;
    $('#pageTitle').textContent = t;
    $('#pageSubtitle').textContent = st;
  });
});

/* ── Geolocation helpers ── */
function loadSavedLocation() {
  try {
    const raw = localStorage.getItem('omni_sensor_location');
    if (raw) return JSON.parse(raw);
  } catch (e) {}
  return null;
}
function saveLocation(lat, lng) {
  try {
    localStorage.setItem('omni_sensor_location', JSON.stringify({ lat, lng, savedAt: Date.now() }));
  } catch (e) {}
  S.sensorMeta.lat = lat;
  S.sensorMeta.lng = lng;
}

async function detectLocation() {
  const saved = loadSavedLocation();
  if (saved) {
    S.sensorMeta.lat = saved.lat;
    S.sensorMeta.lng = saved.lng;
    return;
  }

  // 1. Try browser geolocation (most accurate)
  if (navigator.geolocation) {
    try {
      const pos = await new Promise((res, rej) => navigator.geolocation.getCurrentPosition(res, rej, { timeout: 8000 }));
      S.sensorMeta.lat = pos.coords.latitude;
      S.sensorMeta.lng = pos.coords.longitude;
      saveLocation(S.sensorMeta.lat, S.sensorMeta.lng);
      console.log('[geo] browser location:', S.sensorMeta.lat, S.sensorMeta.lng);
      return;
    } catch (e) {
      console.log('[geo] browser denied/unavailable:', e.message);
    }
  }

  // 2. Fallback: IP geolocation (no API key needed)
  try {
    const resp = await fetch('https://ip-api.com/json/?fields=lat,lon,status');
    const data = await resp.json();
    if (data.status === 'success') {
      S.sensorMeta.lat = data.lat;
      S.sensorMeta.lng = data.lon;
      saveLocation(S.sensorMeta.lat, S.sensorMeta.lng);
      console.log('[geo] IP location:', S.sensorMeta.lat, S.sensorMeta.lng);
      return;
    }
  } catch (e) {
    console.log('[geo] IP lookup failed:', e.message);
  }

  // 3. Hardcoded default (Beirut)
  console.log('[geo] using default location');
}

/* ── Leaflet Map ── */
const map = L.map('map', { zoomControl: false }).setView([33.8938, 35.5018], 14);
L.control.zoom({ position: 'bottomright' }).addTo(map);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
  subdomains: 'abcd', maxZoom: 19
}).addTo(map);
function makeSensorIcon(online) {
  const color = online ? '#0dc9d0' : '#94a3b8';
  const glow = online ? 'rgba(13,201,208,0.25)' : 'rgba(148,163,184,0.15)';
  return L.divIcon({
    className: '',
    html: `<div style="width:14px;height:14px;background:${color};border:2px solid #fff;border-radius:50%;box-shadow:0 0 0 4px ${glow};transition:background 0.3s;"></div>`,
    iconSize: [14,14], iconAnchor: [7,7]
  });
}

function updateSensorOnMap(sensorId, data) {
  const now = Date.now();
  let s = S.sensors[sensorId];
  if (!s) {
    s = {
      lastSeen: now,
      online: true,
      lat: data.lat || S.sensorMeta.lat,
      lng: data.lng || S.sensorMeta.lng,
      marker: null,
      verdict: data.verdict || 'UNKNOWN',
      site: data.site || S.sensorMeta.site,
    };
    s.marker = L.marker([s.lat, s.lng], { icon: makeSensorIcon(true), draggable: true }).addTo(map).bindPopup('');
    s.marker.on('dragend', (e) => {
      const ll = e.target.getLatLng();
      s.lat = ll.lat; s.lng = ll.lng;
      saveLocation(ll.lat, ll.lng);
      toast(`Sensor location updated: ${ll.lat.toFixed(4)}, ${ll.lng.toFixed(4)}`, 'ok');
    });
    s.marker.openPopup();
    S.sensors[sensorId] = s;
  }
  s.lastSeen = now;
  s.online = true;
  s.verdict = data.verdict || s.verdict;
  s.site = data.site || s.site;
  if (data.lat && data.lng) {
    s.lat = data.lat; s.lng = data.lng;
    s.marker.setLatLng([s.lat, s.lng]);
  }
  s.marker.setIcon(makeSensorIcon(true));
  s.marker.setPopupContent(`<b>${sensorId}</b><br>${s.site}<br><span style="color:#0dc9d0">&#9679; ONLINE</span> · ${s.verdict}`);
  updateSensorKPIs();
  updateSensorScreen(sensorId);
}

function updateSensorScreen(sensorId) {
  const s = S.sensors[sensorId];
  if (!s) return;
  // Update static sensor config rows if they exist
  const uptimeEl = $('#sensorUptime');
  if (uptimeEl) uptimeEl.textContent = 'Active now';
}

function markSensorOffline(sensorId) {
  const s = S.sensors[sensorId];
  if (!s || !s.online) return;
  s.online = false;
  s.marker.setIcon(makeSensorIcon(false));
  s.marker.setPopupContent(`<b>${sensorId}</b><br>${s.site}<br><span style="color:#94a3b8">&#9679; OFFLINE</span> · last ${fmtTime(s.lastSeen)}`);
  updateSensorKPIs();
  const uptimeEl = $('#sensorUptime');
  if (uptimeEl) uptimeEl.textContent = 'Disconnected';
}

function checkSensorTimeouts() {
  const now = Date.now();
  for (const [sid, s] of Object.entries(S.sensors)) {
    if (s.online && now - s.lastSeen > SENSOR_TIMEOUT_MS) {
      markSensorOffline(sid);
    }
  }
}

function updateSensorKPIs() {
  const total = Object.keys(S.sensors).length;
  const online = Object.values(S.sensors).filter(s => s.online).length;
  const kpiEl = $('#kpiSensors');
  const unitEl = kpiEl?.nextElementSibling;
  if (kpiEl) kpiEl.textContent = online;
  if (unitEl) unitEl.textContent = `/${total} ONLINE`;

  const mapTag = $('#mapStatusTag');
  if (mapTag) {
    if (online === 0 && total > 0) {
      mapTag.textContent = 'OFFLINE';
      mapTag.className = 'tag unknown';
    } else if (online < total) {
      mapTag.textContent = 'DEGRADED';
      mapTag.className = 'tag warn';
    } else if (total > 0) {
      mapTag.textContent = 'LIVE';
      mapTag.className = 'tag ok';
    } else {
      mapTag.textContent = 'NO DATA';
      mapTag.className = 'tag unknown';
    }
  }
}

/* ── Waveform Canvas ── */
function drawWaveform(canvasId, samples, rms) {
  const cvs = $(canvasId);
  if (!cvs) return;
  const w = cvs.width = cvs.clientWidth || cvs.width;
  const h = cvs.height = cvs.clientHeight || cvs.height;
  const ctx = cvs.getContext('2d');
  ctx.fillStyle = '#0b1f2e'; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  for(let i=1;i<5;i++){ctx.beginPath();ctx.moveTo(0,(h/5)*i);ctx.lineTo(w,(h/5)*i);ctx.stroke();}
  for(let i=1;i<8;i++){ctx.beginPath();ctx.moveTo((w/8)*i,0);ctx.lineTo((w/8)*i,h);ctx.stroke();}
  if(!samples||samples.length<2){
    ctx.fillStyle='rgba(255,255,255,0.3)';ctx.font='13px "IBM Plex Mono"';
    ctx.fillText(S.wsConnected ? 'Waiting for accelerometer frames...' : 'Live waveform available via MQTT WebSocket',20,h/2);return;
  }
  const step=Math.max(1,Math.floor(samples.length/800));
  const maxAmp=Math.max(1,...samples.map(Math.abs));
  ctx.strokeStyle='rgba(13,201,208,0.3)';ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  ctx.strokeStyle='#0dc9d0';ctx.lineWidth=1.5;ctx.beginPath();
  for(let i=0,px=0;i<samples.length&&px<w;i+=step,px++){
    const y=(h/2)-((samples[i]/maxAmp)*(h/2-4));
    if(i===0)ctx.moveTo(px,y);else ctx.lineTo(px,y);
  }
  ctx.stroke();
}

/* ── Inference Rendering ── */
function renderInference(inf) {
  const isStale = inf.stale;
  const displayVerdict = (inf.verdict || '--') + (isStale ? ' (STALE)' : '');
  const vEl=$('#verdictValue');
  if(vEl){vEl.textContent=displayVerdict;vEl.className='verdict-value '+(isStale?'unknown':tagCls(inf.verdict));}
  const tEl=$('#infTag');
  if(tEl){tEl.textContent=displayVerdict;tEl.className='tag '+(isStale?'unknown':tagCls(inf.verdict));}
  const bars=$('#probBars');
  if(bars){bars.innerHTML='';
    const labels={HEALTHY:'Healthy',LEAK:'Leak',CRACK:'Crack',UNKNOWN:'Unknown'};
    for(const[k,v]of Object.entries(inf.probs||{})){
      const pct=Math.round((v||0)*100);
      bars.insertAdjacentHTML('beforeend',`<div class="prob-row"><div class="prob-label">${labels[k]||k}</div><div class="prob-track"><div class="prob-fill ${tagCls(k)}" style="width:${pct}%"></div><span class="prob-val">${pct}%</span></div></div>`);
    }
  }
  const cEl=$('#infConf');if(cEl)cEl.textContent=inf.confidence?((inf.confidence*100).toFixed(1)+'%'):'--';
  const lEl=$('#infLatency');if(lEl)lEl.textContent=inf.latency_ms?Math.round(inf.latency_ms):'--';
  const tmEl=$('#infTime');
  if(tmEl){
    if(isStale){
      const ageSec = Math.round((Date.now() - inf.ts)/1000);
      tmEl.textContent = fmtTime(inf.ts) + ' (' + ageSec + 's ago)';
    } else {
      tmEl.textContent = fmtTime(inf.ts||inf.timestamp);
    }
  }

  const alertPill=$('#kpiAlertPill'),alertVal=$('#kpiAlerts');
  const alerts=S.inferHist.filter(x=>x.verdict!=='HEALTHY').length;
  if(alertPill&&alertVal){if(alerts>0){alertPill.style.display='flex';alertVal.textContent=alerts;}else{alertPill.style.display='none';}}

  updateSensorQuality(inf.features);
  updateHistoryTables();
  updateDispatch(inf);
}

function updateSensorQuality(f) {
  if(!f)return;
  const set=(id,val)=>{const el=$(id);if(el)el.textContent=val;};
  set('#sqMean',f.mean!==undefined?f.mean.toFixed(4)+' g':'--');
  set('#sqRms',f.rms!==undefined?f.rms.toFixed(4)+' g':'--');
  set('#sqPeak',f.peak!==undefined?f.peak.toFixed(4)+' g':'--');
  set('#sqKurt',f.kurtosis!==undefined?f.kurtosis.toFixed(2):'--');
  set('#sqCrest',f.crest_factor!==undefined?f.crest_factor.toFixed(2):'--');
  set('#sqSpec',f.spectral_ratio!==undefined?f.spectral_ratio.toFixed(3):'--');
  set('#sqZcr',f.zcr!==undefined?f.zcr.toFixed(4):'--');
  set('#sqSnr',f.snr!==undefined?f.snr.toFixed(1)+' dB':'--');
  const tEl=$('#sigQualTag');
  if(tEl&&f.rms!==undefined){const ok=f.rms<0.15&&f.kurtosis<5;tEl.textContent=ok?'GOOD':'DEGRADED';tEl.className='tag '+(ok?'ok':'warn');}
}

function updateHistoryTables() {
  const tb=$('#historyTableBody');
  if(tb&&S.inferHist.length>0){
    tb.innerHTML=S.inferHist.slice(0,20).map(inf=>{
      const f=inf.features||{};
      return `<tr><td>${fmtTime(inf.ts)}</td><td>${inf.source||'vibration'}</td><td class="v-${inf.verdict}">${inf.verdict}</td><td>${((inf.confidence||0)*100).toFixed(1)}%</td><td>${Math.round(inf.latency_ms||0)} ms</td><td>RMS:${(f.rms||0).toFixed(3)} K:${(f.kurtosis||0).toFixed(1)}</td></tr>`;
    }).join('');
  }
}

function updateDispatch(inf) {
  if(!inf||inf.verdict==='HEALTHY')return;
  const panel=$('#alertsPanel');
  const countTag=$('#alertCountTag');
  const alerts=S.inferHist.filter(x=>x.verdict!=='HEALTHY');
  if(countTag){countTag.textContent=alerts.length+' OPEN';countTag.className='tag '+(alerts.length?'crit':'ok');}
  if(panel){
    if(alerts.length===0){
      panel.innerHTML=`<div class="alert-card"><div class="alert-icon ok">&#10003;</div><div class="alert-body"><div class="alert-title">All Clear</div><div class="alert-desc">No active alerts. System operating normally.</div></div></div>`;
    }else{
      panel.innerHTML=alerts.slice(0,5).map(a=>`<div class="alert-card"><div class="alert-icon ${tagCls(a.verdict)}">!</div><div class="alert-body"><div class="alert-title">${a.verdict} detected on ${a.sensor_id}</div><div class="alert-desc">Confidence ${((a.confidence||0)*100).toFixed(1)}% at ${fmtTime(a.ts)}</div></div></div>`).join('');
    }
  }
  ['#btnDispatch','#btnSilence','#btnAck'].forEach(id=>{const b=$(id);if(b)b.disabled=alerts.length===0;});
  const atb=$('#alertTableBody');
  if(atb&&alerts.length>0){
    atb.innerHTML=alerts.slice(0,20).map(a=>`<tr><td>${fmtTime(a.ts)}</td><td>${a.sensor_id}</td><td class="v-${a.verdict}">${a.verdict==='LEAK'?'CRITICAL':'WARNING'}</td><td class="v-${a.verdict}">${a.verdict}</td><td>${((a.confidence||0)*100).toFixed(1)}%</td><td><span class="tag crit">OPEN</span></td><td>--</td></tr>`).join('');
  }
}

/* ── Decode base64 int16 LE accelerometer frame ── */
function decodeAccelFrame(b64Payload) {
  try {
    const binary = atob(b64Payload);
    const n = binary.length / 2;
    const samples = new Int16Array(n);
    const view = new DataView(new ArrayBuffer(2));
    for (let i = 0; i < n; i++) {
      view.setUint8(0, binary.charCodeAt(i * 2));
      view.setUint8(1, binary.charCodeAt(i * 2 + 1));
      samples[i] = view.getInt16(0, true); // little-endian
    }
    S.samples = Array.from(samples);
    S.frames++;

    // Update signal meta
    const rms = Math.sqrt(samples.reduce((a, v) => a + v * v, 0) / n);
    const mean = samples.reduce((a, v) => a + v, 0) / n;
    const peak = Math.max(...samples.map(Math.abs));
    const snr = 20 * Math.log10(peak / (rms + 1e-9));
    S.lastRms = rms;

    const rmsEl = $('#signalRms');
    const snrEl = $('#signalSnr');
    const frmEl = $('#signalFrames');
    if (rmsEl) rmsEl.textContent = (rms / 16384).toFixed(4) + ' g';
    if (snrEl) snrEl.textContent = snr.toFixed(1) + ' dB';
    if (frmEl) frmEl.textContent = S.frames;
  } catch (e) {
    console.error('[accel] decode error:', e);
  }
}

/* ── MQTT (bonus, not required) ── */
function tryMQTT() {
  if(S.wsAttempted||typeof window.mqtt==='undefined')return;
  S.wsAttempted=true;
  setMqttStatus('connecting','Trying WS...');
  try{
    const client=window.mqtt.connect(CFG.mqttUrl,{keepalive:30,reconnectPeriod:8000,connectTimeout:12000,clean:true,clientId:'dash-'+Math.random().toString(36).slice(2,6)});
    S.client=client;
    client.on('connect',()=>{S.wsConnected=true;setMqttStatus('online','WS Live');client.subscribe(['sensors/+/accel','sensors/+/result']);});
    client.on('message',(topic,payload)=>{
      if(topic.includes('/result')){
        ingestResult(JSON.parse(payload));
      } else if(topic.includes('/accel')){
        decodeAccelFrame(payload);
      }
    });
    client.on('error',()=>setMqttStatus('offline','WS Failed'));
    client.on('offline',()=>{S.wsConnected=false;setMqttStatus('offline','WS Off');});
  }catch(e){setMqttStatus('offline','WS Error');}
}

/* ── HTTP Polling (primary data source) ── */
async function api(path, opts={}) {
  const url = CFG.apiUrl + path;
  try {
    const resp = await fetch(url, { ...opts, headers: { 'Content-Type': 'application/json', ...opts.headers } });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    return await resp.json();
  } catch (e) {
    console.error('[api] ' + path + ':', e.message);
    throw e;
  }
}

function ingestResult(data) {
  if (!data || !data.verdict) return;
  const sensorId = data.sensor_id || 'unknown';
  const dataTs = data.ts || Date.now();
  const now = Date.now();
  const isStale = now - dataTs > SENSOR_TIMEOUT_MS;

  const inf = { ts: dataTs, verdict: data.verdict, probs: data.probs || {}, confidence: data.confidence || 0, latency_ms: data.latency_ms || 0, features: data.features || {}, sensor_id: sensorId, source: data.source || 'vibration', stale: isStale };
  S.inferHist.unshift(inf);
  if (S.inferHist.length > CFG.maxHistory) S.inferHist.pop();
  renderInference(inf);

  if (isStale) {
    setBridgeStatus('offline', 'Stale data');
    // Do NOT refresh lastSeen — let checkSensorTimeouts mark it offline
  } else {
    setBridgeStatus('online', 'Active');
    updateSensorOnMap(sensorId, data);
  }

  const cnt = $('#infCount'); if (cnt) cnt.textContent = S.inferHist.length;
  const avg = S.inferHist.reduce((a, b) => a + (b.latency_ms || 0), 0) / S.inferHist.length;
  const avgEl = $('#avgLatency'); if (avgEl) avgEl.textContent = Math.round(avg) + ' ms';
}

async function pollLive() {
  try {
    const data = await api('/live');
    if (data) {
      if (data.verdict && data.verdict !== 'WAITING') {
        ingestResult(data);
      }
      if (!S.wsConnected) setMqttStatus('online', 'Polling');
    }
  } catch (e) { /* silent on poll errors */ }
}

async function pollMetrics() {
  try {
    const m = await api('/metrics');
    const cnt = $('#infCount'); if (cnt) cnt.textContent = m.total_inferences || 0;
    const avgEl = $('#avgLatency'); if (avgEl) avgEl.textContent = Math.round(m.avg_latency_ms || 0) + ' ms';
    const ood = $('#oodRate'); if (ood) ood.textContent = m.alert_count ? ((m.alert_count / (m.total_inferences || 1)) * 100).toFixed(1) + '%' : '0.0%';
  } catch (e) { }
}

async function loadTickets() {
  try {
    const tickets = await api('/tickets');
    S.ticketHist = tickets;
    const atb = $('#alertTableBody');
    if (atb && tickets.length > 0) {
      atb.innerHTML = tickets.map(t => `<tr><td>${fmtDate(t.created_at)}</td><td>${t.sensor_id}</td><td class="v-${t.verdict}">${t.severity}</td><td class="v-${t.verdict}">${t.verdict}</td><td>${((t.confidence || 0) * 100).toFixed(1)}%</td><td><span class="tag ${t.status === 'OPEN' ? 'crit' : 'ok'}">${t.status}</span></td><td>${t.false_alarm ? 'False Alarm' : (t.resolution || '--')}</td></tr>`).join('');
    }
    const openCount = tickets.filter(t => t.status === 'OPEN').length;
    const countTag = $('#alertCountTag');
    if (countTag) { countTag.textContent = openCount + ' OPEN'; countTag.className = 'tag ' + (openCount ? 'crit' : 'ok'); }
  } catch (e) { }
}

/* ── Dispatch Actions ── */
$('#btnDispatch').addEventListener('click', async () => {
  const inf = S.inferHist[0];
  if (!inf) return;
  try {
    const res = await api('/tickets', {
      method: 'POST', body: JSON.stringify({
        sensor_id: inf.sensor_id, verdict: inf.verdict, confidence: inf.confidence,
        severity: inf.verdict === 'LEAK' ? 'CRITICAL' : 'WARNING', notes: 'Dispatched from dashboard'
      })
    });
    toast('Ticket ' + res.ticket_id.slice(0, 8) + ' created', 'ok');
    loadTickets();
  } catch (e) { toast('Failed to create ticket: ' + e.message, 'error'); }
});

$('#btnSilence').addEventListener('click', async () => {
  toast('Alert silenced for 30 min', 'ok');
});

$('#btnAck').addEventListener('click', async () => {
  toast('Alert acknowledged', 'ok');
});

/* ── False Alarm Report ── */
async function reportFalseAlarm(correctVerdict) {
  const inf = S.inferHist[0];
  if (!inf) return;
  try {
    await api('/feedback', {
      method: 'POST', body: JSON.stringify({
        sensor_id: inf.sensor_id, verdict: inf.verdict, confidence: inf.confidence,
        false_alarm: true, correct_verdict: correctVerdict, notes: 'Reported from dashboard'
      })
    });
    toast('False alarm recorded for continuous learning', 'ok');
  } catch (e) { toast('Failed to record feedback: ' + e.message, 'error'); }
}

/* ── Data Export ── */
$('#exportBtn')?.addEventListener('click', () => {
  window.open(CFG.apiUrl + '/export/csv', '_blank');
});

/* ── Upload Diagnosis ── */
const uploadZone = $('#uploadZone'), wavInput = $('#wavInput');
if (uploadZone && wavInput) {
  uploadZone.addEventListener('click', () => wavInput.click());
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); const f = e.dataTransfer.files[0]; if (f) processWav(f); });
  wavInput.addEventListener('change', e => { const f = e.target.files[0]; if (f) processWav(f); });
}

async function processWav(file) {
  const resDiv = $('#uploadResult');
  if (resDiv) resDiv.style.display = 'block';
  const vEl = $('#uploadVerdictValue');
  if (vEl) { vEl.textContent = 'Analyzing...'; vEl.className = 'verdict-value'; }
  try {
    const form = new FormData();
    form.append('audio', file);
    form.append('metadata', JSON.stringify({ pipe_material: 'PVC', pressure_bar: 3.0, source: 'manual_upload' }));
    const t0 = performance.now();
    const resp = await fetch(CFG.eepUrl + '/api/v1/diagnose', { method: 'POST', body: form });
    const latency = Math.round(performance.now() - t0);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    const verdict = data.diagnosis?.label || data.label || 'UNKNOWN';
    const probs = data.diagnosis?.probabilities || data.probabilities || {};
    const conf = data.diagnosis?.confidence || data.confidence || 0;
    if (vEl) { vEl.textContent = verdict; vEl.className = 'verdict-value ' + tagCls(verdict); }
    const bars = $('#uploadProbBars');
    if (bars) {
      bars.innerHTML = '';
      const labels = { HEALTHY: 'Healthy', LEAK: 'Leak', CRACK: 'Crack', UNKNOWN: 'Unknown' };
      for (const [k, v] of Object.entries(probs)) { const pct = Math.round((v || 0) * 100); bars.insertAdjacentHTML('beforeend', `<div class="prob-row"><div class="prob-label">${labels[k] || k}</div><div class="prob-track"><div class="prob-fill ${tagCls(k)}" style="width:${pct}%"></div><span class="prob-val">${pct}%</span></div></div>`); }
    }
    $('#uploadConf').textContent = (conf * 100).toFixed(1) + '%';
    $('#uploadLatency').textContent = latency + ' ms';
  } catch (e) {
    if (vEl) { vEl.textContent = 'Error: ' + e.message; vEl.className = 'verdict-value crit'; }
  }
}

/* ── System Health Polling ── */
async function pollHealth() {
  try {
    const resp = await fetch(CFG.eepUrl + '/health', { method: 'GET', cache: 'no-store' });
    const eepTag = $('#eepHealth');
    if (eepTag) { eepTag.textContent = resp.ok ? 'HEALTHY' : 'DEGRADED'; eepTag.className = 'tag ' + (resp.ok ? 'ok' : 'crit'); }
  } catch (e) {
    const eepTag = $('#eepHealth');
    if (eepTag) { eepTag.textContent = 'DOWN'; eepTag.className = 'tag crit'; }
  }
}

function populateConnUrls() {
  const mqtt = $('#connMqtt'), poll = $('#connPoll'), api = $('#connApi');
  if (mqtt) mqtt.textContent = CFG.mqttUrl;
  if (poll) poll.textContent = CFG.apiUrl + '/live';
  if (api) api.textContent = CFG.apiUrl;
}

/* ── Init ── */
detectLocation().then(() => {
  map.setView([S.sensorMeta.lat, S.sensorMeta.lng], 14);
});
tryMQTT();
setInterval(pollLive, CFG.pollInterval);
setInterval(pollMetrics, 10000);
setInterval(loadTickets, 15000);
setInterval(pollHealth, 30000);
setInterval(checkSensorTimeouts, 5000);
pollLive(); pollMetrics(); loadTickets(); pollHealth();
populateConnUrls();

drawWaveform('#waveformCanvas', [], 0);
drawWaveform('#diagWaveformCanvas', [], 0);
setInterval(() => { drawWaveform('#waveformCanvas', S.samples, S.lastRms); drawWaveform('#diagWaveformCanvas', S.samples, S.lastRms); }, 250);

toast('Dashboard v2 loaded. Data source: HTTP polling.', 'ok');
