// Shared data, helpers, primitives for Omni-Sense prototype
// Real values pulled from the project's .md files:
// - Beirut neighborhoods: Hamra, Achrafieh, Verdun, Gemmayzeh, Ras Beirut, Mar Mikhael, Badaro, Ain el-Mreisseh
// - Topology × Fault: Branched/Looped × Circumferential Crack/Longitudinal Crack/Gasket Leak/Orifice Leak
// - Flow rates: 0.18 LPS, 0.47 LPS
// - Models: XGBoost (45%), RF (25%), CNN (25%), Iso.Forest (5%) → fusion; CNN-AE for OOD
// - Metrics: ROC-AUC 0.9907, F1 0.9879, threshold 0.952
// - SLA: CRIT 5min, HIGH 30min, MEDIUM 2h, LOW 8h, INFO 24h

const SENSORS = [
  { id: 'S-HAMRA-001',    site: 'Hamra · Bliss St',           lat: 33.8959, lng: 35.4789, status: 'crit',  battery: 87, snr: 22.1, last: '2s',  cert: 71 },
  { id: 'S-ACHR-002',     site: 'Achrafieh · Sassine',        lat: 33.8869, lng: 35.5192, status: 'warn',  battery: 64, snr: 18.4, last: '4s',  cert: 38 },
  { id: 'S-VERDUN-003',   site: 'Verdun · Dunes',             lat: 33.8794, lng: 35.4836, status: 'ok',    battery: 91, snr: 24.3, last: '1s',  cert: 82 },
  { id: 'S-GEMM-004',     site: 'Gemmayzeh · Gouraud',        lat: 33.8949, lng: 35.5147, status: 'ok',    battery: 79, snr: 21.0, last: '3s',  cert: 64 },
  { id: 'S-RASB-005',     site: 'Ras Beirut · Manara',        lat: 33.9011, lng: 35.4719, status: 'ok',    battery: 88, snr: 23.5, last: '2s',  cert: 76 },
  { id: 'S-MARM-006',     site: 'Mar Mikhael · Armenia',      lat: 33.8993, lng: 35.5215, status: 'ood',   battery: 72, snr: 12.8, last: '6s',  cert: 51 },
  { id: 'S-BADARO-007',   site: 'Badaro · Sami el Solh',      lat: 33.8723, lng: 35.5183, status: 'ok',    battery: 84, snr: 22.7, last: '2s',  cert: 88 },
  { id: 'S-AINM-008',     site: 'Ain el-Mreisseh · Corniche', lat: 33.9012, lng: 35.4878, status: 'off',   battery:  3, snr:  0.0, last: '14m', cert: 22 },
];

const ALERTS = [
  {
    id: 'WO-26041',
    severity: 'CRITICAL',
    sensor: 'S-HAMRA-001',
    site: 'Hamra · Bliss St',
    fault: 'Branched · Orifice Leak',
    flow: 0.47,
    confidence: 0.96,
    pressure_drop: 0.41,
    detected: '14:22:08',
    sla_remaining_pct: 0.32,
    sla_label: '01:42 / 05:00',
  },
  {
    id: 'WO-26039',
    severity: 'HIGH',
    sensor: 'S-ACHR-002',
    site: 'Achrafieh · Sassine',
    fault: 'Looped · Longitudinal Crack',
    flow: 0.18,
    confidence: 0.84,
    pressure_drop: 0.18,
    detected: '13:51:44',
    sla_remaining_pct: 0.55,
    sla_label: '13:42 / 30:00',
  },
  {
    id: 'WO-26037',
    severity: 'MEDIUM',
    sensor: 'S-MARM-006',
    site: 'Mar Mikhael · Armenia',
    fault: 'OOD QUARANTINE',
    flow: null,
    confidence: null,
    pressure_drop: null,
    detected: '13:18:02',
    sla_remaining_pct: 0.78,
    sla_label: '01:34 / 02:00',
    ood: true,
  },
  {
    id: 'WO-26031',
    severity: 'LOW',
    sensor: 'S-GEMM-004',
    site: 'Gemmayzeh · Gouraud',
    fault: 'Branched · Gasket Leak (suspect)',
    flow: 0.18,
    confidence: 0.61,
    pressure_drop: 0.07,
    detected: '11:04:19',
    sla_remaining_pct: 0.42,
    sla_label: '04:38 / 08:00',
  },
];

const FLEET_KPI = {
  sensors_online: 47,
  sensors_total: 52,
  active_alerts: 4,
  ood_today: 11,
  resolved_24h: 18,
  drift_score: 0.041,
  drift_threshold: 0.05,
  rocauc: 0.9907,
  f1: 0.9879,
  threshold: 0.952,
  uptime_pct: 99.74,
  p95_ms: 612,
};

// Tiny SVG helpers
const Sparkline = ({ data, w = 120, h = 28, color = 'var(--signal)', fill = true, dotted = false }) => {
  if (!data || !data.length) return null;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const step = w / (data.length - 1);
  const pts = data.map((v, i) => [i * step, h - ((v - min) / range) * (h - 4) - 2]);
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const fillD = fill ? `${d} L${w},${h} L0,${h} Z` : null;
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      {fill && <path d={fillD} fill={color} opacity="0.14" />}
      <path d={d} fill="none" stroke={color} strokeWidth="1.4" strokeDasharray={dotted ? '2 2' : '0'} />
    </svg>
  );
};

// Waveform (deterministic-ish, looks like vibration data)
const Waveform = ({ w = 380, h = 60, color = 'var(--signal)', density = 1, seed = 7 }) => {
  const N = 240 * density;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const t = i / N;
    // Pseudo-random "vibration": sum of sines + envelope
    const env = Math.sin(t * Math.PI) * 0.85 + 0.15;
    const s =
      Math.sin(t * 80 + seed) * 0.6 +
      Math.sin(t * 230 + seed * 1.7) * 0.3 +
      Math.sin(t * 720 + seed * 0.4) * 0.18 +
      (((Math.sin(i * 0.7 + seed) * 43758.5453) % 1 + 1) % 1 - 0.5) * 0.4;
    const y = h / 2 + s * env * (h / 2 - 2);
    pts.push([i * (w / (N - 1)), y]);
  }
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <line x1="0" y1={h / 2} x2={w} y2={h / 2} stroke="var(--line)" strokeDasharray="2 3" />
      <path d={d} stroke={color} strokeWidth="1" fill="none" opacity="0.95" />
    </svg>
  );
};

// Spectrogram heatmap (CSS gradient based — looks rich without canvas)
const Spectrogram = ({ w = 380, h = 90, intensity = 'normal' }) => {
  // Pre-baked gradient stripes — "leak" pattern shows mid-band hiss
  const cols = 60, rows = 18;
  const cells = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const t = x / cols;
      const f = y / rows;
      let v;
      if (intensity === 'leak') {
        // Strong mid-high band hiss
        v = Math.exp(-Math.pow((f - 0.55) * 3.2, 2)) * (0.55 + 0.45 * Math.sin(t * 9 + f * 3));
        v += 0.18 * Math.exp(-Math.pow((f - 0.78) * 4, 2));
      } else if (intensity === 'ood') {
        // broadband chaos
        v = 0.4 + 0.4 * Math.sin(t * 19 + f * 7) * Math.cos(t * 5 - f * 3);
      } else {
        // baseline murmur
        v = 0.16 * Math.exp(-Math.pow((f - 0.3) * 3, 2)) * (0.6 + 0.4 * Math.sin(t * 4));
      }
      v = Math.max(0, Math.min(1, v));
      cells.push({ x, y, v });
    }
  }
  return (
    <svg width={w} height={h} style={{ display: 'block', borderRadius: 3 }}>
      <rect x="0" y="0" width={w} height={h} fill="oklch(0.12 0.01 240)" />
      {cells.map((c, i) => {
        const cw = w / cols, ch = h / rows;
        const lightness = 0.18 + c.v * 0.55;
        const chroma = 0.05 + c.v * 0.13;
        const hue = intensity === 'ood' ? 25 : intensity === 'leak' ? 200 : 230;
        return (
          <rect key={i} x={c.x * cw} y={(rows - 1 - c.y) * ch} width={cw + 0.5} height={ch + 0.5}
                fill={`oklch(${lightness} ${chroma} ${hue})`} />
        );
      })}
    </svg>
  );
};

const fmtPct = (v) => (v * 100).toFixed(1) + '%';

// Re-export to window for cross-file use
Object.assign(window, {
  SENSORS, ALERTS, FLEET_KPI,
  Sparkline, Waveform, Spectrogram,
  fmtPct,
});
