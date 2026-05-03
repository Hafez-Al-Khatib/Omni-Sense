// screen-diagnose.jsx — Production-aesthetic Diagnose screen
// Matches app.py's flagship feature: upload WAV → analyze → verdict + scores.
// Sky-blue, Orbitron/Rajdhani, animated pipe hero in scope.

const DiagnoseScreen = () => {
  const [stage, setStage] = React.useState('idle'); // idle | uploaded | analyzing | done
  const [verdict, setVerdict] = React.useState(null); // 'leak' | 'clean' | 'ood'
  const [progress, setProgress] = React.useState(0);
  const fileInputRef = React.useRef(null);

  // Three preset audio "files" — feels real, deterministic results
  const PRESETS = [
    { id: 'leak',  name: 'hamra_bliss_14h22.wav',     dur: 12.4, sr: 16000, kind: 'leak',
      iep2: 0.892, iep4: 0.847, ens: 0.876, ood: 0.143, snr: 22.1, conf: 0.96, phys: 0.94 },
    { id: 'clean', name: 'verdun_dunes_baseline.wav', dur: 30.0, sr: 16000, kind: 'clean',
      iep2: 0.142, iep4: 0.198, ens: 0.167, ood: 0.089, snr: 24.3, conf: 0.93, phys: 0.97 },
    { id: 'ood',   name: 'marm_armenia_unknown.wav',  dur: 8.7,  sr: 16000, kind: 'ood',
      iep2: 0.612, iep4: 0.588, ens: 0.601, ood: 0.71,  snr: 12.8, conf: null, phys: 0.41 },
  ];
  const [picked, setPicked] = React.useState(PRESETS[0]);

  // Use refs so a re-render or StrictMode double-invoke can't lose state.
  const pickedRef = React.useRef(picked);
  React.useEffect(() => { pickedRef.current = picked; }, [picked]);
  const rafRef = React.useRef(null);
  const doneTimerRef = React.useRef(null);
  const runIdRef = React.useRef(0);

  const cancelTimer = () => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (doneTimerRef.current) { clearTimeout(doneTimerRef.current); doneTimerRef.current = null; }
  };
  const onPick = (p) => {
    cancelTimer();
    setPicked(p); setStage('uploaded'); setVerdict(null); setProgress(0);
  };
  const reset = () => {
    cancelTimer();
    runIdRef.current++; // invalidate any in-flight run
    setStage('idle'); setVerdict(null); setProgress(0);
  };
  const analyze = () => {
    cancelTimer();
    const myRun = ++runIdRef.current;
    setProgress(0); setVerdict(null); setStage('analyzing');
    const TOTAL_MS = 2400;
    const t0 = performance.now();

    // 1) Deterministic completion: a single setTimeout sets 'done', regardless of RAF.
    doneTimerRef.current = setTimeout(() => {
      if (myRun !== runIdRef.current) return;
      doneTimerRef.current = null;
      setProgress(100);
      setVerdict(pickedRef.current.kind);
      setStage('done');
    }, TOTAL_MS);

    // 2) Progress animation: opportunistic; if RAF is throttled, the bar still
    //    advances via the setTimeout fallback below. Either way, completion is
    //    handled by (1) and does not depend on this loop.
    const tick = () => {
      if (myRun !== runIdRef.current) return;
      const elapsed = performance.now() - t0;
      const pct = Math.min(99, (elapsed / TOTAL_MS) * 100);
      setProgress(pct);
      if (pct < 99) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
  };
  React.useEffect(() => () => cancelTimer(), []);

  return (
    <div className="os-art" style={{ overflow: 'auto', padding: 24 }}>
      {/* HERO */}
      <div className="hero" style={{ marginBottom: 18 }}>
        <div className="hero-inner">
          <div className="hero-logo">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none">
              <circle cx="19" cy="19" r="14" stroke="#1a8fe3" strokeWidth="1.5" strokeDasharray="5 2.5"/>
              <path d="M9 19 Q14 9,19 19 Q24 29,29 19" stroke="#1a8fe3" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
              <circle cx="19" cy="19" r="3.5" fill="#1a8fe3" opacity=".95"/>
              <circle cx="19" cy="19" r="7" fill="none" stroke="#1a8fe3" strokeWidth=".7" opacity=".35"/>
            </svg>
          </div>
          <div>
            <div className="hero-title">DIAGNOSE</div>
            <div className="hero-sub">UPLOAD · ANALYZE · DECIDE — FAST &amp; AUDITABLE</div>
            <div className="hero-badge">
              <div className="hero-bdot"></div>
              EEP ONLINE · CLOUD RUN us-central1 · p95 612 ms
            </div>
          </div>
          <div className="hero-stats">
            <div className="hstat"><div className="hstat-v">0.9907</div><div className="hstat-l">ROC-AUC</div></div>
            <div className="hstat"><div className="hstat-v">0.9879</div><div className="hstat-l">F1</div></div>
            <div className="hstat"><div className="hstat-v">0.952</div><div className="hstat-l">θ</div></div>
          </div>
        </div>
        <div className="pipe-scene">
          <div className="pipe-body"></div>
          <div className="pipe-cap-l"></div>
          <div className="pipe-cap-r"></div>
          <div className="drop drop1"></div><div className="drop drop1b"></div>
          <div className="drop drop2"></div><div className="drop drop2b"></div>
          <div className="drop drop3"></div><div className="drop drop3b"></div>
          <div className="drop drop4"></div>
          <div className="splash splash1"></div><div className="splash splash2"></div>
          <div className="splash splash3"></div><div className="splash splash4"></div>
          <div className="pipe-label">MUNICIPAL WATER NETWORK · SENSOR ARRAY</div>
        </div>
      </div>

      {/* TWO-COLUMN: upload + file info */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16, marginBottom: 16 }}>
        <div className="panel">
          <div className="panel-h"><span>Acoustic Source</span><span className="lbl">.WAV · 16 KHZ · MONO</span></div>
          <div className="panel-b">
            <DropZone onPick={() => onPick(PRESETS[0])} stage={stage} picked={picked} />
            <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div className="lbl" style={{ fontSize: 10 }}>Or pick a captured signal:</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {PRESETS.map(p => (
                  <button key={p.id}
                    onClick={() => onPick(p)}
                    className="btn"
                    style={{
                      borderColor: picked.id === p.id ? '#1a8fe3' : undefined,
                      background: picked.id === p.id ? 'rgba(26,143,227,0.08)' : undefined,
                      fontFamily: "'IBM Plex Mono', monospace", fontSize: 11
                    }}>
                    {p.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-h"><span>File Information</span><span className="lbl">METADATA</span></div>
          <div className="panel-b" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            <Metric label="SAMPLE RATE" value={`${picked.sr}`} unit="Hz" />
            <Metric label="DURATION"    value={picked.dur.toFixed(2)} unit="s" />
            <Metric label="CHANNELS"    value="1" unit="mono" />
            <Metric label="ENCODING"    value="PCM" unit="16-bit" full />
            <Metric label="SOURCE"      value={picked.id === 'leak' ? 'S-HAMRA-001' : picked.id === 'clean' ? 'S-VERDUN-003' : 'S-MARM-006'} unit="" full />
          </div>
        </div>
      </div>

      {/* WAVEFORM + SPECTROGRAM */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div className="panel">
          <div className="panel-h"><span>Waveform</span><span className="lbl">ACCELEROMETER · LIVE</span></div>
          <div className="panel-b">
            <ScrollingWaveform kind={picked.kind} />
          </div>
        </div>
        <div className="panel">
          <div className="panel-h"><span>Log-Power Spectrogram</span><span className="lbl">STFT · 1024 · 512 HOP</span></div>
          <div className="panel-b">
            <SpectrogramLight kind={picked.kind} />
          </div>
        </div>
      </div>

      {/* ANALYZE BUTTON / PROGRESS */}
      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-b" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button
            className="btn primary"
            onClick={analyze}
            disabled={stage === 'analyzing'}
            style={{ flex: 1, padding: '14px 20px', fontFamily: "'Orbitron', sans-serif",
                     fontSize: 14, letterSpacing: 3, fontWeight: 700 }}>
            {stage === 'analyzing' ? 'ANALYZING…' : '⚡ ANALYZE WITH EEP'}
          </button>
          {stage !== 'idle' && (
            <button className="btn" onClick={reset} style={{ padding: '12px 16px' }}>↻ Reset</button>
          )}
        </div>
        {stage === 'analyzing' && (
          <div style={{ padding: '0 16px 16px' }}>
            <ProgressBar value={progress} />
            <div style={{ display: 'flex', gap: 14, marginTop: 8, flexWrap: 'wrap' }}>
              <PipelineStep label="IEP1 · Features"    active={progress > 5}  done={progress > 28} />
              <PipelineStep label="IEP2 · XGBoost"     active={progress > 28} done={progress > 50} />
              <PipelineStep label="IEP4 · RF + CNN"    active={progress > 50} done={progress > 72} />
              <PipelineStep label="OOD · CNN-AE"       active={progress > 72} done={progress > 88} />
              <PipelineStep label="IEP3 · Aggregate"   active={progress > 88} done={progress >= 100} />
            </div>
          </div>
        )}
      </div>

      {/* RESULTS */}
      {stage === 'done' && verdict && (
        <ResultsBlock picked={picked} verdict={verdict} />
      )}
    </div>
  );
};

// === Sub-components ===

const DropZone = ({ onPick, stage, picked }) => {
  const isUploaded = stage !== 'idle';
  return (
    <div
      onClick={onPick}
      style={{
        border: `2px dashed ${isUploaded ? '#1a8fe3' : '#8ecbf3'}`,
        borderRadius: 10,
        padding: '24px 20px',
        background: isUploaded ? 'rgba(26,143,227,0.06)' : 'rgba(255,255,255,0.5)',
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'all 0.18s ease',
      }}>
      <div style={{ fontSize: 32, marginBottom: 6 }}>{isUploaded ? '✓' : '📁'}</div>
      <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 13, fontWeight: 600,
                    color: '#1a8fe3', letterSpacing: 2, marginBottom: 4 }}>
        {isUploaded ? 'FILE LOADED' : 'DROP WAV FILE HERE'}
      </div>
      <div style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: 12, color: '#3a7ab0' }}>
        {isUploaded ? picked.name : 'or click to browse · 16-bit PCM · 30 s max'}
      </div>
    </div>
  );
};

const Metric = ({ label, value, unit, full }) => (
  <div style={{
    background: 'rgba(255,255,255,0.6)',
    border: '1px solid #cee2f3',
    borderRadius: 8,
    padding: '10px 12px',
    gridColumn: full ? '1 / -1' : 'auto',
  }}>
    <div className="lbl" style={{ fontSize: 9, marginBottom: 4 }}>{label}</div>
    <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 16, fontWeight: 700, color: '#0e75c8' }}>
      {value} <span style={{ fontSize: 10, color: '#3a7ab0', fontWeight: 400 }}>{unit}</span>
    </div>
  </div>
);

const ScrollingWaveform = ({ kind }) => {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    let t = 0;
    let raf;
    const isLeak = kind === 'leak';
    const isOod = kind === 'ood';
    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      // gridlines
      ctx.strokeStyle = 'rgba(91,184,245,0.25)'; ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        const yy = (h / 4) * i;
        ctx.moveTo(0, yy); ctx.lineTo(w, yy); ctx.stroke();
      }
      // waveform path
      ctx.beginPath();
      const N = 280;
      for (let i = 0; i < N; i++) {
        const x = (i / N) * w;
        const tt = i / N + t * 0.3;
        const env = Math.sin((i / N) * Math.PI) * 0.85 + 0.25;
        let s;
        if (isLeak) {
          s = Math.sin(tt * 70) * 0.5
            + Math.sin(tt * 230 + 1.7) * 0.35
            + Math.sin(tt * 720) * 0.22
            + (Math.sin(i * 0.91 + t) * 0.5 - 0.25) * 0.55;
        } else if (isOod) {
          s = Math.sin(tt * 19) * 0.7
            + Math.sin(tt * 5 - t) * 0.5
            + (Math.sin(i * 1.7 + t * 2) * 0.5) * 0.8;
        } else {
          s = Math.sin(tt * 40) * 0.18
            + Math.sin(tt * 120 + 0.4) * 0.10
            + (Math.sin(i * 0.4) * 0.5 - 0.25) * 0.12;
        }
        const y = h / 2 + s * env * (h / 2 - 6);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = isOod ? '#e07b2a' : isLeak ? '#d94040' : '#1a8fe3';
      ctx.lineWidth = 1.4;
      ctx.stroke();
      // baseline
      ctx.strokeStyle = 'rgba(58,122,176,0.4)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
      ctx.setLineDash([]);
      t += 0.05;
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [kind]);
  return (
    <canvas ref={ref} width={560} height={150}
      style={{ display: 'block', width: '100%', height: 150, background: 'rgba(255,255,255,0.5)',
               borderRadius: 6, border: '1px solid #cee2f3' }} />
  );
};

const SpectrogramLight = ({ kind }) => {
  // Render an SVG-based heatmap with sky-blue palette
  const cols = 80, rows = 22;
  const cells = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const t = x / cols, f = y / rows;
      let v;
      if (kind === 'leak') {
        v = Math.exp(-Math.pow((f - 0.55) * 3.0, 2)) * (0.55 + 0.45 * Math.sin(t * 9 + f * 3));
        v += 0.18 * Math.exp(-Math.pow((f - 0.78) * 4, 2));
      } else if (kind === 'ood') {
        v = 0.4 + 0.4 * Math.sin(t * 19 + f * 7) * Math.cos(t * 5 - f * 3);
      } else {
        v = 0.16 * Math.exp(-Math.pow((f - 0.3) * 3, 2)) * (0.6 + 0.4 * Math.sin(t * 4));
      }
      v = Math.max(0, Math.min(1, v));
      cells.push({ x, y, v });
    }
  }
  const w = 560, h = 150, cw = w / cols, ch = h / rows;
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`}
      style={{ display: 'block', borderRadius: 6, border: '1px solid #cee2f3', background: '#0a3050' }}>
      {cells.map((c, i) => {
        // Sky-blue heatmap: low = dark navy, high = bright cyan/white
        const lightness = 0.16 + c.v * 0.74;
        const chroma = kind === 'ood' ? 0.05 + c.v * 0.18 : 0.04 + c.v * 0.16;
        const hue = kind === 'ood' ? 50 : kind === 'leak' ? 220 : 240;
        return (
          <rect key={i} x={c.x * cw} y={(rows - 1 - c.y) * ch}
                width={cw + 0.5} height={ch + 0.5}
                fill={`oklch(${lightness} ${chroma} ${hue})`} />
        );
      })}
      {/* axis labels */}
      <text x={6} y={h - 6} fill="rgba(255,255,255,.6)" fontSize="9" fontFamily="IBM Plex Mono">0 Hz</text>
      <text x={6} y={14} fill="rgba(255,255,255,.6)" fontSize="9" fontFamily="IBM Plex Mono">8 kHz</text>
      <text x={w - 36} y={h - 6} fill="rgba(255,255,255,.6)" fontSize="9" fontFamily="IBM Plex Mono">{kind === 'leak' ? '12.4s' : kind === 'ood' ? '8.7s' : '30.0s'}</text>
    </svg>
  );
};

const ProgressBar = ({ value }) => (
  <div style={{ height: 8, background: '#cfe6f8', borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
    <div style={{
      width: `${value}%`, height: '100%',
      background: 'linear-gradient(90deg, #1a8fe3 0%, #3aabf0 50%, #1a8fe3 100%)',
      backgroundSize: '200% 100%',
      animation: 'shimmer 1.6s linear infinite',
      transition: 'width 0.18s ease',
    }} />
  </div>
);

const PipelineStep = ({ label, active, done }) => (
  <div style={{
    display: 'flex', alignItems: 'center', gap: 6,
    fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, letterSpacing: 1,
    color: done ? '#0fa36b' : active ? '#1a8fe3' : '#6c9bc4',
    transition: 'color 0.2s ease',
  }}>
    <span style={{
      width: 10, height: 10, borderRadius: 999,
      background: done ? '#0fa36b' : active ? '#1a8fe3' : 'transparent',
      border: `1.5px solid ${done ? '#0fa36b' : active ? '#1a8fe3' : '#6c9bc4'}`,
      boxShadow: active && !done ? '0 0 8px #1a8fe3' : 'none',
      animation: active && !done ? 'pulse-dot 1s ease-in-out infinite' : 'none',
    }}></span>
    {label}
  </div>
);

const ResultsBlock = ({ picked, verdict }) => {
  const isLeak = verdict === 'leak';
  const isOod = verdict === 'ood';

  return (
    <React.Fragment>
      {/* Verdict + confidence gauge */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16, marginBottom: 16 }}>
        <div className={isLeak ? 'verdict-leak' : isOod ? 'verdict-ood' : 'verdict-clean'}>
          {isLeak  && <span>🚨 LEAK DETECTED</span>}
          {isOod   && <span>⚠ OOD QUARANTINE · 422</span>}
          {!isLeak && !isOod && <span>✓ NO LEAK DETECTED</span>}
          <div style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: 13, fontWeight: 400,
                        letterSpacing: 1.5, marginTop: 6, opacity: 0.92 }}>
            {isLeak  && `Branched · Orifice Leak · est. ${(picked.iep4*0.55).toFixed(2)} LPS`}
            {isOod   && 'Sample outside training distribution · refusing to predict'}
            {!isLeak && !isOod && 'Signal consistent with healthy network baseline'}
          </div>
        </div>
        <div className="panel">
          <div className="panel-h">
            <span>Confidence</span>
            <span className="lbl">{isOod ? 'N/A · QUARANTINED' : 'ENSEMBLE VOTE'}</span>
          </div>
          <div className="panel-b" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <ConfidenceGauge value={isOod ? 0 : picked.conf} ood={isOod} />
            <div style={{ flex: 1 }}>
              <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 24, fontWeight: 800,
                            color: isOod ? '#e07b2a' : '#1a8fe3' }}>
                {isOod ? 'OOD' : `${(picked.conf*100).toFixed(1)}%`}
              </div>
              <div className="lbl" style={{ fontSize: 10, marginTop: 2 }}>
                {isOod ? `OOD score ${picked.ood.toFixed(2)} > θ 0.42` : 'Above decision threshold 0.952'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 4 model scores */}
      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-h"><span>Detailed Analysis · 5-Head Ensemble</span><span className="lbl">FUSION 0.45/0.25/0.25/0.05</span></div>
        <div className="panel-b" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          <ScoreCard label="IEP2 · XGBOOST"    value={picked.iep2} weight="45%" />
          <ScoreCard label="IEP4 · RF + CNN"   value={picked.iep4} weight="25 + 25%" />
          <ScoreCard label="ENSEMBLE"          value={picked.ens}  weight="WEIGHTED" highlight />
          <ScoreCard label="OOD · CNN-AE"      value={picked.ood}  weight={isOod ? 'GATE FIRED' : 'PASS'} ood={isOod} />
        </div>
      </div>

      {/* Quality + raw */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="panel">
          <div className="panel-h"><span>Quality Metrics</span><span className="lbl">SIGNAL QA</span></div>
          <div className="panel-b" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            <Metric label="SNR" value={picked.snr.toFixed(1)} unit="dB" />
            <Metric label="PHYSICS" value={(picked.phys * 100).toFixed(0)} unit="%" />
            <Metric label="LATENCY" value="612" unit="ms" />
          </div>
        </div>
        <div className="panel">
          <div className="panel-h"><span>Raw Predictions · JSON</span><span className="lbl">AUDIT-READY</span></div>
          <div className="panel-b">
            <pre style={{
              margin: 0, padding: 10, fontSize: 10.5, lineHeight: 1.55,
              fontFamily: "'IBM Plex Mono', monospace",
              background: '#0a3050', color: '#a0d4f5', borderRadius: 6,
              overflow: 'auto', maxHeight: 140,
            }}>{JSON.stringify({
  is_leak: isLeak, ood: isOod,
  iep2: picked.iep2, iep4: picked.iep4,
  ensemble: picked.ens, ood_score: picked.ood,
  threshold: 0.952, snr: picked.snr,
  trace_id: '0x' + Math.random().toString(16).slice(2,10),
}, null, 2)}</pre>
          </div>
        </div>
      </div>
    </React.Fragment>
  );
};

const ScoreCard = ({ label, value, weight, highlight, ood }) => (
  <div style={{
    background: highlight ? 'linear-gradient(180deg, rgba(26,143,227,0.18) 0%, rgba(26,143,227,0.05) 100%)' :
                ood       ? 'linear-gradient(180deg, rgba(224,123,42,0.18) 0%, rgba(224,123,42,0.04) 100%)' :
                            'rgba(255,255,255,0.6)',
    border: `1px solid ${highlight ? '#1a8fe3' : ood ? '#e07b2a' : '#cee2f3'}`,
    borderRadius: 8, padding: '12px 14px',
  }}>
    <div className="lbl" style={{ fontSize: 9, marginBottom: 6 }}>{label}</div>
    <div style={{
      fontFamily: "'Orbitron', sans-serif", fontSize: 22, fontWeight: 800,
      color: ood ? '#b85e15' : highlight ? '#0e75c8' : '#1a8fe3',
      marginBottom: 6,
    }}>
      {value.toFixed(3)}
    </div>
    <div style={{ height: 4, background: '#cfe6f8', borderRadius: 2, overflow: 'hidden' }}>
      <div style={{
        width: `${value * 100}%`, height: '100%',
        background: ood ? 'linear-gradient(90deg, #e07b2a, #d94040)'
                        : 'linear-gradient(90deg, #1a8fe3, #3aabf0)',
      }} />
    </div>
    <div className="lbl" style={{ fontSize: 9, marginTop: 6, color: ood ? '#b85e15' : '#3a7ab0' }}>{weight}</div>
  </div>
);

const ConfidenceGauge = ({ value, ood }) => {
  // Animated needle gauge
  const [shown, setShown] = React.useState(0);
  React.useEffect(() => {
    let raf, t = 0;
    const target = ood ? 0 : value;
    const start = 0;
    const dur = 700;
    const t0 = performance.now();
    const tick = () => {
      const k = Math.min(1, (performance.now() - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      setShown(start + (target - start) * eased);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [value, ood]);

  const cx = 70, cy = 70, r = 50;
  const startAngle = Math.PI; // 180°
  const endAngle = 2 * Math.PI; // 360° (top)
  const angle = startAngle + shown * (endAngle - startAngle);
  const nx = cx + Math.cos(angle) * r;
  const ny = cy + Math.sin(angle) * r;
  // arc path
  const arcPath = (a0, a1, color, width) => {
    const x0 = cx + Math.cos(a0) * r, y0 = cy + Math.sin(a0) * r;
    const x1 = cx + Math.cos(a1) * r, y1 = cy + Math.sin(a1) * r;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    return <path d={`M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`}
      stroke={color} strokeWidth={width} fill="none" strokeLinecap="round" />;
  };

  return (
    <svg width="140" height="92" viewBox="0 0 140 92">
      {/* track */}
      {arcPath(startAngle, endAngle, '#cfe6f8', 8)}
      {/* segments */}
      {arcPath(startAngle, startAngle + 0.5 * Math.PI, '#e07b2a', 8)}
      {arcPath(startAngle + 0.5 * Math.PI, startAngle + 0.85 * Math.PI, '#1a8fe3', 8)}
      {arcPath(startAngle + 0.85 * Math.PI, endAngle, '#0fa36b', 8)}
      {/* needle */}
      <line x1={cx} y1={cy} x2={nx} y2={ny}
            stroke={ood ? '#e07b2a' : '#0e75c8'} strokeWidth="2.5" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="5" fill={ood ? '#e07b2a' : '#1a8fe3'} />
      <circle cx={cx} cy={cy} r="2" fill="#fff" />
      <text x={cx} y={88} textAnchor="middle"
            fontFamily="Rajdhani" fontSize="9" fill="#3a7ab0" letterSpacing="2">CONFIDENCE</text>
    </svg>
  );
};

window.DiagnoseScreen = DiagnoseScreen;
