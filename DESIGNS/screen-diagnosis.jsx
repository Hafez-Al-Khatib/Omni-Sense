// Screen 2: Live Diagnosis — 5 ML heads voting → fusion → SHAP → OOD gate
// Bold idea: each ML head is a vertical column whose fill rises in real time
// to its emitted probability; arrows then converge on the fusion verdict.

const LiveDiagnosis = () => {
  const [phase, setPhase] = React.useState(0); // 0..4
  React.useEffect(() => {
    const i = setInterval(() => setPhase(p => (p + 1) % 5), 1800);
    return () => clearInterval(i);
  }, []);

  // Heads with weights from omni/README.md fusion formula
  const heads = [
    { name: 'XGBoost',         tag: 'IEP2', weight: 0.45, score: 0.94, latency: 28, color: 'var(--signal)' },
    { name: 'Random Forest',   tag: 'IEP2', weight: 0.25, score: 0.91, latency: 31, color: 'var(--signal)' },
    { name: 'CNN Spectrogram', tag: 'IEP4', weight: 0.25, score: 0.97, latency: 142, color: 'var(--violet)' },
    { name: 'Isolation Forest',tag: 'IEP2', weight: 0.05, score: 0.88, latency: 19, color: 'var(--ok)' },
  ];
  const fused = heads.reduce((a, h) => a + h.weight * h.score, 0);

  return (
    <div className="os-art" style={{ display: 'grid', gridTemplateColumns: '1fr', gridTemplateRows: '52px 1fr', height: '100%' }}>
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--line-soft)', padding: '0 18px', gap: 16, background: 'var(--bg-1)' }}>
        <button className="btn ghost" style={{ padding: '4px 8px' }}>←</button>
        <span className="tag crit">CRITICAL</span>
        <span style={{ fontWeight: 600 }}>WO-26041 · Hamra · Bliss St</span>
        <span className="lbl">Frame 14:22:08.412 · 0.975s · 16kHz · S-HAMRA-001</span>
        <div className="grow" />
        <span className="tag ok">SLA 01:42 / 05:00</span>
        <button className="btn">Open in CMMS</button>
        <button className="btn primary">Acknowledge</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr 320px', gap: 14, padding: 14, overflow: 'hidden', minHeight: 0 }}>
        {/* LEFT: signal */}
        <div className="col gap-3" style={{ minHeight: 0, overflow: 'auto' }} >
          <div className="panel">
            <div className="panel-h"><span>Vibration · S-HAMRA-001 · A1</span><span className="mono" style={{ color: 'var(--signal)' }}>RMS 0.142 · SNR 22.1dB</span></div>
            <div className="panel-b">
              <Waveform w={350} h={70} color="var(--signal)" seed={2} />
              <div className="lbl" style={{ marginTop: 12 }}>SPECTROGRAM · MID-BAND HISS DETECTED</div>
              <div style={{ marginTop: 8 }}><Spectrogram w={350} h={120} intensity="leak" /></div>
              <div className="row between mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 4 }}>
                <span>0 Hz</span><span>2k</span><span>4k</span><span>6k</span><span>8 kHz</span>
              </div>
            </div>
          </div>
          <div className="panel">
            <div className="panel-h"><span>Physics consistency · SCADA fusion</span><span className="mono ok" style={{ color: 'var(--ok)' }}>MATCH</span></div>
            <div className="panel-b col gap-3">
              <PhysRow label="Pressure (bar)" baseline="3.50" now="3.09" delta="−0.41" mult="×1.18" status="match" />
              <PhysRow label="Flow (L/s)"     baseline="2.40" now="2.87" delta="+0.47" mult="×1.08" status="match" />
              <PhysRow label="Pipe material"  baseline="PVC"  now="PVC"   delta="—"     mult="—"     status="match" />
              <div className="lbl" style={{ marginTop: 4 }}>WNTR EPANET twin · 0.41 bar drop is consistent with branched orifice leak @ 0.47 L/s</div>
            </div>
          </div>
        </div>

        {/* CENTER: ensemble votes → fusion */}
        <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="panel-h"><span>Ensemble inference · 5 heads</span><span className="mono">trace 7f3a · 218ms total</span></div>
          <div style={{ flex: 1, padding: 18, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr) 1.4fr', gap: 14 }}>
            {heads.map((h, i) => (
              <HeadColumn key={h.name} head={h} active={phase >= i} />
            ))}
            {/* Fusion column */}
            <FusionColumn fused={fused} active={phase >= 4} />
          </div>
          <div style={{ padding: 18, paddingTop: 0 }}>
            <div className="lbl">FUSION FORMULA</div>
            <div className="mono" style={{ fontSize: 12, color: 'var(--fg-1)', marginTop: 6, padding: 10, background: 'var(--bg-0)', borderRadius: 4, border: '1px solid var(--line-soft)' }}>
              p_leak = 0.45·p<sub>XGB</sub> + 0.25·p<sub>RF</sub> + 0.25·p<sub>CNN</sub> + 0.05·p<sub>IF</sub> &nbsp;·&nbsp; gated by AE-OOD ≤ 0.42
            </div>
          </div>
        </div>

        {/* RIGHT: SHAP + OOD + actions */}
        <div className="col gap-3" style={{ minHeight: 0, overflow: 'auto' }}>
          <div className="panel">
            <div className="panel-h"><span>OOD safety gate</span><span className="mono ok">PASSED</span></div>
            <div className="panel-b">
              <OodGauge ae={0.18} aeThreshold={0.42} ifScore={0.88} />
              <div className="mono" style={{ fontSize: 11, color: 'var(--fg-1)', marginTop: 10 }}>
                <div>AE recon error <span style={{ color: 'var(--fg-0)' }}>0.18</span> &lt; 0.42 ✓</div>
                <div>Iso.Forest score <span style={{ color: 'var(--fg-0)' }}>0.88</span> &gt; 0.65 ✓</div>
                <div>Mahalanobis d² <span style={{ color: 'var(--fg-0)' }}>4.2</span> &lt; 9.21 ✓</div>
              </div>
            </div>
          </div>
          <div className="panel">
            <div className="panel-h"><span>SHAP · top-5 features</span><span className="mono">IEP2 · XGB</span></div>
            <div className="panel-b col gap-2">
              <ShapBar label="spectral_centroid_mid" value={+0.31} />
              <ShapBar label="kurtosis_a1"           value={+0.22} />
              <ShapBar label="wavelet_db4_lvl3_rms"  value={+0.18} />
              <ShapBar label="band_energy_2k_4k"     value={+0.14} />
              <ShapBar label="rms_envelope_decay"    value={-0.06} />
              <div className="lbl" style={{ marginTop: 4 }}>+ pushes toward leak · − away</div>
            </div>
          </div>
          <div className="panel">
            <div className="panel-h"><span>Decision</span><span className="mono crit" style={{ color: 'var(--crit)' }}>p_leak = 0.96</span></div>
            <div className="panel-b col gap-2">
              <div className="row gap-2"><span className="tag crit">LEAK · BRANCHED · ORIFICE</span></div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
                Threshold θ = 0.952 · margin +0.008<br/>
                Class prior = 0.952 (No_Leak heavy)<br/>
                Decision: <span style={{ color: 'var(--crit)' }}>FIRE TICKET</span>
              </div>
              <div className="row gap-2" style={{ marginTop: 4 }}>
                <button className="btn primary grow">Dispatch crew</button>
                <button className="btn ghost">Quarantine</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const PhysRow = ({ label, baseline, now, delta, mult, status }) => (
  <div className="row between mono" style={{ fontSize: 11, padding: '6px 0', borderBottom: '1px dashed var(--line-soft)' }}>
    <span style={{ color: 'var(--fg-1)' }}>{label}</span>
    <div className="row gap-3">
      <span style={{ color: 'var(--fg-3)', width: 60, textAlign: 'right' }}>{baseline}</span>
      <span style={{ color: 'var(--fg-0)', width: 60, textAlign: 'right' }}>{now}</span>
      <span style={{ color: status === 'match' ? 'var(--ok)' : 'var(--crit)', width: 60, textAlign: 'right' }}>{delta}</span>
      <span className="tag ok" style={{ minWidth: 60, justifyContent: 'center' }}>{mult}</span>
    </div>
  </div>
);

const HeadColumn = ({ head, active }) => {
  const fillH = active ? `${head.score * 100}%` : '0%';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <div className="row gap-2" style={{ alignSelf: 'flex-start' }}>
        <span className="tag" style={{ borderColor: head.color, color: head.color, fontSize: 9 }}>{head.tag}</span>
      </div>
      <div style={{ fontWeight: 600, fontSize: 12, alignSelf: 'flex-start' }}>{head.name}</div>
      <div className="lbl" style={{ alignSelf: 'flex-start' }}>w = {head.weight}</div>
      <div style={{
        flex: 1, width: '100%', minHeight: 200, position: 'relative',
        background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 4, overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute', left: 0, right: 0, bottom: 0, height: fillH,
          background: `linear-gradient(180deg, ${head.color} 0%, oklch(from ${head.color} l c h / 0.3) 100%)`,
          transition: 'height 0.9s cubic-bezier(.4,1.4,.5,1)',
        }} />
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', padding: 8 }}>
          <span className="mono" style={{ fontSize: 18, fontWeight: 600, color: 'oklch(0.10 0.01 240)', mixBlendMode: 'screen' }}>
            {active ? head.score.toFixed(2) : '— —'}
          </span>
        </div>
      </div>
      <div className="mono" style={{ fontSize: 10, color: 'var(--fg-2)' }}>{head.latency} ms</div>
    </div>
  );
};

const FusionColumn = ({ fused, active }) => (
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
    <span className="tag crit" style={{ alignSelf: 'flex-start' }}>FUSION</span>
    <div style={{ fontWeight: 600, fontSize: 13, alignSelf: 'flex-start' }}>p_leak (weighted)</div>
    <div className="lbl" style={{ alignSelf: 'flex-start' }}>θ = 0.952</div>
    <div style={{
      flex: 1, width: '100%', minHeight: 200, position: 'relative',
      background: 'var(--bg-0)', border: '1px solid var(--crit)', borderRadius: 4, overflow: 'hidden'
    }}>
      {/* threshold line */}
      <div style={{ position: 'absolute', left: 0, right: 0, bottom: '95.2%', height: 1, background: 'var(--warn)', zIndex: 2 }} />
      <div style={{ position: 'absolute', right: 4, bottom: 'calc(95.2% + 2px)', fontSize: 9, color: 'var(--warn)' }} className="mono">θ 0.952</div>
      <div style={{
        position: 'absolute', left: 0, right: 0, bottom: 0,
        height: active ? `${fused * 100}%` : '0%',
        background: 'linear-gradient(180deg, var(--crit) 0%, oklch(0.66 0.18 25 / 0.3) 100%)',
        transition: 'height 1.1s cubic-bezier(.4,1.4,.5,1) 0.2s',
      }} />
      <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 8 }}>
        <span className="mono" style={{ fontSize: 32, fontWeight: 700, color: 'var(--fg-0)' }}>
          {active ? fused.toFixed(3) : '0.000'}
        </span>
      </div>
    </div>
    <div className="mono" style={{ fontSize: 10, color: 'var(--crit)' }}>FIRE · margin +{(fused - 0.952).toFixed(3)}</div>
  </div>
);

const OodGauge = ({ ae, aeThreshold, ifScore }) => {
  const aePct = (ae / 1.0) * 100;
  const tPct = (aeThreshold / 1.0) * 100;
  return (
    <div>
      <div className="lbl" style={{ marginBottom: 6 }}>AE RECONSTRUCTION ERROR</div>
      <div style={{ position: 'relative', height: 14, background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 3 }}>
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${aePct}%`, background: 'var(--ok)' }} />
        <div style={{ position: 'absolute', left: `${tPct}%`, top: -2, bottom: -2, width: 1, background: 'var(--warn)' }} />
        <div style={{ position: 'absolute', left: `calc(${tPct}% + 4px)`, top: 16, fontSize: 9, color: 'var(--warn)' }} className="mono">θ 0.42</div>
      </div>
      <div className="row between mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 18 }}>
        <span>0.0</span><span style={{ color: 'var(--fg-0)' }}>{ae.toFixed(2)}</span><span>1.0</span>
      </div>
    </div>
  );
};

const ShapBar = ({ label, value }) => {
  const w = Math.min(Math.abs(value) / 0.4, 1) * 100;
  const pos = value >= 0;
  return (
    <div>
      <div className="row between mono" style={{ fontSize: 10, color: 'var(--fg-1)' }}>
        <span>{label}</span>
        <span style={{ color: pos ? 'var(--crit)' : 'var(--signal)' }}>{value > 0 ? '+' : ''}{value.toFixed(2)}</span>
      </div>
      <div style={{ position: 'relative', height: 6, background: 'var(--bg-0)', borderRadius: 2, marginTop: 2 }}>
        <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--line)' }} />
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          [pos ? 'left' : 'right']: '50%',
          width: `${w / 2}%`,
          background: pos ? 'var(--crit)' : 'var(--signal)',
          borderRadius: 1
        }} />
      </div>
    </div>
  );
};

window.LiveDiagnosis = LiveDiagnosis;
