// Screen 7: "Break my AI" demo — poster session showpiece.
// Bold idea: live waveform, judge "strikes" the pipe, Mahalanobis distance spikes,
// system FIRES the 422 OOD-quarantine response instead of crying "leak."

const BreakMyAi = () => {
  const [event, setEvent] = React.useState(null); // 'spoof' | 'strike' | null
  const [t, setT] = React.useState(0);
  React.useEffect(() => { const i = setInterval(() => setT(x => x + 1), 60); return () => clearInterval(i); }, []);
  React.useEffect(() => {
    if (!event) return;
    const id = setTimeout(() => setEvent(null), 4000);
    return () => clearTimeout(id);
  }, [event]);

  const baseline = 4.2;
  const spike = event === 'strike' ? 38 : event === 'spoof' ? 11 : 0;
  const mDist = baseline + spike * (event ? Math.max(0, Math.min(1, 1 - (t % 67) / 50)) : 0);

  // Build waveform geometry per state
  const N = 200;
  const wfPts = [];
  for (let i = 0; i < N; i++) {
    const x = i / N;
    let y;
    if (event === 'strike') {
      const env = Math.exp(-Math.pow((x - 0.45) * 10, 2));
      y = Math.sin(x * 200) * 0.2 + env * (Math.sin(x * 90) * 0.85 + (Math.sin(i * 1.7) * 43.5 % 1 - 0.5) * 0.7);
    } else if (event === 'spoof') {
      y = Math.sin(x * 120 + 1) * 0.55 + Math.sin(x * 360) * 0.18;
    } else {
      y = Math.sin(x * 80) * 0.18 + Math.sin(x * 220) * 0.08;
    }
    wfPts.push([x * 720, 90 + y * 60]);
  }
  const wfD = wfPts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');

  const verdict = event === 'strike'
    ? { code: '422', label: 'OOD QUARANTINE', detail: 'Kinetic shock · structural impact', color: 'var(--violet)' }
    : event === 'spoof'
    ? { code: '422', label: 'SCADA MISMATCH', detail: 'Audio says LEAK · pressure stable · airborne sound rejected', color: 'var(--violet)' }
    : { code: '200', label: 'NOMINAL', detail: 'Listening · in-distribution · no leak', color: 'var(--ok)' };

  return (
    <div className="os-art" style={{ display: 'grid', gridTemplateRows: '60px 1fr', height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--line-soft)', padding: '0 24px', gap: 16, background: 'var(--bg-1)' }}>
        <span className="tag crit">LIVE DEMO</span>
        <span style={{ fontWeight: 700, fontSize: 16 }}>BREAK · MY · AI</span>
        <span className="lbl">Hardware-in-the-loop · S-DEMO-001 mounted on PVC pipe proxy</span>
        <div className="grow" />
        <span className="mono" style={{ fontSize: 11, color: 'var(--fg-2)' }}>poster session · stand B-12</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 0, height: '100%', minHeight: 0 }}>
        {/* LEFT: pipe + waveform, full bleed */}
        <div style={{ position: 'relative', background: 'var(--bg-0)', overflow: 'hidden', padding: 24, display: 'flex', flexDirection: 'column' }}>
          {/* Pipe illustration */}
          <div className="lbl">DEMO RIG</div>
          <svg width="100%" height="170" viewBox="0 0 720 170" style={{ marginTop: 8 }}>
            {/* Pipe */}
            <rect x="40" y="70" width="640" height="60" rx="30" fill="oklch(0.30 0.012 240)" stroke="var(--line)" />
            <rect x="40" y="70" width="640" height="14" rx="7" fill="oklch(0.36 0.012 240)" />
            {/* sensor */}
            <rect x="350" y="56" width="40" height="22" rx="3" fill="var(--bg-2)" stroke="var(--signal)" />
            <text x="370" y="50" textAnchor="middle" fontSize="10" fill="var(--signal)" fontFamily="IBM Plex Mono">ADXL345</text>
            <line x1="370" y1="78" x2="370" y2="86" stroke="var(--signal)" />
            {/* leak source if active */}
            {event === 'strike' && (
              <g>
                <text x="200" y="56" fontSize="22" fill="var(--violet)" textAnchor="middle">⚒</text>
                <line x1="200" y1="60" x2="200" y2="78" stroke="var(--violet)" strokeWidth="2" strokeDasharray="2 2" />
                {[0, 1, 2, 3].map(i => (
                  <circle key={i} cx="200" cy="100" r={20 + i * 25 + (t % 30)} fill="none" stroke="var(--violet)" strokeOpacity={0.6 - i * 0.15} strokeWidth="1.5" />
                ))}
              </g>
            )}
            {event === 'spoof' && (
              <g>
                <text x="200" y="40" fontSize="20" fill="var(--violet)" textAnchor="middle">📱</text>
                <text x="200" y="58" fontSize="9" fontFamily="IBM Plex Mono" fill="var(--violet)" textAnchor="middle">YOUTUBE</text>
                {[0, 1, 2].map(i => (
                  <ellipse key={i} cx="200" cy="60" rx={10 + i * 8 + (t % 18)} ry={4 + i * 3}
                           fill="none" stroke="var(--violet)" strokeOpacity={0.4 - i * 0.1} />
                ))}
              </g>
            )}
            {/* annotations */}
            <text x="60" y="158" fontSize="10" fontFamily="IBM Plex Mono" fill="var(--fg-2)">PVC Ø160 · 3.5 bar · 16kHz capture</text>
            <text x="680" y="158" fontSize="10" fontFamily="IBM Plex Mono" fill="var(--fg-2)" textAnchor="end">MQTT/mTLS → cloud</text>
          </svg>

          <div className="lbl" style={{ marginTop: 18 }}>VIBRATION · CHANNEL A1 (LIVE)</div>
          <div style={{ marginTop: 6, padding: 14, background: 'oklch(0.10 0.012 240)', border: '1px solid var(--line-soft)', borderRadius: 4 }}>
            <svg width="100%" height="180" viewBox="0 0 720 180">
              <line x1="0" y1="90" x2="720" y2="90" stroke="var(--line)" strokeDasharray="3 3" />
              <path d={wfD} fill="none" stroke={verdict.color} strokeWidth="1.2" />
            </svg>
            <div className="row between mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 4 }}>
              <span>0 ms</span><span>250 ms</span><span>500 ms</span><span>750 ms</span><span>975 ms</span>
            </div>
          </div>

          <div className="lbl" style={{ marginTop: 18 }}>TRY TO BREAK IT — WATCH THE OOD GATE FIRE</div>
          <div className="row gap-3" style={{ marginTop: 8 }}>
            <button className="btn danger" style={{ padding: '12px 18px', fontSize: 13 }} onClick={() => setEvent('strike')}>
              ⚒ Strike pipe with wrench
            </button>
            <button className="btn" style={{ padding: '12px 18px', fontSize: 13, borderColor: 'var(--violet)', color: 'var(--violet)' }} onClick={() => setEvent('spoof')}>
              📱 Play leak video next to mic
            </button>
            <button className="btn ghost" onClick={() => setEvent(null)}>Reset</button>
          </div>
        </div>

        {/* RIGHT: epistemic safety dashboard */}
        <div style={{ borderLeft: '1px solid var(--line-soft)', padding: 24, display: 'flex', flexDirection: 'column', gap: 18, background: 'var(--bg-1)' }}>
          <div>
            <div className="lbl">SYSTEM RESPONSE</div>
            <div style={{
              marginTop: 8, padding: 18,
              background: `oklch(from ${verdict.color} l c h / 0.12)`,
              border: `2px solid ${verdict.color}`, borderRadius: 8
            }}>
              <div className="row between">
                <span className="mono" style={{ fontSize: 36, fontWeight: 700, color: verdict.color, lineHeight: 1 }}>
                  {verdict.code}
                </span>
                <span className="mono" style={{ fontSize: 13, color: verdict.color, fontWeight: 600, letterSpacing: '0.06em' }}>
                  {verdict.label}
                </span>
              </div>
              <div style={{ marginTop: 10, fontSize: 13, color: 'var(--fg-1)' }}>
                {verdict.detail}
              </div>
            </div>
          </div>

          <div>
            <div className="lbl">MAHALANOBIS DISTANCE · ISO.FOREST</div>
            <div className="row between mono" style={{ marginTop: 4, fontSize: 11, color: 'var(--fg-2)' }}>
              <span>0</span><span style={{ color: 'var(--warn)' }}>θ 9.21</span><span>50</span>
            </div>
            <div style={{ height: 16, background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
              <div style={{
                position: 'absolute', left: 0, top: 0, bottom: 0,
                width: `${(mDist / 50) * 100}%`,
                background: mDist > 9.21 ? 'var(--violet)' : 'var(--ok)',
                transition: 'width 0.3s ease'
              }} />
              <div style={{ position: 'absolute', left: '18.4%', top: 0, bottom: 0, width: 1, background: 'var(--warn)' }} />
            </div>
            <div className="mono" style={{ marginTop: 6, fontSize: 12, color: 'var(--fg-0)' }}>
              d² = {mDist.toFixed(2)} {mDist > 9.21 && <span style={{ color: 'var(--violet)' }}>· REJECTED</span>}
            </div>
          </div>

          <div>
            <div className="lbl">PHYSICS CONSISTENCY</div>
            <div style={{ marginTop: 8 }} className="col gap-2">
              <CheckRow label="Pressure drop ↔ acoustic energy" pass={!event || event === 'strike'} />
              <CheckRow label="Piezo impedance match (≠ airborne)" pass={event !== 'spoof'} />
              <CheckRow label="In-distribution kinetic profile" pass={event !== 'strike'} />
              <CheckRow label="SCADA flow correlation" pass={!event} />
            </div>
          </div>

          <div style={{ padding: 14, background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 6 }}>
            <div className="lbl">WHY THIS MATTERS</div>
            <div style={{ marginTop: 6, fontSize: 12, color: 'var(--fg-1)', lineHeight: 1.6 }}>
              An LLM hallucinates and you can't quantify it. A naive classifier returns 0.94 even when the input is nonsense.
              Omni-Sense returns <span className="mono" style={{ color: 'var(--violet)' }}>422 · I don't know</span> with a
              mathematical safety boundary — Isolation Forest + CNN-AE reconstruction. The system <em>refuses</em> to lie.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const CheckRow = ({ label, pass }) => (
  <div className="row between mono" style={{ fontSize: 11, padding: '6px 10px', background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 3 }}>
    <span style={{ color: 'var(--fg-1)' }}>{label}</span>
    <span style={{ color: pass ? 'var(--ok)' : 'var(--violet)', fontWeight: 600 }}>{pass ? '✓ pass' : '✕ fail'}</span>
  </div>
);

window.BreakMyAi = BreakMyAi;
