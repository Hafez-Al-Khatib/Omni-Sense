// Screen 5: MLOps health — drift, OOD rejection, regression gate, audit Merkle chain

const MlopsHealth = () => {
  return (
    <div className="os-art" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '52px 1fr 1fr', gap: 14, padding: 14, height: '100%' }}>
      <div style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--line-soft)', padding: '0 4px 14px', gap: 16 }}>
        <span className="tag sig">MLOPS · MODEL HEALTH</span>
        <span style={{ fontWeight: 600 }}>Production model: iep2-xgb-v17 · iep4-cnn-v9</span>
        <div className="grow" />
        <span className="tag ok">CI green · last build 12 min ago</span>
      </div>

      {/* Drift */}
      <div className="panel">
        <div className="panel-h"><span>Feature drift · 39-d KS test</span><span className="mono ok">0.041 / 0.05</span></div>
        <div className="panel-b">
          <div className="lbl">7-DAY ROLLING KS STATISTIC PER FEATURE</div>
          <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: 'repeat(13, 1fr)', gap: 4 }}>
            {Array.from({ length: 39 }).map((_, i) => {
              const v = 0.005 + (Math.sin(i * 1.7) * 0.5 + 0.5) * 0.06;
              const over = v > 0.05;
              return (
                <div key={i} title={`feat_${i}: ${v.toFixed(3)}`} style={{
                  height: 28,
                  background: `oklch(${0.32 + v * 4} ${0.04 + v * 2} ${over ? 25 : 200} / 0.85)`,
                  borderRadius: 2, border: over ? '1px solid var(--crit)' : '1px solid var(--line-soft)'
                }} />
              );
            })}
          </div>
          <div className="row between mono" style={{ marginTop: 16, fontSize: 11, color: 'var(--fg-1)' }}>
            <div>kurtosis_a1 <span style={{ color: 'var(--fg-0)' }}>0.062</span> ⚠</div>
            <div>spectral_centroid <span style={{ color: 'var(--fg-0)' }}>0.041</span></div>
            <div>wavelet_db4 <span style={{ color: 'var(--fg-0)' }}>0.038</span></div>
          </div>
          <div style={{ marginTop: 14 }}>
            <div className="lbl">SCORE TIMELINE</div>
            <div style={{ marginTop: 6 }}><Sparkline data={[0.022, 0.025, 0.024, 0.029, 0.034, 0.038, 0.041]} w={420} h={36} /></div>
          </div>
        </div>
      </div>

      {/* OOD rejection */}
      <div className="panel">
        <div className="panel-h"><span>OOD rejections · 24h</span><span className="mono">11 / 1,847 (0.6%)</span></div>
        <div className="panel-b">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            <div>
              <div className="lbl">AE RECON ERROR · DISTRIBUTION</div>
              <svg width="100%" height="100" viewBox="0 0 220 100" style={{ marginTop: 8 }}>
                {Array.from({ length: 24 }).map((_, i) => {
                  const t = i / 24;
                  const h = Math.exp(-Math.pow((t - 0.22) * 4.2, 2)) * 80 + 4;
                  const reject = t > 0.42 / 1.0;
                  return <rect key={i} x={i * 9 + 2} y={100 - h} width="7" height={h}
                                fill={reject ? 'var(--violet)' : 'var(--signal)'} opacity="0.9" />;
                })}
                <line x1="92" y1="0" x2="92" y2="100" stroke="var(--warn)" strokeDasharray="3 3" strokeWidth="1.5" />
                <text x="96" y="14" fill="var(--warn)" fontSize="9" fontFamily="IBM Plex Mono">θ 0.42</text>
              </svg>
            </div>
            <div>
              <div className="lbl">REJECTION REASONS</div>
              <div className="col gap-2" style={{ marginTop: 8 }}>
                <RejectRow label="Kinetic shock (wrench/impact)" n={5} pct={45} />
                <RejectRow label="Generator harmonic"             n={3} pct={27} />
                <RejectRow label="Cavitation pump close-by"       n={2} pct={18} />
                <RejectRow label="Sensor mount loose"              n={1} pct={9} />
              </div>
            </div>
          </div>
          <div className="lbl" style={{ marginTop: 14 }}>QUARANTINED FRAMES → REVIEW QUEUE</div>
          <div style={{ marginTop: 6, padding: 10, background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 4 }} className="mono" >
            <div style={{ fontSize: 11, color: 'var(--fg-1)' }}>
              <span style={{ color: 'var(--violet)' }}>quarantine://</span>S-MARM-006/2026-04-27T13:18:02 · awaiting label · queue position 4
            </div>
          </div>
        </div>
      </div>

      {/* Regression gate timeline */}
      <div className="panel">
        <div className="panel-h"><span>Model regression gate · last 14 builds</span><span className="mono ok">F1 0.9879 · ↑0.003</span></div>
        <div className="panel-b">
          <svg width="100%" height="120" viewBox="0 0 480 120" style={{ marginTop: 4 }}>
            <line x1="0" y1="40" x2="480" y2="40" stroke="var(--line-soft)" strokeDasharray="3 4" />
            <text x="6" y="36" fontSize="9" fontFamily="IBM Plex Mono" fill="var(--fg-2)">F1 0.99</text>
            <line x1="0" y1="80" x2="480" y2="80" stroke="var(--line-soft)" strokeDasharray="3 4" />
            <text x="6" y="76" fontSize="9" fontFamily="IBM Plex Mono" fill="var(--fg-2)">F1 0.97</text>
            {[0.985, 0.983, 0.987, 0.984, 0.989, 0.985, 0.971, 0.984, 0.986, 0.987, 0.988, 0.985, 0.987, 0.9879].map((f1, i) => {
              const x = 50 + i * 30;
              const y = 120 - (f1 - 0.95) * 800;
              const fail = f1 < 0.975;
              return (
                <g key={i}>
                  <line x1={x} y1="120" x2={x} y2={y} stroke={fail ? 'var(--crit)' : 'var(--ok)'} strokeWidth="1" opacity="0.5" />
                  <circle cx={x} cy={y} r="4" fill={fail ? 'var(--crit)' : 'var(--ok)'} />
                  {fail && <text x={x - 18} y={y - 8} fontSize="9" fill="var(--crit)" fontFamily="IBM Plex Mono">BLOCKED</text>}
                </g>
              );
            })}
          </svg>
          <div className="row between mono" style={{ marginTop: 6, fontSize: 11, color: 'var(--fg-1)' }}>
            <span>14 builds · 13 passed</span>
            <span>Gate threshold <span style={{ color: 'var(--fg-0)' }}>F1 ≥ 0.975</span> on golden_v1</span>
          </div>
          <div style={{ marginTop: 10, padding: 10, background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 4 }}>
            <div className="row between mono" style={{ fontSize: 11 }}>
              <span style={{ color: 'var(--fg-0)' }}>build #18 · feat/cnn-v9 · 12 min ago</span>
              <span className="tag ok" style={{ fontSize: 9 }}>PASSED</span>
            </div>
            <div className="mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 4 }}>
              ROC-AUC 0.9907 · F1 0.9879 · Leak recall 0.984 · 1,336 recordings · 5-fold CV
            </div>
          </div>
        </div>
      </div>

      {/* Audit Merkle chain */}
      <div className="panel">
        <div className="panel-h"><span>WORM audit · Ed25519 + Merkle chain</span><span className="mono ok">CHAIN VERIFIED</span></div>
        <div className="panel-b">
          <div className="lbl">LAST 6 BLOCKS · HMAC-SHA256 ROLLING</div>
          <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              { h: '7f3a…b21d', t: '14:22:08', ev: 'WO-26041 fired · S-HAMRA-001' },
              { h: '4e91…03c2', t: '14:21:47', ev: 'frame ingest · 472 bytes' },
              { h: '2a08…fe19', t: '14:21:46', ev: 'frame ingest · 472 bytes' },
              { h: '8c52…9a7b', t: '14:21:43', ev: 'WO-26039 acknowledged · Karim H.' },
              { h: 'b314…6e88', t: '14:21:38', ev: 'OOD reject · S-MARM-006' },
              { h: 'e6d2…417f', t: '14:21:34', ev: 'frame ingest · 488 bytes' },
            ].map((b, i) => (
              <div key={i} className="row" style={{ gap: 10, padding: '6px 10px', background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 3, fontSize: 11 }}>
                <span className="mono" style={{ color: 'var(--fg-2)', minWidth: 56 }}>{b.t}</span>
                <span className="mono" style={{ color: 'var(--signal)', minWidth: 96 }}>{b.h}</span>
                <span className="mono" style={{ color: 'var(--fg-1)', flex: 1 }}>{b.ev}</span>
                <span className="dot ok"></span>
              </div>
            ))}
          </div>
          <div className="row between mono" style={{ marginTop: 12, fontSize: 11, color: 'var(--fg-1)' }}>
            <span>chain head <span style={{ color: 'var(--signal)' }}>7f3a…b21d</span></span>
            <span>height 184,772</span>
            <span>HMAC key persisted · OK</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const RejectRow = ({ label, n, pct }) => (
  <div>
    <div className="row between mono" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
      <span>{label}</span>
      <span style={{ color: 'var(--fg-0)' }}>{n}</span>
    </div>
    <div style={{ height: 4, background: 'var(--bg-0)', borderRadius: 2, marginTop: 3, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: 'var(--violet)' }} />
    </div>
  </div>
);

window.MlopsHealth = MlopsHealth;
