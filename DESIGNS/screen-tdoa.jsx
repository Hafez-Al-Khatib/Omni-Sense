// Screen 3: TDOA Spatial Fusion — multi-sensor leak triangulation on pipe schematic
// Bold idea: animated wavefronts emanating from leak point, hyperbolic intersections
// resolving to a ±meter ellipse on the actual pipe geometry.

const TdoaFusion = () => {
  const [t, setT] = React.useState(0);
  React.useEffect(() => { const i = setInterval(() => setT(x => x + 1), 80); return () => clearInterval(i); }, []);

  // Scene: pipe network with 4 sensors, leak at known point
  const sensors = [
    { id: 'S-HAMRA-001', x: 180, y: 120, dist: 47.2, tdoa: 0,    color: 'var(--signal)' },
    { id: 'S-HAMRA-014', x: 560, y: 90,  dist: 82.6, tdoa: 0.024, color: 'var(--signal)' },
    { id: 'S-HAMRA-022', x: 600, y: 360, dist: 94.1, tdoa: 0.031, color: 'var(--signal)' },
    { id: 'S-HAMRA-009', x: 120, y: 380, dist: 68.4, tdoa: 0.014, color: 'var(--signal)' },
  ];
  const leak = { x: 360, y: 220, ellipseW: 38, ellipseH: 18 };

  // animated wavefront radius
  const phase = (t * 4) % 600;

  return (
    <div className="os-art" style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gridTemplateRows: '52px 1fr', height: '100%' }}>
      <div style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--line-soft)', padding: '0 18px', gap: 16, background: 'var(--bg-1)' }}>
        <button className="btn ghost" style={{ padding: '4px 8px' }}>←</button>
        <span className="tag sig">TDOA · SPATIAL FUSION</span>
        <span style={{ fontWeight: 600 }}>Hamra zone · 4 active sensors</span>
        <span className="lbl">Correlation window 12s · Min 2 sensors · Kalman centroid</span>
        <div className="grow" />
        <span className="tag ok">localization confidence 0.91</span>
      </div>

      {/* Map */}
      <div style={{ position: 'relative', background: 'var(--bg-0)', overflow: 'hidden' }}>
        <svg width="100%" height="100%" viewBox="0 0 720 480" preserveAspectRatio="xMidYMid meet" style={{ display: 'block' }}>
          <defs>
            <pattern id="tdoa-grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M 24 0 L 0 0 0 24" fill="none" stroke="oklch(0.20 0.012 240)" strokeWidth="0.5" />
            </pattern>
            <radialGradient id="leak-glow">
              <stop offset="0%" stopColor="var(--crit)" stopOpacity="0.6" />
              <stop offset="60%" stopColor="var(--crit)" stopOpacity="0.18" />
              <stop offset="100%" stopColor="var(--crit)" stopOpacity="0" />
            </radialGradient>
          </defs>

          <rect width="720" height="480" fill="url(#tdoa-grid)" />

          {/* Pipe network — main + branches */}
          <g stroke="oklch(0.36 0.012 240)" strokeWidth="6" fill="none" strokeLinecap="round">
            <path d="M 60 120 L 660 120" />
            <path d="M 60 360 L 660 360" />
            <path d="M 180 120 L 180 360" />
            <path d="M 360 120 L 360 360" />
            <path d="M 540 120 L 540 360" />
          </g>
          <g stroke="oklch(0.50 0.012 240)" strokeWidth="2" fill="none" strokeLinecap="round" strokeDasharray="4 6" opacity="0.7">
            <path d="M 60 120 L 660 120" />
            <path d="M 360 120 L 360 360" />
          </g>

          {/* Pipe labels */}
          <text x="640" y="112" fill="var(--fg-3)" fontSize="10" fontFamily="IBM Plex Mono" textAnchor="end">PVC-Ø160 · main-N</text>
          <text x="640" y="380" fill="var(--fg-3)" fontSize="10" fontFamily="IBM Plex Mono" textAnchor="end">PVC-Ø160 · main-S</text>

          {/* Leak glow */}
          <circle cx={leak.x} cy={leak.y} r="80" fill="url(#leak-glow)" />

          {/* Wavefront circles (one cycle running) */}
          {[0, 150, 300, 450].map(off => {
            const r = (phase + off) % 600;
            const opacity = Math.max(0, 1 - r / 600);
            return (
              <circle key={off} cx={leak.x} cy={leak.y} r={r * 0.5}
                      fill="none" stroke="var(--crit)" strokeWidth="1.2" opacity={opacity * 0.55} />
            );
          })}

          {/* Hyperbolic isochrones (TDOA pairs) */}
          {sensors.slice(0, 3).map((s, i) => {
            const next = sensors[(i + 1) % 4];
            const cx = (s.x + next.x) / 2;
            const cy = (s.y + next.y) / 2;
            const dx = next.x - s.x, dy = next.y - s.y;
            const ang = Math.atan2(dy, dx) * 180 / Math.PI;
            const a = Math.hypot(dx, dy) / 2;
            return (
              <ellipse key={i} cx={cx} cy={cy} rx={a + 20} ry={Math.abs(s.tdoa - next.tdoa) * 8000 + 30}
                       fill="none" stroke="var(--signal)" strokeWidth="0.8" strokeDasharray="3 5"
                       opacity="0.45" transform={`rotate(${ang} ${cx} ${cy})`} />
            );
          })}

          {/* Sensor → leak rays with computed distance */}
          {sensors.map((s, i) => {
            const dx = leak.x - s.x, dy = leak.y - s.y;
            const len = Math.hypot(dx, dy);
            const ux = dx / len, uy = dy / len;
            return (
              <g key={s.id}>
                <line x1={s.x} y1={s.y} x2={leak.x} y2={leak.y}
                      stroke="var(--signal)" strokeWidth="1" strokeDasharray="2 4" opacity="0.5" />
                <text x={s.x + ux * (len / 2) + 6} y={s.y + uy * (len / 2)} fill="var(--signal)" fontSize="10" fontFamily="IBM Plex Mono">
                  {s.dist.toFixed(1)}m · Δt={s.tdoa.toFixed(3)}s
                </text>
              </g>
            );
          })}

          {/* Sensors */}
          {sensors.map(s => (
            <g key={s.id}>
              <circle cx={s.x} cy={s.y} r="14" fill="var(--bg-1)" stroke={s.color} strokeWidth="2" />
              <circle cx={s.x} cy={s.y} r="4" fill={s.color} />
              <text x={s.x} y={s.y - 22} fill="var(--fg-1)" fontSize="11" fontFamily="IBM Plex Mono" textAnchor="middle">{s.id}</text>
            </g>
          ))}

          {/* Leak ellipse (uncertainty) */}
          <ellipse cx={leak.x} cy={leak.y} rx={leak.ellipseW} ry={leak.ellipseH}
                   fill="var(--crit)" fillOpacity="0.15" stroke="var(--crit)" strokeWidth="1.5" strokeDasharray="4 3" />
          <circle cx={leak.x} cy={leak.y} r="6" fill="var(--crit)" />
          <line x1={leak.x - leak.ellipseW} y1={leak.y} x2={leak.x + leak.ellipseW} y2={leak.y}
                stroke="var(--crit)" strokeWidth="0.6" />
          <text x={leak.x} y={leak.y - 28} fill="var(--crit)" fontSize="12" fontFamily="IBM Plex Mono" textAnchor="middle" fontWeight="600">
            LEAK · 33.8961, 35.4791 · ±2.4 m
          </text>
          <text x={leak.x} y={leak.y + 40} fill="var(--fg-1)" fontSize="10" fontFamily="IBM Plex Mono" textAnchor="middle">
            main-N · joint J-14 · depth 1.8m
          </text>
        </svg>

        {/* Legend overlay */}
        <div style={{ position: 'absolute', left: 16, bottom: 16, padding: 12, background: 'oklch(0.18 0.012 240 / 0.92)', backdropFilter: 'blur(6px)', border: '1px solid var(--line-soft)', borderRadius: 6 }}>
          <div className="lbl" style={{ marginBottom: 6 }}>TDOA pipeline</div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--fg-1)', lineHeight: 1.7 }}>
            1 · GCC-PHAT cross-correlate frame pairs<br/>
            2 · solve hyperbolic isochrones (∆t·c)<br/>
            3 · least-squares + Kalman smooth<br/>
            4 · snap to PostGIS pipe geometry
          </div>
        </div>
      </div>

      {/* Right rail: solution + actions */}
      <div style={{ borderLeft: '1px solid var(--line-soft)', display: 'flex', flexDirection: 'column', background: 'var(--bg-0)' }}>
        <div className="panel-h" style={{ borderBottom: '1px solid var(--line-soft)' }}><span>Hypothesis</span><span className="mono">conf 0.91</span></div>
        <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 14, flex: 1, overflow: 'auto' }}>
          <div style={{ padding: 12, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 4 }}>
            <div className="lbl">LOCATION</div>
            <div className="mono" style={{ fontSize: 14, marginTop: 4, color: 'var(--fg-0)' }}>33.8961, 35.4791</div>
            <div className="mono" style={{ fontSize: 11, color: 'var(--fg-2)', marginTop: 2 }}>±2.4 m · 95% CI ellipse</div>
            <hr style={{ border: 0, borderTop: '1px dashed var(--line-soft)', margin: '10px 0' }} />
            <div className="lbl">PIPE</div>
            <div className="mono" style={{ fontSize: 12, marginTop: 4 }}>main-N · J-14 · PVC Ø160</div>
            <div className="mono" style={{ fontSize: 11, color: 'var(--fg-2)' }}>installed 2017 · depth 1.8m</div>
          </div>

          <div className="lbl">SENSORS CONTRIBUTING</div>
          {sensors.map(s => (
            <div key={s.id} className="row between mono" style={{ fontSize: 11, padding: '6px 8px', background: 'var(--bg-1)', borderRadius: 3, border: '1px solid var(--line-soft)' }}>
              <span style={{ color: 'var(--fg-0)' }}>{s.id}</span>
              <span style={{ color: 'var(--fg-2)' }}>{s.dist.toFixed(1)}m</span>
              <span style={{ color: 'var(--signal)' }}>Δt={s.tdoa.toFixed(3)}s</span>
            </div>
          ))}

          <div style={{ padding: 12, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 4 }}>
            <div className="lbl">FLOW ESTIMATE</div>
            <div className="mono" style={{ fontSize: 18, marginTop: 4, color: 'var(--crit)' }}>0.47 L/s</div>
            <div className="mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 2 }}>~ 1,690 L/hr · 40,500 L/day</div>
            <div className="mono" style={{ fontSize: 10, color: 'var(--fg-2)' }}>est. cost USD 81/day at $2.0/m³</div>
          </div>

          <button className="btn primary" style={{ padding: '10px' }}>Dispatch crew · ETA 18min</button>
          <button className="btn">Open in GIS</button>
        </div>
      </div>
    </div>
  );
};

window.TdoaFusion = TdoaFusion;
