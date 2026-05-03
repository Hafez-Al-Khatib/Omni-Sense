// Screen 1: Ops Console — Beirut sensor map + alert queue + KPIs
// REDESIGN v3:
// - Beirut shaped as recognizable peninsula with district polygons
// - Animated KPI counters
// - Sonar-sweep on critical markers
// - Asymmetric layout: hero KPI band on top, map dominates, alert queue right rail
// - Larger Orbitron numerics, color-blocked status tiles instead of uniform dark cards

const useCounter = (target, duration = 1100) => {
  const [v, setV] = React.useState(0);
  React.useEffect(() => {
    const start = performance.now();
    let raf;
    const step = (t) => {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setV(target * eased);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return v;
};

const HeroKpi = ({ label, value, suffix = '', accent = '#1a8fe3', sub, trend }) => {
  const display = useCounter(value);
  const isInt = Number.isInteger(value);
  return (
    <div style={{
      flex: 1, position: 'relative',
      padding: '12px 14px',
      borderRight: '1px solid rgba(91,184,245,0.35)',
      minWidth: 0,
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{
        fontFamily: "'Rajdhani',sans-serif", fontSize: 9.5, fontWeight: 600,
        color: '#3a7ab0', letterSpacing: 0.8, textTransform: 'uppercase',
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
      }}>{label}</div>
      <div style={{
        display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 8, marginTop: 2,
      }}>
        <div style={{
          fontFamily: "'Orbitron',sans-serif", fontSize: 26, fontWeight: 800,
          color: accent, letterSpacing: 0.5, lineHeight: 1, 
          textShadow: `0 0 18px ${accent}33`,
          whiteSpace: 'nowrap',
        }}>
          {isInt ? Math.round(display) : display.toFixed(2)}<span style={{ fontSize: 13, marginLeft: 2, fontWeight: 600 }}>{suffix}</span>
        </div>
        <div style={{ flexShrink: 0, marginBottom: 3 }}>{trend}</div>
      </div>
      {sub && (
        <div style={{
          fontFamily: "'IBM Plex Mono',monospace", fontSize: 9.5,
          color: '#3a7ab0', marginTop: 3, letterSpacing: 0.2,
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>{sub}</div>
      )}
    </div>
  );
};

const Sparkbar = ({ data, accent = '#1a8fe3', w = 56, h = 16 }) => {
  const max = Math.max(...data);
  return (
    <svg width={w} height={h} style={{ display: 'block', flexShrink: 0 }}>
      {data.map((d, i) => {
        const bw = w / data.length - 1;
        const bh = (d / max) * h;
        return <rect key={i} x={i * (bw + 1)} y={h - bh} width={bw} height={bh}
                     fill={accent} opacity={0.35 + 0.65 * (i / data.length)} rx="0.5" />;
      })}
    </svg>
  );
};

const OpsConsole = () => {
  const [selected, setSelected] = React.useState('WO-26041');
  const [tick, setTick] = React.useState(0);
  React.useEffect(() => { const i = setInterval(() => setTick(t => t + 1), 1600); return () => clearInterval(i); }, []);

  const sevColor = (s) => s === 'CRITICAL' ? 'crit' : s === 'HIGH' ? 'warn' : s === 'MEDIUM' ? 'viol' : 'sig';

  return (
    <div className="os-art os-fixed" style={{
      display: 'grid',
      gridTemplateColumns: 'minmax(0, 1fr) 360px',
      gridTemplateRows: '118px minmax(0, 1fr)',
      height: '100%',
      width: '100%',
      background: '#e8f4fd',
    }}>
      {/* HERO KPI BAND — full-width, asymmetric, breaks the card mono-block */}
      <div style={{
        gridColumn: '1 / -1',
        display: 'flex',
        background: 'linear-gradient(135deg, #ffffff 0%, #eaf4fc 50%, #d8ecf9 100%)',
        borderBottom: '1.5px solid #5bb8f5',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Decorative orbital arcs */}
        <svg style={{ position: 'absolute', right: -40, top: -30, opacity: 0.12, pointerEvents: 'none' }}
             width="220" height="220" viewBox="0 0 220 220">
          <circle cx="110" cy="110" r="80" fill="none" stroke="#1a8fe3" strokeWidth="0.8" strokeDasharray="3 3"/>
          <circle cx="110" cy="110" r="50" fill="none" stroke="#1a8fe3" strokeWidth="0.8"/>
          <circle cx="110" cy="110" r="20" fill="none" stroke="#1a8fe3" strokeWidth="0.8"/>
        </svg>

        <HeroKpi label="Sensors" value={47} suffix="/52" accent="#0a7e51"
                 sub="2 OOD · 3 offline"
                 trend={<Sparkbar data={[42,45,46,46,47,47,47,47]} accent="#0fa36b"/>}/>
        <HeroKpi label="Alerts" value={4} accent="#d94040"
                 sub="1 crit · 1 high · 2 med"
                 trend={<Sparkbar data={[2,3,5,4,6,4,3,4]} accent="#d94040"/>}/>
        <HeroKpi label="Resolved 24h" value={12} accent="#0e75c8"
                 sub="MTTR 2h 14m · SLA 96%"
                 trend={<Sparkbar data={[8,10,9,11,12,11,12,12]} accent="#1a8fe3"/>}/>
        <HeroKpi label="ROC-AUC" value={0.974} accent="#0e75c8"
                 sub="F1 0.95 · θ 0.41 · KS 0.04"
                 trend={
                   <span style={{
                     fontFamily: "'Rajdhani',sans-serif", fontSize: 9, fontWeight: 700,
                     color: '#0a7e51', background: 'rgba(15,163,107,0.14)',
                     padding: '2px 6px', borderRadius: 4, letterSpacing: 0.5,
                     whiteSpace: 'nowrap',
                   }}>+0.3% wk</span>
                 }/>
        <HeroKpi label="p95 Latency" value={612} suffix=" ms" accent="#1a8fe3"
                 sub="ingest → dispatch · ≤ 800ms"
                 trend={<Sparkbar data={[680,640,620,610,610,605,612,612]} accent="#1a8fe3"/>}/>
        <HeroKpi label="OOD Queue" value={6} accent="#7c3aed"
                 sub="auto-routed · 2 reviewed"
                 trend={<Sparkbar data={[2,3,4,3,5,5,6,6]} accent="#7c3aed"/>}/>
      </div>

      {/* CENTER: Beirut map */}
      <div style={{ position: 'relative', overflow: 'hidden', background: '#cfe6f8', minWidth: 0, minHeight: 0 }}>
        <BeirutPolygonMap sensors={SENSORS} selected={selected} setSelected={setSelected} tick={tick}/>

        {/* Floating overlay — top-left only, very compact */}
        <div style={{
          position: 'absolute', left: 16, top: 16,
          background: 'rgba(255,255,255,0.94)', backdropFilter: 'blur(8px)',
          border: '1px solid #5bb8f5', borderRadius: 8,
          boxShadow: '0 4px 20px rgba(26,143,227,0.18)',
          padding: '8px 12px',
        }}>
          <div style={{
            fontFamily: "'Orbitron',sans-serif", fontWeight: 800, fontSize: 12,
            letterSpacing: 2.5, color: '#0e75c8', whiteSpace: 'nowrap',
          }}>GREATER BEIRUT</div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8, marginTop: 2,
          }}>
            <span style={{
              fontFamily: "'IBM Plex Mono',monospace", fontSize: 9.5, color: '#0a7e51', fontWeight: 600,
            }}>● {String(14 + Math.floor(tick / 38)).padStart(2, '0')}:22:{String((8 + tick) % 60).padStart(2, '0')}</span>
            <span style={{
              fontFamily: "'Rajdhani',sans-serif", fontSize: 9.5,
              color: '#3a7ab0', letterSpacing: 0.8, fontWeight: 500,
            }}>184 KM · 12 DISTRICTS</span>
          </div>
        </div>

        {/* District health summary — bottom strip (number above name) */}
        <div style={{
          position: 'absolute', left: 16, bottom: 16, right: 16,
          background: 'rgba(255,255,255,0.94)', backdropFilter: 'blur(8px)',
          border: '1px solid #5bb8f5', borderRadius: 8, padding: '8px 14px',
          display: 'flex', gap: 6, justifyContent: 'space-between',
          boxShadow: '0 4px 20px rgba(26,143,227,0.18)',
        }}>
          {[
            { d: 'Hamra',     n: 9,  color: '#0fa36b', s: 'OK' },
            { d: 'Achrafieh', n: 7,  color: '#d94040', s: 'LEAK' },
            { d: 'Verdun',    n: 6,  color: '#0fa36b', s: 'OK' },
            { d: 'Mar Mikhael', n: 5, color: '#7c3aed', s: 'OOD' },
            { d: 'Badaro',    n: 8,  color: '#0fa36b', s: 'OK' },
            { d: 'Ras Beirut', n: 12, color: '#0fa36b', s: 'OK' },
          ].map(d => (
            <div key={d.d} style={{ flex: 1, minWidth: 0, textAlign: 'center', padding: '0 4px', borderLeft: '1px solid rgba(91,184,245,0.3)' }}>
              <div style={{
                fontFamily: "'Orbitron',sans-serif", fontWeight: 800, fontSize: 20,
                color: d.color, lineHeight: 1,
                textShadow: `0 0 6px ${d.color}33`,
              }}>{d.n}</div>
              <div style={{
                fontFamily: "'Rajdhani',sans-serif", fontSize: 9.5, color: '#0b3a5e',
                letterSpacing: 0.4, fontWeight: 600, marginTop: 3,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>{d.d}</div>
              <div style={{
                fontFamily: "'IBM Plex Mono',monospace", fontSize: 8.5, color: d.color,
                fontWeight: 700, letterSpacing: 0.5, marginTop: 1,
              }}>{d.s}</div>
            </div>
          ))}
        </div>
      </div>

      {/* RIGHT: Alert queue */}
      <div style={{
        borderLeft: '1.5px solid #5bb8f5',
        display: 'flex', flexDirection: 'column',
        background: 'linear-gradient(180deg, #ffffff 0%, #eef6fc 100%)',
      }}>
        <div style={{
          padding: '14px 16px', borderBottom: '1.5px solid #5bb8f5',
          background: 'linear-gradient(180deg, #b8ddf7 0%, #d0eaf9 100%)',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontFamily: "'Orbitron',sans-serif", fontWeight: 800, fontSize: 13, letterSpacing: 3, color: '#0e75c8' }}>ALERT QUEUE</div>
              <div style={{ fontFamily: "'Rajdhani',sans-serif", fontSize: 11, color: '#3a7ab0', letterSpacing: 1.5, marginTop: 2 }}>{ALERTS.length} ACTIVE · SORTED BY SEVERITY</div>
            </div>
            <span style={{
              fontFamily: "'Orbitron',sans-serif", fontWeight: 700, fontSize: 22,
              color: '#d94040', letterSpacing: 1,
              textShadow: '0 0 12px rgba(217,64,64,0.4)',
            }}>{ALERTS.length}</span>
          </div>
        </div>
        <div className="scrollbar" style={{ overflowY: 'auto', flex: 1, padding: '6px 10px' }}>
          {ALERTS.map(a => (
            <AlertCard key={a.id} alert={a} sevColor={sevColor} selected={a.id === selected} onClick={() => setSelected(a.id)} />
          ))}
        </div>
        <div style={{
          padding: 14, borderTop: '1.5px solid #5bb8f5',
          background: 'linear-gradient(0deg, rgba(184,221,247,0.4), transparent)',
        }}>
          <div style={{ fontFamily: "'Rajdhani',sans-serif", fontSize: 10, color: '#3a7ab0', letterSpacing: 1.5, fontWeight: 600, marginBottom: 8 }}>SLA · RESPONSE TARGETS</div>
          <div style={{
            fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, color: '#0b3a5e',
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px',
          }}>
            <span>CRITICAL</span><span style={{ color: '#d94040', textAlign: 'right', fontWeight: 600 }}>5 min</span>
            <span>HIGH</span><span style={{ color: '#e07b2a', textAlign: 'right', fontWeight: 600 }}>30 min</span>
            <span>MEDIUM</span><span style={{ color: '#7c3aed', textAlign: 'right', fontWeight: 600 }}>2 hrs</span>
            <span>LOW</span><span style={{ color: '#0e75c8', textAlign: 'right', fontWeight: 600 }}>8 hrs</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// === ALERT CARD ===
const AlertCard = ({ alert, sevColor, selected, onClick }) => {
  const sev = sevColor(alert.severity);
  const sevColorMap = {
    crit: '#d94040', warn: '#e07b2a', viol: '#7c3aed', sig: '#1a8fe3',
  };
  const c = sevColorMap[sev];
  return (
    <div onClick={onClick} style={{
      padding: 12,
      marginBottom: 6,
      cursor: 'pointer',
      background: selected ? `linear-gradient(135deg, ${c}18, ${c}08)` : '#ffffff',
      border: `1px solid ${selected ? c : '#cee3f5'}`,
      borderLeft: `4px solid ${c}`,
      borderRadius: 8,
      boxShadow: selected ? `0 4px 14px ${c}30` : '0 1px 2px rgba(26,143,227,0.06)',
      transition: 'all 0.16s ease',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{
          fontFamily: "'Rajdhani',sans-serif", fontSize: 10, fontWeight: 700,
          color: c, background: `${c}18`,
          padding: '2px 8px', borderRadius: 999, letterSpacing: 1.2,
        }}>{alert.severity}</span>
        <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, color: '#3a7ab0' }}>{alert.id}</span>
      </div>
      <div style={{ marginTop: 8, fontWeight: 700, fontSize: 13, color: '#0b3a5e', fontFamily: "'Rajdhani',sans-serif", letterSpacing: 0.4 }}>{alert.site}</div>
      <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 11, color: '#3a7ab0', marginTop: 2 }}>{alert.fault}</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, color: '#3a7ab0' }}>
        <span>
          {alert.confidence != null ? `conf ${fmtPct(alert.confidence)}` : 'OOD · no inference'}
          {alert.flow != null && ` · ${alert.flow} L/s`}
        </span>
        <span>{alert.detected}</span>
      </div>
      <div style={{ marginTop: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontFamily: "'Rajdhani',sans-serif", fontSize: 9, color: '#3a7ab0', letterSpacing: 1.5, fontWeight: 600 }}>SLA</span>
          <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, color: sev === 'crit' ? c : '#0b3a5e', fontWeight: 600 }}>{alert.sla_label}</span>
        </div>
        <div style={{ height: 5, background: '#e6f0fa', borderRadius: 999, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${alert.sla_remaining_pct * 100}%`,
            background: `linear-gradient(90deg, ${c}, ${c}cc)`,
            transition: 'width 1s linear',
            boxShadow: `0 0 8px ${c}80`,
          }}/>
        </div>
      </div>
    </div>
  );
};

// === BEIRUT POLYGON MAP — district-shaped, recognizable ===
// Beirut peninsula juts west into the Mediterranean. Districts arranged faithfully:
//   Ras Beirut + AUB (NW tip), Hamra (W), Verdun (SW), Manara/Ain el-Mreisseh (N coast),
//   Downtown/Beirut Central (N-center), Gemmayzeh + Mar Mikhael (NE), Achrafieh (E center),
//   Badaro (S center), Sin el Fil (E), Burj Hammoud (NE inland)
const BeirutPolygonMap = ({ sensors, selected, setSelected, tick }) => {
  // viewBox 1000x720 — mirror the Beirut peninsula shape (jutting NW)
  // sensor lat/lng → x,y in viewBox
  // bbox: lng 35.470..35.555, lat 33.870..33.910 (north up)
  const project = (lat, lng) => {
    const x = ((lng - 35.460) / (35.555 - 35.460)) * 1000;
    const y = (1 - (lat - 33.865) / (33.910 - 33.865)) * 720;
    return { x, y };
  };

  // District polygons — hand-tuned to read as Beirut
  const districts = [
    { id: 'ras-beirut', name: 'RAS BEIRUT', d: 'M 110 260 L 175 240 L 235 250 L 270 270 L 275 320 L 240 345 L 180 348 L 130 340 L 100 310 Z', leak: false, watch: false },
    { id: 'hamra',      name: 'HAMRA',      d: 'M 270 270 L 350 265 L 395 285 L 405 335 L 365 360 L 305 360 L 275 320 Z', leak: false, watch: false },
    { id: 'manara',     name: 'AIN EL-MREISSEH', d: 'M 235 250 L 315 215 L 405 215 L 410 260 L 395 285 L 350 265 L 270 270 Z', leak: false, watch: false },
    { id: 'downtown',   name: 'DOWNTOWN',   d: 'M 410 260 L 510 240 L 575 250 L 580 305 L 540 330 L 490 330 L 405 335 L 395 285 Z', leak: false, watch: true },
    { id: 'gemmayzeh',  name: 'GEMMAYZEH',  d: 'M 575 250 L 645 240 L 690 255 L 695 295 L 660 310 L 580 305 Z', leak: false, watch: false },
    { id: 'mar-mikhael', name: 'MAR MIKHAEL', d: 'M 645 240 L 730 235 L 770 260 L 765 295 L 720 305 L 695 295 L 690 255 Z', leak: false, watch: false },
    { id: 'achrafieh',  name: 'ACHRAFIEH',  d: 'M 540 330 L 580 305 L 660 310 L 720 305 L 730 360 L 700 405 L 615 410 L 555 395 Z', leak: true, watch: false },
    { id: 'verdun',     name: 'VERDUN',     d: 'M 240 345 L 305 360 L 365 360 L 405 335 L 425 395 L 395 445 L 320 450 L 260 425 Z', leak: false, watch: false },
    { id: 'badaro',     name: 'BADARO',     d: 'M 425 395 L 555 395 L 615 410 L 600 470 L 540 495 L 470 490 L 415 460 Z', leak: false, watch: false },
    { id: 'sin-el-fil', name: 'SIN EL FIL', d: 'M 700 405 L 770 395 L 820 415 L 815 470 L 760 490 L 700 480 L 670 450 L 615 410 Z', leak: false, watch: false },
    { id: 'burj-hammoud', name: 'BURJ HAMMOUD', d: 'M 730 235 L 815 235 L 865 265 L 870 320 L 840 345 L 770 350 L 765 295 L 770 260 Z', leak: false, watch: false },
  ];

  const districtFill = (d) => {
    if (d.leak) return 'url(#fillLeak)';
    if (d.watch) return 'url(#fillWatch)';
    return 'url(#fillNormal)';
  };

  const sensorColor = (s) => ({
    crit: '#d94040', warn: '#e07b2a', ood: '#7c3aed', off: '#94a8be',
  })[s.status] || '#0fa36b';

  return (
    <svg width="100%" height="100%" viewBox="0 0 1000 700" preserveAspectRatio="xMidYMid slice"
         style={{ display: 'block' }}>
      <defs>
        {/* Sea radial gradient */}
        <radialGradient id="sea" cx="20%" cy="55%" r="90%">
          <stop offset="0%" stopColor="#a4d3f0"/>
          <stop offset="60%" stopColor="#c8e2f5"/>
          <stop offset="100%" stopColor="#d8ecf9"/>
        </radialGradient>
        {/* Land district fills */}
        <linearGradient id="fillNormal" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f4faff"/><stop offset="100%" stopColor="#dbecf8"/>
        </linearGradient>
        <linearGradient id="fillWatch" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#fde8d0"/><stop offset="100%" stopColor="#f5d4a8"/>
        </linearGradient>
        <linearGradient id="fillLeak" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#fbd2d2"/><stop offset="100%" stopColor="#f5a8a8"/>
        </linearGradient>
        <pattern id="dotgrid" width="14" height="14" patternUnits="userSpaceOnUse">
          <circle cx="7" cy="7" r="0.6" fill="#1a8fe3" opacity="0.18"/>
        </pattern>
        <filter id="softShadow"><feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#1a8fe3" floodOpacity="0.18"/></filter>
      </defs>

      {/* Sea background */}
      <rect width="1000" height="720" fill="url(#sea)"/>
      <rect width="1000" height="720" fill="url(#dotgrid)"/>

      {/* Sea ripple lines (parallel to coast) */}
      <g stroke="#1a8fe3" strokeWidth="0.6" fill="none" opacity="0.18">
        <path d="M 0 200 Q 80 215, 110 260" />
        <path d="M 0 280 Q 70 295, 100 310" />
        <path d="M 0 360 Q 60 380, 130 410" />
        <path d="M 0 440 Q 80 460, 180 490" />
      </g>
      <g stroke="#1a8fe3" strokeWidth="0.4" fill="none" opacity="0.10" strokeDasharray="3 4">
        <path d="M 0 240 Q 60 250, 90 280" />
        <path d="M 0 400 Q 70 420, 150 450" />
      </g>

      {/* MEDITERRANEAN label */}
      <text x="40" y="180" fill="#3a7ab0" opacity="0.7" fontFamily="Orbitron, sans-serif"
            fontSize="14" fontWeight="700" letterSpacing="6">MEDITERRANEAN</text>
      <text x="40" y="200" fill="#3a7ab0" opacity="0.5" fontFamily="Rajdhani, sans-serif"
            fontSize="10" letterSpacing="3">SEA · MARE NOSTRUM</text>

      {/* Beirut peninsula outer outline (subtle) */}
      <path d="M 110 260 L 175 240 L 235 250 L 315 215 L 405 215 L 510 240 L 645 240 L 815 235 L 865 265 L 870 320 L 820 415 L 815 470 L 760 490 L 700 480 L 600 470 L 540 495 L 470 490 L 395 445 L 320 450 L 260 425 L 100 310 Z"
            fill="none" stroke="#5bb8f5" strokeWidth="2" opacity="0.4" filter="url(#softShadow)"/>

      {/* District polygons */}
      {districts.map(d => (
        <g key={d.id}>
          <path d={d.d} fill={districtFill(d)} stroke="#5bb8f5" strokeWidth="1.2"
                opacity="0.92"/>
          {/* Subtle inner highlight */}
          <path d={d.d} fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="1" opacity="0.4"/>
        </g>
      ))}

      {/* Pipe spine — passes through districts */}
      <g stroke="#1a8fe3" strokeWidth="2.5" fill="none" opacity="0.55"
         strokeLinecap="round" strokeLinejoin="round">
        <path d="M 150 295 L 320 305 L 480 295 L 620 295 L 770 290 L 820 320" />
        <path d="M 200 410 L 350 415 L 490 425 L 615 425 L 745 425" />
        <path d="M 320 305 L 320 415" />
        <path d="M 490 295 L 490 425" />
        <path d="M 620 295 L 615 425" />
      </g>
      {/* Pipe flow animation — moving dashes */}
      <g stroke="#0fa36b" strokeWidth="1.4" fill="none" strokeDasharray="6 12" opacity="0.85">
        <path d="M 150 295 L 320 305 L 480 295 L 620 295 L 770 290 L 820 320">
          <animate attributeName="stroke-dashoffset" from="0" to="-72" dur="3s" repeatCount="indefinite"/>
        </path>
        <path d="M 200 410 L 350 415 L 490 425 L 615 425 L 745 425">
          <animate attributeName="stroke-dashoffset" from="0" to="72" dur="3.6s" repeatCount="indefinite"/>
        </path>
      </g>

      {/* District labels */}
      {districts.map(d => {
        // compute centroid roughly from path d (use first 3 points as crude approx)
        const pts = d.d.match(/-?\d+(\.\d+)?/g)?.map(Number) || [];
        let cx = 0, cy = 0, n = 0;
        for (let i = 0; i + 1 < pts.length; i += 2) { cx += pts[i]; cy += pts[i+1]; n++; }
        cx /= n; cy /= n;
        // Shorter label for narrow districts
        const label = d.name.length > 12 ? d.name.split(' ')[0] : d.name;
        return (
          <text key={d.id + '-l'} x={cx} y={cy}
                fontFamily="Rajdhani, sans-serif" fontSize="9.5" fontWeight="700"
                fill="#0b3a5e" letterSpacing="1.2"
                textAnchor="middle" opacity="0.78"
                style={{ pointerEvents: 'none' }}>
            {label}
          </text>
        );
      })}

      {/* Sensors */}
      {sensors.map(s => {
        const { x, y } = project(s.lat, s.lng);
        const c = sensorColor(s);
        const isAlert = s.status === 'crit';
        const isWatch = s.status === 'warn';
        const isOOD = s.status === 'ood';
        const isOff = s.status === 'off';

        return (
          <g key={s.id} style={{ cursor: 'pointer' }}>
            {/* Sonar sweep — only critical */}
            {isAlert && (
              <>
                <circle cx={x} cy={y} r="14" fill="none" stroke={c} strokeWidth="1.5" opacity="0.9"
                        style={{ animation: 'sonar 2.6s ease-out infinite', transformOrigin: `${x}px ${y}px` }}/>
                <circle cx={x} cy={y} r="14" fill="none" stroke={c} strokeWidth="1.5" opacity="0.9"
                        style={{ animation: 'sonar 2.6s ease-out infinite 1.3s', transformOrigin: `${x}px ${y}px` }}/>
              </>
            )}
            {isWatch && (
              <circle cx={x} cy={y} r="11" fill="none" stroke={c} strokeWidth="1.2" opacity="0.7"
                      style={{ animation: 'sonar 3.4s ease-out infinite', transformOrigin: `${x}px ${y}px` }}/>
            )}

            {/* Halo */}
            <circle cx={x} cy={y} r="9" fill={c} opacity="0.20"/>
            {/* Outer ring */}
            <circle cx={x} cy={y} r="6" fill="white" stroke={c} strokeWidth="2"/>
            {/* Inner dot */}
            <circle cx={x} cy={y} r="3" fill={c}>
              {(isAlert || isWatch) && (
                <animate attributeName="opacity" values="1;0.4;1" dur="1.4s" repeatCount="indefinite"/>
              )}
            </circle>

            {/* Selected ID badge */}
            {selected && s.id === SENSORS.find(x => x.id === ALERTS.find(a => a.id === selected)?.sensor)?.id && (
              <g>
                <rect x={x + 12} y={y - 22} width="120" height="44" rx="6"
                      fill="white" stroke={c} strokeWidth="1.5"
                      filter="url(#softShadow)"/>
                <text x={x + 18} y={y - 8} fontFamily="Orbitron, sans-serif"
                      fontSize="10" fontWeight="700" fill={c} letterSpacing="1.5">{selected}</text>
                <text x={x + 18} y={y + 4} fontFamily="Rajdhani, sans-serif"
                      fontSize="11" fontWeight="600" fill="#0b3a5e">
                  {ALERTS.find(a => a.id === selected)?.fault}
                </text>
                <text x={x + 18} y={y + 16} fontFamily="IBM Plex Mono, monospace"
                      fontSize="9" fill="#3a7ab0">
                  conf {fmtPct(ALERTS.find(a => a.id === selected)?.confidence || 0)}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Compass rose — top-right */}
      <g transform="translate(940, 60)" opacity="0.8">
        <circle r="22" fill="white" stroke="#5bb8f5" strokeWidth="1.2"/>
        <path d="M 0 -16 L 4 0 L 0 16 L -4 0 Z" fill="#1a8fe3" opacity="0.85"/>
        <path d="M -16 0 L 0 -3 L 16 0 L 0 3 Z" fill="#5bb8f5" opacity="0.5"/>
        <text x="0" y="-26" textAnchor="middle" fontFamily="Orbitron,sans-serif"
              fontSize="9" fontWeight="700" fill="#0e75c8">N</text>
      </g>

      {/* Scale bar — bottom right */}
      <g transform="translate(840, 670)">
        <rect width="80" height="3" fill="#0b3a5e"/>
        <rect width="40" height="3" fill="#1a8fe3"/>
        <text x="0" y="-4" fontFamily="Rajdhani,sans-serif" fontSize="10" fill="#3a7ab0" letterSpacing="1">0</text>
        <text x="36" y="-4" fontFamily="Rajdhani,sans-serif" fontSize="10" fill="#3a7ab0" letterSpacing="1">1km</text>
        <text x="76" y="-4" fontFamily="Rajdhani,sans-serif" fontSize="10" fill="#3a7ab0" letterSpacing="1">2km</text>
      </g>

      <style>{`
        @keyframes sonar {
          0%   { r: 8;  opacity: 0.9; }
          100% { r: 38; opacity: 0;   }
        }
      `}</style>
    </svg>
  );
};

window.OpsConsole = OpsConsole;
