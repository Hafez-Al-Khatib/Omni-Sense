// Screen 4: Field-crew mobile app — iOS frame, technician gets ticket, confirms / false-alarms
// Closes the active-learning loop back to MLflow.

const FieldCrewApp = () => {
  const [tab, setTab] = React.useState('ticket');
  return (
    <IOSDevice height={780}>
      <div style={{ minHeight: '100%', background: 'oklch(0.97 0.005 240)', display: 'flex', flexDirection: 'column', paddingTop: 50 }}>
        {/* App header */}
        <div style={{ padding: '8px 16px 12px', background: 'white', borderBottom: '1px solid oklch(0.92 0.005 240)' }}>
          <div className="row between">
            <div>
              <div style={{ fontSize: 11, color: 'oklch(0.5 0.01 240)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Omni·Field</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: 'oklch(0.18 0.01 240)' }}>Hi, Karim</div>
            </div>
            <div style={{ width: 40, height: 40, borderRadius: 999, background: 'oklch(0.78 0.13 200)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 700 }}>K</div>
          </div>
          <div className="row gap-2" style={{ marginTop: 12 }}>
            <span style={{ padding: '4px 10px', borderRadius: 999, background: 'oklch(0.66 0.18 25 / 0.12)', color: 'oklch(0.55 0.18 25)', fontSize: 11, fontWeight: 600, fontFamily: 'IBM Plex Mono' }}>● ON SHIFT</span>
            <span style={{ padding: '4px 10px', borderRadius: 999, background: 'oklch(0.94 0.005 240)', color: 'oklch(0.4 0.01 240)', fontSize: 11, fontFamily: 'IBM Plex Mono' }}>Truck #4 · Hamra</span>
          </div>
        </div>

        {tab === 'ticket' && <TicketView />}
        {tab === 'list' && <TicketList />}

        {/* tab bar */}
        <div style={{ borderTop: '1px solid oklch(0.92 0.005 240)', background: 'white', padding: '8px 16px 24px', display: 'flex', justifyContent: 'space-around' }}>
          <TabBtn active={tab === 'ticket'} icon="⚠" label="Ticket" onClick={() => setTab('ticket')} />
          <TabBtn active={tab === 'list'}   icon="≡" label="Queue · 3" onClick={() => setTab('list')} />
          <TabBtn icon="◯" label="Map" />
          <TabBtn icon="◔" label="Profile" />
        </div>
      </div>
    </IOSDevice>
  );
};

const TabBtn = ({ active, icon, label, onClick }) => (
  <div onClick={onClick} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, color: active ? 'oklch(0.50 0.18 25)' : 'oklch(0.55 0.01 240)', cursor: 'pointer' }}>
    <div style={{ fontSize: 18 }}>{icon}</div>
    <div style={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }}>{label}</div>
  </div>
);

const TicketView = () => (
  <div style={{ flex: 1, overflow: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 12, background: 'oklch(0.97 0.005 240)' }}>
    {/* Hero ticket card */}
    <div style={{ background: 'oklch(0.18 0.012 240)', color: 'white', borderRadius: 14, padding: 16, boxShadow: '0 4px 12px oklch(0 0 0 / 0.12)' }}>
      <div className="row between">
        <span style={{ background: 'oklch(0.66 0.18 25)', color: 'white', padding: '3px 10px', borderRadius: 4, fontSize: 10, fontWeight: 700, fontFamily: 'IBM Plex Mono', letterSpacing: '0.06em' }}>CRITICAL</span>
        <span style={{ fontSize: 11, fontFamily: 'IBM Plex Mono', color: 'oklch(0.7 0.01 240)' }}>WO-26041</span>
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, marginTop: 14 }}>Hamra · Bliss St</div>
      <div style={{ fontSize: 13, color: 'oklch(0.78 0.01 240)', fontFamily: 'IBM Plex Mono', marginTop: 4 }}>Branched · Orifice Leak</div>
      <hr style={{ border: 0, borderTop: '1px dashed oklch(0.32 0.012 240)', margin: '14px 0' }} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
        <Kpi label="CONF" value="96%" tint="oklch(0.78 0.13 200)" />
        <Kpi label="FLOW" value="0.47 L/s" tint="oklch(0.66 0.18 25)" />
        <Kpi label="SLA" value="01:42" tint="oklch(0.80 0.14 75)" />
      </div>
      <div style={{ marginTop: 14, fontSize: 12, fontFamily: 'IBM Plex Mono', color: 'oklch(0.78 0.01 240)' }}>
        ±2.4 m · main-N joint J-14 · 1.8 m depth
      </div>
    </div>

    {/* Mini map */}
    <div style={{ background: 'white', borderRadius: 12, overflow: 'hidden', border: '1px solid oklch(0.92 0.005 240)' }}>
      <div style={{ height: 140, position: 'relative', background: 'oklch(0.95 0.01 240)' }}>
        <svg width="100%" height="100%" viewBox="0 0 360 140">
          <defs>
            <pattern id="fld-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="oklch(0.92 0.01 240)" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="360" height="140" fill="url(#fld-grid)" />
          <path d="M 20 70 L 340 70" stroke="oklch(0.65 0.01 240)" strokeWidth="6" strokeLinecap="round" />
          <path d="M 180 20 L 180 130" stroke="oklch(0.65 0.01 240)" strokeWidth="6" strokeLinecap="round" />
          <circle cx="180" cy="70" r="40" fill="oklch(0.66 0.18 25 / 0.12)" />
          <ellipse cx="180" cy="70" rx="14" ry="8" fill="none" stroke="oklch(0.55 0.18 25)" strokeWidth="2" strokeDasharray="3 2" />
          <circle cx="180" cy="70" r="4" fill="oklch(0.55 0.18 25)" />
          <circle cx="80" cy="40" r="5" fill="white" stroke="oklch(0.78 0.13 200)" strokeWidth="2" />
          <circle cx="290" cy="100" r="5" fill="white" stroke="oklch(0.78 0.13 200)" strokeWidth="2" />
          <text x="180" y="55" textAnchor="middle" fontSize="9" fontFamily="IBM Plex Mono" fill="oklch(0.55 0.18 25)" fontWeight="600">±2.4m</text>
        </svg>
      </div>
      <div className="row between" style={{ padding: 12 }}>
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'oklch(0.18 0.01 240)' }}>Bliss St · J-14</div>
          <div style={{ fontSize: 11, color: 'oklch(0.5 0.01 240)', fontFamily: 'IBM Plex Mono' }}>Truck 4 · ETA 6 min · 1.2 km</div>
        </div>
        <button style={{ background: 'oklch(0.78 0.13 200)', color: 'oklch(0.15 0.02 240)', border: 0, padding: '8px 14px', borderRadius: 8, fontWeight: 700, fontSize: 12 }}>Navigate</button>
      </div>
    </div>

    {/* Ground truth feedback (the active-learning hook) */}
    <div style={{ background: 'white', borderRadius: 12, padding: 14, border: '1px solid oklch(0.92 0.005 240)' }}>
      <div style={{ fontSize: 11, color: 'oklch(0.5 0.01 240)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.06em' }}>ON-SITE FINDING</div>
      <div style={{ fontSize: 13, color: 'oklch(0.32 0.01 240)', marginTop: 4 }}>Tap once you've inspected the joint. Your call retrains the model tonight.</div>
      <div className="row gap-2" style={{ marginTop: 12 }}>
        <button style={{ flex: 1, background: 'oklch(0.66 0.18 25)', color: 'white', border: 0, padding: '12px', borderRadius: 10, fontWeight: 700, fontSize: 13 }}>✓ Confirm leak</button>
        <button style={{ flex: 1, background: 'white', color: 'oklch(0.32 0.01 240)', border: '1px solid oklch(0.85 0.01 240)', padding: '12px', borderRadius: 10, fontWeight: 600, fontSize: 13 }}>✕ False alarm</button>
      </div>
      <div style={{ fontSize: 10, color: 'oklch(0.5 0.01 240)', fontFamily: 'IBM Plex Mono', marginTop: 10, textAlign: 'center' }}>
        → label routes to MLflow · feedback_log.csv · 1,247 prior labels
      </div>
    </div>
  </div>
);

const Kpi = ({ label, value, tint }) => (
  <div>
    <div style={{ fontSize: 9, color: 'oklch(0.6 0.01 240)', fontFamily: 'IBM Plex Mono', letterSpacing: '0.06em' }}>{label}</div>
    <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'IBM Plex Mono', color: tint, marginTop: 2 }}>{value}</div>
  </div>
);

const TicketList = () => {
  const items = [
    { sev: 'CRIT', site: 'Hamra · Bliss St',     id: 'WO-26041', sla: '01:42', open: true },
    { sev: 'HIGH', site: 'Achrafieh · Sassine',  id: 'WO-26039', sla: '13:42', open: false },
    { sev: 'LOW',  site: 'Gemmayzeh · Gouraud',  id: 'WO-26031', sla: '04:38', open: false },
  ];
  const tint = (s) => s === 'CRIT' ? 'oklch(0.66 0.18 25)' : s === 'HIGH' ? 'oklch(0.80 0.14 75)' : 'oklch(0.78 0.13 200)';
  return (
    <div style={{ flex: 1, overflow: 'auto', padding: 16, background: 'oklch(0.97 0.005 240)' }}>
      <div style={{ fontSize: 11, color: 'oklch(0.5 0.01 240)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>My queue · 3 open</div>
      <div className="col gap-3">
        {items.map(it => (
          <div key={it.id} style={{ background: 'white', borderRadius: 10, padding: 14, border: it.open ? `2px solid ${tint(it.sev)}` : '1px solid oklch(0.92 0.005 240)' }}>
            <div className="row between">
              <span style={{ background: tint(it.sev), color: 'white', padding: '2px 8px', borderRadius: 3, fontSize: 9, fontWeight: 700, fontFamily: 'IBM Plex Mono' }}>{it.sev}</span>
              <span style={{ fontSize: 10, color: 'oklch(0.5 0.01 240)', fontFamily: 'IBM Plex Mono' }}>{it.id}</span>
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, marginTop: 6, color: 'oklch(0.18 0.01 240)' }}>{it.site}</div>
            <div className="row between" style={{ marginTop: 6 }}>
              <span style={{ fontSize: 11, color: 'oklch(0.5 0.01 240)', fontFamily: 'IBM Plex Mono' }}>SLA {it.sla}</span>
              {it.open && <span style={{ fontSize: 11, color: tint(it.sev), fontWeight: 600 }}>● OPEN</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

window.FieldCrewApp = FieldCrewApp;
