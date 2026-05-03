// Screen 6: Sensor fleet detail — drilldown on one edge device
const SensorDetail = () => {
  return (
    <div className="os-art" style={{ display: 'grid', gridTemplateColumns: '320px 1fr 320px', gridTemplateRows: '52px 1fr', height: '100%' }}>
      <div style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--line-soft)', padding: '0 18px', gap: 16, background: 'var(--bg-1)' }}>
        <button className="btn ghost" style={{ padding: '4px 8px' }}>←</button>
        <span className="tag sig">EDGE DEVICE</span>
        <span style={{ fontWeight: 600 }}>S-HAMRA-001 · Hamra · Bliss St</span>
        <div className="grow" />
        <span className="tag ok">ONLINE · mTLS verified</span>
      </div>

      {/* LEFT: device facts */}
      <div style={{ borderRight: '1px solid var(--line-soft)', padding: 16, overflow: 'auto', background: 'var(--bg-0)' }}>
        <div className="lbl">HARDWARE</div>
        <div style={{ marginTop: 8, padding: 12, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 4 }} className="mono">
          <Row k="Model"      v="Raspberry Pi 4 · 4GB" />
          <Row k="Sensor"     v="ADXL345 · I²C 0x53" />
          <Row k="Mount"      v="PVC Ø160 · epoxy clamp" />
          <Row k="Z-axis"     v="⊥ pipe wall ✓" />
          <Row k="Firmware"   v="omni-edge 2.4.1" />
          <Row k="Boot"       v="14d 7h 22m" />
        </div>

        <div className="lbl" style={{ marginTop: 18 }}>POWER</div>
        <div style={{ marginTop: 8, padding: 12, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 4 }}>
          <div className="row between"><span className="lbl">BATTERY</span><span className="mono" style={{ color: 'var(--ok)', fontSize: 14 }}>87%</span></div>
          <div style={{ marginTop: 6, height: 8, background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 3 }}>
            <div style={{ width: '87%', height: '100%', background: 'linear-gradient(90deg, var(--ok), var(--signal))' }} />
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 6 }}>est. 32 days remaining @ 1 frame / 60s · deep-sleep</div>
        </div>

        <div className="lbl" style={{ marginTop: 18 }}>mTLS CERTIFICATE</div>
        <div className="mono" style={{ marginTop: 8, padding: 12, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 4, fontSize: 11 }}>
          <Row k="CN"        v="S-HAMRA-001" />
          <Row k="Issued by" v="Omni-Sense CA" />
          <Row k="Valid"     v="71 / 90 days" tint="ok" />
          <div style={{ marginTop: 8, height: 4, background: 'var(--bg-0)', borderRadius: 2 }}>
            <div style={{ width: '79%', height: '100%', background: 'var(--ok)' }} />
          </div>
          <div style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 6 }}>auto-renews 14 days before expiry</div>
        </div>

        <div className="lbl" style={{ marginTop: 18 }}>QUIET-WINDOW SCHEDULE</div>
        <div style={{ marginTop: 8, padding: 12, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 4 }}>
          <svg width="100%" height="44" viewBox="0 0 280 44">
            {Array.from({ length: 24 }).map((_, h) => {
              const active = h >= 2 && h < 4;
              return <rect key={h} x={h * 11.5} y="8" width="10" height="28"
                            fill={active ? 'var(--signal)' : 'var(--bg-0)'} stroke="var(--line-soft)" strokeWidth="0.5" />;
            })}
            <text x="0" y="42" fontSize="9" fontFamily="IBM Plex Mono" fill="var(--fg-2)">00</text>
            <text x="135" y="42" fontSize="9" fontFamily="IBM Plex Mono" fill="var(--fg-2)">12</text>
            <text x="252" y="42" fontSize="9" fontFamily="IBM Plex Mono" fill="var(--fg-2)">23</text>
          </svg>
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 4 }}>Capture: 02:00 – 04:00 · max SNR · 5-yr battery target</div>
        </div>
      </div>

      {/* CENTER: live signal */}
      <div style={{ padding: 16, overflow: 'auto', background: 'var(--bg-0)' }}>
        <div className="panel">
          <div className="panel-h"><span>Last frame · 14:22:08.412</span><span className="mono"><span className="live-dot" style={{ display: 'inline-block', verticalAlign: 'middle' }}></span> &nbsp;LIVE · 16 kHz · PCM16</span></div>
          <div className="panel-b">
            <div className="lbl">CHANNEL A1 · STRUCTURE-BORNE</div>
            <Waveform w={620} h={68} color="var(--signal)" seed={11} />
            <div className="lbl" style={{ marginTop: 16 }}>CHANNEL A2 · STRUCTURE-BORNE</div>
            <Waveform w={620} h={68} color="var(--signal)" seed={31} />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 14 }}>
              <Mini label="RMS"        value="0.142" />
              <Mini label="SNR"        value="22.1 dB" tint="var(--ok)" />
              <Mini label="VAD"        value="ACTIVE" tint="var(--signal)" />
              <Mini label="Gain norm"  value="0.93" />
            </div>
          </div>
        </div>

        <div className="panel" style={{ marginTop: 14 }}>
          <div className="panel-h"><span>Throughput · 24h</span><span className="mono">14,302 frames · 0 dropped</span></div>
          <div className="panel-b">
            <Sparkline data={[14, 12, 10, 6, 4, 22, 28, 26, 21, 18, 24, 31, 28, 22, 19, 16, 14, 12, 10, 8, 24, 31, 26, 22]} w={620} h={72} />
            <div className="row between mono" style={{ marginTop: 6, fontSize: 10, color: 'var(--fg-2)' }}>
              <span>00:00</span><span>02:00 · QUIET WINDOW</span><span>12:00</span><span>23:00</span>
            </div>
          </div>
        </div>

        <div className="panel" style={{ marginTop: 14 }}>
          <div className="panel-h"><span>MQTT pipeline</span><span className="mono ok" style={{ color: 'var(--ok)' }}>HEALTHY</span></div>
          <div className="panel-b mono" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
            <div>broker: <span style={{ color: 'var(--fg-0)' }}>tls://mqtt.omni-sense.lb:8883</span></div>
            <div>topic: <span style={{ color: 'var(--signal)' }}>edge/beirut/hamra/S-HAMRA-001/frame</span></div>
            <div>QoS 1 · in-flight 0 · last ack 412 ms</div>
            <div style={{ marginTop: 8, color: 'var(--fg-2)' }}>→ Redis Streams → omni-platform → fan-out IEP2/IEP4</div>
          </div>
        </div>
      </div>

      {/* RIGHT: ops actions */}
      <div style={{ borderLeft: '1px solid var(--line-soft)', padding: 16, background: 'var(--bg-0)', overflow: 'auto' }}>
        <div className="lbl">RECENT EVENTS</div>
        <div className="col gap-2" style={{ marginTop: 8 }}>
          {[
            { t: '14:22:08', e: 'WO-26041 fired', tint: 'var(--crit)' },
            { t: '14:21:47', e: 'frame ingest', tint: 'var(--fg-1)' },
            { t: '13:14:02', e: 'OTA firmware check OK', tint: 'var(--ok)' },
            { t: '02:00:00', e: 'quiet-window wake', tint: 'var(--signal)' },
            { t: 'yesterday', e: 'cert auto-rotate', tint: 'var(--ok)' },
          ].map((e, i) => (
            <div key={i} className="mono" style={{ fontSize: 11, padding: '6px 8px', background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 3, display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: e.tint }}>{e.e}</span>
              <span style={{ color: 'var(--fg-2)' }}>{e.t}</span>
            </div>
          ))}
        </div>

        <div className="lbl" style={{ marginTop: 20 }}>OPS ACTIONS</div>
        <div className="col gap-2" style={{ marginTop: 8 }}>
          <button className="btn">Push OTA firmware</button>
          <button className="btn">Rotate cert now</button>
          <button className="btn">Recalibrate baseline</button>
          <button className="btn">Force frame capture</button>
          <button className="btn ghost" style={{ color: 'var(--crit)', borderColor: 'oklch(0.66 0.18 25 / 0.4)' }}>Quarantine sensor</button>
        </div>

        <div className="lbl" style={{ marginTop: 20 }}>RUL · SURVIVAL</div>
        <div style={{ marginTop: 8, padding: 12, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 4 }}>
          <div className="mono" style={{ fontSize: 18, color: 'var(--ok)' }}>P(survive 30d) = 0.97</div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-2)', marginTop: 4 }}>Gumbel survival · CMMS threshold 0.85</div>
        </div>
      </div>
    </div>
  );
};

const Row = ({ k, v, tint }) => (
  <div className="row between mono" style={{ fontSize: 11, padding: '4px 0', borderBottom: '1px dashed var(--line-soft)' }}>
    <span style={{ color: 'var(--fg-2)' }}>{k}</span>
    <span style={{ color: tint === 'ok' ? 'var(--ok)' : 'var(--fg-0)' }}>{v}</span>
  </div>
);
const Mini = ({ label, value, tint = 'var(--fg-0)' }) => (
  <div style={{ padding: 8, background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 3 }}>
    <div className="lbl">{label}</div>
    <div className="mono" style={{ fontSize: 16, color: tint, marginTop: 2 }}>{value}</div>
  </div>
);

window.SensorDetail = SensorDetail;
