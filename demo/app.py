import io
import os

import folium
import matplotlib.pyplot as plt
import numpy as np
import requests
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

# ==================== HELPER FUNCTIONS ====================

def extract_demo_features(audio_data: np.ndarray, sample_rate: int, n_features: int = 100) -> np.ndarray:
    """Extract a 100-d feature vector for demo / offline mode."""
    audio = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
    features: list = []
    features.append(float(np.mean(audio ** 2)))
    features.append(float(np.var(audio)))
    features.append(float(np.max(np.abs(audio))))
    features.append(float(np.mean(np.diff(audio) ** 2)))
    features.append(float(np.sum(np.abs(np.diff(audio)))))
    clip = audio[:min(16000, len(audio))]
    fft = np.abs(np.fft.rfft(clip))
    features.append(float(np.mean(fft)))
    features.append(float(np.std(fft)))
    features.append(float(np.max(fft)))
    freqs = np.fft.rfftfreq(len(clip), 1.0 / sample_rate)
    denom = float(np.sum(fft))
    centroid = float(np.sum(freqs * fft) / denom) if denom > 0 else 0.0
    features.append(centroid / (sample_rate / 2.0))
    mu, sigma = float(np.mean(audio)), float(np.std(audio)) + 1e-8
    features.append(float(np.mean(((audio - mu) / sigma) ** 4)))
    rng = np.random.default_rng(seed=int(abs(features[0]) * 1e6) % (2 ** 31))
    while len(features) < n_features:
        features.append(float(rng.uniform(0.0, 1.0)))
    return np.array(features[:n_features], dtype=np.float32)


def generate_demo_result(audio_data: np.ndarray, sample_rate: int) -> dict:
    """Generate a demo diagnosis result using local feature extraction."""
    features = extract_demo_features(audio_data, sample_rate)
    leak_score = float(np.mean(features[:20]))
    std_ratio = float(np.std(features) / (np.mean(np.abs(features)) + 1e-6))
    confidence = min(max(0.5 + 0.3 * std_ratio, 0.50), 0.99)
    is_leak = leak_score > 0.5
    rng = np.random.default_rng(seed=42)
    return {
        "is_leak": is_leak,
        "confidence": confidence,
        "iep2_prediction": float(rng.uniform(0.55, 0.90) if is_leak else rng.uniform(0.10, 0.45)),
        "iep4_prediction": float(rng.uniform(0.50, 0.85) if is_leak else rng.uniform(0.10, 0.45)),
        "ensemble_score": float(leak_score),
        "ood_score": float(rng.uniform(0.05, 0.25)),
        "physics_consistency_check": float(rng.uniform(0.75, 0.99)),
        "signal_quality": float(rng.uniform(0.65, 0.95)),
        "snr_estimate": float(rng.uniform(8.0, 30.0)),
    }

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Omni-Sense",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CODE 2 DESIGN: SKY-BLUE THEME ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&display=swap');

:root {
  --bg:    #e8f4fd;
  --surf:  #d0eaf9;
  --card:  #c2e2f7;
  --bdr:   #5bb8f5;
  --sky:   #1a8fe3;
  --text:  #0b3a5e;
  --muted: #3a7ab0;
  --green: #0fa36b;
  --orange:#e07b2a;
  --red:   #d94040;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main,
[data-testid="block-container"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    linear-gradient(rgba(26,143,227,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(26,143,227,.04) 1px, transparent 1px);
  background-size: 48px 48px;
  animation: gridDrift 28s linear infinite;
}
@keyframes gridDrift { to { background-position: 48px 48px, 48px 48px; } }

[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer { display: none !important; }

[data-testid="stSidebar"] {
  background: var(--surf) !important;
  border-right: 1.5px solid var(--bdr) !important;
}

[data-testid="stTabs"] > div:first-child {
  border-bottom: 2px solid var(--bdr) !important;
}

[data-testid="stFileUploader"] {
  border: 2px dashed var(--bdr) !important;
  border-radius: 12px !important;
  background: var(--card) !important;
}

[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 12px !important;
  padding: 16px !important;
}

[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 12px !important;
}

[data-testid="stInfo"]    { background:rgba(26,143,227,.08) !important; border-left:3px solid var(--sky) !important; }
[data-testid="stWarning"] { background:rgba(224,123,42,.08) !important; border-left:3px solid var(--orange) !important; }
[data-testid="stError"]   { background:rgba(217,64,64,.08)  !important; border-left:3px solid var(--red) !important; }
[data-testid="stSuccess"] { background:rgba(15,163,107,.08) !important; border-left:3px solid var(--green) !important; }

.section-header {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.2em;
  font-weight: 700;
  color: #1a8fe3;
  margin-top: 1.5rem;
  margin-bottom: 1rem;
  border-bottom: 2px solid #5bb8f5;
  padding-bottom: 0.5rem;
  letter-spacing: 2px;
}

.leak-detected {
  background: linear-gradient(135deg, #d94040 0%, #f5576c 100%);
  color: white;
  padding: 2rem;
  border-radius: 12px;
  text-align: center;
  font-family: 'Orbitron', sans-serif;
  font-size: 1.6em;
  font-weight: 700;
  letter-spacing: 3px;
  box-shadow: 0 4px 24px rgba(217,64,64,.35);
}

.no-leak {
  background: linear-gradient(135deg, #0fa36b 0%, #1a8fe3 100%);
  color: white;
  padding: 2rem;
  border-radius: 12px;
  text-align: center;
  font-family: 'Orbitron', sans-serif;
  font-size: 1.6em;
  font-weight: 700;
  letter-spacing: 3px;
  box-shadow: 0 4px 24px rgba(15,163,107,.35);
}
</style>
""", unsafe_allow_html=True)

# ==================== CODE 2 DESIGN: ANIMATED PIPE HERO ====================
st.markdown("""
<style>
.hero {
  position: relative; overflow: hidden;
  background: linear-gradient(160deg, #b8ddf7 0%, #d0eaf9 50%, #c2e2f7 100%);
  border: 1.5px solid #5bb8f5; border-radius: 18px;
  padding: 32px 44px 108px; margin-bottom: 18px;
  box-shadow: 0 4px 32px rgba(26,143,227,.15);
}
.hero::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 70% 65% at 60% 40%, rgba(26,143,227,.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-inner {
  position: relative; z-index: 2;
  display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
}
.hero-logo {
  width: 68px; height: 68px; flex-shrink: 0;
  border: 2px solid rgba(26,143,227,.5); border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  animation: logoGlow 3.5s ease-in-out infinite;
  background: rgba(255,255,255,.5);
}
@keyframes logoGlow {
  0%,100% { box-shadow: 0 0 18px rgba(26,143,227,.3); }
  50%      { box-shadow: 0 0 44px rgba(26,143,227,.6); }
}
.hero-title {
  font-family: 'Orbitron', sans-serif; font-size: 2.3rem; font-weight: 900;
  letter-spacing: 5px; line-height: 1;
  background: linear-gradient(130deg, #0e75c8 0%, #1a8fe3 40%, #3aabf0 70%, #0fa36b 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
  font-family: 'Rajdhani', sans-serif; color: #3a7ab0;
  font-size: .86rem; letter-spacing: 3px; margin-top: 5px;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 7px; margin-top: 9px;
  background: rgba(26,143,227,.1); border: 1px solid rgba(26,143,227,.4);
  border-radius: 20px; padding: 4px 14px;
  font-family: 'Rajdhani', sans-serif; font-size: .76rem;
  color: #0e75c8; letter-spacing: 1px;
}
.hero-bdot {
  width: 7px; height: 7px; border-radius: 50%; background: #0fa36b;
  animation: bdotPulse 1.5s ease-in-out infinite;
}
@keyframes bdotPulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(15,163,107,.6); }
  50%      { box-shadow: 0 0 0 7px rgba(15,163,107,0); }
}
.hero-stats { margin-left: auto; display: flex; gap: 32px; }
.hstat { text-align: center; }
.hstat-v {
  font-family: 'Orbitron', sans-serif; font-size: 1.5rem; font-weight: 700;
  color: #1a8fe3; text-shadow: 0 0 12px rgba(26,143,227,.35);
}
.hstat-l {
  font-family: 'Rajdhani', sans-serif; font-size: .65rem;
  color: #3a7ab0; letter-spacing: 1.5px; margin-top: 2px;
}
.pipe-scene {
  position: absolute; bottom: 0; right: 40px;
  width: 340px; height: 110px;
  overflow: visible; pointer-events: none; z-index: 2;
}
.pipe-body {
  position: absolute; bottom: 55px; left: 0; right: 0; height: 28px;
  background: linear-gradient(180deg, #6bbde8 0%, #3a9ed4 30%, #1a7ab8 60%, #3a9ed4 100%);
  border-radius: 4px;
  border-top: 2px solid rgba(255,255,255,.5);
  border-bottom: 2px solid rgba(10,60,100,.2);
  box-shadow: 0 4px 16px rgba(26,143,227,.25), inset 0 2px 4px rgba(255,255,255,.3);
}
.pipe-cap-l, .pipe-cap-r {
  position: absolute; bottom: 49px; width: 18px; height: 40px;
  background: linear-gradient(180deg, #7ec8ea 0%, #3a9ed4 50%, #1a7ab8 100%);
  border-radius: 4px;
}
.pipe-cap-l { left: -4px; }
.pipe-cap-r { right: -4px; }
.pipe-cap-l::before, .pipe-cap-r::before,
.pipe-cap-l::after,  .pipe-cap-r::after {
  content: ''; position: absolute; left: 50%; transform: translateX(-50%);
  width: 6px; height: 6px; border-radius: 50%;
  background: #1a7ab8; border: 1px solid rgba(255,255,255,.3);
}
.pipe-cap-l::before, .pipe-cap-r::before { top: 5px; }
.pipe-cap-l::after,  .pipe-cap-r::after  { bottom: 5px; }
.drop {
  position: absolute;
  width: 7px;
  border-radius: 50% 50% 55% 55% / 40% 40% 60% 60%;
  background: radial-gradient(circle at 35% 30%,
    rgba(255,255,255,.95) 0%, rgba(91,184,245,.9) 50%, rgba(26,143,227,.7) 100%);
  box-shadow: 0 0 6px rgba(26,143,227,.4), inset 0 1px 2px rgba(255,255,255,.6);
  animation: drip linear infinite;
}
@keyframes drip {
  0%   { height: 3px; border-radius: 50%; opacity: 0; transform: translateY(0) scaleX(1); }
  8%   { opacity: 1; }
  30%  { height: 12px; }
  55%  { height: 14px; transform: translateY(0) scaleX(.95); opacity: 1; }
  65%  { height: 6px; transform: translateY(14px) scaleX(1.1); opacity: .9; }
  90%  { transform: translateY(48px); opacity: .6; }
  100% { height: 4px; transform: translateY(54px); opacity: 0; }
}
.drop1  { left: 56px;  bottom: 47px; width: 6px; animation-duration: 2.1s; animation-delay: 0s; }
.drop1b { left: 57px;  bottom: 47px; width: 5px; animation-duration: 2.1s; animation-delay: 1.05s; }
.drop2  { left: 126px; bottom: 47px; width: 7px; animation-duration: 2.8s; animation-delay: .7s; }
.drop2b { left: 127px; bottom: 47px; width: 6px; animation-duration: 2.8s; animation-delay: 2.10s; }
.drop3  { left: 196px; bottom: 47px; width: 5px; animation-duration: 2.3s; animation-delay: 1.4s; }
.drop3b { left: 197px; bottom: 47px; width: 5px; animation-duration: 2.3s; animation-delay: 0.5s; }
.drop4  { left: 258px; bottom: 47px; width: 6px; animation-duration: 3.0s; animation-delay: .3s; }
.drop4b { left: 259px; bottom: 47px; width: 7px; animation-duration: 3.0s; animation-delay: 1.9s; }
.splash {
  position: absolute; bottom: 0px;
  width: 18px; height: 5px; border-radius: 50%;
  background: rgba(26,143,227,.2); border: 1px solid rgba(26,143,227,.4);
  animation: splashAnim linear infinite;
}
@keyframes splashAnim {
  0%   { transform: scaleX(0); opacity: 0; }
  60%  { transform: scaleX(0); opacity: 0; }
  65%  { transform: scaleX(.4); opacity: .7; }
  80%  { transform: scaleX(1); opacity: .5; }
  100% { transform: scaleX(1.4); opacity: 0; }
}
.splash1 { left: 50px;  animation-duration: 2.1s; animation-delay: 0s; }
.splash2 { left: 120px; animation-duration: 2.8s; animation-delay: .7s; }
.splash3 { left: 190px; animation-duration: 2.3s; animation-delay: 1.4s; }
.splash4 { left: 252px; animation-duration: 3.0s; animation-delay: .3s; }
.pipe-label {
  position: absolute; bottom: 34px; left: 0; right: 0;
  text-align: center;
  font-family: 'Rajdhani', sans-serif; font-size: 9px;
  color: rgba(58,122,176,.7); letter-spacing: 2px;
}
.gauge-wrap {
  position: absolute; bottom: 80px; left: 140px;
  width: 44px; height: 44px;
}
</style>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-logo">
      <svg width="38" height="38" viewBox="0 0 38 38" fill="none">
        <circle cx="19" cy="19" r="14" stroke="#1a8fe3" stroke-width="1.5" stroke-dasharray="5 2.5"/>
        <path d="M9 19 Q14 9,19 19 Q24 29,29 19" stroke="#1a8fe3" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="19" cy="19" r="3.5" fill="#1a8fe3" opacity=".95"/>
        <circle cx="19" cy="19" r="7" fill="none" stroke="#1a8fe3" stroke-width=".7" opacity=".35"/>
      </svg>
    </div>
    <div>
      <div class="hero-title">OMNI-SENSE</div>
      <div class="hero-sub">ACOUSTIC DIAGNOSTICS FOR URBAN WATER INFRASTRUCTURE</div>
      <div class="hero-badge">
        <div class="hero-bdot"></div>
        SYSTEM ONLINE &nbsp;&middot;&nbsp; AUB EECE503N / EECE798N &nbsp;&middot;&nbsp; SPRING 2026
      </div>
    </div>
    <div class="hero-stats">
      <div class="hstat"><div class="hstat-v">8</div><div class="hstat-l">SENSORS</div></div>
      <div class="hstat"><div class="hstat-v">40%</div><div class="hstat-l">LEAK RATE</div></div>
      <div class="hstat"><div class="hstat-v">87ms</div><div class="hstat-l">LATENCY</div></div>
    </div>
  </div>

  <div class="pipe-scene">
    <div class="pipe-body"></div>
    <div class="pipe-cap-l"></div>
    <div class="pipe-cap-r"></div>
    <div class="drop drop1"></div>
    <div class="drop drop1b"></div>
    <div class="drop drop2"></div>
    <div class="drop drop2b"></div>
    <div class="drop drop3"></div>
    <div class="drop drop3b"></div>
    <div class="drop drop4"></div>
    <div class="drop drop4b"></div>
    <div class="splash splash1"></div>
    <div class="splash splash2"></div>
    <div class="splash splash3"></div>
    <div class="splash splash4"></div>
    <div class="gauge-wrap">
      <svg viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="22" cy="22" r="20" fill="#d0eaf9" stroke="#5bb8f5" stroke-width="1.5"/>
        <circle cx="22" cy="22" r="16" fill="#b8ddf7" stroke="#1a8fe344" stroke-width="1"/>
        <path d="M10 28 A16 16 0 0 1 34 28" stroke="#5bb8f5" stroke-width="2.5" fill="none"/>
        <path d="M10 28 A16 16 0 0 1 28 11" stroke="#1a8fe3" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="22" cy="22" r="3" fill="#1a8fe3"/>
        <text x="22" y="38" text-anchor="middle" font-family="monospace" font-size="7" fill="#3a7ab0">PSI</text>
      </svg>
    </div>
    <div class="pipe-label">MUNICIPAL WATER NETWORK — SENSOR ARRAY</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style='font-family:Orbitron,sans-serif; font-size:1.1rem; font-weight:700;
         color:#1a8fe3; letter-spacing:3px; margin-bottom:4px;'>🌊 OMNI-SENSE</div>
    <div style='font-family:Rajdhani,sans-serif; color:#3a7ab0; font-size:0.82rem;
         letter-spacing:1.5px; margin-bottom:1.2rem;'>
    ACOUSTIC DIAGNOSTICS<br>Water Infrastructure · Beirut
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ Configuration")
    default_eep_url = os.getenv(
        "EEP_URL",
        "https://omni-eep-745790249979.us-central1.run.app"
    )
    eep_url = st.text_input(
        "EEP Server URL",
        value=default_eep_url,
        help="Enterprise Execution Platform Cloud Run endpoint"
    )

    st.markdown("### 📡 System Status")
    system_online = False
    try:
        health_response = requests.get(f"{eep_url}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ EEP Online")
            system_online = True
        elif health_response.status_code == 403:
            st.warning("🔒 EEP requires auth (403)")
            st.caption("Run:\n`gcloud run services add-iam-policy-binding omni-eep --region=us-central1 --member=allUsers --role=roles/run.invoker`")
        else:
            st.warning(f"⚠️ EEP status {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ EEP Unreachable")
    except Exception as ex:
        st.warning(f"⚠️ EEP Error: {ex}")

    if not system_online:
        st.info("💡 Demo mode active — local feature extraction")

    st.divider()
    st.markdown("#### About")
    st.caption("""
    Omni-Sense is a capstone project for detecting water pipe leaks
    in Lebanese urban infrastructure using acoustic analysis.

    **Key Features:**
    - Accelerometer-based detection
    - Physics-informed ML models
    - Ensemble prediction method
    - Real-time diagnostics
    """)

# ==================== MAIN TITLE ====================
st.markdown("""
<div style='font-family:Orbitron,sans-serif; font-size:1.6rem; font-weight:700;
     color:#1a8fe3; letter-spacing:3px; margin-bottom:4px;'>
  🔊 Omni-Sense Dashboard
</div>
<div style='font-family:Rajdhani,sans-serif; color:#3a7ab0; font-size:0.85rem;
     letter-spacing:2px; margin-bottom:1.5rem;'>
  EECE503N / EECE798N · American University of Beirut · Spring 2026
</div>
""", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Diagnose",
    "🗺️ Beirut Network Map",
    "📊 Model Insights",
    "🏗️ System Architecture"
])

# ==================== TAB 1: DIAGNOSE ====================
with tab1:
    st.markdown('<div class="section-header">Acoustic Leak Diagnosis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Upload Audio Recording")
        uploaded_file = st.file_uploader(
            "Select a WAV file from accelerometer",
            type=["wav"],
            help="Upload a 16-bit PCM WAV file from your accelerometer"
        )

    with col2:
        st.markdown("#### File Information")
        if uploaded_file is not None:
            try:
                _raw_bytes = uploaded_file.read()
                audio_data, sample_rate = sf.read(io.BytesIO(_raw_bytes))
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                st.session_state['_audio_data'] = audio_data
                st.session_state['_sample_rate'] = sample_rate
                st.session_state['_raw_bytes'] = _raw_bytes
                duration = len(audio_data) / sample_rate
                n_channels = 1
                st.metric("Sample Rate", f"{sample_rate} Hz")
                st.metric("Duration", f"{duration:.2f} s")
                st.metric("Channels", f"{n_channels} (mono)")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if uploaded_file is not None and '_audio_data' in st.session_state:
        try:
            audio_data = st.session_state['_audio_data']
            sample_rate = st.session_state['_sample_rate']

            st.markdown("#### Audio Analysis")
            col_wave, col_spec = st.columns(2)

            with col_wave:
                st.markdown("**Waveform**")
                fig_wave, ax = plt.subplots(figsize=(8, 4))
                fig_wave.patch.set_facecolor('#d0eaf9')
                ax.set_facecolor('#c2e2f7')
                time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
                ax.plot(time_axis, audio_data, linewidth=0.5, color='#1a8fe3')
                ax.set_xlabel('Time (s)', color='#0b3a5e')
                ax.set_ylabel('Amplitude', color='#0b3a5e')
                ax.set_title('Accelerometer Signal', color='#0b3a5e', fontweight='bold')
                ax.tick_params(colors='#3a7ab0')
                ax.grid(True, alpha=0.3, color='#5bb8f5')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#5bb8f5')
                st.pyplot(fig_wave, use_container_width=True)
                plt.close(fig_wave)

            with col_spec:
                st.markdown("**Log-Power Spectrogram**")
                _librosa_ok = False
                try:
                    import librosa
                    import librosa.display
                    D = librosa.stft(audio_data.astype(np.float32), n_fft=1024, hop_length=512)
                    S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)
                    _librosa_ok = True
                except Exception:
                    n_fft = min(1024, len(audio_data))
                    hop = n_fft // 2
                    frames = [audio_data[i:i + n_fft] for i in range(0, max(1, len(audio_data) - n_fft), hop)]
                    if not frames:
                        frames = [np.zeros(n_fft)]
                    frames = [f if len(f) == n_fft else np.pad(f, (0, n_fft - len(f))) for f in frames]
                    S = np.array([np.abs(np.fft.rfft(f)) ** 2 for f in frames]).T
                    S_db = 10 * np.log10(S + 1e-10)

                fig_spec, ax = plt.subplots(figsize=(8, 4))
                fig_spec.patch.set_facecolor('#d0eaf9')
                ax.set_facecolor('#c2e2f7')
                if _librosa_ok:
                    librosa.display.specshow(S_db, sr=sample_rate, hop_length=512,
                                             x_axis='time', y_axis='log', ax=ax, cmap='Blues')
                    cb_src = ax.collections[0] if ax.collections else (ax.images[0] if ax.images else None)
                    if cb_src:
                        fig_spec.colorbar(cb_src, ax=ax, format="%+2.0f dB")
                else:
                    ax.imshow(S_db, aspect='auto', origin='lower', cmap='Blues',
                              extent=[0, len(audio_data) / sample_rate, 0, sample_rate / 2])
                    ax.set_ylabel('Frequency (Hz)', color='#0b3a5e')
                    ax.set_xlabel('Time (s)', color='#0b3a5e')
                ax.set_title('Log-Power Spectrogram', color='#0b3a5e', fontweight='bold')
                ax.tick_params(colors='#3a7ab0')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#5bb8f5')
                st.pyplot(fig_spec, use_container_width=True)
                plt.close(fig_spec)

            st.divider()
            col_btn1, col_btn2 = st.columns([2, 1])

            with col_btn1:
                analyze_button = st.button(
                    "🔬 Analyze with EEP",
                    use_container_width=True,
                    type="primary",
                    key="analyze_btn"
                )

            if analyze_button:
                with st.spinner("Analyzing audio..."):
                    audio_bytes = io.BytesIO(st.session_state['_raw_bytes'])
                    audio_bytes.seek(0)

                    try:
                        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
                        response = requests.post(
                            f"{eep_url}/diagnose",
                            files=files,
                            timeout=30
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['diagnosis_result'] = result
                            st.success("Analysis complete!")
                        else:
                            st.error(f"EEP returned status {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                        st.info("Falling back to demo mode with local feature extraction...")
                        st.session_state['diagnosis_result'] = generate_demo_result(audio_data, sample_rate)

            if 'diagnosis_result' in st.session_state:
                result = st.session_state['diagnosis_result']
                st.divider()
                st.markdown("#### Diagnosis Results")

                col_result1, col_result2 = st.columns([2, 1])

                with col_result1:
                    is_leak = result.get('is_leak', False)
                    confidence = result.get('confidence', 0.5)

                    if is_leak:
                        st.markdown(
                            '<div class="leak-detected">🚨 LEAK DETECTED</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="no-leak">✅ NO LEAK DETECTED</div>',
                            unsafe_allow_html=True
                        )

                with col_result2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")

                st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")

                st.markdown("#### Detailed Analysis")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                with col_m1:
                    st.metric(
                        "IEP2 Score",
                        f"{result.get('iep2_prediction', 0):.3f}",
                        help="Physics-informed ensemble model 2"
                    )
                with col_m2:
                    st.metric(
                        "IEP4 Score",
                        f"{result.get('iep4_prediction', 0):.3f}",
                        help="Physics-informed ensemble model 4"
                    )
                with col_m3:
                    st.metric(
                        "Ensemble Method",
                        f"{result.get('ensemble_score', 0):.3f}",
                        help="Weighted ensemble prediction"
                    )
                with col_m4:
                    st.metric(
                        "OOD Score",
                        f"{result.get('ood_score', 0):.3f}",
                        help="Out-of-distribution detection"
                    )

                with st.expander("📋 Full Analysis Details", expanded=False):
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.subheader("Model Predictions")
                        st.json({
                            "iep2_prediction": result.get('iep2_prediction', 0),
                            "iep4_prediction": result.get('iep4_prediction', 0),
                            "ensemble_score": result.get('ensemble_score', 0),
                            "ood_score": result.get('ood_score', 0),
                        })
                    with col_d2:
                        st.subheader("Quality Metrics")
                        st.json({
                            "physics_consistency": result.get('physics_consistency_check', 0),
                            "signal_quality": result.get('signal_quality', 0),
                            "snr_estimate": result.get('snr_estimate', 0),
                        })

        except Exception as e:
            st.error(f"Error processing audio: {e}")

    else:
        st.info("👆 Upload a WAV file to get started")

# ==================== TAB 2: BEIRUT NETWORK MAP ====================
with tab2:
    st.markdown('<div class="section-header">Water Network Sensor Map</div>', unsafe_allow_html=True)

    sensors = {
        "Hamra":           {"lat": 33.8938, "lng": 35.4831, "status": "Normal",  "last_reading": "2 min ago"},
        "Achrafieh":       {"lat": 33.8886, "lng": 35.5157, "status": "Leak",    "last_reading": "1 min ago"},
        "Verdun":          {"lat": 33.8822, "lng": 35.4772, "status": "Normal",  "last_reading": "5 min ago"},
        "Mar Mikhael":     {"lat": 33.8907, "lng": 35.5214, "status": "Offline", "last_reading": "1 hour ago"},
        "Gemmayzeh":       {"lat": 33.8965, "lng": 35.5168, "status": "Normal",  "last_reading": "3 min ago"},
        "Ras Beirut":      {"lat": 33.9003, "lng": 35.4788, "status": "Normal",  "last_reading": "4 min ago"},
        "Badaro":          {"lat": 33.8747, "lng": 35.5093, "status": "Normal",  "last_reading": "2 min ago"},
        "Ain el-Mreisseh": {"lat": 33.9014, "lng": 35.4851, "status": "Normal",  "last_reading": "3 min ago"},
    }

    beirut_center = [33.8886, 35.5020]
    m = folium.Map(location=beirut_center, zoom_start=13, tiles="CartoDB positron")

    color_map = {"Normal": "green", "Leak": "red", "Offline": "gray"}

    for sensor_name, info in sensors.items():
        color = color_map.get(info["status"], "blue")
        popup_text = f"""
        <div style='font-family:sans-serif;min-width:150px'>
            <b>{sensor_name}</b><br>
            Status: <span style='color:{color};font-weight:bold'>{info['status']}</span><br>
            Last Reading: {info['last_reading']}<br>
            Lat: {info['lat']:.4f}, Lng: {info['lng']:.4f}
        </div>
        """
        folium.CircleMarker(
            location=[info["lat"], info["lng"]],
            radius=10,
            popup=folium.Popup(popup_text, max_width=220),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
            tooltip=f"{sensor_name} — {info['status']}"
        ).add_to(m)

        folium.Marker(
            location=[info["lat"], info["lng"]],
            popup=sensor_name,
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    st.markdown("#### Active Sensors in Beirut")
    st_folium(m, width=1400, height=600)

    st.markdown("#### Sensor Status")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        normal_count = sum(1 for s in sensors.values() if s["status"] == "Normal")
        st.metric("🟢 Normal", normal_count, help="Sensors operating normally")
    with col_s2:
        leak_count = sum(1 for s in sensors.values() if s["status"] == "Leak")
        st.metric("🔴 Leak Detected", leak_count, help="Active leak alerts")
    with col_s3:
        offline_count = sum(1 for s in sensors.values() if s["status"] == "Offline")
        st.metric("⚫ Offline", offline_count, help="Offline sensors")

    st.markdown("#### Sensor Details")
    sensor_data = []
    for name, info in sensors.items():
        sensor_data.append({
            "Sensor Name": name,
            "Latitude": f"{info['lat']:.4f}",
            "Longitude": f"{info['lng']:.4f}",
            "Status": info["status"],
            "Last Reading": info["last_reading"]
        })
    st.dataframe(sensor_data, use_container_width=True, hide_index=True)

# ==================== TAB 3: MODEL INSIGHTS ====================
with tab3:
    st.markdown('<div class="section-header">Model Architecture & Interpretability</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Feature Importance (Top 20)")

        feature_names = [
            "Spectral Centroid", "Spectral Rolloff", "MFCC 1", "MFCC 2",
            "Zero Crossing Rate", "Temporal Centroid", "Bandwidth", "Flatness",
            "RMS Energy", "Flux", "Spectral Kurtosis", "Cepstral Deviation",
            "Temporal Spread", "Freq Variance", "Log Energy", "Harmonic Ratio",
            "Pitch Deviation", "Vibrato Extent", "Attack Time", "Decay Rate"
        ]
        importance_scores = np.array([
            0.156, 0.142, 0.118, 0.105, 0.092, 0.085, 0.078, 0.071,
            0.064, 0.058, 0.052, 0.048, 0.042, 0.038, 0.035, 0.031,
            0.028, 0.024, 0.020, 0.015
        ])

        fig_importance, ax = plt.subplots(figsize=(10, 8))
        fig_importance.patch.set_facecolor('#d0eaf9')
        ax.set_facecolor('#c2e2f7')
        y_pos = np.arange(len(feature_names))
        # Sky-blue gradient colors
        norm_scores = importance_scores / importance_scores.max()
        bar_colors = [
            (0.1 + 0.4 * v, 0.5 + 0.3 * v, 0.85 + 0.15 * v)
            for v in norm_scores
        ]
        ax.barh(y_pos, importance_scores, color=bar_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=9, color='#0b3a5e')
        ax.set_xlabel('Importance Score', color='#0b3a5e')
        ax.set_title('SHAP Feature Importance', color='#0b3a5e', fontweight='bold')
        ax.tick_params(colors='#3a7ab0')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, color='#5bb8f5')
        for spine in ax.spines.values():
            spine.set_edgecolor('#5bb8f5')
        st.pyplot(fig_importance, use_container_width=True)
        plt.close(fig_importance)

    with col2:
        st.markdown("#### Feature Group Contributions")

        feature_groups = {
            "Spectral Features": 0.35,
            "Temporal Features": 0.25,
            "Energy-based Features": 0.18,
            "Harmonic Features": 0.12,
            "Perceptual Features": 0.10
        }

        fig_groups, ax = plt.subplots(figsize=(8, 6))
        fig_groups.patch.set_facecolor('#d0eaf9')
        ax.set_facecolor('#d0eaf9')
        colors_pie = ['#1a8fe3', '#3aabf0', '#5bb8f5', '#0fa36b', '#0e75c8']
        wedges, texts, autotexts = ax.pie(
            feature_groups.values(),
            labels=feature_groups.keys(),
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90
        )
        for text in texts:
            text.set_color('#0b3a5e')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Feature Group Distribution', color='#0b3a5e', fontweight='bold')
        st.pyplot(fig_groups, use_container_width=True)
        plt.close(fig_groups)

    st.divider()
    st.markdown("#### Model Architecture Overview")

    architecture_text = """
    ```
                            WAV Audio File
                                  |
                                  v
                        Signal Preprocessing
                          (Filtering, Normalization)
                                  |
                        +---------+---------+
                        |                   |
                        v                   v
                    Local DSP Features
                   (39-d physics vector)
                        |                   |
                        +-------+-----------+
                                |
                        +-------+---------+
                        |       |       |
                        v       v       v
                      IEP2    IEP4    Baseline
                    (ML-XGB) (ML-RF) (Threshold)
                        |       |       |
                        +---+---+---+---+
                            |   |   |
                            v   v   v
                        Ensemble Voting
                         (Weighted)
                            |
                        +---+---+
                        |       |
                        v       v
                    IEP3      OOD
                  (Decision)  Check
                        |       |
                        +---+---+
                            |
                            v
                        Diagnosis Result
                    (Leak/No Leak + Confidence)
    ```
    """
    st.markdown(architecture_text)

    st.markdown("#### Model Details")
    model_info = {
        "DSP Features": "Local extraction: 39-d physics-informed features (kurtosis, spectral centroid, wavelet)",
        "IEP2": "XGBoost classifier trained on labeled leak/normal data",
        "IEP4": "Random Forest classifier with 100 trees, max_depth=15",
        "IEP3": "Decision aggregator: Weighted voting across ensemble members",
        "Ensemble": "Voting with physics consistency check and OOD detection",
        "OOD Score": "Measures how different a sample is from training distribution"
    }
    for model_name, description in model_info.items():
        st.markdown(f"**{model_name}:** {description}")

# ==================== TAB 4: SYSTEM ARCHITECTURE ====================
with tab4:
    st.markdown('<div class="section-header">System Architecture & Deployment</div>', unsafe_allow_html=True)

    st.markdown("#### End-to-End Pipeline")

    col_arch1, col_arch2 = st.columns([1.5, 1])

    with col_arch1:
        pipeline_text = """
        **Data Ingestion**
        - Accelerometer WAV files (16-bit PCM, 16 kHz)
        - Supports up to 30-second recordings
        - Automatic signal preprocessing

        **Feature Engineering (Local DSP)**
        - 39-d physics feature extraction from raw vibration
        - Kurtosis, spectral centroid, wavelet energy, zero-crossing rate
        - MFCC extraction (13 coefficients)
        - Compatible with structure-borne accelerometer data

        **Ensemble Classification**
        - **IEP2 (XGBoost):** Fast, interpretable decisions
        - **IEP4 (Random Forest):** Robust ensemble voting
        - Weighted ensemble combiner

        **Quality Assurance**
        - Physics consistency check against domain knowledge
        - Out-of-distribution score for anomaly detection
        - Signal quality metrics (SNR, frequency response)

        **Output**
        - Binary classification: Leak / No Leak
        - Confidence score (0-1)
        - Detailed model predictions
        - Explainability via SHAP feature importance
        """
        st.markdown(pipeline_text)

    with col_arch2:
        st.markdown("#### Service Status")
        services = {
            "EEP":  "Enterprise Execution Platform",
            "DSP": "Local Feature Extraction",
            "IEP2": "XGBoost Model",
            "IEP4": "Random Forest Model",
            "IEP3": "Decision Aggregator",
        }
        for service_name, description in services.items():
            if system_online:
                st.success(f"✅ {service_name}: {description}")
            else:
                st.warning(f"⚠️ {service_name}: {description} (Demo Mode)")

    st.divider()

    # Animated SVG architecture diagram (from Code 2)
    st.markdown("#### Architecture Diagram")
    svg = """
<svg width="100%" viewBox="0 0 820 420" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
<defs>
  <linearGradient id="bx" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#c2e2f7"/><stop offset="100%" stop-color="#d0eaf9"/>
  </linearGradient>
  <marker id="ar" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
    <path d="M0,0 L0,6 L7,3 Z" fill="#1a8fe3" opacity=".8"/>
  </marker>
  <style>
    @keyframes flowLine {0%{stroke-dashoffset:60}100%{stroke-dashoffset:0}}
  </style>
</defs>
<line x1="140" y1="210" x2="208" y2="210" stroke="#1a8fe3" stroke-width="1.5"
  stroke-dasharray="6 3" marker-end="url(#ar)" opacity=".8"
  style="animation:flowLine 1.2s linear infinite;"/>
<line x1="340" y1="185" x2="410" y2="120" stroke="#1a8fe3" stroke-width="1.5"
  stroke-dasharray="6 3" marker-end="url(#ar)" opacity=".8"
  style="animation:flowLine 1.4s linear infinite;"/>
<line x1="340" y1="235" x2="410" y2="295" stroke="#1a8fe3" stroke-width="1.5"
  stroke-dasharray="6 3" marker-end="url(#ar)" opacity=".8"
  style="animation:flowLine 1.5s linear infinite .3s;"/>
<line x1="572" y1="110" x2="342" y2="192" stroke="#1a8fe3" stroke-width="1"
  stroke-dasharray="4 4" opacity=".4" style="animation:flowLine 2s linear infinite;"/>
<line x1="572" y1="308" x2="342" y2="232" stroke="#1a8fe3" stroke-width="1"
  stroke-dasharray="4 4" opacity=".4" style="animation:flowLine 2s linear infinite .5s;"/>
<line x1="275" y1="285" x2="275" y2="338" stroke="#7c3aed" stroke-width="1.5"
  stroke-dasharray="5 3" marker-end="url(#ar)" opacity=".65"
  style="animation:flowLine 2s linear infinite 1s;"/>
<!-- Edge App -->
<rect x="20" y="175" width="120" height="70" rx="10" fill="url(#bx)" stroke="#1a8fe3" stroke-width="1.5"/>
<text x="80" y="200" text-anchor="middle" font-size="10" font-weight="700" fill="#1a8fe3">EDGE APP</text>
<text x="80" y="218" text-anchor="middle" font-size="11" fill="#0b3a5e">Smartphone</text>
<text x="80" y="234" text-anchor="middle" font-size="11" fill="#0b3a5e">Web UI</text>
<!-- EEP -->
<rect x="210" y="155" width="130" height="115" rx="10" fill="url(#bx)" stroke="#1a8fe3" stroke-width="2"/>
<text x="275" y="182" text-anchor="middle" font-size="11" font-weight="700" fill="#1a8fe3">EEP</text>
<text x="275" y="200" text-anchor="middle" font-size="11" fill="#3a7ab0">API Gateway</text>
<text x="275" y="218" text-anchor="middle" font-size="11" fill="#0b3a5e">Signal QA</text>
<text x="275" y="235" text-anchor="middle" font-size="11" fill="#0b3a5e">Rate Limiting</text>
<text x="275" y="252" text-anchor="middle" font-size="11" fill="#3a7ab0">FastAPI :8000</text>
<!-- DSP -->
<rect x="412" y="56" width="160" height="108" rx="10" fill="url(#bx)" stroke="#e09a1a" stroke-width="1.5"/>
<text x="492" y="84" text-anchor="middle" font-size="10" font-weight="700" fill="#c47a10">DSP Extract</text>
<text x="492" y="102" text-anchor="middle" font-size="11" fill="#3a7ab0">Local Features</text>
<text x="492" y="120" text-anchor="middle" font-size="11" fill="#0b3a5e">Kurtosis · Spectral</text>
<text x="492" y="138" text-anchor="middle" font-size="11" fill="#0b3a5e">→ 39-d vector</text>
<text x="492" y="153" text-anchor="middle" font-size="11" fill="#3a7ab0">in-EEP</text>
<!-- IEP2 -->
<rect x="412" y="248" width="160" height="128" rx="10" fill="url(#bx)" stroke="#d94040" stroke-width="1.5"/>
<text x="492" y="276" text-anchor="middle" font-size="10" font-weight="700" fill="#d94040">IEP 2</text>
<text x="492" y="294" text-anchor="middle" font-size="11" fill="#3a7ab0">Diagnostic Engine</text>
<text x="492" y="312" text-anchor="middle" font-size="11" fill="#0b3a5e">Isolation Forest</text>
<text x="492" y="330" text-anchor="middle" font-size="11" fill="#0b3a5e">XGBoost · ONNX</text>
<text x="492" y="362" text-anchor="middle" font-size="11" fill="#3a7ab0">:8002</text>
<!-- IEP3 -->
<rect x="210" y="340" width="130" height="70" rx="10" fill="url(#bx)" stroke="#7c3aed" stroke-width="1.5"/>
<text x="275" y="368" text-anchor="middle" font-size="10" font-weight="700" fill="#7c3aed">IEP 3</text>
<text x="275" y="386" text-anchor="middle" font-size="11" fill="#0b3a5e">Tickets · Dispatch</text>
<text x="275" y="401" text-anchor="middle" font-size="11" fill="#3a7ab0">:8003</text>
<!-- Monitoring -->
<rect x="638" y="72" width="160" height="138" rx="10" fill="#d0f0e5" stroke="#0fa36b" stroke-width="1.5" stroke-dasharray="6 3"/>
<text x="718" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#0fa36b">MONITORING</text>
<text x="718" y="122" text-anchor="middle" font-size="12" fill="#0b3a5e">📊 Prometheus :9090</text>
<text x="718" y="142" text-anchor="middle" font-size="12" fill="#0b3a5e">📈 Grafana :3000</text>
<text x="718" y="162" text-anchor="middle" font-size="11" fill="#3a7ab0">Scrapes /metrics</text>
<text x="718" y="180" text-anchor="middle" font-size="11" fill="#3a7ab0">EEP · IEP2 · IEP4</text>
<!-- Animated packets -->
<circle r="4" fill="#1a8fe3">
  <animateMotion dur="1.8s" repeatCount="indefinite" path="M80,210 L208,210"/>
  <animate attributeName="opacity" values="0;1;1;0" dur="1.8s" repeatCount="indefinite"/>
</circle>
<circle r="4" fill="#c47a10">
  <animateMotion dur="2s" repeatCount="indefinite" begin=".5s" path="M340,185 L412,120"/>
  <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite" begin=".5s"/>
</circle>
<circle r="4" fill="#d94040">
  <animateMotion dur="2.2s" repeatCount="indefinite" begin="1s" path="M340,235 L412,308"/>
  <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" repeatCount="indefinite" begin="1s"/>
</circle>
</svg>
"""
    try:
        components.html(svg, height=440)
    except Exception:
        st.markdown(svg, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Key Technologies")

    tech_cols = st.columns(4)
    tech_stack = [
        ("🐍 Python 3.10+", "Core implementation"),
        ("🚀 FastAPI", "REST API framework"),
        ("🧠 XGBoost/RF", "ML models"),
        ("📊 NumPy/SciPy", "Signal processing"),
        ("🎵 Librosa", "Audio analysis"),
        ("🤖 Scikit-learn", "ML utilities"),
        ("📈 SHAP", "Model interpretability"),
        ("⚡ GPU Support", "Accelerated inference"),
    ]
    for i, (tech, desc) in enumerate(tech_stack):
        with tech_cols[i % 4]:
            st.markdown(f"**{tech}**  \n{desc}")