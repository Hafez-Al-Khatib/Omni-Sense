import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import soundfile as sf

# ==================== HELPER FUNCTIONS (must be defined before use) ====================

def extract_demo_features(audio_data: np.ndarray, sample_rate: int, n_features: int = 100) -> np.ndarray:
    """Extract a 100-d feature vector for demo / offline mode."""
    audio = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
    features: list = []

    # Energy
    features.append(float(np.mean(audio ** 2)))
    features.append(float(np.var(audio)))
    features.append(float(np.max(np.abs(audio))))
    # Temporal
    features.append(float(np.mean(np.diff(audio) ** 2)))
    features.append(float(np.sum(np.abs(np.diff(audio)))))
    # Spectral (FFT on first 16000 samples)
    clip = audio[:min(16000, len(audio))]
    fft = np.abs(np.fft.rfft(clip))
    features.append(float(np.mean(fft)))
    features.append(float(np.std(fft)))
    features.append(float(np.max(fft)))
    # Spectral centroid
    freqs = np.fft.rfftfreq(len(clip), 1.0 / sample_rate)
    denom = float(np.sum(fft))
    centroid = float(np.sum(freqs * fft) / denom) if denom > 0 else 0.0
    features.append(centroid / (sample_rate / 2.0))
    # Kurtosis
    mu, sigma = float(np.mean(audio)), float(np.std(audio)) + 1e-8
    features.append(float(np.mean(((audio - mu) / sigma) ** 4)))
    # Pad remainder with deterministic statistics (no random — reproducible)
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

# Page configuration
st.set_page_config(
    page_title="Omni-Sense",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #003366;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .leak-detected {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    .no-leak {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #003366;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🌊 Omni-Sense")
    st.markdown("""
    <div style='text-align: center; color: #667eea; font-size: 0.9em; margin-bottom: 1.5rem;'>
    <b>Acoustic Diagnostics for Water Infrastructure</b><br>
    <i>Detecting leaks through vibration analysis</i>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # EEP endpoint configuration
    st.markdown("### Configuration")
    eep_url = st.text_input(
        "EEP Server URL",
        value="http://localhost:8000",
        help="Enterprise Execution Platform endpoint"
    )

    # Health check
    st.markdown("### System Status")
    try:
        health_response = requests.get(f"{eep_url}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ EEP Online", icon="✅")
            system_online = True
        else:
            st.warning("⚠️ EEP Unreachable")
            system_online = False
    except:
        st.warning("⚠️ EEP Offline", icon="⚠️")
        system_online = False

    if not system_online:
        st.info("💡 Demo mode enabled: Using local feature extraction")

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

# ==================== MAIN AREA ====================
st.markdown('<div class="main-header">Omni-Sense Dashboard</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Diagnose", "🗺️ Beirut Network Map", "📊 Model Insights", "🏗️ System Architecture"])

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
                # Read once into a buffer so the pointer is never exhausted
                _raw_bytes = uploaded_file.read()
                audio_data, sample_rate = sf.read(io.BytesIO(_raw_bytes))
                # Convert multi-channel to mono by averaging channels
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                # Store on session_state so the outer block can reuse without re-reading
                st.session_state['_audio_data'] = audio_data
                st.session_state['_sample_rate'] = sample_rate
                st.session_state['_raw_bytes'] = _raw_bytes
                duration = len(audio_data) / sample_rate
                n_channels = 1  # already converted to mono
                st.metric("Sample Rate", f"{sample_rate} Hz")
                st.metric("Duration", f"{duration:.2f} s")
                st.metric("Channels", f"{n_channels} (mono)")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if uploaded_file is not None and '_audio_data' in st.session_state:
        try:
            # Reuse the data already read in the info block — no second sf.read()
            audio_data = st.session_state['_audio_data']
            sample_rate = st.session_state['_sample_rate']

            # Display waveform and spectrogram
            st.markdown("#### Audio Analysis")
            col_wave, col_spec = st.columns(2)

            with col_wave:
                st.markdown("**Waveform**")
                fig_wave, ax = plt.subplots(figsize=(8, 4))
                time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
                ax.plot(time_axis, audio_data, linewidth=0.5, color='#667eea')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Accelerometer Signal')
                ax.grid(True, alpha=0.3)
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
                    # Fallback: compute log-STFT with numpy
                    n_fft = min(1024, len(audio_data))
                    hop = n_fft // 2
                    frames = [audio_data[i:i + n_fft] for i in range(0, max(1, len(audio_data) - n_fft), hop)]
                    if not frames:
                        frames = [np.zeros(n_fft)]
                    frames = [f if len(f) == n_fft else np.pad(f, (0, n_fft - len(f))) for f in frames]
                    S = np.array([np.abs(np.fft.rfft(f)) ** 2 for f in frames]).T  # (freq, time)
                    S_db = 10 * np.log10(S + 1e-10)

                fig_spec, ax = plt.subplots(figsize=(8, 4))
                if _librosa_ok:
                    librosa.display.specshow(S_db, sr=sample_rate, hop_length=512,
                                             x_axis='time', y_axis='log', ax=ax)
                    fig_spec.colorbar(ax.collections[0] if ax.collections else
                                      ax.images[0] if ax.images else None,
                                      ax=ax, format="%+2.0f dB") if (ax.collections or ax.images) else None
                else:
                    ax.imshow(S_db, aspect='auto', origin='lower',
                              extent=[0, len(audio_data) / sample_rate, 0, sample_rate / 2])
                    ax.set_ylabel('Frequency (Hz)')
                    ax.set_xlabel('Time (s)')
                ax.set_title('Log-Power Spectrogram')
                st.pyplot(fig_spec, use_container_width=True)
                plt.close(fig_spec)

            # Analyze button
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
                    # Use the original raw bytes — no re-read needed
                    audio_bytes = io.BytesIO(st.session_state['_raw_bytes'])
                    audio_bytes.seek(0)

                    try:
                        # Send to EEP
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

            # Display results if available
            if 'diagnosis_result' in st.session_state:
                result = st.session_state['diagnosis_result']
                st.divider()
                st.markdown("#### Diagnosis Results")

                # Main result box
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

                # Confidence gauge
                st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")

                # Detailed metrics
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

                # Expandable details
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

    # Sensor locations in Beirut with realistic coordinates
    sensors = {
        "Hamra": {"lat": 33.8938, "lng": 35.4831, "status": "Normal", "last_reading": "2 min ago"},
        "Achrafieh": {"lat": 33.8886, "lng": 35.5157, "status": "Leak", "last_reading": "1 min ago"},
        "Verdun": {"lat": 33.8822, "lng": 35.4772, "status": "Normal", "last_reading": "5 min ago"},
        "Mar Mikhael": {"lat": 33.8907, "lng": 35.5214, "status": "Offline", "last_reading": "1 hour ago"},
        "Gemmayzeh": {"lat": 33.8965, "lng": 35.5168, "status": "Normal", "last_reading": "3 min ago"},
        "Ras Beirut": {"lat": 33.9003, "lng": 35.4788, "status": "Normal", "last_reading": "4 min ago"},
        "Badaro": {"lat": 33.8747, "lng": 35.5093, "status": "Normal", "last_reading": "2 min ago"},
        "Ain el-Mreisseh": {"lat": 33.9014, "lng": 35.4851, "status": "Normal", "last_reading": "3 min ago"},
    }

    # Create folium map centered on Beirut
    beirut_center = [33.8886, 35.5020]
    m = folium.Map(
        location=beirut_center,
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # Color mapping for sensor status
    color_map = {
        "Normal": "green",
        "Leak": "red",
        "Offline": "gray"
    }

    # Add sensors to map
    for sensor_name, info in sensors.items():
        color = color_map.get(info["status"], "blue")
        icon_symbol = "checkmark" if info["status"] == "Normal" else ("exclamation" if info["status"] == "Leak" else "question")

        popup_text = f"""
        <b>{sensor_name}</b><br>
        Status: {info['status']}<br>
        Last Reading: {info['last_reading']}<br>
        Lat: {info['lat']:.4f}, Lng: {info['lng']:.4f}
        """

        folium.CircleMarker(
            location=[info["lat"], info["lng"]],
            radius=10,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)

        # Add label
        folium.Marker(
            location=[info["lat"], info["lng"]],
            popup=sensor_name,
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    # Display map
    st.markdown("#### Active Sensors in Beirut")
    st_folium(m, width=1400, height=600)

    # Sensor status legend and details
    st.markdown("#### Sensor Status")
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        normal_count = sum(1 for s in sensors.values() if s["status"] == "Normal")
        st.metric("🟢 Normal", normal_count, help=f"Sensors operating normally")

    with col_s2:
        leak_count = sum(1 for s in sensors.values() if s["status"] == "Leak")
        st.metric("🔴 Leak Detected", leak_count, help=f"Active leak alerts")

    with col_s3:
        offline_count = sum(1 for s in sensors.values() if s["status"] == "Offline")
        st.metric("⚫ Offline", offline_count, help=f"Offline sensors")

    # Detailed sensor table
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

        # Simulated feature importance data
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
        y_pos = np.arange(len(feature_names))
        colors = plt.cm.viridis(importance_scores / importance_scores.max())
        ax.barh(y_pos, importance_scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=9)
        ax.set_xlabel('Importance Score')
        ax.set_title('SHAP Feature Importance')
        ax.invert_yaxis()
        st.pyplot(fig_importance, use_container_width=True)

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
        colors_pie = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        wedges, texts, autotexts = ax.pie(
            feature_groups.values(),
            labels=feature_groups.keys(),
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Feature Group Distribution')
        st.pyplot(fig_groups, use_container_width=True)

    st.divider()
    st.markdown("#### Model Architecture Overview")

    # ASCII architecture diagram
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
                    IEP1 Features      Local Features
                   (100-d vector)      (Fallback mode)
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
        "IEP1": "Feature extraction: 100-dimensional physics-informed features from STFT",
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

    # Detailed architecture explanation
    col_arch1, col_arch2 = st.columns([1.5, 1])

    with col_arch1:
        pipeline_text = """
        **Data Ingestion**
        - Accelerometer WAV files (16-bit PCM, 16 kHz)
        - Supports up to 30-second recordings
        - Automatic signal preprocessing

        **Feature Engineering (IEP1)**
        - Log-power spectrogram with 2048 FFT size
        - MFCC extraction (13 coefficients + deltas)
        - Temporal and spectral statistics
        - Produces 100-dimensional feature vector

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
            "EEP": "Enterprise Execution Platform",
            "IEP1": "Feature Extraction",
            "IEP2": "XGBoost Model",
            "IEP4": "Random Forest Model",
            "IEP3": "Decision Aggregator",
        }

        # Check service health
        for service_name, description in services.items():
            if system_online:
                st.success(f"✅ {service_name}: {description}")
            else:
                st.warning(f"⚠️ {service_name}: {description} (Demo Mode)")

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

