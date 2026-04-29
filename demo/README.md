# Omni-Sense Dashboard

Professional Streamlit dashboard for the Omni-Sense acoustic leak diagnostics platform.

## Quick Start

### Installation
```bash
cd demo
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

### 1. Diagnose Tab
- Upload WAV files from accelerometer recordings
- Real-time waveform visualization
- Log-power spectrogram display
- One-click EEP analysis
- Automatic demo mode with local feature extraction when EEP is offline
- Detailed results with confidence gauge and model predictions
- Full explainability metrics

### 2. Beirut Network Map
- Interactive folium map showing water network sensors
- 8 simulated sensor locations across Beirut
- Real-time status indicators (Normal/Leak/Offline)
- Clickable sensor details
- Summary statistics and detailed sensor table

### 3. Model Insights
- SHAP-based feature importance (top 20 features)
- Feature group distribution breakdown
- Model architecture diagram
- Detailed model documentation

### 4. System Architecture
- End-to-end pipeline overview
- Service health status
- Technology stack
- Component descriptions

## Configuration

Edit the EEP URL in the sidebar to point to your Enterprise Execution Platform:
- Default: `http://localhost:8000`
- Change in the sidebar configuration panel

## Demo Mode

When EEP is unavailable, the dashboard automatically falls back to **demo mode** which:
- Extracts features locally using NumPy
- Generates 100-dimensional feature vector
- Provides realistic diagnosis predictions
- Demonstrates full dashboard functionality

## Architecture

```
WAV Audio → EEP Signal QA → DSP Features (208-d)
    ↓
    ├→ IEP2 (XGBoost + RF ensemble) ─┐
    ├→ IEP4 (CNN Classifier)         ├→ Weighted Ensemble → Diagnosis
    └→ Autoencoder OOD Watchdog      ┘                      (Leak/No_Leak)
         ↓ high-confidence fault
        IEP3 (Dispatch / Active Learning)
```

## Requirements

- Python 3.9+
- Streamlit 1.32+
- NumPy, SciPy, Librosa
- Folium for map visualization
- Requests for EEP communication
- Matplotlib for plotting

See `requirements.txt` for complete dependencies.

## Professional Features

- **Gradient-styled cards** with modern color scheme
- **Responsive layout** using Streamlit columns
- **Real-time metrics** with professional gauges
- **Interactive maps** with sensor clustering
- **Expandable sections** for detailed analysis
- **Demo fallback mode** for testing without backend
- **Clean typography** with section headers and formatting

## Author

Created for the Omni-Sense capstone project - Acoustic diagnostics for Lebanese urban water infrastructure.
