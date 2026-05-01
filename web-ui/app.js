/**
 * Omni-Sense Web UI — Application Logic
 * ========================================
 * Handles audio recording, client-side downsampling,
 * drag-and-drop uploads, and communication with the EEP API.
 */

// ─── Configuration ───────────────────────────────────────
const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : window.location.origin;

const TARGET_SR = 16000;
const RECORD_DURATION_S = 5.0;
const FETCH_TIMEOUT_MS = 30000;

// ─── DOM References ──────────────────────────────────────
const recordBtn = document.getElementById('recordBtn');
const recordTimer = document.getElementById('recordTimer');
const recordInner = document.getElementById('recordInner');
const waveform = document.getElementById('waveform');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadZone = document.getElementById('uploadZone');
const pipeMaterial = document.getElementById('pipeMaterial');
const pressureBar = document.getElementById('pressureBar');
const pressureValue = document.getElementById('pressureValue');
const submitBtn = document.getElementById('submitBtn');
const resultSection = document.getElementById('resultSection');
const oodSection = document.getElementById('oodSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const connectionStatus = document.getElementById('connectionStatus');
const connectionText = document.getElementById('connectionText');
const toastContainer = document.getElementById('toastContainer');

// ─── State ───────────────────────────────────────────────
let mediaRecorder = null;
let audioContext = null;
let audioChunks = [];
let audioBlob = null;
let isRecording = false;
let recordStartTime = null;
let timerInterval = null;
let waveformBars = [];

// ─── Toast Notifications ─────────────────────────────────
function showToast(message, type = 'error', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = '⚠️';
    if (type === 'success') icon = '✅';
    if (type === 'info') icon = 'ℹ️';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `
        <span>${icon}</span>
        <span>${message}</span>
        <button class="toast-close" aria-label="Dismiss">&times;</button>
    `;

    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.remove();
    });

    toastContainer.appendChild(toast);

    if (duration > 0) {
        setTimeout(() => {
            toast.remove();
        }, duration);
    }
}

// ─── Health Check ────────────────────────────────────────
async function checkHealth() {
    connectionStatus.className = 'status-indicator checking';
    connectionText.textContent = 'Checking…';

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const resp = await fetch(`${API_BASE}/health`, { signal: controller.signal });
        clearTimeout(timeoutId);

        if (resp.ok) {
            connectionStatus.className = 'status-indicator';
            connectionText.textContent = 'Online';
        } else {
            connectionStatus.className = 'status-indicator offline';
            connectionText.textContent = 'Degraded';
        }
    } catch (err) {
        connectionStatus.className = 'status-indicator offline';
        connectionText.textContent = 'Offline';
        showToast(
            'Backend appears offline. Analysis will fail unless the server is running.',
            'warning',
            6000
        );
    }
}

checkHealth();

// ─── Step Indicator ──────────────────────────────────────
function updateSteps(step) {
    const steps = document.querySelectorAll('.step');
    const lines = [document.getElementById('line1'), document.getElementById('line2')];

    steps.forEach((s, i) => {
        const num = i + 1;
        s.classList.remove('active', 'completed');
        if (num < step) {
            s.classList.add('completed');
        } else if (num === step) {
            s.classList.add('active');
        }
    });

    lines.forEach((l, i) => {
        l.classList.toggle('completed', i + 1 < step);
    });
}

// ─── Initialize Waveform Bars ────────────────────────────
function initWaveform() {
    waveform.innerHTML = '';
    waveformBars = [];
    for (let i = 0; i < 40; i++) {
        const bar = document.createElement('div');
        bar.className = 'waveform-bar';
        bar.style.height = '4px';
        waveform.appendChild(bar);
        waveformBars.push(bar);
    }
}
initWaveform();

// ─── Pressure Slider ────────────────────────────────────
pressureBar.addEventListener('input', () => {
    pressureValue.textContent = parseFloat(pressureBar.value).toFixed(1);
});

// ─── File Upload ─────────────────────────────────────────
function handleFile(file) {
    if (!file) return;
    const validTypes = ['audio/wav', 'audio/x-wav', 'audio/ogg', 'audio/flac', 'audio/x-flac'];
    const validExts = ['.wav', '.ogg', '.flac'];
    const hasValidExt = validExts.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!validTypes.includes(file.type) && !hasValidExt) {
        showToast('Please upload a WAV, OGG, or FLAC audio file.', 'warning');
        return;
    }

    audioBlob = file;
    fileName.textContent = file.name;
    submitBtn.disabled = false;
    updateSteps(2);
    showToast(`File ready: ${file.name}`, 'success', 3000);
}

fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// ─── Drag & Drop ─────────────────────────────────────────
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => {
        uploadZone.classList.add('drag-over');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, () => {
        uploadZone.classList.remove('drag-over');
    }, false);
});

uploadZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// ─── Recording ──────────────────────────────────────────
recordBtn.addEventListener('click', async () => {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: TARGET_SR,
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            }
        });

        audioContext = new AudioContext({ sampleRate: TARGET_SR });
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 128;
        source.connect(analyser);

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm'
        });

        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });

            // Downsample to 16kHz mono WAV
            audioBlob = await downsampleToWav(webmBlob);
            fileName.textContent = 'Recorded (5s, 16kHz)';
            submitBtn.disabled = false;
            updateSteps(2);
            showToast('Recording captured successfully', 'success', 3000);
        };

        mediaRecorder.start(100);
        isRecording = true;
        recordBtn.classList.add('recording');
        recordStartTime = Date.now();

        // Timer
        timerInterval = setInterval(() => {
            const elapsed = (Date.now() - recordStartTime) / 1000;
            recordTimer.textContent = elapsed.toFixed(1) + 's';

            if (elapsed >= RECORD_DURATION_S) {
                stopRecording();
            }
        }, 100);

        // Waveform visualization
        animateWaveform(analyser);

    } catch (err) {
        console.error('Microphone access denied:', err);
        showToast(
            'Microphone access is required for recording. Please grant permission in your browser settings.',
            'error',
            6000
        );
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;
    recordBtn.classList.remove('recording');
    clearInterval(timerInterval);
    recordTimer.textContent = '5.0s';

    // Reset waveform
    waveformBars.forEach(bar => bar.style.height = '4px');
}

function animateWaveform(analyser) {
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    function draw() {
        if (!isRecording) return;
        requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        const step = Math.floor(dataArray.length / waveformBars.length);
        for (let i = 0; i < waveformBars.length; i++) {
            const value = dataArray[i * step] || 0;
            const height = Math.max(4, (value / 255) * 48);
            waveformBars[i].style.height = height + 'px';
        }
    }
    draw();
}

// ─── Client-Side Downsampling ────────────────────────────
async function downsampleToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const ctx = new OfflineAudioContext(1, 1, TARGET_SR);

    let audioBuffer;
    try {
        audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    } catch {
        // Fallback: return original blob
        return blob;
    }

    // Resample to 16kHz mono
    const targetLength = Math.min(
        audioBuffer.length,
        Math.ceil(RECORD_DURATION_S * audioBuffer.sampleRate)
    );

    const offlineCtx = new OfflineAudioContext(
        1,
        Math.ceil(targetLength * (TARGET_SR / audioBuffer.sampleRate)),
        TARGET_SR
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start();

    const renderedBuffer = await offlineCtx.startRendering();
    const samples = renderedBuffer.getChannelData(0);

    // Encode to WAV
    return encodeWav(samples, TARGET_SR);
}

function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM data
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// ─── Submit to API ──────────────────────────────────────
submitBtn.addEventListener('click', async () => {
    if (!audioBlob) return;

    showLoading(true);
    hideResults();
    updateSteps(3);

    const metadata = JSON.stringify({
        pipe_material: pipeMaterial.value,
        pressure_bar: parseFloat(pressureBar.value),
    });

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    formData.append('metadata', metadata);

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

        const resp = await fetch(`${API_BASE}/api/v1/diagnose`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });
        clearTimeout(timeoutId);

        const data = await resp.json();

        if (resp.status === 200) {
            showDiagnosisResult(data);
        } else if (resp.status === 422 && data.anomaly_score !== undefined) {
            showOODResult(data);
        } else {
            const detail = data.detail || JSON.stringify(data);
            showToast(`Analysis failed: ${detail}`, 'error', 7000);
        }
    } catch (err) {
        console.error('API error:', err);
        if (err.name === 'AbortError') {
            showToast(
                'Request timed out. The server may be overloaded or unreachable.',
                'error',
                7000
            );
        } else if (err.message && err.message.includes('fetch')) {
            showToast(
                'Cannot reach the backend. Make sure the API server is running and CORS is enabled.',
                'error',
                8000
            );
        } else {
            showToast(
                'Failed to connect to the API. Is the server running?',
                'error',
                7000
            );
        }
    } finally {
        showLoading(false);
    }
});

// ─── Display Results ────────────────────────────────────
function showDiagnosisResult(data) {
    resultSection.style.display = 'block';
    oodSection.style.display = 'none';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Confidence gauge
    const confidence = data.confidence || 0;
    const pct = confidence * 100;
    const arcLength = (confidence) * 251; // 251 is full arc
    document.getElementById('gaugeArc').setAttribute('stroke-dasharray', `${arcLength} 251`);
    document.getElementById('gaugeValue').textContent = pct.toFixed(1) + '%';

    // Color based on confidence
    const gaugeColor = confidence > 0.8 ? 'var(--success)' : confidence > 0.5 ? 'var(--warning)' : 'var(--danger)';
    document.getElementById('gaugeArc').setAttribute('stroke', gaugeColor);
    document.getElementById('gaugeValue').style.color = gaugeColor;

    // Details
    const labelEl = document.getElementById('resultLabel');
    labelEl.textContent = data.label || 'Unknown';
    labelEl.className = `result-value badge ${(data.label || '').toLowerCase()}`;

    document.getElementById('resultAnomaly').textContent = (data.anomaly_score || 0).toFixed(4);
    document.getElementById('resultOOD').textContent = data.is_in_distribution ? '✅ Yes' : '❌ No';
    document.getElementById('resultRMS').textContent = data.signal_quality?.rms?.toFixed(4) || '—';
    document.getElementById('resultLatency').textContent = (data.elapsed_ms || 0).toFixed(0) + 'ms';

    // Probability bars
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';
    if (data.probabilities) {
        Object.entries(data.probabilities).forEach(([label, prob]) => {
            const safeLabel = label.toLowerCase().replace(/\s+/g, '-');
            probBars.innerHTML += `
                <div class="prob-bar-row">
                    <span class="prob-label">${label}</span>
                    <div class="prob-track">
                        <div class="prob-fill ${safeLabel}" style="width: ${(prob * 100).toFixed(1)}%"></div>
                    </div>
                    <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                </div>
            `;
        });
    }

    showToast('Analysis complete!', 'success', 4000);
}

function showOODResult(data) {
    resultSection.style.display = 'none';
    oodSection.style.display = 'block';
    oodSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    document.getElementById('oodDetail').textContent =
        data.detail || 'The acoustic signature does not match any known environment.';
    document.getElementById('oodScore').textContent = (data.anomaly_score || 0).toFixed(4);

    showToast('Out-of-distribution sample detected', 'warning', 5000);
}

function hideResults() {
    resultSection.style.display = 'none';
    oodSection.style.display = 'none';
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
    submitBtn.disabled = show;
}

// ─── Calibration ─────────────────────────────────────────
document.getElementById('calibrateBtn')?.addEventListener('click', async () => {
    if (!audioBlob) {
        showToast('Record or upload audio first to use for calibration.', 'warning');
        return;
    }

    showLoading(true);

    const formData = new FormData();
    formData.append('audio', audioBlob, 'ambient.wav');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

        const resp = await fetch(`${API_BASE}/api/v1/calibrate`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });
        clearTimeout(timeoutId);

        const data = await resp.json();

        if (resp.status === 200) {
            showToast(
                `Calibration successful! Threshold: ${data.new_threshold?.toFixed(4)}`,
                'success',
                5000
            );
            hideResults();
        } else {
            showToast(`Calibration failed: ${JSON.stringify(data)}`, 'error');
        }
    } catch (err) {
        showToast('Failed to connect for calibration. Is the server running?', 'error');
    } finally {
        showLoading(false);
    }
});
