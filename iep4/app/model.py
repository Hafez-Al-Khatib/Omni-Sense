"""
Omni-Sense CNN Classifier
===========================
Small 2-D CNN trained on mel spectrograms from 5-second accelerometer recordings.

Architecture:
  Input  : mel spectrogram  (n_mels=64, time_frames≈157)  shape (1, 64, 157)
  Block 1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)     → (32, 32, 78)
  Block 2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)     → (64, 15, 39)
  Block 3: Conv2D(128, 3×3) → BN → ReLU → GlobalAvgPool   → (128,)
  Head   : Dropout(0.4) → Linear(128, n_classes) → Softmax

Rationale for 2D CNN over raw 1D waveform:
  - 80 000-sample 1D waveform is too long for direct conv (large receptive field needed)
  - Mel spectrogram (64×157) is a compact time-frequency representation that
    preserves the structural features captured by IEP1's hand-crafted features,
    but lets the CNN discover its own discriminative filters end-to-end
  - With augmented data (~800 samples), ~50K-parameter 2D CNN converges well

WHY THIS IS DISTINCT FROM IEP1 + IEP2:
  IEP1 + IEP2: hand-crafted physics features → classical ML ensemble
  IEP4:        raw spectrogram → learned convolutional features → deep classifier
  These are genuinely different inductive biases; their ensemble is complementary.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("iep4.model")

# ─── Spectrogram parameters (public aliases for external use by train_cnn.py) ─
SR        = 16000
N_MELS    = 64             # Legacy mel bins (kept for reference)
N_FFT     = 1024
HOP       = 512
N_FRAMES  = (SR * 5) // HOP + 1   # ~157 frames for 5 s
FREQ_BINS = N_FFT // 2 + 1        # 513 — linear STFT bins (preferred for vibration)

# Private aliases kept for backward compat inside this module
_SR = SR
_N_MELS = N_MELS
_N_FFT = N_FFT
_HOP = HOP
_N_FRAMES = N_FRAMES

# ─── Default model path ──────────────────────────────────────────────────────
CNN_PT_PATH   = Path("models/cnn_classifier.pt")
CNN_ONNX_PATH = Path("models/cnn_classifier.onnx")
LABEL_MAP_PATH = Path("models/label_map.json")

_FALLBACK_LABELS: dict[int, str] = {0: "Leak", 1: "No_Leak"}


def _compute_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """
    Compute log-magnitude linear-frequency STFT spectrogram.

    Returns float32 (1, FREQ_BINS, N_FRAMES) normalised to zero-mean/unit-std.

    WHY NOT MEL: mel scale warps to human hearing (speech/music). Pipe vibration
    fault signatures are harmonics evenly spaced in Hz — linear STFT preserves
    those physical relationships.
    """
    import librosa

    y = waveform.astype(np.float32)
    D = librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP, window="hann")
    mag = np.abs(D).astype(np.float32)   # (FREQ_BINS, T)
    log_mag = np.log1p(mag)

    # Pad/trim time axis
    if log_mag.shape[1] < N_FRAMES:
        log_mag = np.pad(log_mag, ((0, 0), (0, N_FRAMES - log_mag.shape[1])))
    else:
        log_mag = log_mag[:, :N_FRAMES]

    log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-8)
    return log_mag[np.newaxis, :, :]   # (1, FREQ_BINS, N_FRAMES)


# Keep mel variant for backward compatibility
def _compute_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Legacy mel spectrogram (kept for backward compat). Use _compute_spectrogram."""
    import librosa
    y = waveform.astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=_SR, n_fft=_N_FFT, hop_length=_HOP,
        n_mels=_N_MELS, fmin=20.0, fmax=_SR / 2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel[np.newaxis, :, :]


class CNNClassifier:
    """
    2-D CNN for vibration fault classification.

    Supports two backends:
      ONNX  — preferred for production (no PyTorch dependency at runtime)
      PyTorch — fallback when ONNX not available

    Degrades gracefully: if no model file exists, is_loaded=False and
    predict() raises NotReadyError.  The EEP treats IEP4 as additive —
    it uses IEP2 alone if IEP4 is unavailable.
    """

    def __init__(self):
        self._session = None          # ONNX session
        self._torch_model = None      # PyTorch model
        self._backend: str | None = None
        self._label_map: dict[int, str] = _FALLBACK_LABELS.copy()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def _load_label_map(self, model_dir: Path) -> None:
        p = model_dir / LABEL_MAP_PATH.name
        if p.exists():
            import json
            with open(p) as f:
                raw = json.load(f)
            self._label_map = {int(k): v for k, v in raw.items()
                               if k != "_decision_threshold"}
            logger.info(f"CNN label map: {self._label_map}")

    def load(self) -> None:
        """
        Try ONNX first, then PyTorch .pt file.
        
        Raises:
            FileNotFoundError: If no model artifacts are found.
        """
        if CNN_ONNX_PATH.exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(CNN_ONNX_PATH),
                    providers=["CPUExecutionProvider"],
                )
                self._backend = "onnx"
                self._load_label_map(CNN_ONNX_PATH.parent)
                logger.info(f"CNN loaded (ONNX) from {CNN_ONNX_PATH}")
                return
            except Exception as exc:
                logger.warning(f"CNN ONNX load failed: {exc}")

        if CNN_PT_PATH.exists():
            try:
                import torch
                model = _build_cnn(n_classes=len(_FALLBACK_LABELS))
                state = torch.load(str(CNN_PT_PATH), map_location="cpu")
                # Handle both raw state_dict and {"model": state_dict, "labels": ...}
                if isinstance(state, dict) and "model" in state:
                    model.load_state_dict(state["model"])
                    if "labels" in state:
                        self._label_map = {int(k): v for k, v in state["labels"].items()}
                else:
                    model.load_state_dict(state)
                model.eval()
                self._torch_model = model
                self._backend = "torch"
                self._load_label_map(CNN_PT_PATH.parent)
                logger.info(f"CNN loaded (PyTorch) from {CNN_PT_PATH}")
                return
            except Exception as exc:
                logger.warning(f"CNN PyTorch load failed: {exc}")

        raise FileNotFoundError(
            f"CNN model not found at {CNN_ONNX_PATH} or {CNN_PT_PATH}. "
            "Run scripts/train_cnn.py to train it."
        )

    def predict(self, waveform: np.ndarray) -> dict:
        """
        Classify a 5-second accelerometer waveform.

        Args:
            waveform: 1D float32 array at 16 kHz

        Returns:
            dict: label, confidence, probabilities, backend
        """
        if not self.is_loaded:
            raise RuntimeError("CNN model not loaded")

        log_mel = _compute_spectrogram(waveform)   # (1, FREQ_BINS, N_FRAMES)
        # Add batch dimension → (1, 1, 64, time_frames)
        x = log_mel[np.newaxis, :, :, :]

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: x.astype(np.float32)})
            proba = output[0][0]   # (n_classes,)
        else:
            import torch
            with torch.no_grad():
                out = self._torch_model(torch.tensor(x))
                proba = torch.softmax(out, dim=1).numpy()[0]

        proba = np.array(proba, dtype=np.float32)
        label_idx = int(np.argmax(proba))
        label = self._label_map.get(label_idx, "unknown")

        probabilities = {
            self._label_map.get(i, str(i)): float(p)
            for i, p in enumerate(proba)
        }

        return {
            "label": label,
            "confidence": float(proba[label_idx]),
            "probabilities": probabilities,
            "backend": self._backend,
        }


# ─── Model architecture ───────────────────────────────────────────────────────

def _build_cnn(n_classes: int = 2, n_mels: int = FREQ_BINS):
    """
    Build the CNN model.  Imported lazily so torch is not required at startup.
    """
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        def forward(self, x):
            return self.net(x)

    class OmniCNN(nn.Module):
        """
        ~50 K parameter CNN.  Designed to train in < 5 min on CPU
        with 500–1000 augmented samples.
        """
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(
                ConvBlock(1, 32),    # (1, 64, T) → (32, 32, T/2)
                ConvBlock(32, 64),   # → (64, 16, T/4)
                ConvBlock(64, 128),  # → (128, 8, T/8)
            )
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.blocks(x)
            x = self.global_pool(x).flatten(1)
            return self.classifier(x)

    return OmniCNN()


# ─── ONNX export helper ───────────────────────────────────────────────────────

def export_to_onnx(pt_path: str | Path, onnx_path: str | Path,
                   n_classes: int = 2) -> None:
    """Export a trained .pt model to ONNX for production serving."""
    import torch

    pt_path = Path(pt_path)
    onnx_path = Path(onnx_path)

    model = _build_cnn(n_classes=n_classes)
    state = torch.load(str(pt_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 1, FREQ_BINS, N_FRAMES)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={"mel_spectrogram": {0: "batch", 3: "time"}},
        opset_version=13,
    )
    logger.info(f"CNN exported to ONNX: {onnx_path}")


# Singleton
cnn_classifier = CNNClassifier()
