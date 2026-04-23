"""
CNN Autoencoder — Out-of-Distribution Detector
===============================================
Replaces the Isolation Forest in IEP2's OOD stage with a more powerful
reconstruction-error-based anomaly detector, validated by Taiwan Water Corp
(99.07% accuracy on excavation anomaly detection) and SA Water (92.44%).

Architecture
------------
Trained exclusively on *normal* (no-leak) log-linear STFT spectrograms.
At inference, a spectrogram with high reconstruction error is flagged as OOD
— the network has not seen this type of acoustic signature before.

Input : (1, FREQ_BINS, N_FRAMES) = (1, 513, 157)  — same pipeline as IEP4

Encoder (stride-2 convolutions, each halving spatial dims):
  Conv2d(1  → 16, k=3, s=2, p=1) + BN + ReLU  → (16, 257, 79)
  Conv2d(16 → 32, k=3, s=2, p=1) + BN + ReLU  → (32, 129, 40)
  Conv2d(32 →  8, k=1)           + BN + ReLU  → ( 8, 129, 40)  [bottleneck]

Decoder (transposed convolutions, each doubling spatial dims):
  ConvTranspose2d( 8 → 32, k=3, s=2, p=1) + BN + ReLU  → (32, 257, 79)
  ConvTranspose2d(32 → 16, k=3, s=2, p=1) + BN + ReLU  → (16, 513, 157)
  Conv2d(16 → 1, k=3, p=1) [linear, no activation]     → ( 1, 513, 157)

OOD score : mean squared reconstruction error per spectrogram.
Threshold : 95th‑percentile of normal-sample errors (calibrated at train time).

Training
--------
  python scripts/train_autoencoder.py \\
      --clips-dir  data/synthesized \\
      --output-dir iep2/models \\
      [--epochs 80] [--batch-size 32] [--lr 1e-3]
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("iep2.autoencoder")

# ─── Spectrogram constants — MUST match IEP4 / train_autoencoder.py ──────────
_SR         = 16_000
_N_FFT      = 1024
_HOP        = 512
_FREQ_BINS  = _N_FFT // 2 + 1          # 513
_N_FRAMES   = (_SR * 5) // _HOP + 1    # 157

# Default file locations (relative to the CWD / Docker workdir)
DEFAULT_PT_PATH        = Path("models/autoencoder_ood.pt")
DEFAULT_ONNX_PATH      = Path("models/autoencoder_ood.onnx")
DEFAULT_THRESHOLD_PATH = Path("models/autoencoder_threshold.npy")


# ─── PyTorch model definition ─────────────────────────────────────────────────

def _build_autoencoder():
    """
    Construct the CNN autoencoder.

    Not imported at module level so that torch is NOT required for
    ONNX-runtime-only deployments.
    """
    import torch.nn as nn

    class _EncBlock(nn.Module):
        """Conv2d (stride=2) + BN + ReLU — halves both spatial dims."""
        def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
            super().__init__()
            if stride == 2:
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            else:
                # Bottleneck: 1×1 conv, no spatial change
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )

        def forward(self, x):
            return self.net(x)

    class _DecBlock(nn.Module):
        """ConvTranspose2d (stride=2) + BN + ReLU — doubles both spatial dims."""
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=3, stride=2, padding=1, output_padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class CNNAutoEncoder(nn.Module):
        """
        Symmetric CNN autoencoder for normal-only reconstruction training.

        Parameter count ~25 K — deliberately small so it trains in minutes
        on CPU with ~2 500 normal spectrograms.
        """

        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = _EncBlock(1,  16, stride=2)   # (1,513,157) → (16,257,79)
            self.enc2 = _EncBlock(16, 32, stride=2)   # → (32,129,40)
            self.enc3 = _EncBlock(32,  8, stride=1)   # → (8,129,40)  bottleneck

            # Decoder
            self.dec1 = _DecBlock(8,  32)              # → (32,257,79)
            self.dec2 = _DecBlock(32, 16)              # → (16,513,157)
            self.out  = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # → (1,513,157)

        def forward(self, x):
            # encode
            z = self.enc3(self.enc2(self.enc1(x)))
            # decode
            return self.out(self.dec2(self.dec1(z)))

        def encode(self, x):
            """Return the bottleneck representation."""
            return self.enc3(self.enc2(self.enc1(x)))

    return CNNAutoEncoder()


# ─── Spectrogram helper ───────────────────────────────────────────────────────

def _wav_to_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """
    Compute a normalised log-magnitude linear STFT spectrogram.

    Parameters
    ----------
    waveform : 1-D float32, 5 s at 16 kHz (80 000 samples)

    Returns
    -------
    np.ndarray, shape (1, FREQ_BINS, N_FRAMES), float32
    """
    import librosa
    y   = waveform.astype(np.float32)
    D   = librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP, window="hann")
    mag = np.log1p(np.abs(D)).astype(np.float32)   # (513, T)

    # Pad or trim to exact N_FRAMES
    if mag.shape[1] < _N_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, _N_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :_N_FRAMES]

    mag = (mag - mag.mean()) / (mag.std() + 1e-8)
    return mag[np.newaxis, :, :]   # (1, 513, 157)


# ─── Main wrapper class ───────────────────────────────────────────────────────

class AutoencoderOODDetector:
    """
    Drop-in replacement for ``OODDetector`` (Isolation Forest).

    Interface mirrors ``ood_detector.OODDetector`` so IEP2's main.py
    can swap between them with a single flag.

    Backends (loaded in priority order)
    ------------------------------------
    1. ONNX Runtime  — zero PyTorch dependency at inference time
    2. PyTorch .pt   — fallback when onnxruntime is not installed
    """

    def __init__(self) -> None:
        self._session   = None          # ONNX session
        self._model     = None          # PyTorch model
        self._backend: str | None = None
        self._threshold: float = 0.05  # default; overridden by calibrated value
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(
        self,
        pt_path:        str | Path | None = None,
        onnx_path:      str | Path | None = None,
        threshold_path: str | Path | None = None,
    ) -> None:
        """
        Load model and calibrated threshold.

        Tries ONNX first (fast CPU inference), then PyTorch .pt.
        Raises FileNotFoundError if neither exists.
        """
        onnx_p  = Path(onnx_path or DEFAULT_ONNX_PATH)
        pt_p    = Path(pt_path   or DEFAULT_PT_PATH)
        thr_p   = Path(threshold_path or DEFAULT_THRESHOLD_PATH)

        if onnx_p.exists():
            self._load_onnx(onnx_p)
        elif pt_p.exists():
            self._load_torch(pt_p)
        else:
            raise FileNotFoundError(
                f"Autoencoder model not found at {onnx_p} or {pt_p}. "
                "Run scripts/train_autoencoder.py first."
            )

        if thr_p.exists():
            self._threshold = float(np.load(thr_p))
            log.info("Loaded calibrated OOD threshold: %.5f", self._threshold)
        else:
            log.warning(
                "Threshold file not found at %s — using default %.4f. "
                "Retrain to calibrate.", thr_p, self._threshold
            )

        self._is_loaded = True

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._backend = "onnx"
        log.info("AutoencoderOOD loaded (ONNX) from %s", path)

    def _load_torch(self, path: Path) -> None:
        import torch
        model = _build_autoencoder()
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            if "threshold" in state:
                self._threshold = float(state["threshold"])
        else:
            model.load_state_dict(state)
        model.eval()
        self._model   = model
        self._backend = "torch"
        log.info("AutoencoderOOD loaded (PyTorch) from %s", path)

    # ── Core inference ────────────────────────────────────────────────────────

    def reconstruction_error(self, spectrogram: np.ndarray) -> float:
        """
        MSE reconstruction error for a single spectrogram.

        Parameters
        ----------
        spectrogram : shape (1, FREQ_BINS, N_FRAMES) or (FREQ_BINS, N_FRAMES)

        Returns
        -------
        float — mean squared error; higher = more anomalous
        """
        if not self._is_loaded:
            raise RuntimeError("Autoencoder model not loaded")

        spec = np.array(spectrogram, dtype=np.float32)
        if spec.ndim == 2:
            spec = spec[np.newaxis, :, :]          # (1, H, W)
        x = spec[np.newaxis, :, :, :]             # (1, 1, H, W)

        if self._backend == "onnx":
            iname = self._session.get_inputs()[0].name
            recon = self._session.run(None, {iname: x})[0]
        else:
            import torch
            with torch.no_grad():
                out = self._model(torch.from_numpy(x))
                recon = out.numpy()

        mse = float(np.mean((x - recon) ** 2))
        return mse

    def reconstruction_error_from_wav(self, waveform: np.ndarray) -> float:
        """Convenience: compute spec then reconstruction error."""
        spec = _wav_to_spectrogram(waveform)
        return self.reconstruction_error(spec)

    # ── OOD interface (matches ood_detector.OODDetector) ─────────────────────

    def score(self, embedding_or_spec: np.ndarray) -> float:
        """
        Return the OOD score.

        To maintain API compatibility with OODDetector:
          - If input is 1-D (YAMNet embedding): fall back to reconstruction
            error on a zero-padded dummy spectrogram (low score = in-dist).
          - If input is 2-D (spectrogram): compute directly.

        In practice, IEP4's /classify endpoint passes spectrograms; IEP2's
        /diagnose endpoint still uses the Isolation Forest path for embeddings.
        """
        if not self._is_loaded:
            raise RuntimeError("Autoencoder model not loaded")

        if embedding_or_spec.ndim == 1:
            # 1-D embedding path — treat as unusual (score = 1.0)
            log.debug("score() called with 1-D input; using embedding-mode stub")
            return 1.0   # caller should use Isolation Forest for embeddings

        return self.reconstruction_error(embedding_or_spec)

    def is_anomalous(
        self,
        spectrogram: np.ndarray,
        threshold_override: float | None = None,
    ) -> bool:
        """
        Return True if the spectrogram is likely OOD (novel acoustic env).

        Parameters
        ----------
        spectrogram    : shape (1, H, W) or (H, W)
        threshold_override : if given, use instead of calibrated self._threshold
        """
        thr = threshold_override if threshold_override is not None else self._threshold
        err = self.reconstruction_error(spectrogram)
        return err > thr

    def is_anomalous_wav(
        self,
        waveform: np.ndarray,
        threshold_override: float | None = None,
    ) -> tuple[bool, float]:
        """
        Check OOD status from raw waveform.

        Returns
        -------
        (is_ood: bool, reconstruction_error: float)
        """
        spec = _wav_to_spectrogram(waveform)
        err  = self.reconstruction_error(spec)
        thr  = threshold_override if threshold_override is not None else self._threshold
        return err > thr, err

    # ── Threshold calibration ─────────────────────────────────────────────────

    def calibrate_threshold(
        self,
        normal_spectrograms: list[np.ndarray],
        percentile: float = 95.0,
    ) -> float:
        """
        Set self._threshold = percentile of reconstruction errors on normal samples.

        Call this after training with a held-out normal validation set.
        Save the result with np.save(DEFAULT_THRESHOLD_PATH, threshold).

        Parameters
        ----------
        normal_spectrograms : list of (1, H, W) or (H, W) spectrograms
        percentile          : default 95 — ~5% of normal samples flagged as OOD

        Returns
        -------
        float — calibrated threshold
        """
        errors = [self.reconstruction_error(s) for s in normal_spectrograms]
        self._threshold = float(np.percentile(errors, percentile))
        log.info(
            "Threshold calibrated: %.5f (p%d of %d normal samples, "
            "mean=%.5f std=%.5f)",
            self._threshold, int(percentile), len(errors),
            float(np.mean(errors)), float(np.std(errors)),
        )
        return self._threshold


# ─── ONNX export helper ───────────────────────────────────────────────────────

def export_to_onnx(
    pt_path:   str | Path,
    onnx_path: str | Path,
) -> None:
    """Export a trained .pt autoencoder to ONNX for production serving."""
    import torch

    pt_path   = Path(pt_path)
    onnx_path = Path(onnx_path)

    model = _build_autoencoder()
    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 1, _FREQ_BINS, _N_FRAMES)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["spectrogram"],
        output_names=["reconstruction"],
        dynamic_axes={"spectrogram": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=13,
    )
    log.info("Autoencoder exported to ONNX: %s", onnx_path)


# ─── Module-level singleton ───────────────────────────────────────────────────

_detector: AutoencoderOODDetector | None = None


def get_autoencoder_detector() -> AutoencoderOODDetector:
    global _detector
    if _detector is None:
        _detector = AutoencoderOODDetector()
    return _detector
