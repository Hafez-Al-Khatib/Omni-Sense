"""
CNN Autoencoder OOD Detector — Training Script
================================================
Trains the ``AutoencoderOODDetector`` exclusively on *normal* spectrograms
(label == "Normal_Operation" or label == "No_Leak") from the augmented dataset.

At the end of training the script:
  1. Saves the PyTorch checkpoint  → iep2/models/autoencoder_ood.pt
  2. Exports to ONNX                → iep2/models/autoencoder_ood.onnx
  3. Calibrates & saves threshold   → iep2/models/autoencoder_threshold.npy
  4. Prints a quick sanity table    (normal vs anomaly reconstruction errors)

Usage
-----
    python scripts/train_autoencoder.py \\
        --clips-dir  data/synthesized \\
        --output-dir iep2/models \\
        [--epochs 80] [--batch-size 32] [--lr 1e-3] [--val-split 0.15]

Design notes
------------
WHY TRAIN ONLY ON NORMAL SAMPLES
  Reconstruction autoencoders learn to minimise error on the training
  distribution. Novel patterns (leaks) that the network never saw produce
  high reconstruction error → reliable OOD signal without labelled anomalies.
  This mirrors Taiwan Water Corp's excavation-event detector (99.07% accuracy)
  and is the standard approach in the vibration monitoring literature.

WHY LOG-LINEAR STFT (not mel)
  Vibration fault signatures are harmonics evenly spaced in Hz. Mel scale
  warps to human hearing, destroying those physical relationships. Linear
  STFT preserves them. Matches iep4/app/model.py exactly.

THRESHOLD CALIBRATION
  95th-percentile of reconstruction errors on a held-out normal validation
  set. That means ~5% of normal frames are flagged OOD (false alarm rate),
  while real anomalies sit well above the threshold (>10× higher error).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("train_autoencoder")

# ─── Audio / spectrogram constants (MUST match iep4/app/model.py) ─────────────
TARGET_SR      = 16_000
TARGET_SAMPLES = TARGET_SR * 5     # 80 000 samples per clip
N_FFT          = 1024
HOP            = 512
FREQ_BINS      = N_FFT // 2 + 1   # 513
N_FRAMES       = TARGET_SAMPLES // HOP + 1  # 157

NORMAL_LABELS = {"Normal_Operation", "No_Leak"}


# ─── Spectrogram ──────────────────────────────────────────────────────────────

def load_spectrogram(wav_path: Path) -> np.ndarray | None:
    """
    Load a WAV clip and return a (1, FREQ_BINS, N_FRAMES) float32 spectrogram.
    Returns None on read/decode errors.
    """
    try:
        import soundfile as sf
        waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as exc:
        log.warning("Could not read %s: %s", wav_path, exc)
        return None

    # Resample if needed (shouldn't happen with correctly prepared clips)
    if sr != TARGET_SR:
        try:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        except Exception:
            log.warning("Resample failed for %s, skipping.", wav_path)
            return None

    # Ensure 1-D mono, padded/trimmed to TARGET_SAMPLES
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if len(waveform) < TARGET_SAMPLES:
        waveform = np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))
    else:
        waveform = waveform[:TARGET_SAMPLES]

    # Log-linear STFT spectrogram
    import librosa
    D   = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP, window="hann")
    mag = np.log1p(np.abs(D)).astype(np.float32)   # (FREQ_BINS, T)

    if mag.shape[1] < N_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, N_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :N_FRAMES]

    mag = (mag - mag.mean()) / (mag.std() + 1e-8)
    return mag[np.newaxis, :, :]   # (1, FREQ_BINS, N_FRAMES)


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_normal_spectrograms(
    clips_dir: Path,
    max_clips: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Return (anomaly_specs, normal_specs) from the synthesized clips directory.

    Normal  = label in NORMAL_LABELS
    Anomaly = everything else (used only for sanity-check evaluation)
    """
    meta_path = clips_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {clips_dir}")

    import pandas as pd
    df = pd.read_csv(meta_path)

    normal_rows  = df[df["label"].isin(NORMAL_LABELS)].reset_index(drop=True)
    anomaly_rows = df[~df["label"].isin(NORMAL_LABELS)].reset_index(drop=True)

    log.info(
        "Dataset — normal: %d  anomaly: %d  total: %d",
        len(normal_rows), len(anomaly_rows), len(df),
    )

    def _load_batch(rows, limit=None):
        specs = []
        for _, row in rows.iterrows():
            if limit and len(specs) >= limit:
                break
            spec = load_spectrogram(clips_dir / row["filename"])
            if spec is not None:
                specs.append(spec)
        return specs

    normal_limit = max_clips if max_clips else None
    normal_specs = _load_batch(normal_rows, normal_limit)
    # Load a sample of anomalies for eval only
    anomaly_specs = _load_batch(anomaly_rows, 200)

    log.info("Loaded %d normal spectrograms.", len(normal_specs))
    log.info("Loaded %d anomaly spectrograms (eval only).", len(anomaly_specs))
    return normal_specs, anomaly_specs


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    clips_dir:  Path,
    output_dir: Path,
    epochs:     int   = 80,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    val_split:  float = 0.15,
    seed:       int   = 42,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Import architecture from iep2/app
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from iep2.app.autoencoder_ood_detector import (
        _build_autoencoder,
        export_to_onnx,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    normal_specs, anomaly_specs = load_normal_spectrograms(clips_dir)

    if len(normal_specs) < 50:
        log.error("Too few normal spectrograms (%d). Need at least 50.", len(normal_specs))
        sys.exit(1)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(normal_specs))
    n_val = max(10, int(len(normal_specs) * val_split))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = np.stack([normal_specs[i] for i in train_idx])   # (N, 1, H, W)
    X_val   = np.stack([normal_specs[i] for i in val_idx])

    log.info("Train: %d  Val: %d", len(X_train), len(X_val))

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ── Model / optimiser ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    model  = _build_autoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", n_params)

    criterion = nn.MSELoss()
    optim     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val_loss = float("inf")
    best_state    = None

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for (batch,) in train_dl:
            batch = batch.to(device)
            optim.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(X_train)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_dl:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_loss /= len(X_val)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  train=%.6f  val=%.6f  lr=%.2e",
                epoch, epochs, train_loss, val_loss,
                scheduler.get_last_lr()[0],
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    log.info("Best val loss: %.6f", best_val_loss)

    # ── Load best checkpoint, compute threshold ───────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    model = model.cpu()

    def _reconstruction_error(spec_np: np.ndarray) -> float:
        x = torch.from_numpy(spec_np[np.newaxis]).float()
        with torch.no_grad():
            r = model(x)
        return float(torch.nn.functional.mse_loss(r, x).item())

    val_errors = [_reconstruction_error(normal_specs[i]) for i in val_idx]
    threshold  = float(np.percentile(val_errors, 95))

    log.info(
        "Threshold (p95 of %d normal val samples): %.5f  "
        "(mean=%.5f, std=%.5f, max=%.5f)",
        len(val_errors), threshold,
        np.mean(val_errors), np.std(val_errors), np.max(val_errors),
    )

    # ── Sanity check ─────────────────────────────────────────────────────────
    if anomaly_specs:
        anom_errors  = [_reconstruction_error(s) for s in anomaly_specs[:100]]
        normal_above = sum(1 for e in val_errors   if e > threshold)
        anom_above   = sum(1 for e in anom_errors  if e > threshold)
        log.info(
            "Sanity — normal above threshold: %d/%d (%.1f%%)   "
            "anomaly above threshold: %d/%d (%.1f%%)",
            normal_above, len(val_errors),  100*normal_above/len(val_errors),
            anom_above,   len(anom_errors), 100*anom_above/len(anom_errors),
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path  = output_dir / "autoencoder_ood.pt"
    onnx_path = output_dir / "autoencoder_ood.onnx"
    thr_path  = output_dir / "autoencoder_threshold.npy"

    torch.save({"model": best_state, "threshold": threshold}, str(pt_path))
    log.info("Saved PyTorch checkpoint: %s", pt_path)

    export_to_onnx(pt_path, onnx_path)
    log.info("Saved ONNX model: %s", onnx_path)

    np.save(str(thr_path), np.array(threshold))
    log.info("Saved threshold: %s  (%.5f)", thr_path, threshold)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  AUTOENCODER OOD DETECTOR — TRAINING COMPLETE")
    print("═" * 72)
    print(f"  Normal training clips  : {len(X_train)}")
    print(f"  Normal validation clips: {len(X_val)}")
    print(f"  Best val MSE loss      : {best_val_loss:.6f}")
    print(f"  OOD threshold (p95)    : {threshold:.5f}")
    if anomaly_specs:
        print(f"  Anomaly detection rate : {anom_above}/{len(anom_errors)}"
              f" ({100*anom_above/len(anom_errors):.1f}%)")
        print(f"  Normal false alarm rate: {normal_above}/{len(val_errors)}"
              f" ({100*normal_above/len(val_errors):.1f}%)")
    print(f"  Saved: {pt_path}")
    print(f"  Saved: {onnx_path}")
    print(f"  Saved: {thr_path}")
    print("═" * 72)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Train CNN Autoencoder OOD detector on normal acoustic clips"
    )
    ap.add_argument("--clips-dir",  default="data/synthesized",
                    help="Directory containing WAV clips + metadata.csv")
    ap.add_argument("--output-dir", default="iep2/models",
                    help="Where to save .pt, .onnx and threshold.npy")
    ap.add_argument("--epochs",     type=int,   default=80)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--val-split",  type=float, default=0.15,
                    help="Fraction of normal clips held out for threshold calibration")
    ap.add_argument("--seed",       type=int,   default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        clips_dir  = Path(args.clips_dir),
        output_dir = Path(args.output_dir),
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        val_split  = args.val_split,
        seed       = args.seed,
    )
