"""
Omni-Sense CNN Training Script
================================
Trains the IEP4 OmniCNN on log-linear STFT spectrograms (NOT mel) derived from
the augmented 5-second accelerometer clips produced by augment_data.py.

WHY LOG-LINEAR STFT (not mel):
  Mel scale warps frequencies to match human hearing (speech/music). Pipe
  vibration physics operates in linear spectral space — the fault-type signature
  is the ratio of energy in specific Hz bands, not perceptual loudness bands.
  Log-linear STFT preserves the physical frequency relationships.

WHY FOCAL LOSS (not weighted CE):
  With 4:1 Leak/No_Leak imbalance, weighted CE still collapses to majority-class
  prediction because 1075 easy Leak gradients overpower 269 No_Leak gradients
  once the model reaches moderate Leak confidence. Focal loss
  FL = -(1-p)^gamma * log(p) down-weights confident-but-easy samples, forcing
  the network to focus on the hard minority class.

WHY RECORDING-LEVEL SPLIT:
  Each source recording produces ~20 augmented clips. A clip-level random split
  puts variants of the same recording in both train and val — the model just needs
  to identify the source recording identity, not learn fault-type features. The
  task difficulty is artificially trivial, causing random-guess collapse when the
  model fails to memorize. We split by unique source_wav so no recording bleeds
  across the fold boundary.

Usage:
    python scripts/train_cnn.py \\
        --clips-dir data/synthesized \\
        --output-dir iep4/models \\
        [--epochs 150] [--batch-size 32] [--lr 3e-4] [--binary]
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("train_cnn")

# ─── Audio constants (must match IEP4) ───────────────────────────────────────
TARGET_SR = 16_000
TARGET_SAMPLES = TARGET_SR * 5
N_FFT = 1024
HOP = 512

# Log-linear STFT: (N_FFT//2 + 1) = 513 frequency bins, ~157 time frames
FREQ_BINS = N_FFT // 2 + 1   # 513
N_FRAMES = TARGET_SAMPLES // HOP + 1   # 157

NO_FAULT_LABELS = {"No_Leak", "Normal_Operation"}


# ─── Spectrogram (log-linear STFT, NOT mel) ──────────────────────────────────

def compute_log_stft(waveform: np.ndarray) -> np.ndarray:
    """
    Log-magnitude linear-frequency STFT spectrogram.

    Returns float32 (1, FREQ_BINS, N_FRAMES) normalised to zero-mean / unit-std.

    Using linear frequency bins (not mel) because vibration fault physics
    operates in linear spectral space — harmonic series of rotating/leaking
    machinery are evenly spaced in Hz, not in mel.
    """
    import librosa

    y = waveform.astype(np.float32)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann")
    mag = np.abs(D).astype(np.float32)                          # (FREQ_BINS, T)
    log_mag = np.log1p(mag)                                     # log(1+|STFT|)

    # Pad / trim time axis to exactly N_FRAMES
    if log_mag.shape[1] < N_FRAMES:
        log_mag = np.pad(log_mag, ((0, 0), (0, N_FRAMES - log_mag.shape[1])))
    else:
        log_mag = log_mag[:, :N_FRAMES]

    # Per-spectrogram normalisation
    log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-8)

    return log_mag[np.newaxis, :, :]   # (1, FREQ_BINS, N_FRAMES)


# ─── Audio loading ────────────────────────────────────────────────────────────

def load_waveform(path: Path) -> np.ndarray:
    """Load WAV → mono float32 @ 16 kHz, pad/trim to 5 s."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr != TARGET_SR:
        ratio = TARGET_SR / sr
        n = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="wrap")
    else:
        audio = audio[:TARGET_SAMPLES]
    return audio


# ─── Dataset ─────────────────────────────────────────────────────────────────

class VibrationDataset:
    def __init__(self, clips_dir: Path, binary: bool = True):
        self.clips_dir = clips_dir
        self.binary = binary
        self.records: list[dict] = []  # {path, label, source_wav}
        self._load_records()

    def _load_records(self):
        metadata_csv = self.clips_dir / "metadata.csv"
        if metadata_csv.exists():
            df = pd.read_csv(metadata_csv)
            for _, row in df.iterrows():
                wav_path = self.clips_dir / row["filename"]
                if wav_path.exists():
                    self.records.append({
                        "path": wav_path,
                        "label": row["label"],
                        "source_wav": row.get("source_wav", "unknown"),
                    })
            logger.info(f"Loaded {len(self.records)} clips from metadata.csv")
        else:
            for class_dir in sorted(self.clips_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for w in class_dir.glob("*.wav"):
                    self.records.append({
                        "path": w, "label": class_dir.name, "source_wav": w.name,
                    })
            logger.info(f"Loaded {len(self.records)} clips from subdirectories")

        if not self.records:
            raise RuntimeError(
                f"No WAV clips found in {self.clips_dir}. Run augment_data.py first."
            )

    def build_label_encoder(self) -> dict[str, int]:
        if self.binary:
            return {"Leak": 0, "No_Leak": 1}
        return {l: i for i, l in enumerate(sorted({r["label"] for r in self.records}))}

    def normalise_label(self, raw: str) -> str:
        if self.binary:
            return "No_Leak" if raw in NO_FAULT_LABELS else "Leak"
        return raw


def recording_level_split(
    records: list[dict],
    val_frac: float,
    binary: bool,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split by unique source_wav — no augmented variant of the same source
    recording can appear in both train and val.

    Stratified: tries to keep the same Leak/No_Leak ratio in both splits.
    """
    from sklearn.model_selection import train_test_split

    # Group records by source_wav
    sources: dict[str, list[dict]] = {}
    for r in records:
        src = r["source_wav"]
        sources.setdefault(src, []).append(r)

    # Build a list of (source_wav, binary_label) for stratified splitting
    src_keys = sorted(sources.keys())
    src_labels = []
    for src in src_keys:
        first_label = sources[src][0]["label"]
        norm = "No_Leak" if first_label in NO_FAULT_LABELS else "Leak"
        src_labels.append(norm)

    # Stratified split at the recording level
    try:
        train_srcs, val_srcs = train_test_split(
            src_keys, test_size=val_frac, random_state=seed, stratify=src_labels
        )
    except ValueError:
        # Not enough samples per class for stratification → fall back to random
        logger.warning("Not enough samples per class for stratified split — using random split")
        train_srcs, val_srcs = train_test_split(src_keys, test_size=val_frac, random_state=seed)

    train_set_srcs = set(train_srcs)
    train_records = [r for r in records if r["source_wav"] in train_set_srcs]
    val_records = [r for r in records if r["source_wav"] not in train_set_srcs]

    logger.info(
        f"Recording-level split: "
        f"{len(train_srcs)} train recordings ({len(train_records)} clips) | "
        f"{len(val_srcs)} val recordings ({len(val_records)} clips)"
    )
    return train_records, val_records


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_loss(
    logits,        # (B, C) raw logits
    targets,       # (B,) int64 class indices
    gamma: float = 2.0,
    alpha: "torch.Tensor | None" = None,  # per-class weight
):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)

    gamma=2.0: standard setting.  Down-weights easy examples by (1-p)^2.
    For binary with p=0.9 (confident correct): weight = (0.1)^2 = 0.01 — 100x
    less influence than a hard example with p=0.5: weight = (0.5)^2 = 0.25.

    This prevents the 4:1 majority-class collapse seen with weighted CE alone.
    """
    import torch
    import torch.nn.functional as F

    log_proba = F.log_softmax(logits, dim=1)         # (B, C)
    proba = torch.exp(log_proba)                     # (B, C)

    # Gather p_t and log(p_t) for the target class
    log_pt = log_proba.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
    pt = proba.gather(1, targets.unsqueeze(1)).squeeze(1)            # (B,)

    focal_weight = (1.0 - pt) ** gamma
    loss = -focal_weight * log_pt

    if alpha is not None:
        alpha_t = alpha.gather(0, targets)
        loss = alpha_t * loss

    return loss.mean()


# ─── SpecAugment (mild) ───────────────────────────────────────────────────────

def spec_augment(spec, freq_mask: int = 6, time_mask: int = 15, rng=None):
    """
    Mild SpecAugment for vibration spectrograms.
    Smaller masks than speech (freq_mask=6 vs 27, time_mask=15 vs 100)
    to avoid masking out fault-specific spectral lines.
    """
    import torch
    if rng is None:
        rng = np.random.default_rng()
    spec = spec.clone()
    _, freq_bins, time_steps = spec.shape

    f = int(rng.integers(0, freq_mask + 1))
    if f > 0:
        f0 = int(rng.integers(0, max(freq_bins - f, 1)))
        spec[:, f0:f0 + f, :] = 0.0

    t = int(rng.integers(0, time_mask + 1))
    if t > 0:
        t0 = int(rng.integers(0, max(time_steps - t, 1)))
        spec[:, :, t0:t0 + t] = 0.0

    return spec


# ─── Training Loop ────────────────────────────────────────────────────────────

def build_model(n_classes: int, freq_bins: int):
    """Instantiate OmniCNN, patching n_mels to freq_bins for log-STFT input."""
    iep4_path = Path(__file__).parent.parent / "iep4"
    if str(iep4_path) not in sys.path:
        sys.path.insert(0, str(iep4_path))
    from app.model import _build_cnn
    return _build_cnn(n_classes=n_classes, n_mels=freq_bins)


def train_epoch(model, loader, optimizer, loss_fn, device, rng, use_augment):
    import torch
    model.train()
    total_loss, n = 0.0, 0
    for specs, targets in loader:
        specs, targets = specs.to(device), targets.to(device)
        if use_augment:
            specs = torch.stack([spec_augment(s, rng=rng) for s in specs])
        optimizer.zero_grad()
        logits = model(specs)
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / max(n, 1)


def eval_epoch(model, loader, loss_fn, device):
    import torch
    from sklearn.metrics import f1_score, classification_report
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for specs, targets in loader:
            specs, targets = specs.to(device), targets.to(device)
            logits = model(specs)
            loss = loss_fn(logits, targets)
            total_loss += loss.item() * len(targets)
            n += len(targets)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    return total_loss / max(n, 1), float(f1), all_targets, all_preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Omni-Sense IEP4 CNN")
    parser.add_argument("--clips-dir", default="data/synthesized")
    parser.add_argument("--output-dir", default="iep4/models")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience in epochs")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = standard weighted CE, 2 = standard focal)")
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--no-binary", dest="binary", action="store_false")
    parser.add_argument("--no-spec-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not found. Install with: pip install torch")
        return

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Spectrogram: log-linear STFT  ({FREQ_BINS} freq bins × {N_FRAMES} frames)")

    # ── Load dataset ──
    dataset = VibrationDataset(clips_dir, binary=args.binary)
    label_map = dataset.build_label_encoder()
    inv_label_map = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    logger.info(f"Classes ({n_classes}): {label_map}")

    # ── Recording-level split (NO clip-level leakage) ──
    train_records, val_records = recording_level_split(
        dataset.records, args.val_split, args.binary, args.seed
    )

    def records_to_tensors(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        specs, labels = [], []
        for i, r in enumerate(records):
            waveform = load_waveform(r["path"])
            spec = compute_log_stft(waveform)
            norm_label = dataset.normalise_label(r["label"])
            labels.append(label_map[norm_label])
            specs.append(spec)
            if (i + 1) % 200 == 0:
                logger.info(f"  {i + 1}/{len(records)} spectrograms")
        return np.array(specs, dtype=np.float32), np.array(labels, dtype=np.int64)

    logger.info("Computing train spectrograms...")
    X_train, y_train = records_to_tensors(train_records)
    logger.info("Computing val spectrograms...")
    X_val, y_val = records_to_tensors(val_records)
    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # Class distribution
    for split_name, y_arr in [("train", y_train), ("val", y_val)]:
        counts = np.bincount(y_arr, minlength=n_classes)
        dist = {inv_label_map[i]: int(c) for i, c in enumerate(counts)}
        logger.info(f"  {split_name} class counts: {dist}")

    # ── DataLoaders ──
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # ── Model ──
    model = build_model(n_classes, freq_bins=FREQ_BINS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"OmniCNN parameters: {n_params:,}  (input channels: 1×{FREQ_BINS}×{N_FRAMES})")

    # ── Focal Loss with per-class alpha (inverse frequency) ──
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    alpha = torch.tensor(1.0 / counts, dtype=torch.float32)
    alpha = (alpha / alpha.sum() * n_classes).to(device)
    logger.info(f"Focal loss alpha: { {inv_label_map[i]: float(a) for i, a in enumerate(alpha)} }")
    logger.info(f"Focal loss gamma: {args.focal_gamma}")

    gamma = args.focal_gamma

    def loss_fn(logits, targets):
        return focal_loss(logits, targets, gamma=gamma, alpha=alpha)

    # ── Optimiser + LR schedule ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ──
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"OmniCNN Training — {n_classes} classes | {len(train_records)} train clips | {args.epochs} max epochs")
    print(f"Loss: Focal (gamma={gamma}) | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"Split: recording-level (no clip-level leakage)")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, rng,
            use_augment=not args.no_spec_augment,
        )
        val_loss, val_f1, val_true, val_pred = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        is_best = val_f1 > best_val_f1
        print(
            f"  Epoch {epoch:>3}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  f1={val_f1:.4f}"
            + (" ★" if is_best else "")
        )

        if is_best:
            best_val_f1 = val_f1
            from sklearn.metrics import classification_report
            print(classification_report(val_true, val_pred,
                                        target_names=[inv_label_map[i] for i in range(n_classes)],
                                        zero_division=0))
            best_state = {
                "model":  {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "labels": {str(i): lbl for i, lbl in inv_label_map.items()},
                "val_f1": val_f1,
                "epoch":  epoch,
                "freq_bins": FREQ_BINS,
                "n_frames":  N_FRAMES,
                "spectrogram": "log_linear_stft",
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    print(f"\n  Best val F1: {best_val_f1:.4f}")

    if best_state is None:
        logger.error("No model state saved.")
        return

    # ── Save .pt ──
    pt_path = output_dir / "cnn_classifier.pt"
    torch.save(best_state, str(pt_path))
    logger.info(f"PyTorch model saved: {pt_path}")

    # ── Save label_map.json ──
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({str(i): lbl for i, lbl in inv_label_map.items()}, f, indent=2)
    logger.info(f"Label map saved: {label_map_path}")

    # ── ONNX export ──
    logger.info("Exporting to ONNX...")
    try:
        iep4_path = Path(__file__).parent.parent / "iep4"
        if str(iep4_path) not in sys.path:
            sys.path.insert(0, str(iep4_path))
        from app.model import export_to_onnx

        # Reload best weights before export
        model.load_state_dict(best_state["model"])
        model.eval()

        onnx_path = output_dir / "cnn_classifier.onnx"
        export_to_onnx(pt_path, onnx_path, n_classes=n_classes)
        logger.info(f"ONNX model saved: {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e} — .pt file still usable by IEP4.")

    print(f"\n{'='*60}")
    print(f"Done!  Best val F1: {best_val_f1:.4f}")
    print(f"  Model:     {pt_path}")
    print(f"  Label map: {label_map_path}")
    print(f"\n  Restart IEP4 to pick up the new model:")
    print(f"  docker compose restart iep4")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
