"""
Utility functions for BirdCLEF 2026.

Covers audio loading, mel-spectrogram conversion,
the Padded cmAP metric, and seed initialization.
"""

import logging
import os
import random
import warnings

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int, duration: float, offset: float | None = None) -> np.ndarray:
    """Load an OGG (or any librosa-supported) audio file.

    Parameters
    ----------
    path:
        Path to the audio file.
    sr:
        Target sample rate in Hz.
    duration:
        Length of the clip to return, in seconds.
    offset:
        Start position in seconds.  If *None* and the file is long enough, a
        random offset is chosen; otherwise the clip starts at 0.

    Returns
    -------
    np.ndarray
        Float32 waveform of exactly ``sr * duration`` samples.  Short files
        are zero-padded on the right.
    """
    target_samples = int(sr * duration)

    if not os.path.exists(path):
        logger.warning("Audio file not found, returning silence: %s", path)
        return np.zeros(target_samples, dtype=np.float32)

    try:
        # Read the full file first to know its length.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, file_sr = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:
        logger.warning("Failed to load %s (%s), returning silence.", path, exc)
        return np.zeros(target_samples, dtype=np.float32)

    total_samples = len(waveform)

    if total_samples == 0:
        logger.warning("Empty audio file: %s", path)
        return np.zeros(target_samples, dtype=np.float32)

    # Determine start sample.
    if offset is not None:
        start = int(offset * sr)
        start = max(0, min(start, max(0, total_samples - 1)))
    elif total_samples > target_samples:
        start = random.randint(0, total_samples - target_samples)
    else:
        start = 0

    clip = waveform[start : start + target_samples]

    # Zero-pad if the clip is shorter than requested.
    if len(clip) < target_samples:
        pad = target_samples - len(clip)
        clip = np.pad(clip, (0, pad), mode="constant")

    return clip.astype(np.float32)


# ---------------------------------------------------------------------------
# Mel-spectrogram
# ---------------------------------------------------------------------------

def audio_to_melspec(waveform: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Convert a waveform to a normalised mel-spectrogram image.

    Parameters
    ----------
    waveform:
        1-D float32 numpy array.
    sr:
        Sample rate of *waveform*.
    cfg:
        Config object/namespace with attributes ``n_mels``, ``n_fft``,
        ``hop_length``, ``fmin``, ``fmax``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(n_mels, time_steps)`` with values in
        ``[0, 1]``.
    """
    melspec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )

    # Convert to dB scale.
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Normalise to [0, 1].
    melspec_min = melspec_db.min()
    melspec_max = melspec_db.max()
    if melspec_max - melspec_min > 1e-6:
        melspec_norm = (melspec_db - melspec_min) / (melspec_max - melspec_min)
    else:
        melspec_norm = np.zeros_like(melspec_db)

    return melspec_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Padded cmAP metric
# ---------------------------------------------------------------------------

def padded_cmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    padding_factor: int = 5,
) -> float:
    """Compute the Padded Class-wise mean Average Precision (cmAP).

    For each class *c*:
    1. Sort samples by predicted score for *c* in descending order.
    2. Prepend ``padding_factor`` synthetic true-positive rows.
    3. Compute Average Precision on the padded sequence.

    The final score is the mean AP across all classes.

    Parameters
    ----------
    y_true:
        Binary ground-truth array of shape ``(n_samples, n_classes)``.
    y_pred:
        Predicted probability array of shape ``(n_samples, n_classes)``.
    padding_factor:
        Number of synthetic true positives prepended per class.

    Returns
    -------
    float
        Mean padded AP across all classes.
    """
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    )

    n_classes = y_true.shape[1]
    ap_scores = []

    for c in range(n_classes):
        true_c = y_true[:, c]
        pred_c = y_pred[:, c]

        # Sort by descending predicted score.
        sort_idx = np.argsort(pred_c)[::-1]
        true_sorted = true_c[sort_idx]

        # Prepend padding_factor synthetic true positives.
        padded_true = np.concatenate(
            [np.ones(padding_factor, dtype=np.float32), true_sorted]
        )

        # Compute AP from cumulative precision/recall.
        n_total = len(padded_true)
        tp_cumsum = np.cumsum(padded_true)
        precision = tp_cumsum / (np.arange(n_total) + 1)
        recall_diff = padded_true  # delta recall at each position

        ap = np.sum(precision * recall_diff) / (tp_cumsum[-1] + 1e-10)
        ap_scores.append(float(ap))

    return float(np.mean(ap_scores))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug("Random seed set to %d.", seed)
