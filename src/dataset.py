"""
Dataset and data-augmentation classes for BirdCLEF 2026.

Classes
-------
BirdCLEFDataset
    Main dataset that loads audio files and returns mel-spectrogram tensors.
MixupDataset
    Wrapper dataset that applies Mixup augmentation on the fly.

Functions
---------
get_transforms(mode, cfg)
    Returns a torchvision transforms pipeline appropriate for the given mode.
"""

import ast
import logging
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import audio_to_melspec, load_audio

logger = logging.getLogger(__name__)

# ImageNet statistics used by most timm pre-trained models.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(mode: str, cfg) -> transforms.Compose:
    """Build a torchvision transforms pipeline.

    For *train* mode a random horizontal flip is included before
    normalisation.  Both modes normalise with ImageNet statistics so that
    pre-trained weights are used correctly.

    Note: SpecAugment is applied inside ``BirdCLEFDataset.__getitem__``
    as a tensor operation *before* these transforms are applied, so the
    pipeline here only handles tensor-level augmentation and normalisation.

    Parameters
    ----------
    mode:
        Either ``'train'`` or ``'val'``/``'test'``.
    cfg:
        Config object (unused here but kept for API consistency).

    Returns
    -------
    transforms.Compose
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                normalize,
            ]
        )
    else:
        return transforms.Compose([normalize])


# ---------------------------------------------------------------------------
# SpecAugment helpers
# ---------------------------------------------------------------------------

def _apply_spec_augment(spec: torch.Tensor, cfg) -> torch.Tensor:
    """Apply SpecAugment frequency and time masking in-place.

    Parameters
    ----------
    spec:
        Tensor of shape ``(1, n_mels, time_steps)`` with values in ``[0, 1]``.
    cfg:
        Config with ``freq_mask_param``, ``time_mask_param``,
        ``num_freq_masks``, ``num_time_masks``.

    Returns
    -------
    torch.Tensor
        Augmented spectrogram of the same shape.
    """
    _, n_mels, time_steps = spec.shape

    # Frequency masking
    for _ in range(cfg.num_freq_masks):
        f = random.randint(0, cfg.freq_mask_param)
        f0 = random.randint(0, max(0, n_mels - f))
        spec[:, f0 : f0 + f, :] = 0.0

    # Time masking
    for _ in range(cfg.num_time_masks):
        t = random.randint(0, cfg.time_mask_param)
        t0 = random.randint(0, max(0, time_steps - t))
        spec[:, :, t0 : t0 + t] = 0.0

    return spec


# ---------------------------------------------------------------------------
# Pink noise augmentation
# ---------------------------------------------------------------------------

def _add_pink_noise(
    waveform: np.ndarray,
    snr_db_min: float,
    snr_db_max: float,
) -> np.ndarray:
    """Add pink (1/f) noise to a waveform at a random SNR.

    Pink noise closely matches the spectral profile of natural environmental
    sounds (wind, rain, water), making it a more realistic augmentation than
    white noise for wildlife recordings.

    Parameters
    ----------
    waveform:
        Float32 waveform array.
    snr_db_min:
        Minimum SNR in dB (higher = less noise).
    snr_db_max:
        Maximum SNR in dB.

    Returns
    -------
    np.ndarray
        Waveform with pink noise added, clipped to [-1, 1].
    """
    n = len(waveform)
    # Generate white noise then shape to 1/f via frequency-domain filtering.
    white = np.random.randn(n).astype(np.float32)

    # Build 1/sqrt(f) filter (pink noise has 1/f power spectral density).
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division-by-zero at DC
    pink_filter = 1.0 / np.sqrt(freqs)
    pink_filter[0] = 0.0  # zero out DC component

    pink = np.fft.irfft(np.fft.rfft(white) * pink_filter, n=n).astype(np.float32)

    # Scale noise to achieve the desired SNR.
    signal_rms = np.sqrt(np.mean(waveform ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(pink ** 2)) + 1e-9
    target_snr = np.random.uniform(snr_db_min, snr_db_max)
    noise_scale = signal_rms / (noise_rms * (10 ** (target_snr / 20.0)))

    return np.clip(waveform + noise_scale * pink, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Secondary label helpers
# ---------------------------------------------------------------------------

def _parse_secondary_labels(raw: str) -> list:
    """Parse the ``secondary_labels`` column from BirdCLEF metadata.

    The column stores values like ``"['earspa1', 'houspa']"`` or ``"[]"`` —
    a Python list serialised to a string.  ``ast.literal_eval`` is used for
    safe parsing.

    Parameters
    ----------
    raw:
        Raw string value from the DataFrame cell.

    Returns
    -------
    list[str]
        List of species codes, possibly empty.
    """
    raw = raw.strip()
    if not raw or raw in ("[]", "nan", ""):
        return []
    try:
        result = ast.literal_eval(raw)
        return [s.strip() for s in result if isinstance(s, str) and s.strip()]
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class BirdCLEFDataset(Dataset):
    """PyTorch Dataset for BirdCLEF 2026 audio clips.

    Each item is a ``(image_tensor, label_vector)`` pair where:

    * ``image_tensor`` — float32 tensor of shape ``(3, n_mels, time_steps)``
      (three identical channels so ImageNet-pretrained backbones work).
    * ``label_vector`` — float32 tensor of shape ``(num_classes,)``
      with ``1.0`` at the index of ``primary_label`` and
      ``secondary_label_weight`` at any secondary label indices
      (training only; validation uses hard primary labels only).

    Parameters
    ----------
    df:
        DataFrame with at least ``filename`` (relative to
        ``cfg.train_audio_dir``) and ``primary_label`` columns.
        During inference ``primary_label`` may be absent.
    cfg:
        Config object.
    label_encoder:
        Fitted ``sklearn.preprocessing.LabelEncoder``.
    mode:
        ``'train'`` enables random segment extraction, SpecAugment, pink
        noise, and secondary labels; ``'val'``/``'test'`` starts from
        offset 0 and uses only the primary label.
    """

    def __init__(self, df, cfg, label_encoder, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.label_encoder = label_encoder
        self.mode = mode
        self.num_classes = len(label_encoder.classes_)
        self.transforms = get_transforms(mode, cfg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # ------------------------------------------------------------------
        # Build the audio file path.
        # ------------------------------------------------------------------
        filename = row["filename"]
        audio_path = os.path.join(self.cfg.train_audio_dir, filename)

        # ------------------------------------------------------------------
        # Load audio.
        # ------------------------------------------------------------------
        # Use the row's 'offset' column if present (for pseudo-labelled data),
        # otherwise use random offset for train and fixed 0 for val/test.
        if "offset" in row.index and not _is_missing(row["offset"]):
            offset = float(row["offset"])
        else:
            offset = None if self.mode == "train" else 0.0

        waveform = load_audio(
            path=audio_path,
            sr=self.cfg.sample_rate,
            duration=self.cfg.duration,
            offset=offset,
        )

        # ------------------------------------------------------------------
        # Pink noise augmentation (training only).
        # ------------------------------------------------------------------
        if self.mode == "train":
            noise_prob = getattr(self.cfg, "pink_noise_prob", 0.0)
            if noise_prob > 0.0 and random.random() < noise_prob:
                waveform = _add_pink_noise(
                    waveform,
                    snr_db_min=getattr(self.cfg, "pink_noise_snr_db_min", 5.0),
                    snr_db_max=getattr(self.cfg, "pink_noise_snr_db_max", 20.0),
                )

        # ------------------------------------------------------------------
        # Convert to mel-spectrogram.
        # ------------------------------------------------------------------
        melspec = audio_to_melspec(waveform, self.cfg.sample_rate, self.cfg)
        # shape: (n_mels, time_steps)

        # Convert to tensor and add channel dim -> (1, n_mels, time_steps)
        spec_tensor = torch.from_numpy(melspec).unsqueeze(0)

        # ------------------------------------------------------------------
        # SpecAugment (training only).
        # ------------------------------------------------------------------
        if self.mode == "train":
            spec_tensor = _apply_spec_augment(spec_tensor, self.cfg)

        # ------------------------------------------------------------------
        # Replicate channel 3x for ImageNet pre-trained models.
        # shape: (3, n_mels, time_steps)
        # ------------------------------------------------------------------
        spec_tensor = spec_tensor.repeat(3, 1, 1)

        # ------------------------------------------------------------------
        # Apply torchvision transforms (normalisation, etc.).
        # ------------------------------------------------------------------
        spec_tensor = self.transforms(spec_tensor)

        # ------------------------------------------------------------------
        # Build label vector.
        # ------------------------------------------------------------------
        label_vector = torch.zeros(self.num_classes, dtype=torch.float32)

        # Primary label — hard positive (weight = 1.0).
        if "primary_label" in row.index and not _is_missing(row["primary_label"]):
            try:
                label_idx = self.label_encoder.transform([row["primary_label"]])[0]
                label_vector[label_idx] = 1.0
            except ValueError:
                logger.warning(
                    "Unknown primary label '%s' at index %d — skipping.",
                    row["primary_label"],
                    idx,
                )

        # Secondary labels — soft positives (training only).
        # Validation keeps hard primary labels to align with pcmAP evaluation.
        if self.mode == "train":
            secondary_weight = getattr(self.cfg, "secondary_label_weight", 0.0)
            if secondary_weight > 0.0 and "secondary_labels" in row.index:
                sec_raw = row["secondary_labels"]
                if not _is_missing(sec_raw):
                    sec_labels = _parse_secondary_labels(str(sec_raw))
                    for sl in sec_labels:
                        try:
                            sec_idx = self.label_encoder.transform([sl])[0]
                            # Don't overwrite a primary label's higher weight.
                            if label_vector[sec_idx] < secondary_weight:
                                label_vector[sec_idx] = secondary_weight
                        except ValueError:
                            pass  # species not in training set — skip silently

        return spec_tensor, label_vector


def _is_missing(value) -> bool:
    """Return True if *value* is NaN or an empty string."""
    if value is None:
        return True
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return str(value).strip() == ""


# ---------------------------------------------------------------------------
# Mixup dataset wrapper
# ---------------------------------------------------------------------------

class MixupDataset(Dataset):
    """Wrapper dataset that applies Mixup augmentation.

    Each call to ``__getitem__`` fetches a sample from the underlying
    *dataset* and mixes it with a second randomly chosen sample using a
    Beta(alpha, alpha) mixing coefficient.

    Parameters
    ----------
    dataset:
        Any ``torch.utils.data.Dataset`` whose ``__getitem__`` returns
        ``(image_tensor, label_vector)``.
    alpha:
        Beta distribution parameter.  Set to ``0`` to disable mixing.
    """

    def __init__(self, dataset: Dataset, alpha: float = 0.4):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, y1 = self.dataset[idx]

        if self.alpha <= 0.0:
            return x1, y1

        # Choose a second sample uniformly at random.
        idx2 = random.randint(0, len(self.dataset) - 1)
        x2, y2 = self.dataset[idx2]

        # Sample mixing coefficient from Beta distribution.
        lam = float(np.random.beta(self.alpha, self.alpha))
        lam = max(lam, 1.0 - lam)  # keep the dominant sample dominant

        x_mix = lam * x1 + (1.0 - lam) * x2
        y_mix = lam * y1 + (1.0 - lam) * y2

        return x_mix, y_mix


# ---------------------------------------------------------------------------
# Combined dataset (real + pseudo-labelled data)
# ---------------------------------------------------------------------------

class CombinedDataset(Dataset):
    """Combines a real dataset and a pseudo-labelled dataset with controlled mixing.

    At each ``__getitem__`` call, samples from the pseudo-labelled dataset
    with probability *pseudo_ratio* and from the real dataset otherwise.
    The effective dataset length equals that of the real dataset so that
    epoch semantics remain consistent.

    Parameters
    ----------
    real_ds:
        Dataset of real (human-annotated) samples.
    pseudo_ds:
        Dataset of pseudo-labelled samples (may be empty).
    pseudo_ratio:
        Fraction of each batch drawn from pseudo-labelled data.
    """

    def __init__(self, real_ds: Dataset, pseudo_ds: Dataset, pseudo_ratio: float = 0.3):
        self.real_ds = real_ds
        self.pseudo_ds = pseudo_ds
        self.pseudo_ratio = pseudo_ratio

    def __len__(self) -> int:
        return len(self.real_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pseudo_ratio > 0.0 and len(self.pseudo_ds) > 0:
            if random.random() < self.pseudo_ratio:
                pseudo_idx = random.randint(0, len(self.pseudo_ds) - 1)
                return self.pseudo_ds[pseudo_idx]
        return self.real_ds[idx]
