"""
Pseudo-labelling pipeline for BirdCLEF 2026.

Usage
-----
    python src/pseudo_label.py --config configs/improved.yaml \\
        --soundscapes_dir /kaggle/input/birdclef-2026/unlabeled_soundscapes \\
        --output_csv     /kaggle/working/pseudo_labels.csv

Overview
--------
Pseudo-labelling uses trained fold models to generate predictions on
unlabeled soundscapes, then keeps only high-confidence windows as
additional training examples for a subsequent training round.

Pipeline
~~~~~~~~
1. Load all fold checkpoints from ``cfg.output_dir``.
2. Load the label encoder saved during training.
3. Slide 5-second windows over every OGG in ``soundscapes_dir``.
4. For each window, run ensemble inference and record the max probability
   and predicted primary label.
5. Keep windows whose ``max(probs) > confidence_threshold``.
6. Cap each class at ``per_class_cap`` examples to prevent dominant
   soundscape species from overwhelming rare ones.
7. Write a CSV whose columns mirror ``train_metadata.csv`` so that
   ``BirdCLEFDataset`` can load it without modification.

Round strategy
~~~~~~~~~~~~~~
- Round 1: train on real data → generate pseudo-labels (threshold ≥ 0.5)
- Round 2: train on real + pseudo → generate new pseudo-labels (threshold ≥ 0.5)
- Round 3: final model (improvements are marginal after round 3)
"""

import argparse
import collections
import logging
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Allow running as a script from the repo root or from inside src/.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import load_config
from src.inference import load_fold_models, predict_window, preprocess_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only windows where max(probs) exceeds this threshold are kept.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Secondary-label threshold: species with prob > this (excluding primary)
# are added to the secondary_labels list of the pseudo-labelled row.
SECONDARY_THRESHOLD = 0.3

# Maximum pseudo-label examples per class (prevents dominant classes from
# drowning rare ones).  Set to None to disable.
DEFAULT_PER_CLASS_CAP = None  # set at runtime based on real data statistics


# ---------------------------------------------------------------------------
# Main pseudo-labelling routine
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    cfg,
    soundscapes_dir: str,
    output_csv: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    per_class_cap: int | None = None,
) -> pd.DataFrame:
    """Generate pseudo-labels for unlabelled soundscapes.

    Parameters
    ----------
    cfg:
        Config namespace (same as used for training).
    soundscapes_dir:
        Directory containing unlabelled OGG soundscape files.
    output_csv:
        Path to write the pseudo-label CSV.
    confidence_threshold:
        Minimum ``max(probs)`` for a window to be included.
    per_class_cap:
        Maximum number of pseudo-labelled examples per class.
        If None, no cap is applied.

    Returns
    -------
    pd.DataFrame
        Pseudo-label DataFrame written to *output_csv*.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load label encoder.
    # ------------------------------------------------------------------
    le_path = os.path.join(cfg.output_dir, "label_encoder.pkl")
    if not os.path.exists(le_path):
        logger.error("LabelEncoder not found at %s. Run train.py first.", le_path)
        sys.exit(1)

    with open(le_path, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    logger.info("Loaded LabelEncoder with %d classes.", num_classes)

    # ------------------------------------------------------------------
    # Load fold models.
    # ------------------------------------------------------------------
    models = load_fold_models(cfg, num_classes, device)

    # ------------------------------------------------------------------
    # Discover unlabelled soundscape files.
    # ------------------------------------------------------------------
    if not os.path.isdir(soundscapes_dir):
        logger.error("Soundscapes directory not found: %s", soundscapes_dir)
        sys.exit(1)

    ogg_files = sorted(
        f for f in os.listdir(soundscapes_dir) if f.lower().endswith(".ogg")
    )
    logger.info("Found %d OGG files in %s.", len(ogg_files), soundscapes_dir)

    if not ogg_files:
        logger.error("No OGG files found.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Sliding window inference.
    # ------------------------------------------------------------------
    window_samples = int(cfg.sample_rate * cfg.duration)
    # No overlap during pseudo-labelling (speed > marginal quality gain).
    step_samples = window_samples

    # class_name -> list of (filename, offset, probs_vector)
    candidates: dict = collections.defaultdict(list)
    total_windows = 0
    kept_windows = 0

    for fname in tqdm(ogg_files, desc="Pseudo-label files"):
        file_path = os.path.join(soundscapes_dir, fname)
        # Store the absolute path so BirdCLEFDataset can locate the file
        # regardless of which audio_dir is configured.
        abs_path = os.path.abspath(file_path)

        try:
            import librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_waveform, _ = librosa.load(
                    file_path, sr=cfg.sample_rate, mono=True
                )
        except Exception as exc:
            logger.warning("Failed to load %s (%s), skipping.", fname, exc)
            continue

        total_samples = len(full_waveform)
        if total_samples == 0:
            continue

        start = 0
        while start < total_samples:
            window = full_waveform[start : start + window_samples]
            if len(window) < window_samples:
                window = np.pad(
                    window, (0, window_samples - len(window)), mode="constant"
                )

            offset_seconds = start / cfg.sample_rate
            total_windows += 1

            tensor = preprocess_window(window, cfg.sample_rate, cfg, device)
            probs = predict_window(models, tensor, device, cfg.fp16, use_tta=False)

            max_prob = float(probs.max())
            if max_prob < confidence_threshold:
                start += step_samples
                continue

            # Primary label: species with highest probability.
            primary_idx = int(probs.argmax())
            primary_label = le.classes_[primary_idx]

            candidates[primary_label].append(
                {
                    "filename": abs_path,
                    "offset": offset_seconds,
                    "primary_label": primary_label,
                    "probs": probs,
                    "confidence": max_prob,
                }
            )
            kept_windows += 1
            start += step_samples

    logger.info(
        "Windows: total=%d, kept (confidence >= %.2f)=%d (%.1f%%)",
        total_windows,
        confidence_threshold,
        kept_windows,
        100.0 * kept_windows / max(total_windows, 1),
    )

    # ------------------------------------------------------------------
    # Apply per-class cap.
    # ------------------------------------------------------------------
    rows = []
    for class_name, items in candidates.items():
        # Sort by confidence descending, keep top-K.
        items_sorted = sorted(items, key=lambda x: x["confidence"], reverse=True)
        if per_class_cap is not None:
            items_sorted = items_sorted[:per_class_cap]

        for item in items_sorted:
            probs = item["probs"]
            primary_idx = int(probs.argmax())

            # Secondary labels: all species with prob > SECONDARY_THRESHOLD
            # (excluding the primary species itself).
            secondary = [
                le.classes_[i]
                for i in range(num_classes)
                if i != primary_idx and probs[i] > SECONDARY_THRESHOLD
            ]

            rows.append(
                {
                    "filename": item["filename"],
                    "offset": item["offset"],
                    "primary_label": item["primary_label"],
                    "secondary_labels": str(secondary),
                    "pseudo_confidence": item["confidence"],
                    "source": "pseudo",
                }
            )

    if not rows:
        logger.warning("No pseudo-labelled examples generated. Try lowering --threshold.")
        return pd.DataFrame()

    pseudo_df = pd.DataFrame(rows)
    pseudo_df.to_csv(output_csv, index=False)

    class_counts = pseudo_df["primary_label"].value_counts()
    logger.info(
        "Pseudo-label summary: %d rows, %d classes. "
        "Top-5 classes: %s",
        len(pseudo_df),
        len(class_counts),
        class_counts.head(5).to_dict(),
    )
    logger.info("Pseudo-labels saved to: %s", output_csv)
    return pseudo_df


# ---------------------------------------------------------------------------
# Per-class cap helper
# ---------------------------------------------------------------------------

def compute_per_class_cap(
    real_metadata_csv: str,
    multiplier: float = 5.0,
) -> int:
    """Compute a per-class cap based on the real training data distribution.

    The cap is set to ``multiplier × median(class_count)`` so that the
    pseudo-labelled data for any single class cannot exceed 5× the typical
    real-data class size.

    Parameters
    ----------
    real_metadata_csv:
        Path to the real ``train_metadata.csv``.
    multiplier:
        Cap multiplier relative to the median class count.

    Returns
    -------
    int
        Per-class cap value.
    """
    df = pd.read_csv(real_metadata_csv)
    counts = df["primary_label"].value_counts()
    cap = int(counts.median() * multiplier)
    logger.info(
        "Per-class cap: median_count=%d × %.1f = %d",
        int(counts.median()),
        multiplier,
        cap,
    )
    return cap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BirdCLEF 2026 pseudo-label generation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/improved.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--soundscapes_dir",
        type=str,
        required=True,
        help="Directory containing unlabelled OGG soundscape files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to {output_dir}/pseudo_labels.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--auto_cap",
        action="store_true",
        help="Automatically compute per-class cap from real training data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)
    cfg = load_config(config_path)

    output_csv = args.output_csv or os.path.join(cfg.output_dir, "pseudo_labels.csv")

    per_class_cap = None
    if args.auto_cap:
        per_class_cap = compute_per_class_cap(cfg.train_metadata)

    generate_pseudo_labels(
        cfg=cfg,
        soundscapes_dir=args.soundscapes_dir,
        output_csv=output_csv,
        confidence_threshold=args.threshold,
        per_class_cap=per_class_cap,
    )
