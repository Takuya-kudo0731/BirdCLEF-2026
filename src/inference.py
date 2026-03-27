"""
Inference script for BirdCLEF 2026 baseline.

Usage
-----
    python src/inference.py --config configs/baseline.yaml
    python src/inference.py --config configs/improved.yaml

For every OGG file in ``cfg.test_soundscapes_dir`` the script:
1. Loads the full audio.
2. Slides a 5-second window (with optional overlap) across the recording.
3. Runs each window through every saved fold model and averages the
   sigmoid probabilities.
4. Multiple overlapping windows that fall in the same 5-second bucket are
   averaged together before writing the submission row.
5. Writes one submission row per 5-second bucket with
   ``row_id = {filename_stem}_{start_seconds}``.

The final DataFrame is saved to ``{output_dir}/submission.csv``.
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
from torch.cuda.amp import autocast
from tqdm import tqdm

# Allow running as a script from the repo root or from inside src/.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import load_config
from src.model import BirdCLEFModel
from src.utils import audio_to_melspec, load_audio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ImageNet normalisation constants (must match those used during training).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_fold_models(cfg, num_classes: int, device: torch.device) -> list:
    """Load all fold checkpoints from ``cfg.output_dir``."""
    models = []
    for fold in range(cfg.n_folds):
        ckpt_path = os.path.join(cfg.output_dir, f"fold{fold}_best.pth")
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found, skipping fold %d: %s", fold, ckpt_path)
            continue

        model = BirdCLEFModel(cfg, num_classes=num_classes).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
        logger.info("Loaded fold %d from %s", fold, ckpt_path)

    if not models:
        logger.error(
            "No fold checkpoints found in '%s'. Run train.py first.", cfg.output_dir
        )
        sys.exit(1)

    return models


# ---------------------------------------------------------------------------
# Window inference
# ---------------------------------------------------------------------------

def preprocess_window(
    waveform: np.ndarray,
    sr: int,
    cfg,
    device: torch.device,
) -> torch.Tensor:
    """Convert a waveform window to a normalised (3, H, W) tensor on *device*."""
    melspec = audio_to_melspec(waveform, sr, cfg)
    # (n_mels, time_steps) -> (1, n_mels, time_steps) -> (3, n_mels, time_steps)
    tensor = torch.from_numpy(melspec).unsqueeze(0).repeat(3, 1, 1)
    # ImageNet normalisation.
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    tensor = tensor.to(device)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)  # add batch dim -> (1, 3, H, W)


@torch.no_grad()
def predict_window(
    models: list,
    tensor: torch.Tensor,
    device: torch.device,
    fp16: bool,
    use_tta: bool = False,
) -> np.ndarray:
    """Run *tensor* through all models and return the mean probability vector.

    Parameters
    ----------
    models:
        List of loaded ``BirdCLEFModel`` instances.
    tensor:
        Input tensor of shape ``(1, 3, H, W)``.
    device:
        Torch device.
    fp16:
        Whether to use FP16 autocast (CUDA only).
    use_tta:
        If True, also run a horizontally flipped copy and average both.
        Horizontal flip = time reversal of the spectrogram.

    Returns
    -------
    np.ndarray
        Mean probability vector of shape ``(num_classes,)``.
    """
    preds = []
    for model in models:
        if fp16 and device.type == "cuda":
            with autocast():
                probs = model.predict(tensor)
        else:
            probs = model.predict(tensor)
        preds.append(probs.cpu().numpy())

        if use_tta:
            tensor_flip = torch.flip(tensor, dims=[-1])  # time reversal
            if fp16 and device.type == "cuda":
                with autocast():
                    probs_flip = model.predict(tensor_flip)
            else:
                probs_flip = model.predict(tensor_flip)
            preds.append(probs_flip.cpu().numpy())

    return np.mean(preds, axis=0).squeeze(0)  # shape: (num_classes,)


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def run_inference(cfg) -> None:
    """Run full inference pipeline and write submission.csv."""
    os.makedirs(cfg.output_dir, exist_ok=True)

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
    # Load sample submission to get expected columns and row_ids.
    # ------------------------------------------------------------------
    if not os.path.exists(cfg.sample_submission):
        logger.error("Sample submission not found: %s", cfg.sample_submission)
        sys.exit(1)

    sample_sub = pd.read_csv(cfg.sample_submission)
    # The species columns are everything except 'row_id'.
    species_cols = [c for c in sample_sub.columns if c != "row_id"]
    logger.info(
        "Sample submission has %d rows, %d species columns.",
        len(sample_sub),
        len(species_cols),
    )

    # Map species name -> column index in our label encoder.
    # Species not in the encoder get a constant probability of 0.
    species_to_le_idx: dict = {}
    for sp in species_cols:
        try:
            idx = int(le.transform([sp])[0])
            species_to_le_idx[sp] = idx
        except ValueError:
            pass  # species not in training data
    logger.info(
        "%d / %d submission species are in the training label encoder.",
        len(species_to_le_idx),
        len(species_cols),
    )

    # ------------------------------------------------------------------
    # Load fold models.
    # ------------------------------------------------------------------
    models = load_fold_models(cfg, num_classes, device)

    # ------------------------------------------------------------------
    # Discover test soundscape files.
    # ------------------------------------------------------------------
    if not os.path.isdir(cfg.test_soundscapes_dir):
        logger.error(
            "Test soundscapes directory not found: %s", cfg.test_soundscapes_dir
        )
        sys.exit(1)

    test_files = sorted(
        f
        for f in os.listdir(cfg.test_soundscapes_dir)
        if f.lower().endswith(".ogg")
    )
    logger.info("Found %d test soundscape files.", len(test_files))

    if not test_files:
        logger.error("No OGG files found in %s.", cfg.test_soundscapes_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Sliding window parameters.
    # ------------------------------------------------------------------
    window_samples = int(cfg.sample_rate * cfg.duration)
    overlap_seconds = getattr(cfg, "inference_overlap", 0.0)
    overlap_samples = int(cfg.sample_rate * overlap_seconds)
    step_samples = max(1, window_samples - overlap_samples)
    use_tta = getattr(cfg, "inference_tta", False)

    logger.info(
        "Sliding window: duration=%ds, overlap=%.1fs, step=%ds, TTA=%s",
        cfg.duration,
        overlap_seconds,
        step_samples // cfg.sample_rate,
        use_tta,
    )

    # ------------------------------------------------------------------
    # Inference loop.
    # ------------------------------------------------------------------
    # With overlapping windows, multiple windows can map to the same 5-second
    # bucket.  We accumulate predictions per bucket and average at the end.
    # bucket key: "{file_stem}_{bucket_start_seconds}"
    bucket_preds: dict = collections.defaultdict(list)

    for fname in tqdm(test_files, desc="Test files"):
        file_path = os.path.join(cfg.test_soundscapes_dir, fname)
        file_stem = os.path.splitext(fname)[0]

        # Load the full audio file (no duration cap).
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
            logger.warning("Empty audio file: %s, skipping.", fname)
            continue

        # Slide windows with the configured step size.
        start = 0
        while start < total_samples:
            end = start + window_samples
            window = full_waveform[start:end]

            # Zero-pad the last window if it is shorter than 5 seconds.
            if len(window) < window_samples:
                window = np.pad(
                    window, (0, window_samples - len(window)), mode="constant"
                )

            # Map this window to its canonical 5-second bucket.
            # All windows whose start falls in [k*duration, (k+1)*duration)
            # contribute to the same bucket.
            bucket_idx = start // window_samples
            bucket_start_seconds = bucket_idx * cfg.duration
            row_id = f"{file_stem}_{bucket_start_seconds}"

            # Build input tensor and run models.
            tensor = preprocess_window(window, cfg.sample_rate, cfg, device)
            probs = predict_window(models, tensor, device, cfg.fp16, use_tta)

            bucket_preds[row_id].append(probs)

            start += step_samples

    # ------------------------------------------------------------------
    # Average predictions within each bucket and build submission rows.
    # ------------------------------------------------------------------
    all_rows: list = []
    for row_id, pred_list in bucket_preds.items():
        # Average all overlapping-window predictions for this bucket.
        avg_probs = np.mean(pred_list, axis=0)

        row: dict = {"row_id": row_id}
        for sp in species_cols:
            if sp in species_to_le_idx:
                row[sp] = float(avg_probs[species_to_le_idx[sp]])
            else:
                row[sp] = 0.0
        all_rows.append(row)

    # ------------------------------------------------------------------
    # Build and save submission DataFrame.
    # ------------------------------------------------------------------
    if not all_rows:
        logger.error("No predictions were generated. Check your test data.")
        sys.exit(1)

    submission = pd.DataFrame(all_rows, columns=["row_id"] + species_cols)

    # Fill any row_ids from the sample submission that we did not predict.
    sample_row_ids = set(sample_sub["row_id"].tolist())
    pred_row_ids = set(submission["row_id"].tolist())
    missing = sample_row_ids - pred_row_ids
    if missing:
        logger.warning(
            "%d row_ids from sample submission were not predicted; filling with 0.",
            len(missing),
        )
        missing_df = pd.DataFrame(
            [{c: (0.0 if c != "row_id" else rid) for c in submission.columns}
             for rid in missing]
        )
        submission = pd.concat([submission, missing_df], ignore_index=True)

    # Sort to match sample_submission order where possible.
    submission = submission.set_index("row_id")
    submission = submission.reindex(
        sample_sub["row_id"].tolist(), fill_value=0.0
    ).reset_index()
    if "index" in submission.columns:
        submission = submission.rename(columns={"index": "row_id"})

    out_path = os.path.join(cfg.output_dir, "submission.csv")
    submission.to_csv(out_path, index=False)
    logger.info("Submission saved to: %s  (%d rows)", out_path, len(submission))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 inference script.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)
    cfg = load_config(config_path)
    run_inference(cfg)
