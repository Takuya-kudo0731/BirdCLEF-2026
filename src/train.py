"""
Training script for BirdCLEF 2026 baseline.

Usage
-----
    python src/train.py --config configs/baseline.yaml [--fold <int>]

The script performs stratified k-fold cross-validation on the training
metadata CSV.  For each fold it trains a BirdCLEFModel and saves the
checkpoint with the best Padded cmAP to ``{output_dir}/fold{k}_best.pth``.
A fitted ``LabelEncoder`` is saved to ``{output_dir}/label_encoder.pkl``
for later use during inference.
"""

import argparse
import logging
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as a script from the repo root or from inside src/.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import load_config
from src.dataset import BirdCLEFDataset, CombinedDataset, MixupDataset
from src.model import BirdCLEFModel
from src.utils import padded_cmap, set_seed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss for multilabel classification.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Applied element-wise and averaged over the batch.  Focal Loss
    downweights easy negatives (high-confidence correct predictions) and
    focuses learning on hard, uncertain examples — especially useful for
    the severe class imbalance found in BirdCLEF datasets where some
    species have fewer than 10 training recordings.

    Parameters
    ----------
    gamma:
        Focusing parameter.  Higher values reduce the relative loss for
        well-classified examples.  Typical range: 0.5–5.0.
    alpha:
        Weighting factor for the positive class.  Compensates for
        class imbalance at the binary level.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BirdCLEFModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg,
    epoch: int,
) -> float:
    """Run one training epoch.

    Returns
    -------
    float
        Mean loss over all batches.
    """
    model.train()
    running_loss = 0.0
    n_batches = len(loader)
    grad_clip = getattr(cfg, "grad_clip", 1.0)

    progress = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for batch_idx, (images, labels) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg.fp16 and device.type == "cuda":
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: BirdCLEFModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg,
) -> tuple:
    """Run validation and compute loss + Padded cmAP.

    Returns
    -------
    tuple[float, float]
        ``(mean_loss, padded_cmap_score)``
    """
    model.eval()
    running_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    progress = tqdm(loader, desc="Validation", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cfg.fp16 and device.type == "cuda":
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        running_loss += loss.item()
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    score = padded_cmap(y_true, y_pred, padding_factor=5)
    mean_loss = running_loss / max(len(loader), 1)
    return mean_loss, score


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def build_optimizer(model: BirdCLEFModel, cfg) -> Adam:
    """Build an Adam optimizer with optional differential learning rates.

    If the config supplies ``learning_rate_backbone`` and
    ``learning_rate_head``, the backbone parameters are trained at the
    lower rate (preserving pretrained features) and the new head is trained
    at the higher rate.  This avoids catastrophic forgetting of the
    ImageNet-pretrained representations early in training.

    Falls back to a single learning rate (``cfg.learning_rate``) if the
    differential keys are absent.

    Parameters
    ----------
    model:
        ``BirdCLEFModel`` instance (must have ``.backbone`` and ``.head``
        attributes as created by the updated ``model.py``).
    cfg:
        Config namespace.

    Returns
    -------
    Adam
    """
    lr_backbone = getattr(cfg, "learning_rate_backbone", None)
    lr_head = getattr(cfg, "learning_rate_head", None)
    base_lr = getattr(cfg, "learning_rate", 1e-3)

    if lr_backbone is not None and lr_head is not None:
        backbone_params = list(model.backbone.parameters())
        if model.pool is not None:
            backbone_params += list(model.pool.parameters())
        head_params = list(model.head.parameters())

        param_groups = [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]
        logger.info(
            "Differential LR — backbone: %.2e | head: %.2e", lr_backbone, lr_head
        )
    else:
        param_groups = model.parameters()
        logger.info("Single LR — %.2e", base_lr)

    return Adam(param_groups, lr=base_lr, weight_decay=cfg.weight_decay)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(cfg, fold_override: int | None = None) -> None:
    """Full training routine with stratified k-fold cross-validation."""

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load metadata.
    # ------------------------------------------------------------------
    logger.info("Loading metadata from: %s", cfg.train_metadata)
    if not os.path.exists(cfg.train_metadata):
        logger.error("Metadata file not found: %s", cfg.train_metadata)
        sys.exit(1)

    df = pd.read_csv(cfg.train_metadata)
    logger.info("Dataset size: %d rows, columns: %s", len(df), list(df.columns))

    # Drop rows with missing primary_label.
    df = df.dropna(subset=["primary_label"]).reset_index(drop=True)
    logger.info("After dropping missing labels: %d rows.", len(df))

    # ------------------------------------------------------------------
    # Encode labels.
    # ------------------------------------------------------------------
    le = LabelEncoder()
    le.fit(df["primary_label"])
    num_classes = len(le.classes_)
    cfg.num_classes = num_classes
    logger.info("Number of classes: %d", num_classes)

    le_path = os.path.join(cfg.output_dir, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    logger.info("LabelEncoder saved to: %s", le_path)

    # ------------------------------------------------------------------
    # Build criterion.
    # ------------------------------------------------------------------
    if getattr(cfg, "use_focal_loss", False):
        criterion = FocalLoss(
            gamma=getattr(cfg, "focal_gamma", 2.0),
            alpha=getattr(cfg, "focal_alpha", 0.25),
        )
        logger.info(
            "Loss: FocalLoss (gamma=%.1f, alpha=%.2f)",
            cfg.focal_gamma,
            cfg.focal_alpha,
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Loss: BCEWithLogitsLoss")

    # ------------------------------------------------------------------
    # K-Fold split.
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    folds = list(skf.split(df, df["primary_label"]))

    fold_range = range(cfg.n_folds)
    if fold_override is not None:
        fold_range = [fold_override]

    for fold in fold_range:
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold, cfg.n_folds - 1)
        logger.info("=" * 60)

        train_idx, val_idx = folds[fold]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        logger.info("Train samples: %d | Val samples: %d", len(train_df), len(val_df))

        # --------------------------------------------------------------
        # Datasets & loaders.
        # --------------------------------------------------------------
        train_ds_base = BirdCLEFDataset(train_df, cfg, le, mode="train")

        # Mix in pseudo-labelled data if a CSV path is configured.
        pseudo_csv = getattr(cfg, "pseudo_labels_path", None)
        if pseudo_csv and os.path.exists(pseudo_csv):
            pseudo_df = pd.read_csv(pseudo_csv)
            # Re-use train_audio_dir as the root; pseudo CSV stores filenames
            # relative to the soundscapes dir, so override audio root if needed.
            pseudo_ds_base = BirdCLEFDataset(pseudo_df, cfg, le, mode="train")
            pseudo_ratio = getattr(cfg, "pseudo_mix_ratio", 0.3)
            train_ds_base = CombinedDataset(train_ds_base, pseudo_ds_base, pseudo_ratio)
            logger.info(
                "Pseudo-label mixing enabled: %d pseudo rows (ratio=%.2f)",
                len(pseudo_df),
                pseudo_ratio,
            )

        train_ds = MixupDataset(train_ds_base, alpha=cfg.mixup_alpha)
        val_ds = BirdCLEFDataset(val_df, cfg, le, mode="val")

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )

        # --------------------------------------------------------------
        # Model, criterion, optimiser, scheduler.
        # --------------------------------------------------------------
        model = BirdCLEFModel(cfg, num_classes=num_classes).to(device)
        optimizer = build_optimizer(model, cfg)

        # CosineAnnealingLR scales all param groups from their initial LR
        # down to eta_min.
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
            eta_min=1e-7,
        )
        scaler = GradScaler(enabled=(cfg.fp16 and device.type == "cuda"))

        best_score = -1.0
        patience_counter = 0
        best_ckpt_path = os.path.join(cfg.output_dir, f"fold{fold}_best.pth")

        # --------------------------------------------------------------
        # Epoch loop.
        # --------------------------------------------------------------
        for epoch in range(1, cfg.num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, cfg, epoch
            )
            val_loss, val_score = validate(model, val_loader, criterion, device, cfg)
            scheduler.step()

            # Log current LRs for both param groups.
            current_lrs = [pg["lr"] for pg in optimizer.param_groups]
            lr_str = " | ".join(f"{lr:.2e}" for lr in current_lrs)
            logger.info(
                "Epoch %02d | train_loss=%.4f | val_loss=%.4f | val_cmap=%.4f | lr=[%s]",
                epoch,
                train_loss,
                val_loss,
                val_score,
                lr_str,
            )

            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_score": val_score,
                        "num_classes": num_classes,
                        "label_encoder_classes": le.classes_.tolist(),
                    },
                    best_ckpt_path,
                )
                logger.info(
                    "  -> New best score %.4f. Checkpoint saved to %s",
                    best_score,
                    best_ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        patience_counter,
                    )
                    break

        logger.info(
            "Fold %d finished. Best val Padded cmAP: %.4f", fold, best_score
        )

    logger.info("All folds complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train only this single fold (0-indexed). If omitted, all folds are run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        # Resolve relative to the repo root (parent of src/).
        config_path = os.path.join(_REPO_ROOT, config_path)
    cfg = load_config(config_path)
    train(cfg, fold_override=args.fold)
