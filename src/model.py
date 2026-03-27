"""
Neural network model for BirdCLEF 2026.

The backbone is any timm-compatible architecture specified in the config.
Supports optional GeM (Generalized Mean) pooling in place of the default
Global Average Pooling, which typically improves retrieval-style tasks.
"""

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GeM Pooling
# ---------------------------------------------------------------------------

class GeMPooling(nn.Module):
    """Generalized Mean Pooling for 2-D feature maps.

    Computes ``(mean(x^p))^(1/p)`` where ``p`` is a learnable scalar
    initialised to *p_init*.

    * At ``p=1`` this reduces to Global Average Pooling.
    * As ``p → ∞`` it approaches Global Max Pooling.

    The learnable ``p`` lets the model find the optimal aggregation
    strategy during training.

    Parameters
    ----------
    p:
        Initial value of the exponent parameter.
    eps:
        Small constant added before exponentiation to avoid log(0).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a ``(B, C, H, W)`` feature map to ``(B, C)``.

        Parameters
        ----------
        x:
            Feature map of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Pooled tensor of shape ``(batch, channels)``.
        """
        return (
            F.adaptive_avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                output_size=1,
            )
            .pow(1.0 / self.p)
            .flatten(1)
        )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BirdCLEFModel(nn.Module):
    """Bird-call classification model built on a timm backbone.

    The architecture is split into three components:

    1. ``backbone`` — feature extractor (timm model without head/pooling).
    2. ``pool``     — either ``GeMPooling`` (if ``cfg.use_gem_pooling`` is
       True) or ``None`` (backbone handles pooling via ``num_classes=0``).
    3. ``head``     — ``nn.Linear(in_features, num_classes)`` classifier.

    When GeM pooling is enabled ``backbone.forward_features(x)`` is called
    to get raw ``(B, C, H, W)`` feature maps, which are then passed through
    ``pool`` before the head.  Otherwise ``backbone(x)`` is called directly,
    returning pooled ``(B, in_features)`` features.

    Parameters
    ----------
    cfg:
        Config object with at least ``model_name`` (str), ``pretrained``
        (bool), and optionally ``use_gem_pooling`` (bool) / ``gem_p``
        (float) attributes.
    num_classes:
        Number of output classes.
    """

    def __init__(self, cfg, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.use_gem = getattr(cfg, "use_gem_pooling", False)

        logger.info(
            "Building model '%s' (pretrained=%s, num_classes=%d, gem=%s).",
            cfg.model_name,
            cfg.pretrained,
            num_classes,
            self.use_gem,
        )

        if self.use_gem:
            # Remove the backbone's default head AND pooling so we can apply
            # GeM pooling on the raw (B, C, H, W) feature maps.
            self.backbone = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                num_classes=0,
                global_pool="",
                in_chans=3,
            )
            self.pool = GeMPooling(p=getattr(cfg, "gem_p", 3.0))
        else:
            # Use the backbone's built-in Global Average Pooling.
            self.backbone = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                num_classes=0,
                in_chans=3,
            )
            self.pool = None

        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (for use with ``BCEWithLogitsLoss``).

        Parameters
        ----------
        x:
            Float tensor of shape ``(batch, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch, num_classes)``.
        """
        if self.use_gem:
            features = self.backbone.forward_features(x)  # (B, C, H, W)
            pooled = self.pool(features)                   # (B, C)
        else:
            pooled = self.backbone(x)                      # (B, C) via avg pool

        return self.head(pooled)                           # (B, num_classes)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities.

        Parameters
        ----------
        x:
            Float tensor of shape ``(batch, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Probability tensor of shape ``(batch, num_classes)`` in ``[0, 1]``.
        """
        return torch.sigmoid(self.forward(x))
