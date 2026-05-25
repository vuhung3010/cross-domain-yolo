"""AdvGRL: dynamic gradient-reversal weight per DA-Detect (Adv_GRL).

The effective GRL weight applied at each iter is:
    lambda_adv = lambda_0 * min(beta, 1/L_c)     if L_c <= alpha   (hard regime)
    lambda_adv = lambda_0                        otherwise         (easy regime)

This caps lambda_adv at lambda_0 * beta (default 0.1 * 30 = 3.0), NOT at beta.
See spec §2.1.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_DEFAULT_ALPHA: float | None = None


def default_alpha() -> float:
    """alpha = BCE-with-logits([0.7, 0.3], [1.0, 0.0])  (DA-Detect's bce constant)."""
    global _DEFAULT_ALPHA
    if _DEFAULT_ALPHA is None:
        pred = torch.tensor([[0.7, 0.3]])
        label = torch.tensor([[1.0, 0.0]])
        _DEFAULT_ALPHA = F.binary_cross_entropy_with_logits(pred, label).item()
    return _DEFAULT_ALPHA


def compute_lambda_adv(L_c: float, lambda_0: float = 0.1, alpha: float = None,
                       beta: float = 30.0, eps: float = 1e-7) -> float:
    """Compute the AdvGRL effective weight for this iter.

    Args:
        L_c: scalar DA-classifier loss from the detached forward (Python float).
        lambda_0: base GRL weight (e.g. 0.1).
        alpha: gate threshold; if None, uses default_alpha() ≈ 0.6286.
        beta: cap on the 1/L_c factor.
        eps: division-by-zero guard.

    Returns:
        Python float — the effective lambda. Pass with a negative sign into
        gradient_scalar() to perform gradient reversal.
    """
    if alpha is None:
        alpha = default_alpha()
    L_c = float(L_c)
    if L_c <= alpha:
        adv_threshold = min(beta, 1.0 / (L_c + eps))
        return lambda_0 * adv_threshold
    return lambda_0


from torch import nn
from utils.domain_grl import gradient_scalar
from utils.domain_loss import da_img_loss


def advgrl_step(
    backbone_feat: torch.Tensor,
    source_count: int,
    classifier: nn.Module,
    *,
    use_advgrl: bool,
    lambda_0: float,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, float, float]:
    """Two-pass AdvGRL: detached forward for L_c, then GRL-attached forward.

    Args:
        backbone_feat: concatenated source+target features, shape [B_s+B_t, C, H, W].
        source_count: B_s — number of source rows at the start.
        classifier: DAImgHead (or similar) producing [B, 1, H, W] logits.
        use_advgrl: if False, lambda_adv is just lambda_0 (plain fixed-GRL).
        lambda_0, alpha, beta: AdvGRL hyperparameters.

    Returns:
        (loss_da_image, lambda_adv, L_c_value)
        - loss_da_image: scalar tensor for backprop.
        - lambda_adv:    Python float, the effective GRL weight this iter.
        - L_c_value:     Python float, the detached classifier loss (for logging).
    """
    # Pass 1: detached — compute scalar L_c for AdvGRL gating.
    pred_detached = classifier(backbone_feat.detach())
    L_c_tensor = da_img_loss(pred_detached, source_count=source_count)
    L_c = float(L_c_tensor.item())

    # Decide lambda_adv.
    if use_advgrl:
        lambda_adv = compute_lambda_adv(L_c, lambda_0=lambda_0, alpha=alpha, beta=beta)
    else:
        lambda_adv = lambda_0

    # Pass 2: GRL-attached — actual gradients flow back through gradient_scalar.
    feat_grl = gradient_scalar(backbone_feat, -lambda_adv)
    pred = classifier(feat_grl)
    loss_da_image = da_img_loss(pred, source_count=source_count)

    return loss_da_image, lambda_adv, L_c
