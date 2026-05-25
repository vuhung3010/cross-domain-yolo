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


def default_alpha() -> float:
    """alpha = BCE-with-logits([0.7, 0.3], [1.0, 0.0])  (DA-Detect's bce constant)."""
    pred = torch.tensor([[0.7, 0.3]])
    label = torch.tensor([[1.0, 0.0]])
    return F.binary_cross_entropy_with_logits(pred, label).item()


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
