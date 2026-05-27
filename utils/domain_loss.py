"""Domain-adaptation loss functions.

da_img_loss      — BCE-with-logits between dense classifier output and dense domain labels
triplet_img_loss — image-level triplet metric regularization (anchor=source, positive=target, negative=aux)
"""
import torch
import torch.nn.functional as F
from torch import nn


def da_img_loss(logits: torch.Tensor, source_count: int) -> torch.Tensor:
    """Image-level DA loss (DANN-style BCE).

    Args:
        logits: classifier output of shape [B, 1, H, W] where the first
                `source_count` rows are source images and the remainder are target.
        source_count: number of source samples in the batch (B_s).

    Returns:
        Scalar BCE-with-logits loss.

    Labels are dense (shape matches logits): 1.0 for source rows, 0.0 for target.
    This matches both the existing utils/damain_loss.DA_loss convention and
    DA-Detect's DAImgHead loss.
    """
    labels = torch.zeros_like(logits)
    labels[:source_count] = 1.0
    return F.binary_cross_entropy_with_logits(logits, labels)


def triplet_img_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Image-level triplet loss: max(d(A,P) - d(A,N) + margin, 0).

    Args:
        anchor, positive, negative: [N, D] tensors (typically N=1 batch-centroid).
        margin: triplet margin (delta in spec).
    """
    return nn.TripletMarginLoss(margin=margin, p=2)(anchor, positive, negative)
