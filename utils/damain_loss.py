"""Deprecation shim — use utils.domain_loss instead.

Kept to avoid breaking old imports. The original DA_loss is re-exported with
its legacy signature; new code should call utils.domain_loss.da_img_loss.
"""
import torch
import torch.nn.functional as F


def DA_loss(features, target):
    """Legacy DA loss: features is a list of [B,C,H,W] tensors, target is 0 or 1."""
    loss = 0.0
    for feature in features:
        N, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1)
        label = torch.zeros_like(feature) if target == 0 else torch.ones_like(feature)
        feature_end = feature.reshape(N, -1)
        label_end = label.reshape(N, -1)
        loss = loss + F.binary_cross_entropy_with_logits(feature_end, label_end)
    return loss / len(features)
