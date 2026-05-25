"""Stub for RepulsionLoss — the full implementation is not required for the
current adaptation plan.  The function signature matches the call-sites in
utils/loss.py so that imports resolve without errors.
"""

import torch


def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1):
    """Return zero repulsion-loss tensors.

    Parameters
    ----------
    pbox : torch.Tensor  – predicted boxes
    gtbox : torch.Tensor – ground-truth boxes
    deta : float         – smooth-ln delta
    pnms : float         – predicted-box NMS threshold
    gtnms : float        – GT-box NMS threshold

    Returns
    -------
    (lrepGT, lrepBox) : tuple[torch.Tensor, torch.Tensor]
    """
    device = pbox.device if isinstance(pbox, torch.Tensor) else torch.device('cpu')
    zero = torch.tensor(0.0, device=device)
    return zero, zero
