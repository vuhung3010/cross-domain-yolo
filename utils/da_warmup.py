"""DA-loss warmup scaling — pure helper used by train_GRL.py.

The scalar returned by `compute_da_warmup_scale` multiplies `loss_da_image`
before it is added to the total loss. Modes:
    off:  always 1.0 (no warmup — DA loss applied from iter 0).
    gate: 0.0 while ni <= nw, then 1.0. Hard cutoff at the LR-warmup boundary.
    ramp: linear from 0.0 at ni=0 to 1.0 at ni=nw, clamped at 1.0 after.

`nw == 0` short-circuits to 1.0 (no warmup window) for both gate and ramp.
"""
from __future__ import annotations

_VALID_MODES = ('off', 'gate', 'ramp')


def compute_da_warmup_scale(ni: int, nw: int, mode: str) -> float:
    """Return the scalar to multiply `loss_da_image` by at iteration `ni`.

    Args:
        ni: current global iteration index (matches `ni` in train_GRL.py).
        nw: number of LR-warmup iterations (matches `nw` in train_GRL.py).
        mode: one of 'off', 'gate', 'ramp'.

    Returns:
        Python float in [0.0, 1.0].
    """
    if mode not in _VALID_MODES:
        raise ValueError(f'Unknown mode {mode!r}; expected one of {_VALID_MODES}')
    if mode == 'off':
        return 1.0
    if nw <= 0:
        return 1.0
    if mode == 'gate':
        return 0.0 if ni <= nw else 1.0
    # mode == 'ramp' (only remaining option)
    if ni >= nw:
        return 1.0
    return max(0, ni) / nw
