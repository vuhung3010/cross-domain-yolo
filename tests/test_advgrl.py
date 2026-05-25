"""Unit tests for utils/advgrl.compute_lambda_adv."""
import math
import pytest
from utils.advgrl import compute_lambda_adv, default_alpha


LAMBDA_0 = 0.1
BETA = 30.0
ALPHA = default_alpha()  # BCE([0.7,0.3],[1,0]) ≈ 0.6286


def test_default_alpha_close_to_0_6286():
    assert math.isclose(default_alpha(), 0.6286, abs_tol=0.001)


def test_easy_regime_uses_lambda_0():
    """L_c > alpha -> return lambda_0 unchanged."""
    assert compute_lambda_adv(0.7,  LAMBDA_0, ALPHA, BETA) == LAMBDA_0
    assert compute_lambda_adv(1.0,  LAMBDA_0, ALPHA, BETA) == LAMBDA_0
    assert compute_lambda_adv(10.0, LAMBDA_0, ALPHA, BETA) == LAMBDA_0


def test_hard_regime_uses_lambda_0_times_min():
    """L_c <= alpha -> lambda_0 * min(beta, 1/L_c)."""
    # L_c = 0.3 -> 1/L_c ≈ 3.33; min(30, 3.33) = 3.33; lambda_adv = 0.333
    v = compute_lambda_adv(0.3, LAMBDA_0, ALPHA, BETA)
    assert math.isclose(v, LAMBDA_0 * (1.0 / 0.3), rel_tol=1e-5)


def test_cap_at_lambda_0_times_beta():
    """L_c -> 0 -> lambda_adv saturates at lambda_0 * beta = 3.0."""
    v = compute_lambda_adv(1e-9, LAMBDA_0, ALPHA, BETA)
    assert math.isclose(v, LAMBDA_0 * BETA, rel_tol=1e-3)
    # Also at the boundary L_c=0 with eps guard:
    v0 = compute_lambda_adv(0.0, LAMBDA_0, ALPHA, BETA)
    assert v0 <= LAMBDA_0 * BETA + 1e-6


def test_alpha_boundary():
    """At L_c == alpha, the function takes the hard-regime branch (<= comparison, not <)."""
    v_at = compute_lambda_adv(ALPHA, LAMBDA_0, ALPHA, BETA)
    v_above = compute_lambda_adv(ALPHA + 1e-6, LAMBDA_0, ALPHA, BETA)
    # At the boundary, hard branch fires: v_at = lambda_0 * (1/alpha) > lambda_0.
    assert v_at > LAMBDA_0
    assert math.isclose(v_at, LAMBDA_0 * (1.0 / ALPHA), rel_tol=1e-3)
    # Just above, easy branch fires: v_above == lambda_0 exactly.
    assert v_above == LAMBDA_0


def test_always_le_lambda_0_times_beta():
    """The cap invariant — assertion smoke tests rely on this."""
    for L_c in [0.0, 1e-6, 0.01, 0.1, 0.3, 0.6, 0.6286, 0.7, 1.0, 5.0]:
        v = compute_lambda_adv(L_c, LAMBDA_0, ALPHA, BETA)
        assert v <= LAMBDA_0 * BETA + 1e-6, f'L_c={L_c} gave {v}'


import torch
from models.da_classifier import DAImgHead
from utils.advgrl import advgrl_step


def test_advgrl_step_returns_three_things():
    head = DAImgHead(in_channels=64)
    feat = torch.randn(4, 64, 2, 2, requires_grad=True)
    loss, lambda_adv, L_c = advgrl_step(
        feat, source_count=2, classifier=head,
        use_advgrl=True, lambda_0=0.1, alpha=default_alpha(), beta=30.0,
    )
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert isinstance(lambda_adv, float)
    assert isinstance(L_c, float)
    assert lambda_adv <= 0.1 * 30.0 + 1e-6


def test_advgrl_step_gradient_flows_back():
    """Backprop reaches backbone_feat with the sign-flipped weight."""
    head = DAImgHead(in_channels=64)
    feat = torch.randn(4, 64, 2, 2, requires_grad=True)
    loss, _, _ = advgrl_step(
        feat, source_count=2, classifier=head,
        use_advgrl=False, lambda_0=0.1, alpha=default_alpha(), beta=30.0,
    )
    loss.backward()
    assert feat.grad is not None
    assert feat.grad.abs().sum() > 0


def test_advgrl_step_fixed_lambda_when_off():
    """use_advgrl=False -> lambda_adv == lambda_0 regardless of L_c."""
    head = DAImgHead(in_channels=64)
    feat = torch.randn(4, 64, 2, 2)
    _, lambda_adv, _ = advgrl_step(
        feat, source_count=2, classifier=head,
        use_advgrl=False, lambda_0=0.07, alpha=default_alpha(), beta=30.0,
    )
    assert lambda_adv == 0.07
