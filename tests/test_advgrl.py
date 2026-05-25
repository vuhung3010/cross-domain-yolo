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
    """At L_c == alpha, the function takes the hard-regime branch."""
    v_at = compute_lambda_adv(ALPHA, LAMBDA_0, ALPHA, BETA)
    v_above = compute_lambda_adv(ALPHA + 1e-6, LAMBDA_0, ALPHA, BETA)
    assert v_at != v_above or math.isclose(v_at, LAMBDA_0, rel_tol=1e-3)


def test_always_le_lambda_0_times_beta():
    """The cap invariant — assertion smoke tests rely on this."""
    for L_c in [0.0, 1e-6, 0.01, 0.1, 0.3, 0.6, 0.6286, 0.7, 1.0, 5.0]:
        v = compute_lambda_adv(L_c, LAMBDA_0, ALPHA, BETA)
        assert v <= LAMBDA_0 * BETA + 1e-6, f'L_c={L_c} gave {v}'
