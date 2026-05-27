# tests/test_da_warmup.py
"""Tests for utils.da_warmup.compute_da_warmup_scale."""
from __future__ import annotations

import math
import pytest

from utils.da_warmup import compute_da_warmup_scale


class TestOffMode:
    def test_off_returns_one_at_iter_zero(self):
        assert compute_da_warmup_scale(ni=0, nw=1000, mode='off') == 1.0

    def test_off_returns_one_mid_warmup(self):
        assert compute_da_warmup_scale(ni=500, nw=1000, mode='off') == 1.0

    def test_off_returns_one_post_warmup(self):
        assert compute_da_warmup_scale(ni=5000, nw=1000, mode='off') == 1.0


class TestGateMode:
    def test_gate_returns_zero_at_iter_zero(self):
        assert compute_da_warmup_scale(ni=0, nw=1000, mode='gate') == 0.0

    def test_gate_returns_zero_mid_warmup(self):
        assert compute_da_warmup_scale(ni=999, nw=1000, mode='gate') == 0.0

    def test_gate_returns_zero_at_boundary(self):
        # ni == nw is still inside the warmup window (matches train_GRL.py: `if ni <= nw`).
        assert compute_da_warmup_scale(ni=1000, nw=1000, mode='gate') == 0.0

    def test_gate_returns_one_just_past_boundary(self):
        assert compute_da_warmup_scale(ni=1001, nw=1000, mode='gate') == 1.0

    def test_gate_returns_one_post_warmup(self):
        assert compute_da_warmup_scale(ni=5000, nw=1000, mode='gate') == 1.0


class TestRampMode:
    def test_ramp_returns_zero_at_iter_zero(self):
        assert compute_da_warmup_scale(ni=0, nw=1000, mode='ramp') == 0.0

    def test_ramp_returns_half_at_midpoint(self):
        assert math.isclose(compute_da_warmup_scale(ni=500, nw=1000, mode='ramp'), 0.5)

    def test_ramp_returns_one_at_boundary(self):
        assert compute_da_warmup_scale(ni=1000, nw=1000, mode='ramp') == 1.0

    def test_ramp_clamps_to_one_post_warmup(self):
        assert compute_da_warmup_scale(ni=5000, nw=1000, mode='ramp') == 1.0

    def test_ramp_is_monotonic(self):
        scales = [compute_da_warmup_scale(ni=i, nw=1000, mode='ramp') for i in range(0, 1100, 100)]
        for a, b in zip(scales, scales[1:]):
            assert a <= b

    def test_ramp_clamps_to_zero_for_negative_ni(self):
        # Contract: return value is always in [0.0, 1.0], even for nonsensical ni < 0.
        assert compute_da_warmup_scale(ni=-1, nw=1000, mode='ramp') == 0.0
        assert compute_da_warmup_scale(ni=-100, nw=1000, mode='ramp') == 0.0


class TestInvalid:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match='mode'):
            compute_da_warmup_scale(ni=0, nw=1000, mode='banana')

    def test_unknown_mode_raises_even_when_nw_zero(self):
        # Mode validation must happen before the nw=0 short-circuit.
        with pytest.raises(ValueError, match='mode'):
            compute_da_warmup_scale(ni=0, nw=0, mode='banana')

    def test_zero_nw_with_ramp_returns_one(self):
        # Edge case: nw=0 means "no warmup". Ramp should immediately be at 1.0
        # rather than dividing by zero. Same return as 'off'.
        assert compute_da_warmup_scale(ni=0, nw=0, mode='ramp') == 1.0
        assert compute_da_warmup_scale(ni=5, nw=0, mode='ramp') == 1.0

    def test_zero_nw_with_gate_returns_one(self):
        # nw=0 means no warmup window — gate is immediately open.
        assert compute_da_warmup_scale(ni=0, nw=0, mode='gate') == 1.0
