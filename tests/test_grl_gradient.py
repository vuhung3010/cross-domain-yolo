"""Pin gradient_scalar behavior: forward is identity, backward scales by weight."""
import torch
from utils.domain_grl import gradient_scalar


def test_forward_is_identity():
    x = torch.randn(2, 3, 4, 4)
    y = gradient_scalar(x, -0.1)
    assert torch.allclose(y, x)


def test_backward_scales_by_weight():
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    # Use small weight -0.1 — typical GRL sign-flip + scale.
    y = gradient_scalar(x, -0.1)
    y.sum().backward()
    # dL/dx via identity-forward is grad_output (== 1 everywhere from sum().backward()),
    # scaled by weight = -0.1.
    expected = torch.full_like(x, -0.1)
    assert torch.allclose(x.grad, expected)


def test_backward_dynamic_weight():
    """Critical for AdvGRL: weight can be a Python float computed per iter."""
    for w in [-3.0, -1.0, -0.1, 0.0, 0.5]:
        x = torch.randn(2, 3, requires_grad=True)
        y = gradient_scalar(x, w)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, w))
