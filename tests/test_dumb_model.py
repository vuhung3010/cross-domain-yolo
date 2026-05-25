"""Tests for the dumb model and DAImgHead."""
import torch
from models.da_classifier import DAImgHead


def test_da_img_head_output_shape():
    head = DAImgHead(in_channels=1024)
    x = torch.randn(2, 1024, 4, 4)
    y = head(x)
    assert y.shape == (2, 1, 4, 4)


def test_da_img_head_grad_flows():
    head = DAImgHead(in_channels=1024)
    x = torch.randn(2, 1024, 4, 4, requires_grad=True)
    head(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
