"""Shared pytest fixtures."""
from pathlib import Path
import pytest
import torch

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def source_manifest():
    return str(FIXTURES / 'source.txt')


@pytest.fixture
def target_manifest():
    return str(FIXTURES / 'target.txt')


@pytest.fixture
def aux_manifest():
    return str(FIXTURES / 'aux.txt')


@pytest.fixture
def tiny_backbone_feat(device):
    """[2, 1024, 2, 2] — mimics YOLOv5-L SPPF output at very small img size."""
    torch.manual_seed(0)
    return torch.randn(2, 1024, 2, 2, device=device, requires_grad=True)
