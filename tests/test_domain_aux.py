"""Tests for utils/domain_aux.py path resolution."""
from utils.domain_aux import resolve_da_path


def test_absolute_path_unchanged():
    assert resolve_da_path('/abs/path/imgs', '/root') == '/abs/path/imgs'


def test_relative_joined_with_root():
    assert resolve_da_path('images/train', '/root/data') == '/root/data/images/train'


def test_no_root_returns_input():
    assert resolve_da_path('images/train', None) == 'images/train'
