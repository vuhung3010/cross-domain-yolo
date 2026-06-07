import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent


def run_train_grl_flags(*flags):
    return subprocess.run(
        [sys.executable, 'train_GRL.py', *flags],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


def assert_flag_error(flags, expected):
    result = run_train_grl_flags(*flags)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert expected in output


def test_da_img_faithful_requires_da_img():
    assert_flag_error(['--da-img-faithful'], '--da-img-faithful requires --da-img')


def test_da_img_faithful_allows_advgrl_past_flag_validation():
    result = run_train_grl_flags('--da-img', '--da-img-faithful', '--advgrl', '--cfg', 'missing.yaml')
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert '--da-img-faithful conflicts with --advgrl' not in output
    assert 'missing.yaml' in output or 'No such file' in output or 'does not exist' in output


def test_da_img_faithful_allows_aux_past_flag_validation():
    result = run_train_grl_flags('--da-img', '--da-img-faithful', '--aux', '--cfg', 'missing.yaml')
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert '--da-img-faithful conflicts with --aux' not in output
    assert 'missing.yaml' in output or 'No such file' in output or 'does not exist' in output


def test_da_img_faithful_triplet_still_requires_aux():
    assert_flag_error(
        ['--da-img', '--da-img-faithful', '--triplet-img'],
        '--triplet-img requires --aux',
    )


def test_da_img_faithful_allows_aux_triplet_past_flag_validation():
    result = run_train_grl_flags('--da-img', '--da-img-faithful', '--aux', '--triplet-img', '--cfg', 'missing.yaml')
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert '--da-img-faithful conflicts with --triplet-img' not in output
    assert 'missing.yaml' in output or 'No such file' in output or 'does not exist' in output


def test_da_img_faithful_requires_warmup_off():
    assert_flag_error(
        ['--da-img', '--da-img-faithful', '--da-img-warmup', 'ramp'],
        '--da-img-faithful requires --da-img-warmup off',
    )


def test_da_feat_layers_requires_da_img():
    assert_flag_error(
        ['--da-feat-layers', 'neck-p4'],
        '--da-feat-layers requires --da-img',
    )


def test_da_feat_layers_neck_requires_faithful_mode():
    assert_flag_error(
        ['--da-img', '--da-feat-layers', 'neck-p4'],
        '--da-feat-layers neck-* currently requires --da-img-faithful',
    )


def test_da_feat_layers_neck_p4_allows_past_flag_validation():
    result = run_train_grl_flags(
        '--da-img', '--da-img-faithful', '--da-feat-layers', 'neck-p4', '--cfg', 'missing.yaml',
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert '--da-feat-layers requires --da-img' not in output
    assert '--da-feat-layers neck-* currently requires --da-img-faithful' not in output
    assert 'missing.yaml' in output or 'No such file' in output or 'does not exist' in output


def test_da_feat_layers_rejects_unknown_choice():
    result = run_train_grl_flags('--da-img', '--da-img-faithful', '--da-feat-layers', 'banana')
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert 'invalid choice' in output
