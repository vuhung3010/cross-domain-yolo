"""End-to-end smoke test: run train_GRL.py for ~1 epoch on toy fixtures across
all six valid flag combinations.

Slow-ish (a couple of minutes total). Skipped automatically if CUDA is not available.
"""
import os
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / 'tests' / 'fixtures'


@pytest.fixture(scope='module', autouse=True)
def ensure_fixtures():
    """Regenerate fixtures if missing."""
    if not (FIXTURES / 'source.txt').exists():
        subprocess.run([sys.executable, str(FIXTURES / 'make_fixtures.py')], check=True)


@pytest.fixture(scope='module')
def toy_yaml(tmp_path_factory):
    p = tmp_path_factory.mktemp('cfg') / 'toy.yaml'
    p.write_text(f'''path: {FIXTURES}
train: source/images
val:   source/images
target: target/images
aux:    aux/images
nc: 1
names: ['toy']
''')
    return str(p)


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('flags,name', [
    ([],                                                                           'baseline'),
    (['--da-img'],                                                                 'daimg'),
    (['--da-img', '--advgrl'],                                                     'advgrl'),
    (['--da-img', '--aux'],                                                        'aux'),
    (['--da-img', '--aux', '--triplet-img'],                                       'triplet'),
    (['--da-img', '--advgrl', '--aux', '--triplet-img'],                           'full'),
])
def test_train_GRL_smoke(toy_yaml, tmp_path, flags, name):
    cmd = [
        sys.executable, 'train_GRL.py',
        '--cfg', 'configs/domain/yolov5l_GRL.yaml',
        '--data', toy_yaml,
        '--epochs', '1', '--batch-size', '2', '--img', '64',
        '--name', f'smoke_{name}',
        '--project', str(tmp_path),
    ] + flags
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f'FAILED [{name}]:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}'
