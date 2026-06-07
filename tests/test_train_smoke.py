"""End-to-end smoke test: run train_GRL.py for ~1 epoch on toy fixtures across
all six valid flag combinations.

Slow-ish (a couple of minutes total). Skipped automatically if CUDA is not available.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture(scope='module', autouse=True)
def ensure_fixtures():
    """Always regenerate fixtures (overwrite-safe, <1s) so partial deletions self-heal."""
    sys.path.insert(0, str(FIXTURES))
    try:
        import make_fixtures  # type: ignore
        make_fixtures.main()
    finally:
        sys.path.pop(0)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('flags,name', [
    ([],                                                                           'baseline'),
    (['--da-img'],                                                                 'daimg'),
    (['--da-img', '--da-img-faithful'],                                             'daimg_faithful'),
    (['--da-img', '--da-img-faithful', '--advgrl'],                                'daimg_faithful_advgrl'),
    (['--da-img', '--da-img-faithful', '--advgrl', '--da-feat-layers', 'neck-p4'],  'daimg_faithful_advgrl_neck_p4'),
    (['--da-img', '--da-img-faithful', '--advgrl', '--da-feat-layers', 'neck-all'], 'daimg_faithful_advgrl_neck_all'),
    (['--da-img', '--da-img-faithful', '--advgrl', '--advgrl-alpha', '0.75', '--da-feat-layers', 'neck-all', '--da-img-obj-gate'], 'daimg_faithful_advgrl_neck_all_objgate'),
    (['--da-img', '--da-img-faithful', '--aux'],                                    'daimg_faithful_aux'),
    (['--da-img', '--da-img-faithful', '--aux', '--triplet-img'],                   'daimg_faithful_triplet'),
    (['--da-img', '--da-img-faithful', '--advgrl', '--aux', '--triplet-img'],       'daimg_faithful_full'),
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
    result = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=180,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, f'FAILED [{name}]:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}'

    # Positive artifact checks — guard against vacuous-pass on a no-op exit.
    run_dir = tmp_path / f'smoke_{name}'
    ckpt = run_dir / 'weights' / 'last.pt'
    assert ckpt.exists(), f'[{name}] checkpoint missing: {ckpt}'

    if '--da-img' in flags:
        da_csv = run_dir / 'da_losses.csv'
        assert da_csv.exists(), f'[{name}] da_losses.csv missing: {da_csv}'
        # Header + at least one data row.
        lines = da_csv.read_text().splitlines()
        assert len(lines) > 1, f'[{name}] da_losses.csv has no data rows (got {len(lines)} lines)'
        if '--da-img-faithful' in flags:
            header = lines[0].split(',')
            first_row = lines[1].split(',')
            row = dict(zip(header, first_row))
            assert row['da_scale'] == '1.0'
            if '--advgrl' in flags:
                assert row['L_c'] != ''
                assert 0.0 < float(row['lambda_adv']) <= 3.0
            else:
                assert row['lambda_adv'] == '0.1'
        if '--triplet-img' in flags:
            header = lines[0].split(',')
            first_row = lines[1].split(',')
            row = dict(zip(header, first_row))
            assert row['loss_triplet_img'] != ''
            assert float(row['loss_triplet_img']) >= 0.0
