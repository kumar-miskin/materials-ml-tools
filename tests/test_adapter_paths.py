"""All conversion CLIs must preserve their source and output sidecars."""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from materials_ml_tools import adapters


CLIS = [
    (adapters.main, 'read_nequip_xyz', []),
    (adapters.mace_main, 'read_mace_xyz', ['--energy-key', 'REF_energy', '--forces-key', 'REF_forces']),
    (adapters.matgl_main, 'read_matgl_xyz', ['--reference-energy-key', 'ref_energy', '--reference-forces-key', 'ref_forces']),
]


@pytest.mark.parametrize('cli,reader,extra', CLIS)
@pytest.mark.parametrize('collision', ['same', 'resolved', 'symlink', 'hardlink', 'metadata_input', 'metadata_alias', 'outputs_alias'])
def test_adapter_rejects_collisions_before_reading(tmp_path, monkeypatch, cli, reader, extra, collision):
    source = tmp_path / 'source.xyz'
    output = tmp_path / 'predictions.csv'
    source.write_bytes(b'original XYZ bytes\n')
    if collision == 'same':
        output = source
    elif collision == 'resolved':
        output = tmp_path / '.' / 'source.xyz'
    elif collision == 'symlink':
        output.symlink_to(source)
    elif collision == 'hardlink':
        os.link(source, output)
    elif collision == 'metadata_input':
        source = output.with_suffix(output.suffix + '.meta.json')
        source.write_bytes(b'original XYZ bytes\n')
    elif collision == 'metadata_alias':
        output.with_suffix(output.suffix + '.meta.json').symlink_to(source)
    else:
        output.write_bytes(b'old CSV bytes\n')
        os.link(output, output.with_suffix(output.suffix + '.meta.json'))
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

    def do_not_read(*args, **kwargs):
        pytest.fail('collision must be rejected before parsing input')

    monkeypatch.setattr(adapters, reader, do_not_read)
    with pytest.raises(SystemExit) as exc:
        cli([str(source), '--out', str(output), '--energy-unit', 'eV', '--force-unit', 'eV/Angstrom', *extra])
    assert exc.value.code == 2
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


@pytest.mark.parametrize('cli,reader,extra', CLIS)
def test_adapter_distinct_existing_outputs_still_allowed(tmp_path, monkeypatch, cli, reader, extra):
    source, output = tmp_path / 'source.xyz', tmp_path / 'predictions.csv'
    source.write_bytes(b'original XYZ bytes\n')
    sidecar = output.with_suffix(output.suffix + '.meta.json')
    output.write_text('old CSV'); sidecar.write_text('old metadata')
    monkeypatch.setattr(adapters, reader, lambda *a, **k: (pd.DataFrame({'value': [1]}), {'frames': 1}))
    cli([str(source), '--out', str(output), '--energy-unit', 'eV', '--force-unit', 'eV/Angstrom', *extra])
    assert source.read_bytes() == b'original XYZ bytes\n'
    assert output.read_text() == 'value\n1\n'
    assert json.loads(sidecar.read_text()) == {'frames': 1}
