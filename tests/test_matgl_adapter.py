"""ASE calculator output matches MatGL PESCalculator's energy/forces contract."""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from materials_ml_tools.adapters import COLUMNS, matgl_main, read_matgl_xyz
from materials_ml_tools.metrics import evaluate


def evaluated_frames():
    result = []
    for symbols, et, ep, ft, fp in [
        ("LiH", -4., -3., [[0,0,0],[0,0,0]], [[1,0,0],[0,2,0]]),
        ("H2O", -2., -2.3, [[0,0,0]]*3, [[0,0,0],[0,0,0],[0,0,-2]]),
    ]:
        at = Atoms(symbols)
        at.info['ref_energy'] = et
        at.arrays['ref_forces'] = np.array(ft, dtype=float)
        at.calc = SinglePointCalculator(at, energy=ep, forces=np.array(fp, dtype=float))
        result.append(at)
    return result


def read(path):
    return read_matgl_xyz(path, reference_energy_key='ref_energy', reference_forces_key='ref_forces', energy_unit='eV', force_unit='eV/Angstrom')


def test_standard_ase_calculator_output_to_flat_metrics(tmp_path):
    path=tmp_path/'eval.xyz'; write(path,evaluated_frames(),format='extxyz')
    frame,meta=read(path)
    assert list(frame.columns)==COLUMNS
    assert frame.structure_id.tolist()==['eval:0']*2+['eval:1']*3
    assert frame.element.tolist()==['Li','H','H','H','O']
    assert meta['frames']==2 and len(meta['source_sha256'])==64
    scores=evaluate(frame)
    assert scores['energy_mae_per_atom']==pytest.approx(.3)
    assert scores['force_vector_rmse']==pytest.approx((9/5)**.5)


def test_cli_writes_source_metadata(tmp_path):
    path=tmp_path/'eval.xyz'; write(path,evaluated_frames(),format='extxyz')
    out=tmp_path/'flat.csv'
    matgl_main([str(path),'--reference-energy-key','ref_energy','--reference-forces-key','ref_forces','--energy-unit','eV','--force-unit','eV/Angstrom','--out',str(out)])
    assert out.read_text().splitlines()[0]==','.join(COLUMNS)
    meta=json.loads((tmp_path/'flat.csv.meta.json').read_text())
    assert meta['reference_energy_key']=='ref_energy'


def test_rejects_missing_reference_and_prediction_as_label(tmp_path):
    path=tmp_path/'eval.xyz'; at=evaluated_frames()[0];del at.arrays['ref_forces'];write(path,at,format='extxyz')
    with pytest.raises(ValueError,match='missing reference'):
        read(path)
    with pytest.raises(ValueError,match='must differ'):
        read_matgl_xyz(path,reference_energy_key='energy',reference_forces_key='ref_forces',energy_unit='eV',force_unit='eV/Angstrom')


def test_rejects_nonfinite_forces(tmp_path):
    path=tmp_path/'eval.xyz';at=evaluated_frames()[0];at.arrays['ref_forces'][0,0]=float('nan');write(path,at,format='extxyz')
    with pytest.raises(ValueError,match='finite shape'):
        read(path)
