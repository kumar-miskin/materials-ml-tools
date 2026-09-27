"""MACE CLI output uses info/arrays fields, not an ASE calculator."""
import json

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.io import write

from materials_ml_tools.adapters import COLUMNS, mace_main, read_mace_xyz
from materials_ml_tools.metrics import evaluate


def frames(prefix="MACE_"):
    first = Atoms("LiH", positions=[[0, 0, 0], [0, 0, 1]])
    first.info.update(REF_energy=-4.0, **{prefix + "energy": -3.0})
    first.arrays["REF_forces"] = np.zeros((2, 3))
    first.arrays[prefix + "forces"] = np.array([[1., 0, 0], [0, 2., 0]])
    second = Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    second.info.update(REF_energy=-2.0, **{prefix + "energy": -2.3})
    second.arrays["REF_forces"] = np.zeros((3, 3))
    second.arrays[prefix + "forces"] = np.array([[0., 0, 0], [0, 0, 0], [0, 0, -2.]])
    return [first, second]


def convert(path, **kwargs):
    return read_mace_xyz(path, energy_key="REF_energy", forces_key="REF_forces",
                         energy_unit="eV", force_unit="eV/Angstrom", **kwargs)


def test_mace_eval_output_and_metrics(tmp_path):
    src = tmp_path / "out.xyz"
    write(src, frames(), format="extxyz")
    table, meta = convert(src)
    assert list(table.columns) == COLUMNS
    assert table.structure_id.tolist() == ["out:0"] * 2 + ["out:1"] * 3
    assert table.element.tolist() == ["Li", "H", "H", "H", "O"]
    assert table.n_atoms.tolist() == [2, 2, 3, 3, 3]
    assert meta["frames"] == 2 and len(meta["source_sha256"]) == 64
    got = evaluate(table)
    assert got["energy_mae_per_atom"] == pytest.approx(.3)
    assert got["force_component_rmse"] == pytest.approx((9 / 15) ** .5)
    assert got["force_vector_rmse"] == pytest.approx((9 / 5) ** .5)


def test_custom_prefix_and_cli_metadata(tmp_path):
    src = tmp_path / "pinned.xyz"
    write(src, frames("pinned_"), format="extxyz")
    out = tmp_path / "flat.csv"
    mace_main([str(src), "--energy-key", "REF_energy", "--forces-key", "REF_forces",
               "--prediction-prefix", "pinned_", "--energy-unit", "eV",
               "--force-unit", "eV/Angstrom", "--out", str(out)])
    assert out.read_text().splitlines()[0] == ",".join(COLUMNS)
    assert json.loads((tmp_path / "flat.csv.meta.json").read_text())["prediction_prefix"] == "pinned_"


def test_missing_fields_and_reference_collision(tmp_path):
    src = tmp_path / "missing.xyz"
    one = frames()[0]
    del one.arrays["REF_forces"]
    write(src, one, format="extxyz")
    with pytest.raises(ValueError, match="frame 0: missing field.*REF_forces"):
        convert(src)
    with pytest.raises(ValueError, match="reference keys must differ"):
        read_mace_xyz(src, energy_key="MACE_energy", forces_key="REF_forces",
                      energy_unit="eV", force_unit="eV/Angstrom")


def test_reject_nonfinite_energy_or_force(tmp_path):
    src = tmp_path / "bad.xyz"
    one = frames()[0]
    one.info["MACE_energy"] = np.nan
    write(src, one, format="extxyz")
    with pytest.raises(ValueError, match="finite scalar energy"):
        convert(src)
    one.info["MACE_energy"] = -3.0
    one.arrays["MACE_forces"][0, 0] = np.inf
    write(src, one, format="extxyz")
    with pytest.raises(ValueError, match="finite shape"):
        convert(src)


def test_no_frames_and_no_units(tmp_path):
    src = tmp_path / "empty.xyz"
    src.write_text("")
    with pytest.raises(ValueError, match="no frames"):
        convert(src)
    with pytest.raises(ValueError, match="energy_unit is required"):
        read_mace_xyz(src, energy_key="REF_energy", forces_key="REF_forces",
                      energy_unit="", force_unit="eV/Angstrom")
