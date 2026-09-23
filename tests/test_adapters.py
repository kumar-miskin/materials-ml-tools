from pathlib import Path
import json

import pytest

pytest.importorskip("ase")

from materials_ml_tools.adapters import COLUMNS, main, read_nequip_xyz
from materials_ml_tools.metrics import evaluate

FIXTURE = Path(__file__).parent / "data" / "nequip_test_dataset0.xyz"


def test_reads_nequip_writer_output_into_flat_contract():
    table, meta = read_nequip_xyz(FIXTURE, energy_unit="eV", force_unit="eV/Angstrom")
    assert list(table.columns) == COLUMNS
    assert table.structure_id.tolist() == ["nequip_test_dataset0:0"] * 2 + ["nequip_test_dataset0:1"] * 3
    assert table.element.tolist() == ["Li", "H", "O", "H", "H"]
    assert table.n_atoms.tolist() == [2, 2, 3, 3, 3]
    assert meta["frames"] == 2
    assert meta["energy_unit"] == "eV" and meta["force_unit"] == "eV/Angstrom"
    assert len(meta["source_sha256"]) == 64


def test_metrics_match_hand_calculation():
    # frame 0: dE = +1.0 over 2 atoms; force errors Li (1,0,0), H (0,2,0)
    # frame 1: dE = -0.3 over 3 atoms; force errors O 0, H 0, H (0,0,-2)
    table, _ = read_nequip_xyz(FIXTURE, energy_unit="eV", force_unit="eV/Angstrom")
    got = evaluate(table)
    assert got["structures"] == 2 and got["atoms"] == 5
    assert got["energy_mae_per_atom"] == pytest.approx((0.5 + 0.1) / 2)
    assert got["force_component_rmse"] == pytest.approx((9 / 15) ** .5)  # all 15 Cartesian components
    assert got["force_vector_mae"] == pytest.approx(5 / 5)  # norms 1, 2, 0, 0, 2
    assert got["force_vector_rmse"] == pytest.approx((9 / 5) ** .5)
    assert got["force_component_rmse_by_element"] == pytest.approx({"Li": (1 / 3) ** .5, "H": (8 / 9) ** .5, "O": 0.0})


@pytest.mark.parametrize("energy_unit,force_unit", [("", "eV/Angstrom"), ("eV", "  "), (None, "eV/Angstrom")])
def test_missing_units_are_rejected(energy_unit, force_unit):
    with pytest.raises(ValueError, match="unit"):
        read_nequip_xyz(FIXTURE, energy_unit=energy_unit, force_unit=force_unit)


def test_missing_reference_fields_are_rejected(tmp_path):
    bad = tmp_path / "no_reference.xyz"
    bad.write_text(
        "1\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.0 pbc="T T T"\n'
        "H 0.0 0.0 0.0 0.1 0.0 0.0\n"
    )
    with pytest.raises(ValueError, match="output_fields_from_original_dataset"):
        read_nequip_xyz(bad, energy_unit="eV", force_unit="eV/Angstrom")


def test_cli_writes_csv_and_metadata(tmp_path):
    out = tmp_path / "predictions.csv"
    main([str(FIXTURE), "--energy-unit", "eV", "--force-unit", "eV/Angstrom", "--out", str(out)])
    assert out.read_text().splitlines()[0] == ",".join(COLUMNS)
    meta = json.loads((tmp_path / "predictions.csv.meta.json").read_text())
    assert meta["source_format"] == "nequip.train.callbacks.TestTimeXYZFileWriter"
