from pathlib import Path
import pytest
import pandas as pd
from materials_ml_tools.metrics import evaluate


def test_explicit_metric_conventions():
    frame = pd.DataFrame({
        "structure_id": ["a", "a"], "element": ["Li", "H"], "n_atoms": [2, 2],
        "energy_true": [10., 10.], "energy_pred": [12., 12.],
        "fx_true": [0., 0.], "fy_true": [0., 0.], "fz_true": [0., 0.],
        "fx_pred": [1., 0.], "fy_pred": [0., 2.], "fz_pred": [0., 0.],
    })
    got = evaluate(frame)
    assert got["energy_mae_per_atom"] == 1.0
    assert round(got["force_component_rmse"], 6) == round((5 / 6) ** .5, 6)
    assert got["force_vector_mae"] == 1.5
    assert set(got["force_component_rmse_by_element"]) == {"H", "Li"}


def test_vector_rmse_is_sqrt3_times_component_rmse():
    frame = pd.DataFrame({
        "structure_id": ["a", "a", "b"], "element": ["Li", "H", "H"], "n_atoms": [2, 2, 1],
        "energy_true": [10., 10., 3.], "energy_pred": [12., 12., 3.],
        "fx_true": [0., 0., 0.], "fy_true": [0., 0., 0.], "fz_true": [0., 0., 0.],
        "fx_pred": [1., 0., .3], "fy_pred": [0., 2., -.4], "fz_pred": [0., 0., 1.2],
    })
    got = evaluate(frame)
    assert abs(got["force_vector_rmse"] - 3 ** .5 * got["force_component_rmse"]) < 1e-12


def test_non_default_index_does_not_misalign_elements():
    frame = pd.DataFrame({
        "structure_id": ["a", "a"], "element": ["Li", "H"], "n_atoms": [2, 2],
        "energy_true": [0., 0.], "energy_pred": [0., 0.],
        "fx_true": [0., 0.], "fy_true": [0., 0.], "fz_true": [0., 0.],
        "fx_pred": [3., 0.], "fy_pred": [0., 0.], "fz_pred": [0., 0.],
    }, index=[10, 20])
    got = evaluate(frame)
    assert got["force_component_rmse_by_element"] == {"H": 0.0, "Li": 3.0 ** .5}


def test_repeated_structure_energy_must_agree():
    frame = pd.read_csv("examples/predictions.csv")
    frame.loc[1, "energy_pred"] = 9999
    with pytest.raises(ValueError, match="repeated structure energies disagree"):
        evaluate(frame)


@pytest.mark.parametrize("value", [0, 1, 2.5, 999])
def test_n_atoms_must_match_rows(value):
    frame = pd.read_csv("examples/predictions.csv")
    frame["n_atoms"] = frame["n_atoms"].astype(float)
    frame.loc[0, "n_atoms"] = value
    with pytest.raises(ValueError, match="n_atoms"):
        evaluate(frame)


def test_nonfinite_measurement_fails_before_json_serialization():
    frame = pd.read_csv("examples/predictions.csv")
    frame.loc[0, "fx_pred"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(frame)


def test_empty_prediction_table_fails():
    frame = pd.read_csv("examples/predictions.csv")
    with pytest.raises(ValueError, match="empty"):
        evaluate(frame.iloc[:0])


def test_numeric_strings_use_validated_values_without_changing_input():
    frame = pd.read_csv("examples/predictions.csv")
    expected = evaluate(frame)
    numeric = frame.columns.difference(["structure_id", "element"])
    frame[numeric] = frame[numeric].astype(str)
    original = frame.copy(deep=True)
    assert evaluate(frame) == expected
    pd.testing.assert_frame_equal(frame, original)


def test_unused_categorical_structure_and_element_are_not_samples():
    frame = pd.read_csv("examples/predictions.csv")
    expected = evaluate(frame)
    for column in ("structure_id", "element"):
        frame[column] = pd.Categorical(
            frame[column], categories=[*frame[column].unique(), "unused"]
        )
    original = frame.copy(deep=True)
    assert evaluate(frame) == expected
    assert "unused" not in evaluate(frame)["force_component_rmse_by_element"]
    pd.testing.assert_frame_equal(frame, original)


def test_numeric_strings_cannot_hide_disagreeing_energy():
    frame = pd.read_csv("examples/predictions.csv").astype(str)
    frame.loc[1, "energy_pred"] = "999"
    with pytest.raises(ValueError, match="repeated structure energies disagree"):
        evaluate(frame)


def test_energy_rmse_and_force_mae_conventions():
    frame=pd.DataFrame({
        'structure_id':['a','a','b'],'element':['H','H','Li'],'n_atoms':[2,2,1],
        'energy_true':[0.,0.,0.],'energy_pred':[2.,2.,3.],
        'fx_true':[0.,0.,0.],'fy_true':[0.,0.,0.],'fz_true':[0.,0.,0.],
        'fx_pred':[3.,0.,0.],'fy_pred':[0.,4.,0.],'fz_pred':[0.,0.,2.],
    })
    got=evaluate(frame)
    # Energies use equal structure weighting: errors per atom 1 and 3.
    assert got['energy_rmse_per_atom']==pytest.approx(5**.5)
    assert got['force_component_mae']==pytest.approx(9/9)
    # No fixed conversion between component MAE and vector MAE.
    assert got['force_vector_mae']==pytest.approx(3)


def test_new_metrics_are_zero_for_exact_predictions():
    frame=pd.read_csv('examples/predictions.csv')
    for key in ('energy','fx','fy','fz'):frame[key+'_pred']=frame[key+'_true']
    got=evaluate(frame)
    assert got['energy_rmse_per_atom']==0
    assert got['force_component_mae']==0


def test_per_structure_breakdown_values():
    from materials_ml_tools.metrics import per_structure
    frame = pd.DataFrame({
        "structure_id": ["a", "a", "b"], "element": ["Li", "H", "H"], "n_atoms": [2, 2, 1],
        "energy_true": [10., 10., 3.], "energy_pred": [12., 12., 4.],
        "fx_true": [0., 0., 0.], "fy_true": [0., 0., 0.], "fz_true": [0., 0., 0.],
        "fx_pred": [1., 0., .3], "fy_pred": [0., 2., -.4], "fz_pred": [0., 0., 1.2],
    })
    got = per_structure(frame)
    assert list(got["structure_id"]) == ["a", "b"]
    a, b = got.iloc[0], got.iloc[1]
    assert a["n_atoms"] == 2 and a["composition"] == "H1 Li1"
    assert a["energy_error"] == 2.0 and a["energy_error_per_atom"] == 1.0
    assert a["force_component_rmse"] == pytest.approx((5 / 6) ** .5)
    assert a["force_vector_rmse"] == pytest.approx((5 / 2) ** .5)
    assert b["n_atoms"] == 1 and b["composition"] == "H1"
    assert b["energy_error"] == 1.0 and b["energy_error_per_atom"] == 1.0
    assert b["force_component_rmse"] == pytest.approx(((0.09 + 0.16 + 1.44) / 3) ** .5)
    assert b["force_vector_rmse"] == pytest.approx((0.09 + 0.16 + 1.44) ** .5)


def test_per_structure_uses_the_same_validation_contract():
    from materials_ml_tools.metrics import per_structure
    frame = pd.read_csv("examples/predictions.csv")
    with pytest.raises(ValueError, match="empty"):
        per_structure(frame.iloc[:0])
    frame.loc[1, "energy_pred"] = 9999
    with pytest.raises(ValueError, match="repeated structure energies disagree"):
        per_structure(frame)


def test_per_structure_excludes_unused_categorical_elements():
    from materials_ml_tools.metrics import per_structure
    frame = pd.read_csv("examples/predictions.csv")
    frame["element"] = pd.Categorical(frame["element"], categories=[*frame["element"].unique(), "Xx"])
    got = per_structure(frame)
    assert all("Xx" not in composition for composition in got["composition"])


def test_per_structure_cli_writes_breakdown(tmp_path):
    from materials_ml_tools.metrics import main
    out = tmp_path / "per_structure.csv"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", ["metrics", "examples/predictions.csv", "--per-structure", str(out)])
        main()
    written = pd.read_csv(out)
    assert list(written.columns) == [
        "structure_id", "n_atoms", "composition", "energy_true", "energy_pred",
        "energy_error", "energy_error_per_atom", "force_component_rmse", "force_vector_rmse",
    ]
    assert len(written) == pd.read_csv("examples/predictions.csv")["structure_id"].nunique()


@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
@pytest.mark.parametrize("breakdown", [False, True])
def test_finite_extreme_errors_do_not_overflow_or_underflow(magnitude, breakdown):
    """Squaring first loses finite, representable norms at either float extreme."""
    from materials_ml_tools.metrics import per_structure
    frame = pd.DataFrame({
        "structure_id": ["a"], "element": ["H"], "n_atoms": [1],
        "energy_true": [0.], "energy_pred": [magnitude],
        **{f"f{axis}_true": [0.] for axis in "xyz"},
        **{f"f{axis}_pred": [magnitude] for axis in "xyz"},
    })
    original = frame.copy(deep=True)
    if breakdown:
        got = per_structure(frame).iloc[0]
    else:
        got = evaluate(frame)
        assert got["energy_rmse_per_atom"] == pytest.approx(magnitude, abs=0)
        assert got["force_vector_mae"] == pytest.approx(3 ** .5 * magnitude, abs=0)
        assert got["force_component_rmse_by_element"]["H"] == pytest.approx(magnitude, abs=0)
    assert got["force_component_rmse"] == pytest.approx(magnitude, abs=0)
    assert got["force_vector_rmse"] == pytest.approx(3 ** .5 * magnitude, abs=0)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
def test_scaled_rms_preserves_mixed_signs_and_zero_components(magnitude):
    from materials_ml_tools.metrics import _rms
    import numpy as np
    assert _rms(np.array([magnitude, -magnitude, 0.])) == pytest.approx(
        magnitude * (2 / 3) ** .5, abs=0
    )
    assert _rms(np.zeros((2, 3))) == 0.


@pytest.mark.parametrize("collision", ["input-json", "input-csv", "outputs"])
def test_metrics_cli_rejects_path_collisions_before_writing(tmp_path, monkeypatch, collision):
    from materials_ml_tools.metrics import main
    source = tmp_path / "predictions.csv"
    source.write_bytes(Path("examples/predictions.csv").read_bytes())
    original = source.read_bytes()
    out, breakdown = tmp_path / "metrics.json", tmp_path / "breakdown.csv"
    if collision == "input-json":
        out = source
    elif collision == "input-csv":
        breakdown = source
    else:
        breakdown = out
    monkeypatch.setattr("sys.argv", ["metrics", str(source), "--out", str(out),
                                     "--per-structure", str(breakdown)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert source.read_bytes() == original
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "breakdown.csv").exists()


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_metrics_cli_rejects_filesystem_alias_of_input(tmp_path, monkeypatch, alias_kind):
    import os
    from materials_ml_tools.metrics import main
    source = tmp_path / "predictions.csv"
    source.write_bytes(Path("examples/predictions.csv").read_bytes())
    original = source.read_bytes()
    alias = tmp_path / "alias.json"
    if alias_kind == "symlink":
        alias.symlink_to(source)
    else:
        os.link(source, alias)
    monkeypatch.setattr("sys.argv", ["metrics", str(source), "--out", str(alias)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert source.read_bytes() == alias.read_bytes() == original


def test_metrics_cli_writes_distinct_json_and_csv_outputs(tmp_path, monkeypatch):
    import json
    from materials_ml_tools.metrics import main
    out, breakdown = tmp_path / "metrics.json", tmp_path / "breakdown.csv"
    monkeypatch.setattr("sys.argv", ["metrics", "examples/predictions.csv", "--out", str(out),
                                     "--per-structure", str(breakdown)])
    main()
    assert json.loads(out.read_text())["atoms"] == 2
    assert list(pd.read_csv(breakdown)["structure_id"]) == ["example-1"]
