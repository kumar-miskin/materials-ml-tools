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
