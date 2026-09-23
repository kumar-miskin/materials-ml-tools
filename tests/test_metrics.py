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
