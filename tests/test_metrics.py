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
