# Materials ML Tools

Small, model-agnostic utilities for evaluating published machine-learning interatomic potentials. The first module makes error reporting more comparable across MACE, NequIP, Allegro, MatGL and FAIR-Chem workflows by calculating the same metrics from a flat prediction table.

## Why this exists

Force errors are often reported as either component-wise RMSE or vector-magnitude error. Those are different quantities and can change a model comparison. Energy errors can also be quoted per structure or per atom. This package names the convention in every output field instead of collapsing them into a single ambiguous "MAE".

## Input

One CSV row per atom:

```text
structure_id,element,n_atoms,energy_true,energy_pred,fx_true,fy_true,fz_true,fx_pred,fy_pred,fz_pred
```

Energy is repeated for atoms in the same structure. The CLI de-duplicates structures before calculating energy metrics.

```bash
python -m materials_ml_tools.metrics examples/predictions.csv
```

## Output conventions

For structure s with N_s atoms and force error e_i = F_pred,i - F_true,i on atom i (N atoms in total):

| Field | Formula | Aggregated over |
|---|---|---|
| `energy_mae_per_atom` | mean_s \|E_pred,s - E_true,s\| / N_s | structures |
| `force_component_rmse` | sqrt( sum_i sum_{a in x,y,z} e_ia^2 / (3N) ) | all 3N Cartesian components |
| `force_vector_mae` | mean_i \|\|e_i\|\| | atoms (Euclidean norm per atom) |
| `force_vector_rmse` | sqrt( sum_i \|\|e_i\|\|^2 / N ) | atoms |
| `force_component_rmse_by_element` | component RMSE restricted to atoms of each element | components, per element |

`force_vector_rmse` is always sqrt(3) times `force_component_rmse` for the same errors, because the numerator is identical and only the denominator changes (N vs 3N). There is no fixed conversion between component MAE and vector MAE, so an MAE should never be compared across papers or codes without knowing which one it is. NequIP's `misc/parity_plot.py`, for example, flattens the Nx3 force array before averaging, so its force MAE/RMSE are component-wise.

## NequIP adapter

`materials_ml_tools.adapters` converts the extended-XYZ file written by `nequip.train.callbacks.TestTimeXYZFileWriter` into the input table above. Configure the writer to keep the reference values:

```yaml
callbacks:
  - _target_: nequip.train.callbacks.TestTimeXYZFileWriter
    out_file: ${hydra:runtime.output_dir}/test
    output_fields_from_original_dataset: [total_energy, forces]
    chemical_symbols: ${chemical_symbols}
```

Then:

```bash
pip install -e '.[nequip]'
python -m materials_ml_tools.adapters test_dataset0.xyz --energy-unit eV --force-unit eV/Angstrom --out predictions.csv
python -m materials_ml_tools.metrics predictions.csv
```

The XYZ file does not record units, so `--energy-unit` and `--force-unit` are required and are written with the source SHA-256 and frame count to `predictions.csv.meta.json`. Structure IDs are `<file stem>:<frame index>` in file order. Files missing `original_dataset_energy` or `original_dataset_forces` are rejected rather than guessed.

Units are inherited from the input and must be recorded by the caller. The tool does not imply benchmark comparability when datasets, splits, reference methods, or units differ.

## Public ecosystem reviewed

The interface is deliberately framework-neutral. Useful public projects include:

- MACE: https://github.com/ACEsuit/mace
- NequIP: https://github.com/mir-group/nequip
- Allegro: https://github.com/mir-group/allegro
- MatGL: https://github.com/materialsvirtuallab/matgl
- FAIR-Chem: https://github.com/facebookresearch/fairchem
- ASE: https://gitlab.com/ase/ase
- pymatgen: https://github.com/materialsproject/pymatgen

This repository uses only public APIs, documentation and example-shaped data. It contains no unpublished research data.

## MACE evaluation adapter

After `mace_eval_configs --configs input.xyz --model model.model --output output.xyz`,
convert its extended XYZ to the same metrics input. MACE writes predictions to
`MACE_energy` and `MACE_forces` by default (`--info_prefix` changes the prefix).
The reference energy and force field names depend on the input data, so name
both explicitly instead of guessing them:

```bash
pip install -e '.[dev]'
materials-ml-mace-xyz output.xyz --energy-key REF_energy --forces-key REF_forces \
    --energy-unit eV --force-unit eV/Angstrom --out predictions.csv
python -m materials_ml_tools.metrics predictions.csv
```

Substitute the *actual* reference field names in your XYZ; `REF_energy` and
`REF_forces` are examples, not defaults. For a custom MACE `--info_prefix`, pass
`--prediction-prefix` with the same value. The adapter requires all four fields,
finite scalar energies, and finite Nx3 forces per frame. It records keys, units,
file SHA-256 and frame count beside the CSV. It does not run MACE or infer units,
model provenance, split membership, or reference methods; do not compare scores
across runs without checking those conditions.
