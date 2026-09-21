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

- `energy_mae_per_atom`: mean absolute error of structure energy divided by atom count
- `force_component_rmse`: RMSE over all Cartesian force components
- `force_vector_mae`: mean Euclidean norm of each atom's force-error vector
- per-element component RMSE, to expose composition-dependent failures

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
