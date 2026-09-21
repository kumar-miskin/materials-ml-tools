from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"structure_id", "element", "n_atoms", "energy_true", "energy_pred", *(f"f{a}_{k}" for a in "xyz" for k in ("true", "pred"))}


def evaluate(df: pd.DataFrame) -> dict:
    missing = sorted(REQUIRED - set(df.columns))
    if missing: raise ValueError(f"missing columns: {', '.join(missing)}")
    structures = df.drop_duplicates("structure_id")
    energy_per_atom = (structures.energy_pred - structures.energy_true) / structures.n_atoms
    true = df[["fx_true", "fy_true", "fz_true"]].to_numpy(float)
    pred = df[["fx_pred", "fy_pred", "fz_pred"]].to_numpy(float)
    error = pred - true
    per_element = {}
    for element, rows in df.groupby("element").groups.items():
        e = error[list(rows)]
        per_element[str(element)] = float(np.sqrt(np.mean(e**2)))
    return {
        "structures": int(len(structures)),
        "atoms": int(len(df)),
        "energy_mae_per_atom": float(np.mean(np.abs(energy_per_atom))),
        "force_component_rmse": float(np.sqrt(np.mean(error**2))),
        "force_vector_mae": float(np.mean(np.linalg.norm(error, axis=1))),
        "force_component_rmse_by_element": per_element,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Calculate explicit MLIP error conventions")
    p.add_argument("csv", type=Path); p.add_argument("--out", type=Path)
    args = p.parse_args(); result = evaluate(pd.read_csv(args.csv))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out: args.out.write_text(text + "\n")
    else: print(text)

if __name__ == "__main__": main()
