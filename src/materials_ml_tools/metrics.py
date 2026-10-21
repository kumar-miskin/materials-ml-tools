from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"structure_id", "element", "n_atoms", "energy_true", "energy_pred", *(f"f{a}_{k}" for a in "xyz" for k in ("true", "pred"))}


def _validated(df: pd.DataFrame) -> pd.DataFrame:
    """Check the prediction-table contract and return a cleaned copy.

    The caller's frame is never mutated. The returned frame has a fresh
    positional index and float-valued numeric columns, so downstream groupby
    and positional lookups behave identically for any input index or dtype.
    """
    missing = sorted(REQUIRED - set(df.columns))
    if missing: raise ValueError(f"missing columns: {', '.join(missing)}")
    # A repeated structure energy is a contract, not a hint to keep the first row.
    # Otherwise a corrupted atom row silently changes (or disappears from) the score.
    if df.empty:
        raise ValueError("prediction table must not be empty")
    if df[["structure_id", "element"]].isna().any().any():
        raise ValueError("structure_id and element must not be null")
    numeric_columns = sorted(REQUIRED - {"structure_id", "element"})
    try:
        numbers = df[numeric_columns].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction table contains non-numeric measurements") from exc
    if not np.isfinite(numbers.to_numpy()).all():
        raise ValueError("prediction table contains non-finite measurements")
    counts = numbers["n_atoms"]
    if (counts < 1).any() or (counts != np.floor(counts)).any():
        raise ValueError("n_atoms must be a positive integer")
    # groupby().groups returns index labels; positional lookups below need a clean index
    df = df.reset_index(drop=True).copy()
    # Use the validated numeric values, not the original object/string columns.
    # Keep caller input untouched; numeric strings are accepted by validation.
    for column in numeric_columns:
        df[column] = numbers[column].to_numpy()
    counts = counts.reset_index(drop=True)
    for structure_id, group in df.groupby("structure_id", sort=False, observed=True):
        if len(group) != counts.loc[group.index].iloc[0] or counts.loc[group.index].nunique() != 1:
            raise ValueError(f"{structure_id}: n_atoms must equal the number of atom rows")
        if group[["energy_true", "energy_pred"]].nunique().ne(1).any():
            raise ValueError(f"{structure_id}: repeated structure energies disagree")
    return df


def _rms(values: np.ndarray) -> float:
    """Root mean square without squaring unscaled floating-point values."""
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean((values / scale) ** 2)))


def evaluate(df: pd.DataFrame) -> dict:
    df = _validated(df)
    structures = df.drop_duplicates("structure_id")
    energy_per_atom = (structures.energy_pred - structures.energy_true) / structures.n_atoms
    true = df[["fx_true", "fy_true", "fz_true"]].to_numpy(float)
    pred = df[["fx_pred", "fy_pred", "fz_pred"]].to_numpy(float)
    error = pred - true
    per_element = {}
    for element, rows in df.groupby("element", observed=True).groups.items():
        e = error[list(rows)]
        per_element[str(element)] = _rms(e)
    return {
        "structures": int(len(structures)),
        "atoms": int(len(df)),
        "energy_mae_per_atom": float(np.mean(np.abs(energy_per_atom))),
        "energy_rmse_per_atom": _rms(energy_per_atom.to_numpy()),
        "force_component_mae": float(np.mean(np.abs(error))),
        "force_component_rmse": _rms(error),
        "force_vector_mae": float(np.mean(np.hypot.reduce(error, axis=1))),
        "force_vector_rmse": float(np.sqrt(3) * _rms(error)),
        "force_component_rmse_by_element": per_element,
    }


def per_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Per-structure error breakdown for locating outlier structures.

    One row per structure: the repeated structure energy (validated to agree
    on every atom row), its error, the error per atom, and the structure's
    force component and vector RMSE. ``composition`` lists observed element
    counts alphabetically as ``<symbol><count>`` separated by spaces (unused
    categorical levels are not observations and are excluded). Structure IDs
    keep the ``<file stem>:<frame index>`` form the adapters assign, so rows
    join back to the source XYZ frames. The aggregate energy scores average
    per-structure errors (equal weight per structure), while the aggregate
    force scores pool all 3N Cartesian components across atoms, so larger
    structures contribute more force components. This table is where both
    weightings are inspected, not changed.
    """
    df = _validated(df)
    error = df[["fx_pred", "fy_pred", "fz_pred"]].to_numpy(float) - df[
        ["fx_true", "fy_true", "fz_true"]].to_numpy(float)
    rows = []
    for structure_id, group in df.groupby("structure_id", sort=False, observed=True):
        e = error[group.index]
        n_atoms = int(group["n_atoms"].iloc[0])
        energy_true = float(group["energy_true"].iloc[0])
        energy_pred = float(group["energy_pred"].iloc[0])
        observed = group["element"].value_counts()
        composition = " ".join(
            f"{element}{count}" for element, count in sorted(observed[observed > 0].items())
        )
        rows.append({
            "structure_id": structure_id,
            "n_atoms": n_atoms,
            "composition": composition,
            "energy_true": energy_true,
            "energy_pred": energy_pred,
            "energy_error": energy_pred - energy_true,
            "energy_error_per_atom": (energy_pred - energy_true) / n_atoms,
            "force_component_rmse": _rms(e),
            "force_vector_rmse": float(np.sqrt(3) * _rms(e)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Calculate explicit MLIP error conventions")
    p.add_argument("csv", type=Path); p.add_argument("--out", type=Path)
    p.add_argument("--per-structure", type=Path, dest="per_structure",
                   help="also write the per-structure error breakdown to this CSV path")
    args = p.parse_args()
    # Check all paths before reading or writing: otherwise a valid invocation
    # can destroy its source CSV or silently replace one output with another.
    paths = [("input CSV", args.csv)]
    if args.out:
        paths.append(("JSON output", args.out))
    if args.per_structure:
        paths.append(("per-structure output", args.per_structure))
    for i, (label, path) in enumerate(paths):
        for other_label, other in paths[:i]:
            if path.resolve() == other.resolve() or (
                path.exists() and other.exists() and path.samefile(other)
            ):
                p.error(f"{label} and {other_label} must be different files")
    table = pd.read_csv(args.csv)
    result = evaluate(table)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out: args.out.write_text(text + "\n")
    else: print(text)
    if args.per_structure:
        per_structure(table).to_csv(args.per_structure, index=False)

if __name__ == "__main__": main()
