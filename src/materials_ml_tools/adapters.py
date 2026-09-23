"""Adapters from public MLIP output formats to the flat prediction table.

The flat table is the input contract of :func:`materials_ml_tools.metrics.evaluate`:
one row per atom, with the structure energy repeated on each atom row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import REQUIRED

COLUMNS = [
    "structure_id", "element", "n_atoms", "energy_true", "energy_pred",
    "fx_true", "fy_true", "fz_true", "fx_pred", "fy_pred", "fz_pred",
]
assert set(COLUMNS) == REQUIRED

REF_ENERGY = "original_dataset_energy"
REF_FORCES = "original_dataset_forces"


def _require_unit(name: str, value: str | None) -> str:
    if value is None or not str(value).strip():
        raise ValueError(
            f"{name} is required: NequIP XYZ output does not record units, "
            "so they must come from the training config or dataset documentation"
        )
    return str(value).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_nequip_xyz(path: str | Path, *, energy_unit: str, force_unit: str) -> tuple[pd.DataFrame, dict]:
    """Read a file written by ``nequip.train.callbacks.TestTimeXYZFileWriter``.

    NequIP writes model predictions as the ASE calculator results and the
    reference values as ``original_dataset_energy`` (frame info) and
    ``original_dataset_forces`` (per-atom arrays). The writer must be configured
    with ``output_fields_from_original_dataset: [total_energy, forces]``.

    Returns the flat prediction table and a metadata dict with units, source
    path, SHA-256 and frame count. Structure IDs are ``<file stem>:<frame index>``
    in file order, which is the order the writer appended test batches.
    """
    import ase.io  # optional dependency: pip install 'materials-ml-tools[nequip]'

    energy_unit = _require_unit("energy_unit", energy_unit)
    force_unit = _require_unit("force_unit", force_unit)
    path = Path(path)
    rows = []
    n_frames = 0
    for index, frame in enumerate(ase.io.iread(str(path), index=":", format="extxyz")):
        n_frames += 1
        where = f"{path.name} frame {index}"
        if REF_ENERGY not in frame.info or REF_FORCES not in frame.arrays:
            raise ValueError(
                f"{where}: missing {REF_ENERGY!r} or {REF_FORCES!r}; set "
                "output_fields_from_original_dataset: [total_energy, forces] on the writer"
            )
        if frame.calc is None:
            raise ValueError(f"{where}: no predicted energy/forces found")
        try:
            energy_pred = float(frame.get_potential_energy())
            forces_pred = np.asarray(frame.get_forces(), dtype=float)
        except Exception as exc:  # ASE raises PropertyNotImplementedError
            raise ValueError(f"{where}: no predicted energy/forces found") from exc
        forces_true = np.asarray(frame.arrays[REF_FORCES], dtype=float)
        n_atoms = len(frame)
        if forces_true.shape != (n_atoms, 3) or forces_pred.shape != (n_atoms, 3):
            raise ValueError(f"{where}: force arrays must have shape ({n_atoms}, 3)")
        energy_true = float(frame.info[REF_ENERGY])
        structure_id = f"{path.stem}:{index}"
        for symbol, f_true, f_pred in zip(frame.get_chemical_symbols(), forces_true, forces_pred):
            rows.append((structure_id, symbol, n_atoms, energy_true, energy_pred, *f_true, *f_pred))
    if not rows:
        raise ValueError(f"{path}: no frames found")
    frame_table = pd.DataFrame(rows, columns=COLUMNS)
    metadata = {
        "source_format": "nequip.train.callbacks.TestTimeXYZFileWriter",
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "frames": n_frames,
        "energy_unit": energy_unit,
        "force_unit": force_unit,
    }
    return frame_table, metadata


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert NequIP test-time XYZ output to the flat prediction table")
    p.add_argument("xyz", type=Path)
    p.add_argument("--energy-unit", required=True, help="e.g. eV; must match the training data")
    p.add_argument("--force-unit", required=True, help="e.g. eV/Angstrom; must match the training data")
    p.add_argument("--out", type=Path, required=True, help="CSV path; metadata is written to <out>.meta.json")
    args = p.parse_args(argv)
    table, metadata = read_nequip_xyz(args.xyz, energy_unit=args.energy_unit, force_unit=args.force_unit)
    table.to_csv(args.out, index=False)
    args.out.with_suffix(args.out.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
