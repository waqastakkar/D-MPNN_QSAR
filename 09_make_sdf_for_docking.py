#!/usr/bin/env python3
"""Generate docking-ready 3D SDF from selected top compounds."""

from __future__ import annotations

import argparse
import csv
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def write_manifest(path: Path, args: argparse.Namespace, n_success: int, n_fail: int) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "rdkit": getattr(Chem, "__version__", "unknown"),
        "args": " ".join(sys.argv[1:]),
        "seed": args.seed,
        "n_success": n_success,
        "n_fail": n_fail,
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert top compounds to 3D SDF for docking")
    p.add_argument("--in_csv", default="screening/top_for_docking.csv")
    p.add_argument("--out_sdf", default="screening/top_for_docking.sdf")
    p.add_argument("--id_col", default="Catalog_NO")
    p.add_argument("--name_col", default="Name")
    p.add_argument("--smiles_col", default="canonical_smiles")
    p.add_argument("--n_confs", type=int, default=10)
    p.add_argument("--max_attempts", type=int, default=50)
    p.add_argument("--prune_rms_thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minimize", type=str2bool, default=True)
    p.add_argument("--out_failures_csv", default="screening/top_for_docking_3d_failures.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    in_csv = Path(args.in_csv)
    out_sdf = Path(args.out_sdf)
    fail_csv = Path(args.out_failures_csv)

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    fail_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    for c in [args.id_col, args.name_col, args.smiles_col]:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    failures: List[Dict[str, Any]] = []
    n_success = 0

    writer = Chem.SDWriter(str(out_sdf))
    for _, row in df.iterrows():
        cid = str(row.get(args.id_col, ""))
        nm = str(row.get(args.name_col, ""))
        smi = str(row.get(args.smiles_col, "")).strip()

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append({**row.to_dict(), "reason": "invalid_smiles"})
            continue

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(args.seed)
        params.pruneRmsThresh = float(args.prune_rms_thresh)
        if hasattr(params, "maxAttempts"):
            params.maxAttempts = int(args.max_attempts)
        elif hasattr(params, "maxAttemptsPerAtom"):
            params.maxAttemptsPerAtom = int(args.max_attempts)

        try:
            conf_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=int(args.n_confs),
                    params=params,
                    maxAttempts=int(args.max_attempts),
                )
            )
        except TypeError:
            conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(args.n_confs), params=params))
        if not conf_ids:
            failures.append({**row.to_dict(), "reason": "embed_fail"})
            continue

        if args.minimize:
            try:
                mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
                if mp is not None:
                    AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s", numThreads=1)
                else:
                    AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1)
            except Exception:
                try:
                    AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1)
                except Exception:
                    failures.append({**row.to_dict(), "reason": "mmff_fail"})
                    continue

        conf_energies = []
        for cid_conf in conf_ids:
            e = np.nan
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid_conf)
                if ff is not None:
                    e = float(ff.CalcEnergy())
            except Exception:
                pass
            conf_energies.append((cid_conf, e))

        best_conf = sorted(conf_energies, key=lambda x: (np.nan_to_num(x[1], nan=1e12), x[0]))[0][0]
        out_mol = Chem.Mol(mol)
        out_mol.RemoveAllConformers()
        out_mol.AddConformer(mol.GetConformer(best_conf), assignId=True)
        out_mol.SetProp("_Name", cid)

        for col in df.columns:
            val = row.get(col, "")
            out_mol.SetProp(str(col), "" if pd.isna(val) else str(val))
        out_mol.SetProp(args.id_col, cid)
        out_mol.SetProp(args.name_col, nm)

        writer.write(out_mol)
        n_success += 1

    writer.close()

    pd.DataFrame(failures).to_csv(fail_csv, index=False)
    write_manifest(out_sdf.parent / "run_manifest_make_sdf.csv", args, n_success=n_success, n_fail=len(failures))

    print(f"[DONE] Wrote SDF: {out_sdf}")
    print(f"[DONE] Wrote failures CSV: {fail_csv}")


if __name__ == "__main__":
    main()
