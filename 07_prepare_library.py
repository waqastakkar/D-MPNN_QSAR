#!/usr/bin/env python3
"""Prepare and QC a screening library from CSV.

This script preserves all original columns and adds standardized SMILES/QC fields.
"""

from __future__ import annotations

import argparse
import csv
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover
    rdMolStandardize = None

NATURE = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}


@dataclass
class ProcessResult:
    ok: bool
    canonical_smiles: str
    n_frags: int
    kept_fragment_smiles: str
    removed_salt: bool
    reason: str
    mw: float
    logp: float


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def setup_plot_style(args: argparse.Namespace) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "svg.fonttype": "none",
            "font.size": args.base_font,
            "axes.titlesize": args.title_size,
            "axes.labelsize": args.label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
        }
    )


def auto_smiles_col(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"SMILES column '{requested}' not found.")
        return requested
    for c in ["rdkit_canonical_smiles", "SMILES", "smiles"]:
        if c in df.columns:
            return c
    raise ValueError("No SMILES column found. Expected one of rdkit_canonical_smiles, SMILES, smiles.")


def _largest_fragment_manual(mol: Chem.Mol) -> Optional[Chem.Mol]:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return sorted(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()), reverse=True)[0]


def _largest_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if rdMolStandardize is not None:
        try:
            chooser = rdMolStandardize.LargestFragmentChooser()
            out = chooser.choose(mol)
            if out is not None:
                return out
        except Exception:
            pass
    return _largest_fragment_manual(mol)


def process_smiles(smiles: str, remove_salts: bool, largest_fragment: bool, force_canonical: bool) -> ProcessResult:
    smi = str(smiles).strip()
    if not smi:
        return ProcessResult(False, "", 0, "", False, "empty_smiles", np.nan, np.nan)

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ProcessResult(False, "", 0, "", False, "invalid_smiles", np.nan, np.nan)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return ProcessResult(False, "", 0, "", False, "sanitize_fail", np.nan, np.nan)

    n_frags = len(Chem.GetMolFrags(mol))
    frag_mol = mol
    if remove_salts and largest_fragment:
        frag_mol = _largest_fragment(mol)
        if frag_mol is None:
            return ProcessResult(False, "", n_frags, "", True, "empty_after_fragment", np.nan, np.nan)

    if frag_mol is None or frag_mol.GetNumAtoms() == 0:
        return ProcessResult(False, "", n_frags, "", bool(n_frags > 1), "empty_after_fragment", np.nan, np.nan)

    kept = Chem.MolToSmiles(frag_mol, canonical=True, isomericSmiles=True)
    if not kept:
        return ProcessResult(False, "", n_frags, "", bool(n_frags > 1), "empty_after_fragment", np.nan, np.nan)

    canon = Chem.MolToSmiles(frag_mol, canonical=True, isomericSmiles=True) if force_canonical else kept
    try:
        mw = float(Descriptors.MolWt(frag_mol))
        logp = float(Crippen.MolLogP(frag_mol))
    except Exception:
        mw, logp = np.nan, np.nan

    return ProcessResult(True, canon, n_frags, kept, bool(n_frags > 1), "", mw, logp)


def write_manifest(path: Path, args: argparse.Namespace) -> None:
    rows = [{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "rdkit": getattr(Chem, "__version__", "unknown"),
        "args": " ".join(sys.argv[1:]),
        "seed": 0,
    }]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_qc_plot(clean_df: pd.DataFrame, counts: Dict[str, int], out_svg: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(clean_df["MW"].dropna(), bins=30, color=NATURE["blue"], alpha=0.9, edgecolor="black")
    axes[0].set_title("MW Distribution")
    axes[0].set_xlabel("Molecular Weight")
    axes[0].set_ylabel("Count")

    axes[1].hist(clean_df["LogP"].dropna(), bins=30, color=NATURE["green"], alpha=0.9, edgecolor="black")
    axes[1].set_title("LogP Distribution")
    axes[1].set_xlabel("LogP")
    axes[1].set_ylabel("Count")

    labels = ["kept", "failed", "dedup_removed"]
    vals = [counts.get(k, 0) for k in labels]
    colors = [NATURE["blue"], NATURE["orange"], NATURE["purple"]]
    axes[2].bar(labels, vals, color=colors, edgecolor="black")
    axes[2].set_title("Library Processing Summary")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare and clean library CSV for screening")
    p.add_argument("--in_csv", default="library.csv")
    p.add_argument("--outdir", default="screening")
    p.add_argument("--id_col", default="Catalog_NO")
    p.add_argument("--name_col", default="Name")
    p.add_argument("--smiles_col", default="auto")
    p.add_argument("--dedup_on", choices=["canonical_smiles", "id", "both"], default="canonical_smiles")
    p.add_argument("--remove_salts", type=str2bool, default=True)
    p.add_argument("--largest_fragment", type=str2bool, default=True)
    p.add_argument("--force_rdkit_canonical", type=str2bool, default=True)
    p.add_argument("--drop_invalid", type=str2bool, default=True)

    p.add_argument("--title_size", type=float, default=18)
    p.add_argument("--label_size", type=float, default=16)
    p.add_argument("--tick_size", type=float, default=14)
    p.add_argument("--legend_size", type=float, default=14)
    p.add_argument("--base_font", type=float, default=14)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots" / "svg").mkdir(parents=True, exist_ok=True)

    setup_plot_style(args)

    print(f"[INFO] Loading {args.in_csv}")
    df = pd.read_csv(args.in_csv)

    for must in [args.id_col, args.name_col]:
        if must not in df.columns:
            raise ValueError(f"Required column missing: {must}")

    smiles_col = auto_smiles_col(df, args.smiles_col)
    print(f"[INFO] Using SMILES column: {smiles_col}")

    proc_rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        orig = row.to_dict()
        res = process_smiles(
            smiles=row.get(smiles_col, ""),
            remove_salts=args.remove_salts,
            largest_fragment=args.largest_fragment,
            force_canonical=args.force_rdkit_canonical,
        )
        merged = dict(orig)
        merged.update(
            {
                "input_smiles_col": smiles_col,
                "input_smiles": row.get(smiles_col, ""),
                "canonical_smiles": res.canonical_smiles,
                "rdkit_canonical_smiles": res.canonical_smiles,
                "n_frags": res.n_frags,
                "kept_fragment_smiles": res.kept_fragment_smiles,
                "removed_salt": bool(res.removed_salt),
                "is_valid": bool(res.ok),
                "MW": res.mw,
                "LogP": res.logp,
            }
        )
        if res.ok:
            proc_rows.append(merged)
        else:
            fail = dict(merged)
            fail["reason"] = res.reason
            fail_rows.append(fail)

    clean_df = pd.DataFrame(proc_rows)
    fail_df = pd.DataFrame(fail_rows)

    dedup_removed_df = pd.DataFrame(columns=clean_df.columns)
    if not clean_df.empty:
        before = len(clean_df)
        if args.dedup_on == "canonical_smiles":
            key_cols = ["canonical_smiles"]
        elif args.dedup_on == "id":
            key_cols = [args.id_col]
        else:
            key_cols = ["canonical_smiles", args.id_col]

        keep_mask = ~clean_df.duplicated(subset=key_cols, keep="first")
        dedup_removed_df = clean_df.loc[~keep_mask].copy()
        clean_df = clean_df.loc[keep_mask].copy()
        print(f"[INFO] Deduplicated {before - len(clean_df)} rows on {key_cols}")

    if args.drop_invalid:
        pass
    else:
        if not fail_df.empty:
            keep_cols = [c for c in clean_df.columns if c in fail_df.columns]
            repaired = fail_df[keep_cols].copy()
            clean_df = pd.concat([clean_df, repaired], ignore_index=True)

    clean_df.to_csv(outdir / "library_clean.csv", index=False)
    fail_df.to_csv(outdir / "library_failures.csv", index=False)
    dedup_removed_df.to_csv(outdir / "library_dedup_removed.csv", index=False)

    counts = {"kept": int(len(clean_df)), "failed": int(len(fail_df)), "dedup_removed": int(len(dedup_removed_df))}
    make_qc_plot(clean_df, counts, outdir / "plots" / "svg" / "library_qc.svg")
    write_manifest(outdir / "run_manifest_prepare_library.csv", args)

    print(f"[DONE] Wrote: {outdir / 'library_clean.csv'}")
    print(f"[DONE] Wrote: {outdir / 'library_failures.csv'}")
    print(f"[DONE] Wrote: {outdir / 'library_dedup_removed.csv'}")
    print(f"[DONE] Wrote: {outdir / 'plots' / 'svg' / 'library_qc.svg'}")


if __name__ == "__main__":
    main()
