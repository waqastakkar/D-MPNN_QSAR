#!/usr/bin/env python3
"""Prepare ChEMBL QSAR datasets with full QC, reproducibility, and plotting."""

from __future__ import annotations

import argparse
import hashlib
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "RDKit is required for this script. Please install rdkit in the current environment."
    ) from exc

NATURE_PALETTE: Dict[str, str] = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}


@dataclass
class PipelineCounts:
    raw_rows: int = 0
    dropped_missing_smiles_value: int = 0
    dropped_non_ic50: int = 0
    dropped_unknown_units: int = 0
    dropped_non_positive_ic50: int = 0
    dropped_invalid_smiles: int = 0
    after_relation_handling: int = 0
    after_standardization: int = 0
    after_rdkit: int = 0
    after_dedup: int = 0


@dataclass
class PipelineContext:
    args: argparse.Namespace
    outdir: Path
    seeds: List[int]
    counts: PipelineCounts
    ic50_filter_possible: bool
    relation_strategy_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ChEMBL QSAR dataset for regression/classification with full QC outputs."
    )
    parser.add_argument("--input", required=True, type=str, help="Path to raw ChEMBL CSV file")
    parser.add_argument("--outdir", default="qsar_out", type=str, help="Output directory")
    parser.add_argument(
        "--task",
        default="regression",
        choices=["regression", "classification", "both"],
        help="Modeling task mode",
    )
    parser.add_argument(
        "--activity_threshold_uM",
        default=1.0,
        type=float,
        help="Classification threshold in uM (Active if IC50_uM <= threshold)",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        type=str,
        help="Comma-separated seeds for downstream reproducibility bookkeeping",
    )
    parser.add_argument(
        "--dedup_strategy",
        default="median",
        choices=["median", "mean", "keep_best"],
        help="How to aggregate duplicated canonical SMILES",
    )
    parser.add_argument(
        "--keep_relations",
        default="eq_only",
        choices=["eq_only", "censor_to_value", "drop_non_eq"],
        help="How to handle relation operators in activity values",
    )
    parser.add_argument(
        "--svg_only",
        default=True,
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        help="Keep SVG output only by default (set false if wanted).",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        default=False,
        help="Additionally save PNG versions of plots.",
    )
    parser.add_argument("--title_size", default=18, type=int)
    parser.add_argument("--label_size", default=16, type=int)
    parser.add_argument("--tick_size", default=14, type=int)
    parser.add_argument("--legend_size", default=14, type=int)
    parser.add_argument("--base_font", default=14, type=int)
    return parser.parse_args()


def parse_seed_list(seeds: str) -> List[int]:
    parts = [p.strip() for p in seeds.split(",") if p.strip()]
    if not parts:
        raise ValueError("--seeds must contain at least one integer.")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid --seeds value '{seeds}'. Use comma-separated integers.") from exc


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def map_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates: Dict[str, Sequence[str]] = {
        "molecule_chembl_id": ["molecule_chembl_id", "chembl_id", "moleculechemblid"],
        "canonical_smiles": ["canonical_smiles", "smiles", "molecule_smiles"],
        "standard_value": ["standard_value", "value", "standardvalue"],
        "standard_units": ["standard_units", "units", "standardunits"],
        "standard_relation": ["standard_relation", "relation", "standardrelation"],
        "standard_type": ["standard_type", "type", "standardtype"],
    }
    normalized_to_raw = {canonicalize_name(c): c for c in df.columns}
    mapped: Dict[str, str] = {}
    missing_required: List[str] = []

    required = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_relation",
    ]

    for key, aliases in candidates.items():
        found = None
        for alias in aliases:
            norm = canonicalize_name(alias)
            if norm in normalized_to_raw:
                found = normalized_to_raw[norm]
                break
        if found is not None:
            mapped[key] = found
        elif key in required:
            missing_required.append(key)

    if missing_required:
        raise ValueError(
            "Missing required columns after robust mapping: "
            + ", ".join(missing_required)
            + ". Found columns: "
            + ", ".join(df.columns.astype(str))
        )
    return mapped


def standardize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    work = df.copy()
    for internal, raw_col in mapping.items():
        work[internal] = work[raw_col]
    return work


def normalize_units(units: Any) -> str:
    if pd.isna(units):
        return ""
    u = str(units).strip().replace("μ", "u").replace("µ", "u")
    return u.lower()


def convert_to_nM(value: float, unit_norm: str) -> Optional[float]:
    if unit_norm == "nm":
        return value
    if unit_norm == "um":
        return value * 1_000.0
    if unit_norm == "m":
        return value * 1_000_000_000.0
    return None


def apply_relation_filter(df: pd.DataFrame, keep_relations: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    work = df.copy()
    work["standard_relation"] = work["standard_relation"].astype(str).str.strip()

    before = (
        work["standard_relation"]
        .value_counts(dropna=False)
        .rename_axis("relation")
        .reset_index(name="count_before")
    )

    if keep_relations in {"eq_only", "drop_non_eq"}:
        kept = work[work["standard_relation"] == "="].copy()
        note = "Kept only '=' relation rows."
    elif keep_relations == "censor_to_value":
        kept = work[work["standard_relation"].isin(["=", "<", ">", "<=", ">="])].copy()
        note = "Retained '=', '<', '>', '<=', '>=' and censored non-'=' by using numeric value as-is."
    else:
        raise ValueError(f"Unsupported relation strategy: {keep_relations}")

    after = (
        kept["standard_relation"]
        .value_counts(dropna=False)
        .rename_axis("relation")
        .reset_index(name="count_after")
    )
    relation_report = before.merge(after, on="relation", how="outer").fillna(0)
    relation_report[["count_before", "count_after"]] = relation_report[["count_before", "count_after"]].astype(int)
    return kept, relation_report, before, note


def unit_filter_and_convert(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["unit_norm"] = work["standard_units"].map(normalize_units)

    units_before = work["unit_norm"].value_counts(dropna=False).rename_axis("unit_norm").reset_index(name="count_before")

    work["standard_value"] = pd.to_numeric(work["standard_value"], errors="coerce")
    work = work.dropna(subset=["standard_value"]).copy()

    work["IC50_nM"] = [convert_to_nM(v, u) for v, u in zip(work["standard_value"], work["unit_norm"])]
    kept = work.dropna(subset=["IC50_nM"]).copy()
    kept["IC50_nM"] = kept["IC50_nM"].astype(float)

    units_after = kept["unit_norm"].value_counts(dropna=False).rename_axis("unit_norm").reset_index(name="count_after")
    units_report = units_before.merge(units_after, on="unit_norm", how="outer").fillna(0)
    units_report[["count_before", "count_after"]] = units_report[["count_before", "count_after"]].astype(int)
    return kept, units_report


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[work["IC50_nM"] > 0].copy()
    work["IC50_M"] = work["IC50_nM"] * 1e-9
    work["pIC50"] = -np.log10(work["IC50_M"])
    work["IC50_uM"] = work["IC50_nM"] / 1_000.0
    return work


def rdkit_process(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        raw_smiles = str(row["canonical_smiles"]).strip()
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            invalid_rows.append(
                {
                    "molecule_chembl_id": row.get("molecule_chembl_id", ""),
                    "canonical_smiles": raw_smiles,
                    "standard_value": row.get("standard_value", np.nan),
                    "standard_units": row.get("standard_units", ""),
                }
            )
            continue

        rdkit_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if not scaffold:
                scaffold = "[NO_SCAFFOLD]"
        except Exception:
            scaffold = "[NO_SCAFFOLD]"

        rec = row.to_dict()
        rec["rdkit_canonical_smiles"] = rdkit_smiles
        rec["scaffold"] = scaffold
        rec["MolWt"] = float(Descriptors.MolWt(mol))
        rec["LogP"] = float(Crippen.MolLogP(mol))
        rec["HBD"] = int(Lipinski.NumHDonors(mol))
        rec["HBA"] = int(Lipinski.NumHAcceptors(mol))
        rec["TPSA"] = float(rdMolDescriptors.CalcTPSA(mol))
        rec["RotB"] = int(Lipinski.NumRotatableBonds(mol))
        rec["Rings"] = int(rdMolDescriptors.CalcNumRings(mol))
        records.append(rec)

    return pd.DataFrame(records), pd.DataFrame(invalid_rows)


def deduplicate_dataset(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if strategy not in {"median", "mean", "keep_best"}:
        raise ValueError(f"Unsupported dedup strategy: {strategy}")

    grouped = df.groupby("rdkit_canonical_smiles", dropna=False)
    output_rows: List[pd.Series] = []
    report_rows: List[Dict[str, Any]] = []

    for key, group in grouped:
        group = group.copy().sort_values("pIC50", ascending=False)
        n = len(group)

        if strategy == "median":
            agg_pic50 = float(group["pIC50"].median())
            agg_ic50_nm = float(10 ** (9 - agg_pic50))
            representative = group.iloc[0].copy()
            representative["pIC50"] = agg_pic50
            representative["IC50_nM"] = agg_ic50_nm
            representative["IC50_M"] = agg_ic50_nm * 1e-9
            representative["IC50_uM"] = agg_ic50_nm / 1_000.0
        elif strategy == "mean":
            agg_pic50 = float(group["pIC50"].mean())
            agg_ic50_nm = float(10 ** (9 - agg_pic50))
            representative = group.iloc[0].copy()
            representative["pIC50"] = agg_pic50
            representative["IC50_nM"] = agg_ic50_nm
            representative["IC50_M"] = agg_ic50_nm * 1e-9
            representative["IC50_uM"] = agg_ic50_nm / 1_000.0
        else:  # keep_best
            best_idx = group["IC50_nM"].idxmin()
            representative = group.loc[best_idx].copy()
            agg_pic50 = float(representative["pIC50"])
            agg_ic50_nm = float(representative["IC50_nM"])

        output_rows.append(representative)

        for _, row in group.iterrows():
            report_rows.append(
                {
                    "rdkit_canonical_smiles": key,
                    "molecule_chembl_id": row.get("molecule_chembl_id", ""),
                    "canonical_smiles": row.get("canonical_smiles", ""),
                    "IC50_nM_member": row.get("IC50_nM", np.nan),
                    "pIC50_member": row.get("pIC50", np.nan),
                    "n_group": n,
                    "dedup_strategy": strategy,
                    "aggregated_IC50_nM": agg_ic50_nm,
                    "aggregated_pIC50": agg_pic50,
                }
            )

    dedup = pd.DataFrame(output_rows).reset_index(drop=True)
    duplicates_report = pd.DataFrame(report_rows)
    return dedup, duplicates_report


def add_classification_label(df: pd.DataFrame, threshold_uM: float) -> Tuple[pd.DataFrame, float]:
    threshold_nM = threshold_uM * 1_000.0
    work = df.copy()
    work["Active"] = (work["IC50_nM"] <= threshold_nM).astype(int)
    return work, threshold_nM


def configure_plot_style(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": args.base_font,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
        }
    )


def bold_ticks(ax: plt.Axes, tick_size: int) -> None:
    ax.tick_params(axis="both", labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def save_figure(fig: plt.Figure, path_svg: Path, also_png: bool, path_png: Optional[Path]) -> None:
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg, format="svg", bbox_inches="tight")
    if also_png and path_png is not None:
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plots(df: pd.DataFrame, args: argparse.Namespace, outdir: Path, include_classification: bool) -> None:
    configure_plot_style(args)
    svg_dir = outdir / "plots" / "svg"
    png_dir = outdir / "plots" / "png"

    # A) pIC50 histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["pIC50"].dropna(), bins=30, color=NATURE_PALETTE["blue"], edgecolor="black", alpha=0.9)
    ax.set_title("Distribution of pIC50", fontsize=args.title_size, fontweight="bold")
    ax.set_xlabel("pIC50 (-log10[M])", fontsize=args.label_size, fontweight="bold")
    ax.set_ylabel("Count", fontsize=args.label_size, fontweight="bold")
    bold_ticks(ax, args.tick_size)
    fig.tight_layout()
    save_figure(fig, svg_dir / "pIC50_hist.svg", args.png, png_dir / "pIC50_hist.png")

    # B) IC50 histogram nM with log-scaled x
    fig, ax = plt.subplots(figsize=(7, 5))
    positive = df["IC50_nM"].replace([np.inf, -np.inf], np.nan).dropna()
    positive = positive[positive > 0]
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 30) if len(positive) > 1 else 10
    ax.hist(positive, bins=bins, color=NATURE_PALETTE["green"], edgecolor="black", alpha=0.9)
    ax.set_xscale("log")
    ax.set_title("Distribution of IC50 (nM)", fontsize=args.title_size, fontweight="bold")
    ax.set_xlabel("IC50 (nM, log scale)", fontsize=args.label_size, fontweight="bold")
    ax.set_ylabel("Count", fontsize=args.label_size, fontweight="bold")
    bold_ticks(ax, args.tick_size)
    fig.tight_layout()
    save_figure(fig, svg_dir / "ic50_hist_nM.svg", args.png, png_dir / "ic50_hist_nM.png")

    # C) Active balance
    if include_classification and "Active" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        counts = df["Active"].value_counts().reindex([0, 1], fill_value=0)
        labels = ["Inactive (0)", "Active (1)"]
        colors = [NATURE_PALETTE["orange"], NATURE_PALETTE["blue"]]
        ax.bar(labels, counts.values, color=colors, edgecolor="black")
        ax.set_title("Class Balance", fontsize=args.title_size, fontweight="bold")
        ax.set_xlabel("Class", fontsize=args.label_size, fontweight="bold")
        ax.set_ylabel("Count", fontsize=args.label_size, fontweight="bold")
        bold_ticks(ax, args.tick_size)
        fig.tight_layout()
        save_figure(fig, svg_dir / "active_balance.svg", args.png, png_dir / "active_balance.png")

    # D) top scaffold counts
    fig, ax = plt.subplots(figsize=(10, 6))
    scaffold_counts = df["scaffold"].fillna("[NO_SCAFFOLD]").value_counts().head(20)
    ax.barh(scaffold_counts.index[::-1], scaffold_counts.values[::-1], color=NATURE_PALETTE["purple"], edgecolor="black")
    ax.set_title("Top 20 Murcko Scaffolds", fontsize=args.title_size, fontweight="bold")
    ax.set_xlabel("Count", fontsize=args.label_size, fontweight="bold")
    ax.set_ylabel("Scaffold", fontsize=args.label_size, fontweight="bold")
    bold_ticks(ax, args.tick_size)
    fig.tight_layout()
    save_figure(fig, svg_dir / "scaffold_counts.svg", args.png, png_dir / "scaffold_counts.png")

    # E) MolWt vs pIC50
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["MolWt"], df["pIC50"], s=20, alpha=0.8, color=NATURE_PALETTE["cyan"], edgecolor="black", linewidth=0.3)
    ax.set_title("Molecular Weight vs pIC50", fontsize=args.title_size, fontweight="bold")
    ax.set_xlabel("MolWt (g/mol)", fontsize=args.label_size, fontweight="bold")
    ax.set_ylabel("pIC50 (-log10[M])", fontsize=args.label_size, fontweight="bold")
    bold_ticks(ax, args.tick_size)
    fig.tight_layout()
    save_figure(fig, svg_dir / "mw_vs_pIC50.svg", args.png, png_dir / "mw_vs_pIC50.png")

    # F) TPSA vs pIC50
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["TPSA"], df["pIC50"], s=20, alpha=0.8, color=NATURE_PALETTE["red2"], edgecolor="black", linewidth=0.3)
    ax.set_title("TPSA vs pIC50", fontsize=args.title_size, fontweight="bold")
    ax.set_xlabel("TPSA (Å²)", fontsize=args.label_size, fontweight="bold")
    ax.set_ylabel("pIC50 (-log10[M])", fontsize=args.label_size, fontweight="bold")
    bold_ticks(ax, args.tick_size)
    fig.tight_layout()
    save_figure(fig, svg_dir / "tpsa_vs_pIC50.svg", args.png, png_dir / "tpsa_vs_pIC50.png")


def build_manifest(
    ctx: PipelineContext,
    input_path: Path,
    relation_counts_before: pd.DataFrame,
    relation_report: pd.DataFrame,
    units_report: pd.DataFrame,
    rdkit_version: str,
    versions: Dict[str, str],
) -> pd.DataFrame:
    relation_summary = "; ".join(
        f"{row['relation']}={int(row['count_before'])}" for _, row in relation_counts_before.iterrows()
    )
    manifest_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_filename": input_path.name,
        "input_sha256": sha256_of_file(input_path),
        "number_of_raw_rows": ctx.counts.raw_rows,
        "number_dropped_missing_smiles_value": ctx.counts.dropped_missing_smiles_value,
        "counts_per_relation_raw": relation_summary,
        "keep_relations_strategy": ctx.args.keep_relations,
        "relation_strategy_note": ctx.relation_strategy_note,
        "number_invalid_smiles_dropped": ctx.counts.dropped_invalid_smiles,
        "number_after_dedup": ctx.counts.after_dedup,
        "dedup_strategy": ctx.args.dedup_strategy,
        "python_version": platform.python_version(),
        "rdkit_version": rdkit_version,
        "pandas_version": versions["pandas"],
        "numpy_version": versions["numpy"],
        "seeds_list": ",".join(str(s) for s in ctx.seeds),
        "activity_threshold_uM": ctx.args.activity_threshold_uM,
        "number_dropped_non_ic50": ctx.counts.dropped_non_ic50,
        "number_dropped_unknown_units": ctx.counts.dropped_unknown_units,
        "number_dropped_non_positive_ic50": ctx.counts.dropped_non_positive_ic50,
        "after_relation_handling": ctx.counts.after_relation_handling,
        "after_standardization": ctx.counts.after_standardization,
        "after_rdkit": ctx.counts.after_rdkit,
        "units_report_rows": len(units_report),
        "relations_report_rows": len(relation_report),
        "ic50_filter_possible": ctx.ic50_filter_possible,
    }
    return pd.DataFrame([manifest_row])


def write_manifest_txt(manifest_df: pd.DataFrame, path: Path) -> None:
    row = manifest_df.iloc[0].to_dict()
    lines = ["QSAR Preparation Run Manifest", "=" * 40]
    for key, value in row.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    random.seed(seeds[0])
    np.random.seed(seeds[0])

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ctx = PipelineContext(
        args=args,
        outdir=outdir,
        seeds=seeds,
        counts=PipelineCounts(),
        ic50_filter_possible=False,
        relation_strategy_note="",
    )

    print("[1/8] Reading input...")
    raw_df = pd.read_csv(input_path)
    ctx.counts.raw_rows = len(raw_df)

    print("[2/8] Mapping/standardizing columns...")
    mapping = map_columns(raw_df)
    df = standardize_columns(raw_df, mapping)

    df["canonical_smiles"] = df["canonical_smiles"].astype(str).str.strip()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    mask_missing = (
        df["canonical_smiles"].replace("", np.nan).isna()
        | df["standard_value"].isna()
        | df["standard_units"].isna()
    )
    ctx.counts.dropped_missing_smiles_value = int(mask_missing.sum())
    df = df[~mask_missing].copy()

    if "standard_type" in df.columns:
        ctx.ic50_filter_possible = True
        before = len(df)
        df = df[df["standard_type"].astype(str).str.contains("IC50", case=False, na=False)].copy()
        ctx.counts.dropped_non_ic50 = before - len(df)
    else:
        print("[INFO] standard_type column not present; IC50-specific type filtering not possible.")

    print("[3/8] Handling relations and units...")
    df, relation_report, relation_counts_before, relation_note = apply_relation_filter(df, args.keep_relations)
    ctx.relation_strategy_note = relation_note
    ctx.counts.after_relation_handling = len(df)

    df, units_report = unit_filter_and_convert(df)
    ctx.counts.dropped_unknown_units = int(units_report["count_before"].sum() - units_report["count_after"].sum())

    before_positive = len(df)
    df = compute_targets(df)
    ctx.counts.dropped_non_positive_ic50 = before_positive - len(df)
    ctx.counts.after_standardization = len(df)

    df.to_csv(outdir / "master_raw_standardized.csv", index=False)

    print("[4/8] RDKit parsing/canonicalization...")
    valid_rdkit, invalid_examples = rdkit_process(df)
    ctx.counts.dropped_invalid_smiles = len(df) - len(valid_rdkit)
    ctx.counts.after_rdkit = len(valid_rdkit)

    valid_rdkit.to_csv(outdir / "master_valid_rdkit.csv", index=False)
    invalid_examples.head(100).to_csv(outdir / "invalid_smiles_examples.csv", index=False)

    print("[5/8] Deduplicating compounds...")
    dedup_df, dup_report = deduplicate_dataset(valid_rdkit, args.dedup_strategy)

    include_classification = args.task in {"classification", "both"}
    if include_classification:
        dedup_df, threshold_nM = add_classification_label(dedup_df, args.activity_threshold_uM)
        class_counts = dedup_df["Active"].value_counts().to_dict()
        print(f"[INFO] Class balance Active=1: {class_counts.get(1, 0)}, Active=0: {class_counts.get(0, 0)}")
    else:
        threshold_nM = args.activity_threshold_uM * 1_000.0
        dedup_df["Active"] = np.nan

    required_cols = [
        "molecule_chembl_id",
        "rdkit_canonical_smiles",
        "canonical_smiles",
        "pIC50",
        "IC50_nM",
        "Active",
        "scaffold",
        "MolWt",
        "LogP",
        "HBD",
        "HBA",
        "TPSA",
        "RotB",
        "Rings",
    ]
    for col in required_cols:
        if col not in dedup_df.columns:
            dedup_df[col] = np.nan
    dedup_df = dedup_df[required_cols + [c for c in dedup_df.columns if c not in required_cols]]

    dedup_df.to_csv(outdir / "master_dedup.csv", index=False)
    dup_report.to_csv(outdir / "duplicates_report.csv", index=False)
    relation_report.to_csv(outdir / "relations_report.csv", index=False)
    units_report.to_csv(outdir / "units_report.csv", index=False)
    ctx.counts.after_dedup = len(dedup_df)

    print("[6/8] Writing QC summary + manifest...")
    qc_rows = [
        {"metric": "raw_rows", "value": ctx.counts.raw_rows},
        {"metric": "dropped_missing_smiles_value", "value": ctx.counts.dropped_missing_smiles_value},
        {"metric": "dropped_non_ic50", "value": ctx.counts.dropped_non_ic50},
        {"metric": "after_relation_handling", "value": ctx.counts.after_relation_handling},
        {"metric": "dropped_unknown_units", "value": ctx.counts.dropped_unknown_units},
        {"metric": "dropped_non_positive_ic50", "value": ctx.counts.dropped_non_positive_ic50},
        {"metric": "after_standardization", "value": ctx.counts.after_standardization},
        {"metric": "dropped_invalid_smiles", "value": ctx.counts.dropped_invalid_smiles},
        {"metric": "after_rdkit", "value": ctx.counts.after_rdkit},
        {"metric": "after_dedup", "value": ctx.counts.after_dedup},
        {"metric": "activity_threshold_uM", "value": args.activity_threshold_uM},
        {"metric": "activity_threshold_nM", "value": threshold_nM},
        {"metric": "dedup_strategy", "value": args.dedup_strategy},
        {"metric": "keep_relations", "value": args.keep_relations},
        {"metric": "task", "value": args.task},
    ]
    if include_classification:
        class_counts = dedup_df["Active"].value_counts().to_dict()
        qc_rows.extend(
            [
                {"metric": "class_inactive_0", "value": class_counts.get(0, 0)},
                {"metric": "class_active_1", "value": class_counts.get(1, 0)},
            ]
        )
    pd.DataFrame(qc_rows).to_csv(outdir / "data_qc_summary.csv", index=False)

    versions = {"pandas": pd.__version__, "numpy": np.__version__}
    rdkit_version_str = getattr(rdBase, "rdkitVersion", "unknown")

    manifest_df = build_manifest(
        ctx=ctx,
        input_path=input_path,
        relation_counts_before=relation_counts_before,
        relation_report=relation_report,
        units_report=units_report,
        rdkit_version=rdkit_version_str,
        versions=versions,
    )
    manifest_df.to_csv(outdir / "run_manifest.csv", index=False)
    write_manifest_txt(manifest_df, outdir / "run_manifest.txt")

    print("[7/8] Generating plots...")
    make_plots(dedup_df, args, outdir, include_classification=include_classification)

    print("[8/8] Finalizing outputs...")
    print(outdir / "master_dedup.csv")
    print(outdir / "plots")
    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)
