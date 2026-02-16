#!/usr/bin/env python3
"""
QSAR post-processing for ChEMBL target extracts (SQLite output CSVs).

What it does (production level):
1) Loads ChEMBL extracted CSV (raw or qsar_ready).
2) Converts standard_value + standard_units to nM (keeps only nM/uM/µM by default).
3) Computes pIC50 (or more generally pActivity) = -log10(value_in_M).
4) Aggregates per compound (keeps best potency = max pIC50) with configurable strategy.
5) Assigns Active (1) / Inactive (0) based on pIC50 threshold (default >= 6).
6) Computes RDKit + Lipinski properties (MW, LogP, HBD, HBA, TPSA, RotB, Rings).
7) Produces publication-ready SVG figures:
   - Active vs Inactive counts (clean horizontal bar)
   - Radar (spider) plot: mean properties active vs inactive (raw ranges shown per axis)
   - Bubble scatter: MW vs LogP (bubble size = chosen property), legend shows counts
8) All text in plots Times New Roman + bold; font sizes fully controllable by CLI args.
9) Writes processed tables to CSV.

Requirements:
  pip install pandas numpy matplotlib rdkit-pypi
"""

from __future__ import annotations

import argparse
import os
import math
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute pIC50, active/inactive labels, Lipinski props, and figures (SVG).")
    p.add_argument("--input", required=True, help="Input CSV from ChEMBL extraction (e.g., CHEMBL284_qsar_ready.csv)")
    p.add_argument("--outdir", default="qsar_out", help="Output directory")
    p.add_argument("--endpoint", default="IC50", help="Endpoint standard_type to use (default: IC50)")
    p.add_argument("--threshold", type=float, default=6.0, help="Active threshold for pIC50 (default >= 6.0)")
    p.add_argument("--units_keep", default="nM,uM,µM", help="Allowed standard_units list to keep before conversion")
    p.add_argument("--relation_keep", default="=", help="Allowed standard_relation (default '=')")

    # Aggregation choices
    p.add_argument("--aggregate", default="best",
                   choices=["best", "median", "mean"],
                   help="How to aggregate multiple records per compound (default: best=max pIC50)")
    p.add_argument("--prefer_pchembl", action="store_true",
                   help="If set, uses pchembl_value when present and compatible with endpoint; else compute from value+unit.")

    # Figure control
    p.add_argument("--font_base", type=float, default=14.0, help="Base font size")
    p.add_argument("--title_size", type=float, default=18.0, help="Title font size")
    p.add_argument("--label_size", type=float, default=14.0, help="Axis label font size")
    p.add_argument("--tick_size", type=float, default=12.0, help="Tick font size")
    p.add_argument("--legend_size", type=float, default=12.0, help="Legend font size")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster formats (SVG is vector but kept for consistency)")

    # Scatter / bubble options
    p.add_argument("--bubble_min", type=float, default=30.0, help="Min bubble size")
    p.add_argument("--bubble_max", type=float, default=500.0, help="Max bubble size")
    p.add_argument("--bubble_by", default="TPSA", choices=["TPSA", "MolWt", "LogP", "RotB", "HBA", "HBD"],
                   help="Which property drives bubble size (default: TPSA)")

    # Output control
    p.add_argument("--svg", action="store_true", help="If set, output SVG (recommended).")
    p.add_argument("--png", action="store_true", help="If set, also output PNG.")
    return p.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_plot_style(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": args.font_base,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": args.title_size,
        "axes.labelsize": args.label_size,
        "xtick.labelsize": args.tick_size,
        "ytick.labelsize": args.tick_size,
        "legend.fontsize": args.legend_size,
        "svg.fonttype": "none",
    })


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def to_nM(value: float, unit: str) -> Optional[float]:
    if value is None:
        return None
    u = str(unit).strip()
    if u == "nM":
        return value
    if u in ("uM", "µM"):
        return value * 1000.0
    if u == "pM":
        return value / 1000.0
    if u == "mM":
        return value * 1e6
    return None


def nM_to_p(value_nM: float) -> Optional[float]:
    if value_nM is None or value_nM <= 0:
        return None
    value_M = value_nM * 1e-9
    return -math.log10(value_M)


# -----------------------------
# RDKit properties
# -----------------------------
def compute_props(smiles: str) -> Optional[Dict[str, float]]:
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)

    v = 0
    if mw > 500: v += 1
    if logp > 5: v += 1
    if hbd > 5: v += 1
    if hba > 10: v += 1

    return {
        "MolWt": mw,
        "LogP": logp,
        "HBD": float(hbd),
        "HBA": float(hba),
        "TPSA": tpsa,
        "RotB": float(rotb),
        "Rings": float(rings),
        "Ro5_Violations": float(v),
        "Ro5_Pass": float(1 if v == 0 else 0),
    }


# -----------------------------
# Main processing
# -----------------------------
def load_and_prepare(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)

    required_cols = ["molecule_chembl_id", "canonical_smiles", "standard_type",
                     "standard_relation", "standard_value", "standard_units"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    keep_units = {u.strip() for u in args.units_keep.split(",") if u.strip()}
    keep_rel = {r.strip() for r in args.relation_keep.split(",") if r.strip()}

    df = df[df["standard_type"].astype(str).str.upper() == args.endpoint.upper()].copy()
    df = df[df["standard_relation"].astype(str).isin(keep_rel)].copy()
    df = df[df["standard_units"].astype(str).isin(keep_units)].copy()

    if args.prefer_pchembl and "pchembl_value" in df.columns:
        df["pIC50"] = df["pchembl_value"].apply(safe_float)
    else:
        df["pIC50"] = np.nan

    def _calc_pic50(row) -> Optional[float]:
        if pd.notna(row.get("pIC50")):
            return float(row["pIC50"])
        v = safe_float(row["standard_value"])
        u = row["standard_units"]
        v_nM = to_nM(v, u)
        if v_nM is None:
            return None
        return nM_to_p(v_nM)

    df["pIC50"] = df.apply(_calc_pic50, axis=1)
    df = df[df["pIC50"].notna()].copy()
    df["Active"] = (df["pIC50"].astype(float) >= float(args.threshold)).astype(int)
    return df


def aggregate_per_compound(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    def mode_smiles(x: pd.Series) -> str:
        x = x.dropna().astype(str)
        x = x[x.str.strip() != ""]
        if len(x) == 0:
            return ""
        return x.value_counts().idxmax()

    agg_map = {"best": "max", "median": "median", "mean": "mean"}
    how = agg_map[args.aggregate]

    # --- compute the aggregated pIC50 as before (NO CHANGE) ---
    g = df.groupby("molecule_chembl_id", as_index=False).agg(
        canonical_smiles=("canonical_smiles", mode_smiles),
        pIC50=("pIC50", how),
        Active=("Active", "max"),
        n_records=("pIC50", "count"),
    )

    # --- NEW: attach standard_value + standard_units from a representative row ---
    # Strategy:
    #   - if aggregate=best: take the row with MAX pIC50 for that compound
    #   - if aggregate=mean/median: take the row whose pIC50 is closest to aggregated pIC50
    df2 = df.copy()
    df2["pIC50"] = df2["pIC50"].astype(float)

    if args.aggregate == "best":
        rep = (
            df2.sort_values(["molecule_chembl_id", "pIC50"], ascending=[True, False])
               .groupby("molecule_chembl_id", as_index=False)
               .first()[["molecule_chembl_id", "standard_value", "standard_units"]]
        )
    else:
        # merge aggregated pIC50 back, then choose closest record
        tmp = df2.merge(g[["molecule_chembl_id", "pIC50"]], on="molecule_chembl_id", suffixes=("", "_agg"))
        tmp["abs_diff"] = (tmp["pIC50"] - tmp["pIC50_agg"]).abs()

        rep = (
            tmp.sort_values(["molecule_chembl_id", "abs_diff"], ascending=[True, True])
               .groupby("molecule_chembl_id", as_index=False)
               .first()[["molecule_chembl_id", "standard_value", "standard_units"]]
        )

    # merge representative IC50 columns into compound-level table
    g = g.merge(rep, on="molecule_chembl_id", how="left")

    # keep your Active definition consistent with aggregated pIC50
    g["Active"] = (g["pIC50"].astype(float) >= float(args.threshold)).astype(int)
    return g

def add_rdkit_properties(df: pd.DataFrame) -> pd.DataFrame:
    props = []
    for smi in df["canonical_smiles"].astype(str).tolist():
        d = compute_props(smi)
        props.append(d if d is not None else {k: np.nan for k in
                                              ["MolWt", "LogP", "HBD", "HBA", "TPSA", "RotB", "Rings",
                                               "Ro5_Violations", "Ro5_Pass"]})
    props_df = pd.DataFrame(props)
    out = pd.concat([df.reset_index(drop=True), props_df.reset_index(drop=True)], axis=1)
    out = out[out["MolWt"].notna()].copy()
    return out


# -----------------------------
# Plotting
# -----------------------------
def save_figure(fig: plt.Figure, outpath_base: str, args: argparse.Namespace) -> None:
    if args.svg or (not args.png):
        fig.savefig(outpath_base + ".svg", format="svg", bbox_inches="tight")
    if args.png:
        fig.savefig(outpath_base + ".png", format="png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


# ===== CHANGED PLOTS ONLY (everything else unchanged) =====
def plot_active_inactive_counts(df: pd.DataFrame, outbase: str, args: argparse.Namespace) -> None:
    """
    Cleaner class balance plot: horizontal bars + count shown.
    """
    n_active = int((df["Active"] == 1).sum())
    n_inactive = int((df["Active"] == 0).sum())

    fig = plt.figure(figsize=(7.5, 4.8))
    ax = fig.add_subplot(111)

    labels = [f"Active (n={n_active})", f"Inactive (n={n_inactive})"]
    values = [n_active, n_inactive]

    ax.barh(labels, values)
    ax.set_title("Dataset Class Balance", fontweight="bold")
    ax.set_xlabel("Number of compounds", fontweight="bold")

    for y, v in enumerate(values):
        ax.text(v, y, f"  {v}", va="center", ha="left", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, outbase, args)


def radar_plot_means(df: pd.DataFrame, outbase: str, args: argparse.Namespace) -> None:
    props = ["MolWt", "LogP", "RotB", "TPSA", "HBA", "HBD"]

    max_map = {
        "MolWt": 600.0,   # Da
        "LogP": 7.0,
        "RotB": 12.0,
        "TPSA": 180.0,    # Å^2
        "HBA": 10.0,
        "HBD": 5.0,
    }

    tick_map = {
        "MolWt": [150, 300, 450, 600],
        "LogP":  [2.7, 4.8, 7.0],
        "RotB":  [3, 6, 9, 12],
        "TPSA":  [45.4, 90.7, 135.2, 180.4],
        "HBA":   [2, 5, 8, 10],
        "HBD":   [1, 2, 4, 5],
    }

    data = df[props + ["Active"]].copy()
    active_raw = data[data["Active"] == 1][props].mean()
    inactive_raw = data[data["Active"] == 0][props].mean()

    def norm(p: str, v: float) -> float:
        m = max_map[p]
        return float(np.clip(v / m, 0.0, 1.0)) if m > 0 else 0.0

    active_scaled = np.array([norm(p, float(active_raw[p])) for p in props], dtype=float)
    inactive_scaled = np.array([norm(p, float(inactive_raw[p])) for p in props], dtype=float)

    active_vals = active_scaled.tolist() + [active_scaled[0]]
    inactive_vals = inactive_scaled.tolist() + [inactive_scaled[0]]

    angles = np.linspace(0, 2 * np.pi, len(props), endpoint=False).tolist()
    angles += [angles[0]]

    fig = plt.figure(figsize=(10.0, 8.0))
    ax = fig.add_subplot(111, polar=True)

    # Background bands
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.fill(angles, [r] * len(angles), alpha=0.03)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontweight="bold")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["Molecular Weight", "LogP", "RotBonds", "TPSA", "HBA", "HBD"],
        fontweight="bold"
    )

    # ---- overlap fix: outward offsets + small angle tweak for MolWt & TPSA
    # radial text offset in "radius units"
    radial_bump = {
        "MolWt": 0.06,   # push outward a bit more
        "TPSA":  0.09,   # push outward more to separate from MolWt
        "default": 0.04
    }
    # tiny angular tweak (radians) to separate tick text clouds
    angle_bump = {
        "MolWt": -0.06,  # rotate slightly counterclockwise
        "TPSA":  +0.06,  # rotate slightly clockwise
    }

    for i, p in enumerate(props):
        ang0 = angles[i]
        ang = ang0 + angle_bump.get(p, 0.0)
        bump = radial_bump.get(p, radial_bump["default"])

        ticks = tick_map[p]
        for t in ticks:
            r = float(np.clip((t / max_map[p]) + bump, 0.0, 1.08))

            if p == "MolWt":
                label = f"{int(round(t))} Da"
            elif p == "TPSA":
                label = f"{t:.1f} Å$^2$"
            else:
                label = f"{t:g}"

            # align depending on side of plot
            side = np.cos(ang)
            ha = "left" if side > 0.25 else ("right" if side < -0.25 else "center")

            ax.text(
                ang,
                r,
                label,
                ha=ha,
                va="center",
                fontweight="bold",
                rotation=np.degrees(ang),
                rotation_mode="anchor"
            )

    n_active = int((df["Active"] == 1).sum())
    n_inactive = int((df["Active"] == 0).sum())

    ax.plot(angles, inactive_vals, linewidth=2.5, label=f"Inactive (n={n_inactive})")
    ax.fill(angles, inactive_vals, alpha=0.12)

    ax.plot(angles, active_vals, linewidth=2.5, label=f"Active (n={n_active})")
    ax.fill(angles, active_vals, alpha=0.12)

    ax.set_title("Spider Plot of Physicochemical Properties", fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=False)

    save_figure(fig, outbase, args)


def bubble_scatter(df: pd.DataFrame, outbase: str, args: argparse.Namespace) -> None:
    """
    Bubble scatter with:
    - Proper transparency to avoid overplotting
    - Neutral (gray) bubble-size legend
    - Class legend shows counts
    """

    required = ["MolWt", "LogP", args.bubble_by, "Active"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required property for scatter: {c}")

    x = df["MolWt"].astype(float).values
    y = df["LogP"].astype(float).values
    b = df[args.bubble_by].astype(float).values
    cls = df["Active"].astype(int).values

    n_active = int((cls == 1).sum())
    n_inactive = int((cls == 0).sum())

    # bubble size scaling
    b_min, b_max = np.nanmin(b), np.nanmax(b)
    if not np.isfinite(b_min) or not np.isfinite(b_max) or b_max == b_min:
        sizes = np.full_like(x, (args.bubble_min + args.bubble_max) / 2.0, dtype=float)
    else:
        sizes = args.bubble_min + (b - b_min) * (args.bubble_max - args.bubble_min) / (b_max - b_min)
        sizes = np.clip(sizes, args.bubble_min, args.bubble_max)

    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111)

    mask_a = cls == 1
    mask_i = cls == 0

    # ---- KEY FIX: add transparency (alpha) ----
    ax.scatter(
        x[mask_i], y[mask_i],
        s=sizes[mask_i],
        alpha=0.35,
        label=f"Inactive (n={n_inactive})"
    )
    ax.scatter(
        x[mask_a], y[mask_a],
        s=sizes[mask_a],
        alpha=0.35,
        label=f"Active (n={n_active})"
    )

    ax.set_title(f"MolWt vs LogP (Bubble size = {args.bubble_by})", fontweight="bold")
    ax.set_xlabel("Molecular Weight (MolWt)", fontweight="bold")
    ax.set_ylabel("LogP", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------- Neutral bubble size legend (FIX) --------
    q = np.nanpercentile(b, [25, 50, 75])
    q = np.unique(np.round(q, 2))

    def size_for(val):
        return float(
            np.clip(
                args.bubble_min + (val - b_min) * (args.bubble_max - args.bubble_min) / (b_max - b_min),
                args.bubble_min,
                args.bubble_max,
            )
        )

    handles = []
    labels = []
    for val in q[:3]:
        handles.append(
            ax.scatter([], [], s=size_for(val), color="gray", alpha=0.4)
        )
        labels.append(f"{args.bubble_by} ~ {val}")

    legend1 = ax.legend(frameon=False, loc="upper left")
    ax.add_artist(legend1)
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="lower right",
        title="Bubble size guide"
    )

    save_figure(fig, outbase, args)

# -----------------------------
# Run
# -----------------------------
def main() -> int:
    args = parse_args()
    ensure_outdir(args.outdir)
    set_plot_style(args)

    # 1) Load & compute pIC50 and labels
    df_rows = load_and_prepare(args)

    # Save row-level file
    row_level_path = os.path.join(args.outdir, "row_level_with_pIC50.csv")
    df_rows.to_csv(row_level_path, index=False)

    # 2) Aggregate per compound
    df_cmpd = aggregate_per_compound(df_rows, args)
    cmpd_level_path = os.path.join(args.outdir, "compound_level_pIC50.csv")
    df_cmpd.to_csv(cmpd_level_path, index=False)

    # 3) RDKit properties + Lipinski
    df_props = add_rdkit_properties(df_cmpd)
    props_path = os.path.join(args.outdir, "compound_level_with_properties.csv")
    df_props.to_csv(props_path, index=False)

    # 4) Summary counts
    counts = df_props["Active"].value_counts().to_dict()
    summary = {
        "n_compounds_total": int(df_props.shape[0]),
        "n_active": int(counts.get(1, 0)),
        "n_inactive": int(counts.get(0, 0)),
        "endpoint": args.endpoint,
        "threshold_pIC50": float(args.threshold),
        "aggregate": args.aggregate,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    # 5) Figures
    plot_active_inactive_counts(df_props, os.path.join(args.outdir, "fig_class_balance"), args)
    radar_plot_means(df_props, os.path.join(args.outdir, "fig_spider_properties_active_vs_inactive"), args)
    bubble_scatter(df_props, os.path.join(args.outdir, "fig_bubble_mw_vs_logp"), args)

    print("DONE")
    print(f"Row-level:      {row_level_path}")
    print(f"Compound-level: {cmpd_level_path}")
    print(f"Properties:     {props_path}")
    print(f"Summary:        {os.path.join(args.outdir, 'summary.csv')}")
    print(f"Figures:        {args.outdir}/fig_*.svg (and/or .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
