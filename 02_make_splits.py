#!/usr/bin/env python3
"""Generate reproducible QSAR dataset splits with scaffold/random/stratified methods."""

from __future__ import annotations

import argparse
import hashlib
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import rdBase

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - handled with actionable runtime errors
    Chem = None  # type: ignore[assignment]
    MurckoScaffold = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]
    RDKIT_AVAILABLE = False


NO_SCAFFOLD = "[NO_SCAFFOLD]"
SPLITS = ("train", "val", "test")
PALETTE = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}
SPLIT_COLORS = {
    "train": PALETTE["blue"],
    "val": PALETTE["green"],
    "test": PALETTE["orange"],
}


@dataclass
class SplitResult:
    seed: int
    method_used: str
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index
    target_counts: Dict[str, int]
    warnings: List[str]
    dropped_small_groups_count: int
    dropped_small_groups_rows: int
    groups_table: pd.DataFrame
    predesignation: pd.DataFrame
    overlap_rows_pass: bool
    overlap_groups_pass: bool


@dataclass
class PlotStyle:
    title_size: int
    label_size: int
    tick_size: int
    legend_size: int
    base_font: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible QSAR data splits.")
    parser.add_argument("--input", default="qsar_out/master_dedup.csv", help="Input CSV path")
    parser.add_argument("--outdir", default="splits", help="Output directory")
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated seed list, e.g. 0,1,2",
    )
    parser.add_argument(
        "--method",
        default="scaffold",
        choices=["scaffold", "random", "stratified"],
        help="Splitting method",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--group_col", default="scaffold")
    parser.add_argument("--smiles_col", default="rdkit_canonical_smiles")
    parser.add_argument("--id_col", default="molecule_chembl_id")
    parser.add_argument("--target_col", default="pIC50")
    parser.add_argument("--label_col", default="Active")
    parser.add_argument("--min_group_size", type=int, default=1)
    parser.add_argument("--plots_all_seeds", action="store_true", default=False)
    parser.add_argument("--svg_only", action="store_true", default=True)
    parser.add_argument("--png", action="store_true", default=False)
    parser.add_argument("--title_size", type=int, default=18)
    parser.add_argument("--label_size", type=int, default=16)
    parser.add_argument("--tick_size", type=int, default=14)
    parser.add_argument("--legend_size", type=int, default=14)
    parser.add_argument("--base_font", type=int, default=14)
    return parser.parse_args()


def parse_seeds(seeds_str: str) -> List[int]:
    try:
        seeds = [int(x.strip()) for x in seeds_str.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"Invalid --seeds value '{seeds_str}'. Use comma-separated integers.") from exc
    if not seeds:
        raise ValueError("No seeds provided. Use --seeds with at least one integer.")
    return seeds


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float, tol: float = 1e-8) -> Dict[str, float]:
    for name, value in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Ratios must sum to 1.0 within tolerance {tol}, got {total:.12f} "
            f"({train_ratio}, {val_ratio}, {test_ratio})."
        )
    return {"train": train_ratio, "val": val_ratio, "test": test_ratio}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + f". Found columns: {', '.join(df.columns.astype(str).tolist())}"
        )


def compute_scaffold(smiles: str) -> str:
    if not RDKIT_AVAILABLE:
        raise RuntimeError(
            "RDKit is required to compute scaffolds when --group_col is missing. "
            "Install RDKit or provide a valid scaffold column."
        )
    if pd.isna(smiles):
        return NO_SCAFFOLD
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return NO_SCAFFOLD
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold else NO_SCAFFOLD


def ensure_group_column(df: pd.DataFrame, group_col: str, smiles_col: str) -> pd.DataFrame:
    out = df.copy()
    if group_col not in out.columns:
        check_required_columns(out, [smiles_col])
        print(f"[INFO] '{group_col}' not found. Computing Murcko scaffolds from '{smiles_col}' in-memory...")
        out[group_col] = out[smiles_col].map(compute_scaffold)
    out[group_col] = out[group_col].fillna(NO_SCAFFOLD).replace("", NO_SCAFFOLD)
    return out


def allocate_counts_exact(n: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {k: ratios[k] * n for k in SPLITS}
    floored = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(floored.values())
    fracs = sorted(((raw[k] - floored[k], k) for k in SPLITS), reverse=True)
    for i in range(remainder):
        floored[fracs[i % len(fracs)][1]] += 1
    return floored


def greedy_group_split(
    df: pd.DataFrame,
    group_col: str,
    ratios: Dict[str, float],
    seed: int,
    min_group_size: int,
    id_col: str,
) -> SplitResult:
    rng = np.random.RandomState(seed)
    warnings: List[str] = []
    group_to_indices: Dict[str, List[int]] = {}
    for idx, group in zip(df.index, df[group_col].astype(str)):
        group_to_indices.setdefault(group, []).append(idx)

    dropped_groups = [g for g, idxs in group_to_indices.items() if len(idxs) < min_group_size]
    dropped_rows = sum(len(group_to_indices[g]) for g in dropped_groups)
    if dropped_groups:
        warnings.append(
            f"Dropped {len(dropped_groups)} groups with size < {min_group_size} (rows dropped: {dropped_rows})."
        )

    kept_groups = [g for g in sorted(group_to_indices.keys()) if g not in set(dropped_groups)]
    if not kept_groups:
        raise ValueError("No groups remaining after min_group_size filtering.")

    shuffle_order = rng.permutation(len(kept_groups))
    shuffled_groups = [kept_groups[i] for i in shuffle_order]
    shuffled_positions = {g: pos for pos, g in enumerate(shuffled_groups)}
    sorted_groups = sorted(
        shuffled_groups,
        key=lambda g: (-len(group_to_indices[g]), shuffled_positions[g]),
    )

    n = sum(len(group_to_indices[g]) for g in sorted_groups)
    target = allocate_counts_exact(n, ratios)

    assigned: Dict[str, str] = {}
    split_to_indices: Dict[str, List[int]] = {k: [] for k in SPLITS}

    for group in sorted_groups:
        size = len(group_to_indices[group])
        capacities = {s: target[s] - len(split_to_indices[s]) for s in SPLITS}
        best_split = max(SPLITS, key=lambda s: (capacities[s], target[s], -len(split_to_indices[s]), s == "train"))
        assigned[group] = best_split
        split_to_indices[best_split].extend(group_to_indices[group])

    groups_table_rows = []
    for group in sorted(group_to_indices.keys()):
        idxs = group_to_indices[group]
        split_name = assigned.get(group, "dropped")
        rep_smiles = None
        if idxs:
            row = df.loc[idxs[0]]
            rep_smiles = row.get("rdkit_canonical_smiles", None)
        groups_table_rows.append(
            {
                "group_id": group,
                "group_size": len(idxs),
                "assigned_split": split_name,
                "representative_smiles": rep_smiles,
            }
        )
    groups_table = pd.DataFrame(groups_table_rows)

    pre_rows = []
    for split_name in SPLITS:
        for idx in split_to_indices[split_name]:
            row = df.loc[idx]
            pre_rows.append(
                {
                    id_col: row.get(id_col, idx),
                    "group_id": str(row[group_col]),
                    "split": split_name,
                }
            )
    predesignation = pd.DataFrame(pre_rows)

    train_idx = pd.Index(split_to_indices["train"])
    val_idx = pd.Index(split_to_indices["val"])
    test_idx = pd.Index(split_to_indices["test"])

    overlap_rows_pass = (
        len(set(train_idx).intersection(val_idx)) == 0
        and len(set(train_idx).intersection(test_idx)) == 0
        and len(set(val_idx).intersection(test_idx)) == 0
    )

    split_groups = {
        "train": set(df.loc[train_idx, group_col].astype(str).tolist()),
        "val": set(df.loc[val_idx, group_col].astype(str).tolist()),
        "test": set(df.loc[test_idx, group_col].astype(str).tolist()),
    }
    overlap_groups_pass = (
        len(split_groups["train"].intersection(split_groups["val"])) == 0
        and len(split_groups["train"].intersection(split_groups["test"])) == 0
        and len(split_groups["val"].intersection(split_groups["test"])) == 0
    )

    return SplitResult(
        seed=seed,
        method_used="scaffold",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        target_counts=target,
        warnings=warnings,
        dropped_small_groups_count=len(dropped_groups),
        dropped_small_groups_rows=dropped_rows,
        groups_table=groups_table,
        predesignation=predesignation,
        overlap_rows_pass=overlap_rows_pass,
        overlap_groups_pass=overlap_groups_pass,
    )


def random_row_split(df: pd.DataFrame, ratios: Dict[str, float], seed: int, id_col: str, group_col: str) -> SplitResult:
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(df.index.to_numpy())
    target = allocate_counts_exact(len(shuffled), ratios)
    n_train = target["train"]
    n_val = target["val"]

    train_idx = pd.Index(shuffled[:n_train])
    val_idx = pd.Index(shuffled[n_train : n_train + n_val])
    test_idx = pd.Index(shuffled[n_train + n_val :])

    overlap_rows_pass = (
        len(set(train_idx).intersection(val_idx)) == 0
        and len(set(train_idx).intersection(test_idx)) == 0
        and len(set(val_idx).intersection(test_idx)) == 0
    )

    group_available = group_col in df.columns
    if group_available:
        split_groups = {
            "train": set(df.loc[train_idx, group_col].astype(str)),
            "val": set(df.loc[val_idx, group_col].astype(str)),
            "test": set(df.loc[test_idx, group_col].astype(str)),
        }
        overlap_groups_pass = (
            len(split_groups["train"].intersection(split_groups["val"])) == 0
            and len(split_groups["train"].intersection(split_groups["test"])) == 0
            and len(split_groups["val"].intersection(split_groups["test"])) == 0
        )
    else:
        overlap_groups_pass = True

    all_df = pd.concat(
        [
            pd.DataFrame({"_idx": train_idx, "split": "train"}),
            pd.DataFrame({"_idx": val_idx, "split": "val"}),
            pd.DataFrame({"_idx": test_idx, "split": "test"}),
        ],
        ignore_index=True,
    )
    map_df = all_df.merge(df[[id_col, group_col]] if group_available else df[[id_col]], left_on="_idx", right_index=True)
    if group_available:
        map_df = map_df.rename(columns={group_col: "group_id"})
    else:
        map_df["group_id"] = map_df["_idx"].astype(str)

    if group_available:
        grp = map_df.groupby("group_id")
        gt = grp.agg(group_size=("_idx", "count"), split_n=("split", "nunique")).reset_index()
        gt["assigned_split"] = grp["split"].agg(lambda x: x.iloc[0] if x.nunique() == 1 else "mixed").values
    else:
        gt = pd.DataFrame({"group_id": map_df["group_id"], "group_size": 1, "split_n": 1, "assigned_split": map_df["split"]})
    gt["representative_smiles"] = None

    predesignation = map_df[[id_col, "group_id", "split"]].copy()

    return SplitResult(
        seed=seed,
        method_used="random",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        target_counts=target,
        warnings=[],
        dropped_small_groups_count=0,
        dropped_small_groups_rows=0,
        groups_table=gt[["group_id", "group_size", "assigned_split", "representative_smiles"]],
        predesignation=predesignation,
        overlap_rows_pass=overlap_rows_pass,
        overlap_groups_pass=overlap_groups_pass,
    )


def stratified_split(
    df: pd.DataFrame,
    ratios: Dict[str, float],
    seed: int,
    id_col: str,
    group_col: str,
    label_col: str,
) -> SplitResult:
    if label_col not in df.columns:
        res = random_row_split(df=df, ratios=ratios, seed=seed, id_col=id_col, group_col=group_col)
        res.method_used = "random"
        res.warnings.append(f"label_col '{label_col}' missing; fell back to random split.")
        return res

    unique_classes = sorted(df[label_col].dropna().unique().tolist())
    if len(unique_classes) < 2:
        res = random_row_split(df=df, ratios=ratios, seed=seed, id_col=id_col, group_col=group_col)
        res.method_used = "random"
        res.warnings.append(f"label_col '{label_col}' has <2 classes; fell back to random split.")
        return res

    rng = np.random.RandomState(seed)
    split_to_indices: Dict[str, List[int]] = {k: [] for k in SPLITS}

    for class_value in unique_classes:
        class_idx = df.index[df[label_col] == class_value].to_numpy()
        class_idx = rng.permutation(class_idx)
        target = allocate_counts_exact(len(class_idx), ratios)
        n_train = target["train"]
        n_val = target["val"]
        split_to_indices["train"].extend(class_idx[:n_train].tolist())
        split_to_indices["val"].extend(class_idx[n_train : n_train + n_val].tolist())
        split_to_indices["test"].extend(class_idx[n_train + n_val :].tolist())

    for split_name in SPLITS:
        split_to_indices[split_name] = rng.permutation(split_to_indices[split_name]).tolist()

    target_global = allocate_counts_exact(len(df), ratios)

    res = random_row_split(df=df, ratios=ratios, seed=seed, id_col=id_col, group_col=group_col)
    res.method_used = "stratified"
    res.train_idx = pd.Index(split_to_indices["train"])
    res.val_idx = pd.Index(split_to_indices["val"])
    res.test_idx = pd.Index(split_to_indices["test"])
    res.target_counts = target_global

    overlap_rows_pass = (
        len(set(res.train_idx).intersection(res.val_idx)) == 0
        and len(set(res.train_idx).intersection(res.test_idx)) == 0
        and len(set(res.val_idx).intersection(res.test_idx)) == 0
    )
    res.overlap_rows_pass = overlap_rows_pass

    split_frames = []
    for name, idx in (("train", res.train_idx), ("val", res.val_idx), ("test", res.test_idx)):
        block = df.loc[idx, [id_col]].copy()
        block["split"] = name
        block["_idx"] = idx.values
        if group_col in df.columns:
            block["group_id"] = df.loc[idx, group_col].astype(str).values
        else:
            block["group_id"] = block["_idx"].astype(str)
        split_frames.append(block)
    mapping = pd.concat(split_frames, ignore_index=True)

    grp = mapping.groupby("group_id")
    gt = grp.agg(group_size=("_idx", "count")).reset_index()
    gt["assigned_split"] = grp["split"].agg(lambda x: x.iloc[0] if x.nunique() == 1 else "mixed").values
    gt["representative_smiles"] = None

    if group_col in df.columns:
        split_groups = {
            "train": set(df.loc[res.train_idx, group_col].astype(str)),
            "val": set(df.loc[res.val_idx, group_col].astype(str)),
            "test": set(df.loc[res.test_idx, group_col].astype(str)),
        }
        res.overlap_groups_pass = (
            len(split_groups["train"].intersection(split_groups["val"])) == 0
            and len(split_groups["train"].intersection(split_groups["test"])) == 0
            and len(split_groups["val"].intersection(split_groups["test"])) == 0
        )
    else:
        res.overlap_groups_pass = True

    res.groups_table = gt[["group_id", "group_size", "assigned_split", "representative_smiles"]]
    res.predesignation = mapping[[id_col, "group_id", "split"]].copy()
    return res


def get_split_dataframes(df: pd.DataFrame, result: SplitResult) -> Dict[str, pd.DataFrame]:
    split_map = {
        "train": df.loc[result.train_idx].copy(),
        "val": df.loc[result.val_idx].copy(),
        "test": df.loc[result.test_idx].copy(),
    }
    return split_map


def summarize_seed(
    split_map: Dict[str, pd.DataFrame],
    seed: int,
    method: str,
    group_col: str,
    target_col: str,
    label_col: str,
    result: SplitResult,
) -> pd.DataFrame:
    rows = []
    total_rows = sum(len(v) for v in split_map.values())
    for split_name in SPLITS:
        sdf = split_map[split_name]
        row = {
            "seed": seed,
            "method": method,
            "split": split_name,
            "rows": len(sdf),
            "ratio_achieved": (len(sdf) / total_rows) if total_rows > 0 else np.nan,
            "unique_groups": sdf[group_col].astype(str).nunique() if group_col in sdf.columns else np.nan,
            "pIC50_mean": sdf[target_col].mean() if target_col in sdf.columns else np.nan,
            "pIC50_std": sdf[target_col].std() if target_col in sdf.columns else np.nan,
            "pIC50_min": sdf[target_col].min() if target_col in sdf.columns else np.nan,
            "pIC50_max": sdf[target_col].max() if target_col in sdf.columns else np.nan,
            "n_active": np.nan,
            "n_inactive": np.nan,
            "pct_active": np.nan,
            "group_size_min": np.nan,
            "group_size_median": np.nan,
            "group_size_max": np.nan,
            "row_overlap_check_pass": bool(result.overlap_rows_pass),
            "group_overlap_check_pass": bool(result.overlap_groups_pass),
            "warnings": " | ".join(result.warnings) if result.warnings else "",
        }

        if group_col in sdf.columns and not sdf.empty:
            gsz = sdf.groupby(group_col).size()
            row["group_size_min"] = gsz.min()
            row["group_size_median"] = float(gsz.median())
            row["group_size_max"] = gsz.max()

        if label_col in sdf.columns:
            active = int((sdf[label_col] == 1).sum())
            inactive = int((sdf[label_col] == 0).sum())
            denom = active + inactive
            row["n_active"] = active
            row["n_inactive"] = inactive
            row["pct_active"] = (100.0 * active / denom) if denom else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def init_plot_style(style: PlotStyle) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = style.base_font
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"


def apply_axis_style(ax: plt.Axes, style: PlotStyle, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=style.title_size, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=style.label_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=style.label_size, fontweight="bold")
    ax.tick_params(labelsize=style.tick_size)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def save_figure(fig: plt.Figure, out_svg: Path, out_png: Optional[Path] = None) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)


def plot_seed_artifacts(
    summary_df: pd.DataFrame,
    split_map: Dict[str, pd.DataFrame],
    groups_table: pd.DataFrame,
    seed: int,
    target_col: str,
    label_col: str,
    svg_dir: Path,
    png_dir: Optional[Path],
    style: PlotStyle,
) -> None:
    counts = summary_df.set_index("split")["rows"].reindex(SPLITS)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(SPLITS, counts.values, color=[SPLIT_COLORS[s] for s in SPLITS], edgecolor="black", linewidth=1.2)
    apply_axis_style(ax, style, f"Split Counts (Seed {seed})", "Split", "Row Count")
    save_figure(
        fig,
        svg_dir / f"split_counts_seed_{seed}.svg",
        (png_dir / f"split_counts_seed_{seed}.png") if png_dir else None,
    )

    if target_col in next(iter(split_map.values())).columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = 30
        for split_name in SPLITS:
            values = split_map[split_name][target_col].dropna().values
            if len(values) > 0:
                ax.hist(
                    values,
                    bins=bins,
                    alpha=0.5,
                    label=split_name,
                    color=SPLIT_COLORS[split_name],
                    edgecolor="black",
                    linewidth=0.8,
                )
        apply_axis_style(ax, style, f"pIC50 Distribution by Split (Seed {seed})", "pIC50", "Frequency")
        leg = ax.legend(fontsize=style.legend_size, frameon=True)
        for txt in leg.get_texts():
            txt.set_fontweight("bold")
        save_figure(
            fig,
            svg_dir / f"pIC50_distribution_by_split_seed_{seed}.svg",
            (png_dir / f"pIC50_distribution_by_split_seed_{seed}.png") if png_dir else None,
        )

    if label_col in next(iter(split_map.values())).columns:
        sr = summary_df.set_index("split")["pct_active"].reindex(SPLITS)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(SPLITS, sr.values, color=[SPLIT_COLORS[s] for s in SPLITS], edgecolor="black", linewidth=1.2)
        apply_axis_style(ax, style, f"% Active by Split (Seed {seed})", "Split", "% Active")
        save_figure(
            fig,
            svg_dir / f"active_ratio_by_split_seed_{seed}.svg",
            (png_dir / f"active_ratio_by_split_seed_{seed}.png") if png_dir else None,
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    g_sizes = groups_table["group_size"].dropna().values
    if len(g_sizes) == 0:
        g_sizes = np.array([0])
    ax.hist(g_sizes, bins=min(30, max(5, int(np.sqrt(len(g_sizes))))), color=PALETTE["purple"], edgecolor="black")
    apply_axis_style(ax, style, f"Scaffold Group Size Distribution (Seed {seed})", "Group Size", "Frequency")
    save_figure(
        fig,
        svg_dir / f"group_size_distribution_seed_{seed}.svg",
        (png_dir / f"group_size_distribution_seed_{seed}.png") if png_dir else None,
    )


def plot_aggregate(
    all_seed_summary: pd.DataFrame,
    label_col_exists: bool,
    svg_dir: Path,
    png_dir: Optional[Path],
    style: PlotStyle,
) -> None:
    agg_counts = all_seed_summary.groupby("split")["rows"].agg(["mean", "std"]).reindex(SPLITS)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SPLITS))
    ax.bar(
        x,
        agg_counts["mean"].values,
        yerr=agg_counts["std"].fillna(0).values,
        capsize=5,
        color=[SPLIT_COLORS[s] for s in SPLITS],
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    apply_axis_style(ax, style, "Split Counts Across Seeds (Mean ± SD)", "Split", "Row Count")
    save_figure(
        fig,
        svg_dir / "split_counts_across_seeds.svg",
        (png_dir / "split_counts_across_seeds.png") if png_dir else None,
    )

    if label_col_exists and "pct_active" in all_seed_summary.columns:
        agg_active = all_seed_summary.groupby("split")["pct_active"].agg(["mean", "std"]).reindex(SPLITS)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            x,
            agg_active["mean"].values,
            yerr=agg_active["std"].fillna(0).values,
            capsize=5,
            color=[SPLIT_COLORS[s] for s in SPLITS],
            edgecolor="black",
            linewidth=1.2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(SPLITS)
        apply_axis_style(ax, style, "% Active Across Seeds (Mean ± SD)", "Split", "% Active")
        save_figure(
            fig,
            svg_dir / "active_ratio_across_seeds.svg",
            (png_dir / "active_ratio_across_seeds.png") if png_dir else None,
        )


def write_manifest(
    manifest_rows: List[Dict[str, object]],
    outdir: Path,
) -> None:
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    txt_path = outdir / "run_manifest.txt"
    lines = ["QSAR split run manifest", "=" * 80, ""]
    for _, row in manifest_df.iterrows():
        lines.append(
            " | ".join(
                [
                    f"seed={row['seed']}",
                    f"method={row['method_used']}",
                    f"train={row['train_count']}",
                    f"val={row['val_count']}",
                    f"test={row['test_count']}",
                    f"warnings={row['warnings']}",
                ]
            )
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    try:
        seeds = parse_seeds(args.seeds)
        ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    except Exception as exc:
        raise SystemExit(f"[ERROR] Argument validation failed: {exc}") from exc

    random.seed(0)
    np.random.seed(0)

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"[ERROR] Input file not found: {input_path}")

    print(f"[INFO] Reading input dataset: {input_path}")
    df = pd.read_csv(input_path)
    check_required_columns(df, [args.id_col, args.smiles_col])

    if args.method == "scaffold":
        df = ensure_group_column(df, args.group_col, args.smiles_col)

    file_hash = sha256_file(input_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    style = PlotStyle(
        title_size=args.title_size,
        label_size=args.label_size,
        tick_size=args.tick_size,
        legend_size=args.legend_size,
        base_font=args.base_font,
    )
    init_plot_style(style)

    plots_svg_dir = outdir / "plots" / "svg"
    plots_png_dir = (outdir / "plots" / "png") if args.png else None
    plots_svg_dir.mkdir(parents=True, exist_ok=True)
    if plots_png_dir:
        plots_png_dir.mkdir(parents=True, exist_ok=True)

    all_seed_summary_rows: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    split_index_rows: List[Dict[str, object]] = []

    print(f"[INFO] Running split method='{args.method}' for seeds: {seeds}")
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        if args.method == "scaffold":
            result = greedy_group_split(
                df=df,
                group_col=args.group_col,
                ratios=ratios,
                seed=seed,
                min_group_size=args.min_group_size,
                id_col=args.id_col,
            )
        elif args.method == "random":
            result = random_row_split(df=df, ratios=ratios, seed=seed, id_col=args.id_col, group_col=args.group_col)
        else:
            result = stratified_split(
                df=df,
                ratios=ratios,
                seed=seed,
                id_col=args.id_col,
                group_col=args.group_col,
                label_col=args.label_col,
            )

        seed_dir = outdir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        split_map = get_split_dataframes(df, result)

        for split_name in SPLITS:
            sdf = split_map[split_name].copy()
            sdf["split"] = split_name
            sdf["seed"] = seed
            out_path = seed_dir / f"{split_name}.csv"
            sdf.to_csv(out_path, index=False)
            split_index_rows.append({"seed": seed, "split": split_name, "path": str(out_path)})

        summary_df = summarize_seed(
            split_map=split_map,
            seed=seed,
            method=result.method_used,
            group_col=args.group_col,
            target_col=args.target_col,
            label_col=args.label_col,
            result=result,
        )
        summary_df.to_csv(seed_dir / "split_summary.csv", index=False)

        result.groups_table.to_csv(seed_dir / "groups_table.csv", index=False)
        result.predesignation.to_csv(seed_dir / "predesignation_table.csv", index=False)

        all_seed_summary_rows.append(summary_df)

        unique_groups = {
            "train": int(split_map["train"][args.group_col].astype(str).nunique()) if args.group_col in split_map["train"].columns else np.nan,
            "val": int(split_map["val"][args.group_col].astype(str).nunique()) if args.group_col in split_map["val"].columns else np.nan,
            "test": int(split_map["test"][args.group_col].astype(str).nunique()) if args.group_col in split_map["test"].columns else np.nan,
        }

        manifest_rows.append(
            {
                "timestamp": now_iso,
                "input_filename": str(input_path),
                "input_sha256": file_hash,
                "total_rows_input": int(len(df)),
                "method_used": result.method_used,
                "seed": seed,
                "train_ratio": ratios["train"],
                "val_ratio": ratios["val"],
                "test_ratio": ratios["test"],
                "group_col": args.group_col,
                "dropped_small_groups_count": result.dropped_small_groups_count,
                "dropped_small_groups_rows": result.dropped_small_groups_rows,
                "train_count": len(split_map["train"]),
                "val_count": len(split_map["val"]),
                "test_count": len(split_map["test"]),
                "unique_groups_train": unique_groups["train"],
                "unique_groups_val": unique_groups["val"],
                "unique_groups_test": unique_groups["test"],
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "rdkit_version": rdBase.rdkitVersion if RDKIT_AVAILABLE else "not_available",
                "warnings": " | ".join(result.warnings) if result.warnings else "",
            }
        )

        should_plot_seed = args.plots_all_seeds or seed == seeds[0]
        if should_plot_seed:
            plot_seed_artifacts(
                summary_df=summary_df,
                split_map=split_map,
                groups_table=result.groups_table,
                seed=seed,
                target_col=args.target_col,
                label_col=args.label_col,
                svg_dir=plots_svg_dir,
                png_dir=plots_png_dir,
                style=style,
            )

        print(
            f"[INFO] Seed {seed}: train={len(split_map['train'])}, "
            f"val={len(split_map['val'])}, test={len(split_map['test'])}, method_used={result.method_used}"
        )

    all_seed_summary = pd.concat(all_seed_summary_rows, ignore_index=True)
    all_seed_summary.to_csv(outdir / "all_seeds_summary.csv", index=False)
    pd.DataFrame(split_index_rows).to_csv(outdir / "splits_index.csv", index=False)
    write_manifest(manifest_rows=manifest_rows, outdir=outdir)

    label_exists = args.label_col in df.columns
    plot_aggregate(
        all_seed_summary=all_seed_summary,
        label_col_exists=label_exists,
        svg_dir=plots_svg_dir,
        png_dir=plots_png_dir,
        style=style,
    )

    print(f"[INFO] Split files written under: {outdir.resolve()}")
    print(f"[INFO] Plots written under: {plots_svg_dir.resolve()}")
    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
