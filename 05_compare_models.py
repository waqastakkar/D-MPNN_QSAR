#!/usr/bin/env python3
"""Compare baseline QSAR models versus D-MPNN across seeds and scaffold splits."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import ttest_rel, wilcoxon

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import rdkit

    RDKIT_AVAILABLE = True
except Exception as exc:
    RDKIT_AVAILABLE = False
    RDKIT_IMPORT_ERROR = exc


PALETTE = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}


@dataclass
class Args:
    splits_dir: Path
    baselines_dir: Path
    dmpnn_dir: Path
    outdir: Path
    seeds: List[int]
    primary_metric: str
    split: str
    baseline_pick: str
    dmpnn_name: str
    run_wilcoxon: bool
    run_paired_ttest: bool
    run_bootstrap_ci: bool
    bootstrap_n: int
    confidence: float
    run_ad: bool
    ad_radius: int
    ad_nbits: int
    title_size: int
    label_size: int
    tick_size: int
    legend_size: int
    base_font: int
    svg_only: bool
    png: bool


def parse_bool_flag(value: str) -> bool:
    """Parse permissive string booleans for CLI flags."""
    value_l = str(value).strip().lower()
    if value_l in {"1", "true", "t", "yes", "y"}:
        return True
    if value_l in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> Args:
    """Parse CLI arguments and return structured args."""
    parser = argparse.ArgumentParser(description="Compare baseline QSAR models with D-MPNN.")
    parser.add_argument("--splits_dir", type=Path, default=Path("splits"))
    parser.add_argument("--baselines_dir", type=Path, default=Path("baselines_out"))
    parser.add_argument("--dmpnn_dir", type=Path, default=Path("dmpnn_out"))
    parser.add_argument("--outdir", type=Path, default=Path("compare_out"))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--primary_metric", choices=["rmse", "mae", "r2", "pearson"], default="rmse")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--baseline_pick", default="auto")
    parser.add_argument("--dmpnn_name", default="dmpnn")
    parser.add_argument("--run_wilcoxon", type=parse_bool_flag, default=True)
    parser.add_argument("--run_paired_ttest", type=parse_bool_flag, default=True)
    parser.add_argument("--run_bootstrap_ci", type=parse_bool_flag, default=True)
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--run_ad", type=parse_bool_flag, default=True)
    parser.add_argument("--ad_radius", type=int, default=2)
    parser.add_argument("--ad_nbits", type=int, default=2048)
    parser.add_argument("--title_size", type=int, default=18)
    parser.add_argument("--label_size", type=int, default=16)
    parser.add_argument("--tick_size", type=int, default=14)
    parser.add_argument("--legend_size", type=int, default=14)
    parser.add_argument("--base_font", type=int, default=14)
    parser.add_argument("--svg_only", type=parse_bool_flag, default=True)
    parser.add_argument("--png", type=parse_bool_flag, default=False)

    ns = parser.parse_args()
    seeds = [int(s.strip()) for s in ns.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided in --seeds.")
    return Args(
        splits_dir=ns.splits_dir,
        baselines_dir=ns.baselines_dir,
        dmpnn_dir=ns.dmpnn_dir,
        outdir=ns.outdir,
        seeds=seeds,
        primary_metric=ns.primary_metric,
        split=ns.split,
        baseline_pick=ns.baseline_pick,
        dmpnn_name=ns.dmpnn_name,
        run_wilcoxon=ns.run_wilcoxon,
        run_paired_ttest=ns.run_paired_ttest,
        run_bootstrap_ci=ns.run_bootstrap_ci,
        bootstrap_n=ns.bootstrap_n,
        confidence=ns.confidence,
        run_ad=ns.run_ad,
        ad_radius=ns.ad_radius,
        ad_nbits=ns.ad_nbits,
        title_size=ns.title_size,
        label_size=ns.label_size,
        tick_size=ns.tick_size,
        legend_size=ns.legend_size,
        base_font=ns.base_font,
        svg_only=ns.svg_only,
        png=ns.png,
    )


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_columns(df: pd.DataFrame, required: Sequence[str], source_name: str) -> None:
    """Ensure required columns are present in DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_name}: {missing}")


def identify_id_column(df: pd.DataFrame) -> Optional[str]:
    """Choose identifier column if present."""
    candidates = ["molecule_chembl_id", "id", "molecule_id", "compound_id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def identify_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Choose smiles column if present."""
    candidates = ["rdkit_canonical_smiles", "smiles", "canonical_smiles"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_standardize_predictions(path: Path, default_model: Optional[str] = None) -> pd.DataFrame:
    """Load predictions file and normalize key columns/types."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path)
    ensure_columns(df, ["seed", "split", "y_true", "y_pred"], str(path))
    if "model" not in df.columns:
        if default_model is None:
            raise ValueError(f"Column 'model' missing and no default provided in {path}")
        df["model"] = default_model
    df["seed"] = df["seed"].astype(int)
    df["split"] = df["split"].astype(str)
    df["model"] = df["model"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="raise")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="raise")
    return df


def filter_split_and_seeds(df: pd.DataFrame, split: str, seeds: Sequence[int], name: str) -> pd.DataFrame:
    """Filter predictions by split and seeds with validation."""
    out = df[(df["split"] == split) & (df["seed"].isin(seeds))].copy()
    if out.empty:
        raise ValueError(f"No rows found for split='{split}' and seeds={list(seeds)} in {name}")
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson}


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, confidence: float, rng: np.random.Generator) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    if values.size == 0:
        return np.nan, np.nan
    means = np.empty(n_bootstrap, dtype=float)
    n = values.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = np.nanmean(values[idx])
    alpha = 1.0 - confidence
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def select_baseline_model(baseline_df: pd.DataFrame, split: str, baseline_pick: str) -> str:
    """Select baseline model either explicitly or by lowest mean RMSE."""
    available = sorted(baseline_df["model"].unique().tolist())
    if baseline_pick != "auto":
        if baseline_pick not in available:
            raise ValueError(f"Requested baseline model '{baseline_pick}' not found. Available: {available}")
        return baseline_pick

    rows: List[Dict[str, object]] = []
    for (seed, model), grp in baseline_df.groupby(["seed", "model"]):
        metrics = compute_metrics(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
        rows.append({"seed": seed, "model": model, **metrics})
    met_df = pd.DataFrame(rows)
    agg = met_df.groupby("model", as_index=False)["rmse"].mean().sort_values("rmse", ascending=True)
    if agg.empty:
        raise ValueError(f"Could not select baseline model for split '{split}': no data after filtering.")
    return str(agg.iloc[0]["model"])


def align_two_models_for_seed(
    base_seed_df: pd.DataFrame,
    dmp_seed_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Align baseline and D-MPNN rows for one seed by ID if possible, else row order."""
    id_col = identify_id_column(base_seed_df) or identify_id_column(dmp_seed_df)

    if id_col and id_col in base_seed_df.columns and id_col in dmp_seed_df.columns:
        b = base_seed_df.copy()
        d = dmp_seed_df.copy()
        b[id_col] = b[id_col].astype(str)
        d[id_col] = d[id_col].astype(str)
        merged = pd.merge(
            b,
            d,
            on=id_col,
            suffixes=("_baseline", "_dmpnn"),
            how="inner",
        )
        if merged.empty:
            raise ValueError(f"Seed {seed}: no overlap found between baseline and dmpnn using id '{id_col}'.")
        if not np.allclose(merged["y_true_baseline"].to_numpy(), merged["y_true_dmpnn"].to_numpy(), equal_nan=False):
            raise ValueError(
                f"Seed {seed}: y_true mismatch between baseline and dmpnn after ID-based alignment on '{id_col}'."
            )
        merged["alignment_mode"] = "id"
        merged["alignment_id_col"] = id_col
        return merged

    print(f"[WARN] Seed {seed}: no common ID column; aligning baseline and dmpnn by row order.")
    if len(base_seed_df) != len(dmp_seed_df):
        raise ValueError(
            f"Seed {seed}: cannot align by row order due to different row counts "
            f"(baseline={len(base_seed_df)}, dmpnn={len(dmp_seed_df)})."
        )

    b = base_seed_df.reset_index(drop=True).copy()
    d = dmp_seed_df.reset_index(drop=True).copy()
    if not np.allclose(b["y_true"].to_numpy(), d["y_true"].to_numpy(), equal_nan=False):
        raise ValueError(f"Seed {seed}: y_true mismatch when aligned by row order.")

    merged = pd.DataFrame(
        {
            "row_id": np.arange(len(b)),
            "y_true_baseline": b["y_true"].to_numpy(),
            "y_pred_baseline": b["y_pred"].to_numpy(),
            "y_true_dmpnn": d["y_true"].to_numpy(),
            "y_pred_dmpnn": d["y_pred"].to_numpy(),
        }
    )
    for extra_col in ["rdkit_canonical_smiles", "smiles"]:
        if extra_col in b.columns:
            merged[f"{extra_col}_baseline"] = b[extra_col].to_numpy()
        if extra_col in d.columns:
            merged[f"{extra_col}_dmpnn"] = d[extra_col].to_numpy()
    merged["alignment_mode"] = "row_order"
    merged["alignment_id_col"] = ""
    return merged


def set_plot_style(args: Args) -> None:
    """Apply global plotting style constraints."""
    rcParams["svg.fonttype"] = "none"
    rcParams["font.family"] = "Times New Roman"
    rcParams["font.size"] = args.base_font
    rcParams["font.weight"] = "bold"
    rcParams["axes.labelweight"] = "bold"
    rcParams["axes.titleweight"] = "bold"


def style_axis(ax: plt.Axes, args: Args) -> None:
    """Apply axis-level typography style."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(args.tick_size)
        label.set_fontweight("bold")
    ax.title.set_fontsize(args.title_size)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(args.label_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(args.label_size)
    ax.yaxis.label.set_fontweight("bold")


def save_figure(fig: plt.Figure, out_svg: Path, args: Args) -> None:
    """Save figure in required deterministic formats."""
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    if args.png and not args.svg_only:
        fig.savefig(out_svg.with_suffix(".png"), dpi=300)
    plt.close(fig)


def compute_nn_similarity(train_smiles: Sequence[str], test_smiles: Sequence[str], radius: int, nbits: int) -> np.ndarray:
    """Compute nearest-neighbor Tanimoto similarity from test to train set."""
    if not RDKIT_AVAILABLE:
        raise RuntimeError(f"RDKit is required for AD analysis but unavailable: {RDKIT_IMPORT_ERROR}")

    train_fps = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            continue
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
    if not train_fps:
        raise ValueError("No valid training fingerprints could be created for AD analysis.")

    sims = np.full(len(test_smiles), np.nan, dtype=float)
    for i, smi in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        similarities = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        sims[i] = max(similarities) if similarities else np.nan
    return sims


def binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute binned x/y means for trend lines."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    mids = []
    means = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sel = (x >= lo) & (x <= hi)
        else:
            sel = (x >= lo) & (x < hi)
        if np.any(sel):
            mids.append((lo + hi) / 2.0)
            means.append(float(np.nanmean(y[sel])))
    return np.array(mids), np.array(means)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    set_plot_style(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_svg_dir = args.outdir / "plots" / "svg"
    plot_svg_dir.mkdir(parents=True, exist_ok=True)

    print("[1/10] Loading predictions...")
    baseline_path = args.baselines_dir / "all_predictions_regression.csv"
    dmpnn_path = args.dmpnn_dir / "all_predictions_regression.csv"

    baseline_raw = load_and_standardize_predictions(baseline_path)
    dmpnn_raw = load_and_standardize_predictions(dmpnn_path, default_model=args.dmpnn_name)

    baseline_df = filter_split_and_seeds(baseline_raw, args.split, args.seeds, "baselines")
    dmpnn_df = filter_split_and_seeds(dmpnn_raw, args.split, args.seeds, "dmpnn")

    if dmpnn_df["model"].nunique() > 1:
        print(f"[WARN] D-MPNN file has multiple model labels: {dmpnn_df['model'].unique().tolist()}; forcing to {args.dmpnn_name}.")
    dmpnn_df["model"] = args.dmpnn_name

    baseline_df.to_csv(args.outdir / "standardized_baselines_predictions.csv", index=False)
    dmpnn_df.to_csv(args.outdir / "standardized_dmpnn_predictions.csv", index=False)

    print("[2/10] Selecting baseline model...")
    chosen_baseline = select_baseline_model(baseline_df, args.split, args.baseline_pick)
    print(f"Selected baseline: {chosen_baseline}")

    print("[3/10] Computing per-seed metrics for all models...")
    all_df = pd.concat([baseline_df, dmpnn_df], ignore_index=True)
    per_seed_rows = []
    for (seed, model, split), grp in all_df.groupby(["seed", "model", "split"]):
        m = compute_metrics(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
        per_seed_rows.append({"seed": int(seed), "model": model, "split": split, **m})
    per_seed_metrics = pd.DataFrame(per_seed_rows).sort_values(["seed", "model"]) if per_seed_rows else pd.DataFrame()
    if per_seed_metrics.empty:
        raise ValueError("No per-seed metrics computed. Check input predictions.")
    per_seed_metrics.to_csv(args.outdir / "per_seed_metrics.csv", index=False)

    print("[4/10] Computing aggregate metrics and winner...")
    rng = np.random.default_rng(2025)
    agg_rows = []
    metrics_list = ["rmse", "mae", "r2", "pearson"]
    for (model, split), grp in per_seed_metrics.groupby(["model", "split"]):
        for metric in metrics_list:
            vals = grp[metric].to_numpy(dtype=float)
            ci_low, ci_high = (np.nan, np.nan)
            if args.run_bootstrap_ci:
                ci_low, ci_high = bootstrap_ci(vals, args.bootstrap_n, args.confidence, rng)
            agg_rows.append(
                {
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_seeds": int(len(vals)),
                }
            )
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(args.outdir / "aggregate_metrics.csv", index=False)

    split_primary = aggregate[(aggregate["split"] == args.split) & (aggregate["metric"] == args.primary_metric)].copy()
    if split_primary.empty:
        raise ValueError("No aggregate rows for primary metric.")
    lower_better = args.primary_metric in {"rmse", "mae"}
    split_primary = split_primary.sort_values("mean", ascending=lower_better)
    winner_model = str(split_primary.iloc[0]["model"])
    pd.DataFrame(
        [{"primary_metric": args.primary_metric, "split": args.split, "winner_model": winner_model}]
    ).to_csv(args.outdir / "winner_summary.csv", index=False)

    print("[5/10] Aligning chosen baseline vs dmpnn for paired analyses...")
    base_sel = baseline_df[baseline_df["model"] == chosen_baseline].copy()
    dmp_sel = dmpnn_df.copy()

    aligned_seed_dfs: List[pd.DataFrame] = []
    for seed in args.seeds:
        b_seed = base_sel[base_sel["seed"] == seed].copy()
        d_seed = dmp_sel[dmp_sel["seed"] == seed].copy()
        if b_seed.empty or d_seed.empty:
            raise ValueError(f"Missing baseline or dmpnn rows for seed {seed} on split {args.split}.")
        aligned = align_two_models_for_seed(b_seed, d_seed, seed)
        aligned["seed"] = seed
        aligned_seed_dfs.append(aligned)

    aligned_df = pd.concat(aligned_seed_dfs, ignore_index=True)

    print("[6/10] Statistical tests on per-seed paired primary metric...")
    stat_rows = []
    pair_metrics = per_seed_metrics[(per_seed_metrics["split"] == args.split) & (per_seed_metrics["model"].isin([chosen_baseline, args.dmpnn_name]))]
    pivot = pair_metrics.pivot(index="seed", columns="model", values=args.primary_metric)
    if chosen_baseline not in pivot.columns or args.dmpnn_name not in pivot.columns:
        raise ValueError("Could not build paired per-seed vectors for stats tests.")
    paired = pivot[[chosen_baseline, args.dmpnn_name]].dropna()
    bvals = paired[chosen_baseline].to_numpy(dtype=float)
    dvals = paired[args.dmpnn_name].to_numpy(dtype=float)
    direction = "dmpnn_better" if (
        np.nanmean(dvals) < np.nanmean(bvals) if lower_better else np.nanmean(dvals) > np.nanmean(bvals)
    ) else "baseline_better"

    if args.run_wilcoxon:
        if SCIPY_AVAILABLE:
            try:
                stat, pval = wilcoxon(bvals, dvals)
                stat_rows.append(
                    {
                        "metric": args.primary_metric,
                        "test_name": "wilcoxon_signed_rank",
                        "n": len(bvals),
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "baseline_mean": float(np.nanmean(bvals)),
                        "dmpnn_mean": float(np.nanmean(dvals)),
                        "direction": direction,
                    }
                )
            except ValueError as exc:
                print(f"[WARN] Wilcoxon skipped: {exc}")
        else:
            print("[WARN] SciPy unavailable; Wilcoxon skipped.")

    if args.run_paired_ttest:
        if SCIPY_AVAILABLE:
            stat, pval = ttest_rel(bvals, dvals, nan_policy="omit")
            stat_rows.append(
                {
                    "metric": args.primary_metric,
                    "test_name": "paired_ttest",
                    "n": len(bvals),
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "baseline_mean": float(np.nanmean(bvals)),
                    "dmpnn_mean": float(np.nanmean(dvals)),
                    "direction": direction,
                }
            )
        else:
            print("[WARN] SciPy unavailable; paired t-test skipped.")

    pd.DataFrame(stat_rows).to_csv(args.outdir / "stat_tests.csv", index=False)

    print("[7/10] Building error table...")
    id_col = identify_id_column(base_sel) if identify_id_column(base_sel) == identify_id_column(dmp_sel) else None
    smiles_col = identify_smiles_column(base_sel) or identify_smiles_column(dmp_sel)

    err_rows = []
    for seed in args.seeds:
        ali = aligned_df[aligned_df["seed"] == seed]
        if id_col and id_col in ali.columns:
            ids = ali[id_col].to_numpy()
        elif "row_id" in ali.columns:
            ids = ali["row_id"].to_numpy()
        else:
            ids = np.arange(len(ali))

        def pick_smiles(colset, suffix):
            for c in [f"smiles_{suffix}", f"rdkit_canonical_smiles_{suffix}", f"canonical_smiles_{suffix}"]:
                if c in colset:
                    return c
            return None

        smi_baseline_col = pick_smiles(set(ali.columns), "baseline")
        smi_dmpnn_col = pick_smiles(set(ali.columns), "dmpnn")

        for model_name, yt_col, yp_col, smi_col in [
            (chosen_baseline, "y_true_baseline", "y_pred_baseline", smi_baseline_col),
            (args.dmpnn_name, "y_true_dmpnn", "y_pred_dmpnn", smi_dmpnn_col),
        ]:
            y_true = ali[yt_col].to_numpy(dtype=float)
            y_pred = ali[yp_col].to_numpy(dtype=float)
            residual = y_true - y_pred
            abs_error = np.abs(residual)
            for i in range(len(ali)):
                row = {
                    "seed": seed,
                    "model": model_name,
                    "y_true": y_true[i],
                    "y_pred": y_pred[i],
                    "residual": residual[i],
                    "abs_error": abs_error[i],
                    "id": ids[i],
                }
                if smi_col and smi_col in ali.columns:
                    row["smiles"] = ali.iloc[i][smi_col]
                err_rows.append(row)
    error_df = pd.DataFrame(err_rows)
    error_df.to_csv(args.outdir / "error_table.csv", index=False)

    print("[8/10] Generating plots...")
    two_models = [chosen_baseline, args.dmpnn_name]

    # A) primary metric bar
    fig, ax = plt.subplots(figsize=(6, 5))
    prim = aggregate[(aggregate["split"] == args.split) & (aggregate["metric"] == args.primary_metric) & (aggregate["model"].isin(two_models))]
    prim = prim.set_index("model").loc[two_models].reset_index()
    ax.bar(
        prim["model"],
        prim["mean"],
        yerr=prim["std"],
        color=[PALETTE["blue"], PALETTE["orange"]],
        edgecolor="black",
        linewidth=1.0,
    )
    ax.set_title(f"Primary Metric ({args.primary_metric.upper()})")
    ax.set_xlabel("Model")
    ax.set_ylabel(args.primary_metric.upper())
    style_axis(ax, args)
    save_figure(fig, plot_svg_dir / "metric_bar_primary.svg", args)

    # B) all metrics grouped bars
    fig, ax = plt.subplots(figsize=(9, 5))
    metric_order = ["rmse", "mae", "r2", "pearson"]
    x = np.arange(len(metric_order))
    w = 0.35
    for j, model in enumerate(two_models):
        subset = aggregate[(aggregate["split"] == args.split) & (aggregate["model"] == model)].set_index("metric")
        means = [subset.loc[m, "mean"] if m in subset.index else np.nan for m in metric_order]
        stds = [subset.loc[m, "std"] if m in subset.index else 0.0 for m in metric_order]
        ax.bar(
            x + (j - 0.5) * w,
            means,
            w,
            yerr=stds,
            label=model,
            color=PALETTE["blue"] if model == chosen_baseline else PALETTE["orange"],
            edgecolor="black",
            linewidth=1.0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_order], fontweight="bold")
    ax.set_title("Model Comparison Across Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean ± SD")
    leg = ax.legend(prop={"size": args.legend_size, "weight": "bold"})
    for t in leg.get_texts():
        t.set_fontweight("bold")
    style_axis(ax, args)
    save_figure(fig, plot_svg_dir / "metric_all.svg", args)

    # C) abs error boxplot
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [
        error_df.loc[error_df["model"] == chosen_baseline, "abs_error"].to_numpy(),
        error_df.loc[error_df["model"] == args.dmpnn_name, "abs_error"].to_numpy(),
    ]
    bp = ax.boxplot(data, labels=two_models, patch_artist=True)
    for patch, color in zip(bp["boxes"], [PALETTE["blue"], PALETTE["orange"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title("Absolute Error Distribution")
    ax.set_xlabel("Model")
    ax.set_ylabel("Absolute Error")
    style_axis(ax, args)
    save_figure(fig, plot_svg_dir / "error_boxplot_abs.svg", args)

    # D) residual histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in [(chosen_baseline, PALETTE["blue"]), (args.dmpnn_name, PALETTE["orange"])]:
        vals = error_df.loc[error_df["model"] == model, "residual"].to_numpy(dtype=float)
        ax.hist(vals, bins=30, alpha=0.45, label=model, color=color, edgecolor="black", linewidth=0.4)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Count")
    leg = ax.legend(prop={"size": args.legend_size, "weight": "bold"})
    for t in leg.get_texts():
        t.set_fontweight("bold")
    style_axis(ax, args)
    save_figure(fig, plot_svg_dir / "residual_hist.svg", args)

    print("[9/10] Applicability domain analysis...")
    ad_df = pd.DataFrame()
    if args.run_ad:
        ad_rows = []
        for seed in args.seeds:
            split_seed_train = args.splits_dir / f"seed_{seed}" / "train.csv"
            if not split_seed_train.exists():
                raise FileNotFoundError(f"AD analysis requires train split file: {split_seed_train}")
            train_df = pd.read_csv(split_seed_train)
            train_smiles_col = identify_smiles_column(train_df)
            if train_smiles_col is None:
                raise ValueError(f"No SMILES column found in {split_seed_train}; expected one of rdkit_canonical_smiles/smiles.")
            train_smiles = train_df[train_smiles_col].astype(str).tolist()

            err_seed = error_df[error_df["seed"] == seed].copy()
            if "smiles" in err_seed.columns:
                smiles_for_seed = err_seed[["id", "smiles"]].dropna().drop_duplicates()
            else:
                smiles_for_seed = pd.DataFrame(columns=["id", "smiles"])
            if smiles_for_seed.empty:
                split_seed_test = args.splits_dir / f"seed_{seed}" / "test.csv"
                if not split_seed_test.exists():
                    raise FileNotFoundError(
                        f"No smiles in predictions and test split missing for seed {seed}: {split_seed_test}"
                    )
                test_df = pd.read_csv(split_seed_test)
                test_smiles_col = identify_smiles_column(test_df)
                test_id_col = identify_id_column(test_df)
                if test_smiles_col is None:
                    raise ValueError(f"No SMILES column found in {split_seed_test} for AD analysis.")
                if test_id_col is None:
                    raise ValueError(f"No ID column in {split_seed_test}; cannot map test molecules for AD analysis.")
                smiles_for_seed = test_df[[test_id_col, test_smiles_col]].rename(columns={test_id_col: "id", test_smiles_col: "smiles"})

            nn = compute_nn_similarity(
                train_smiles=train_smiles,
                test_smiles=smiles_for_seed["smiles"].astype(str).tolist(),
                radius=args.ad_radius,
                nbits=args.ad_nbits,
            )
            sim_df = smiles_for_seed.copy()
            sim_df["nn_similarity"] = nn
            merged = err_seed.merge(sim_df, on=["id", "smiles"], how="left")
            for _, r in merged.iterrows():
                ad_rows.append(
                    {
                        "seed": int(r["seed"]),
                        "model": r["model"],
                        "id": r["id"],
                        "nn_similarity": r["nn_similarity"],
                        "abs_error": r["abs_error"],
                        "y_true": r["y_true"],
                        "y_pred": r["y_pred"],
                    }
                )

        ad_df = pd.DataFrame(ad_rows)
        ad_df.to_csv(args.outdir / "applicability_domain.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 5))
        for model, color in [(chosen_baseline, PALETTE["blue"]), (args.dmpnn_name, PALETTE["orange"])]:
            sub = ad_df[ad_df["model"] == model]
            xvals = sub["nn_similarity"].to_numpy(dtype=float)
            yvals = sub["abs_error"].to_numpy(dtype=float)
            ax.scatter(xvals, yvals, s=12, alpha=0.25, color=color, label=f"{model} points")
            bx, by = binned_trend(xvals, yvals, n_bins=10)
            if bx.size > 0:
                ax.plot(bx, by, color=color, linewidth=2.5, label=f"{model} binned mean")
        ax.set_title("Absolute Error vs Nearest-Neighbor Similarity")
        ax.set_xlabel("Nearest-Neighbor Tanimoto Similarity")
        ax.set_ylabel("Absolute Error")
        leg = ax.legend(prop={"size": max(args.legend_size - 1, 8), "weight": "bold"})
        for t in leg.get_texts():
            t.set_fontweight("bold")
        style_axis(ax, args)
        save_figure(fig, plot_svg_dir / "error_vs_similarity.svg", args)
    else:
        print("[INFO] AD analysis disabled by --run_ad False.")

    print("[10/10] Writing run manifest...")
    manifest_row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seeds": ",".join(map(str, args.seeds)),
        "split": args.split,
        "chosen_baseline_model": chosen_baseline,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
        "sklearn_version": sys.modules["sklearn"].__version__,
        "rdkit_version": getattr(rdkit, "__version__", "unavailable") if RDKIT_AVAILABLE else "unavailable",
        "scipy_version": sys.modules.get("scipy").__version__ if SCIPY_AVAILABLE else "unavailable",
        "baseline_predictions_sha256": sha256_file(baseline_path),
        "dmpnn_predictions_sha256": sha256_file(dmpnn_path),
    }
    manifest_df = pd.DataFrame([manifest_row])
    manifest_df.to_csv(args.outdir / "run_manifest.csv", index=False)

    txt_path = args.outdir / "run_manifest.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for k, v in manifest_row.items():
            f.write(f"{k}: {v}\n")

    print("Output directory:", args.outdir.resolve())
    print("DONE")


if __name__ == "__main__":
    main()
