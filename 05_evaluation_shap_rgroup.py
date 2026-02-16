#!/usr/bin/env python3
"""Final evaluation + interpretability for QSAR baselines vs D-MPNN.

This script consolidates predictions, computes final metrics and statistical tests,
generates publication-style figures, and runs interpretability analyses including
SHAP, R-group/substituent analysis, MMPA-style transform summaries, and
applicability-domain diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")


NATURE_PALETTE = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}


@dataclass
class PrimaryModelSelection:
    source: str
    model_name: str


def str2bool(v: str | bool) -> bool:
    """Parse truthy/falsy strings for argparse booleans."""
    if isinstance(v, bool):
        return v
    x = str(v).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {v}")


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file path."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(*paths: Path) -> None:
    """Create directories if absent."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def configure_plot_style(args: argparse.Namespace) -> None:
    """Set mandatory figure style settings."""
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.titlesize"] = args.title_size
    mpl.rcParams["axes.labelsize"] = args.label_size
    mpl.rcParams["xtick.labelsize"] = args.tick_size
    mpl.rcParams["ytick.labelsize"] = args.tick_size
    mpl.rcParams["legend.fontsize"] = args.legend_size
    mpl.rcParams["font.size"] = args.base_font


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    p = argparse.ArgumentParser(description="Final evaluation + SHAP + R-group for QSAR")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--baselines_dir", default="baselines_out")
    p.add_argument("--dmpnn_dir", default="dmpnn_out")
    p.add_argument("--outdir", default="final_out")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--task", default="regression", choices=["regression", "classification", "both"])

    p.add_argument("--smiles_col", default="rdkit_canonical_smiles")
    p.add_argument("--id_col", default="molecule_chembl_id")
    p.add_argument("--target_col", default="pIC50")
    p.add_argument("--label_col", default="Active")

    p.add_argument("--run_shap", type=str2bool, default=True)
    p.add_argument("--run_rgroup", type=str2bool, default=True)
    p.add_argument("--run_mmpa", type=str2bool, default=True)
    p.add_argument("--max_shap_samples", type=int, default=1000)
    p.add_argument("--top_k_bits", type=int, default=50)
    p.add_argument("--top_k_rgroups", type=int, default=30)
    p.add_argument("--min_rgroup_count", type=int, default=10)
    p.add_argument("--max_cores", type=int, default=50)

    p.add_argument("--primary_model_source", default="dmpnn", choices=["dmpnn", "baselines"])
    p.add_argument("--primary_model_name", default="auto")
    p.add_argument("--fallback_train_from_splits", type=str2bool, default=True)

    p.add_argument("--title_size", type=int, default=18)
    p.add_argument("--label_size", type=int, default=16)
    p.add_argument("--tick_size", type=int, default=14)
    p.add_argument("--legend_size", type=int, default=14)
    p.add_argument("--base_font", type=int, default=14)
    p.add_argument("--svg_only", type=str2bool, default=True)
    p.add_argument("--png", type=str2bool, default=False)
    return p.parse_args()


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV if file exists, else None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def standardize_prediction_table(
    df: pd.DataFrame,
    id_col: str,
    smiles_col: str,
    target_col: str,
    default_model: str,
) -> pd.DataFrame:
    """Map prediction table into required schema."""
    table = df.copy()

    seed_col = _find_col(table, ["seed"])
    model_col = _find_col(table, ["model", "model_name", "estimator"])
    split_col = _find_col(table, ["split", "subset"])
    idc = _find_col(table, ["id", "mol_id", "molecule_id", id_col])
    smc = _find_col(table, ["smiles", "canonical_smiles", smiles_col])
    ytc = _find_col(table, ["y_true", "true", "target", target_col])
    ypc = _find_col(table, ["y_pred", "pred", "prediction", "yhat"])

    missing = []
    if seed_col is None:
        missing.append("seed")
    if split_col is None:
        missing.append("split")
    if idc is None:
        missing.append("id")
    if smc is None:
        missing.append("smiles")
    if ytc is None:
        missing.append("y_true")
    if ypc is None:
        missing.append("y_pred")
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "seed": table[seed_col],
            "model": table[model_col] if model_col is not None else default_model,
            "split": table[split_col],
            "id": table[idc],
            "smiles": table[smc],
            "y_true": pd.to_numeric(table[ytc], errors="coerce"),
            "y_pred": pd.to_numeric(table[ypc], errors="coerce"),
        }
    )
    out = out.dropna(subset=["seed", "split", "id", "smiles", "y_true", "y_pred"]).copy()
    out["seed"] = out["seed"].astype(int)
    out["model"] = out["model"].astype(str)
    out["split"] = out["split"].astype(str).str.lower()
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics with robust Pearson handling."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pear = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pear = np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pear}


def compute_per_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics grouped by model/seed/split."""
    rows: List[Dict[str, object]] = []
    for (model, seed, split), g in df.groupby(["model", "seed", "split"]):
        m = regression_metrics(g["y_true"].values, g["y_pred"].values)
        rows.append({"model": model, "seed": int(seed), "split": split, **m, "n": len(g)})
    return pd.DataFrame(rows)


def aggregate_metrics(per_seed: pd.DataFrame, split: str) -> pd.DataFrame:
    """Aggregate per-seed metrics into mean/std and display strings for a split."""
    sub = per_seed[per_seed["split"] == split].copy()
    out_rows = []
    for model, g in sub.groupby("model"):
        row: Dict[str, object] = {"model": model}
        for metric in ["RMSE", "MAE", "R2", "Pearson"]:
            mu = g[metric].mean()
            sd = g[metric].std(ddof=1)
            row[f"{metric}_mean"] = mu
            row[f"{metric}_std"] = sd
            row[f"{metric}_mean_std"] = f"{mu:.4f}±{(0.0 if np.isnan(sd) else sd):.4f}"
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["RMSE_mean", "R2_mean"], ascending=[True, False]).reset_index(drop=True)
    return out


def choose_primary_model(
    per_seed: pd.DataFrame,
    source: str,
    primary_model_name: str,
    baseline_models: set[str],
    dmpnn_models: set[str],
) -> PrimaryModelSelection:
    """Choose primary model according to CLI rules."""
    allowed = dmpnn_models if source == "dmpnn" else baseline_models
    if primary_model_name != "auto":
        if primary_model_name not in allowed:
            raise ValueError(
                f"Requested primary model '{primary_model_name}' not found in source={source}."
            )
        return PrimaryModelSelection(source=source, model_name=primary_model_name)

    sub = per_seed[(per_seed["split"] == "test") & (per_seed["model"].isin(allowed))]
    if sub.empty:
        raise ValueError(f"No test rows available for primary model selection from source={source}.")
    agg = sub.groupby("model").agg(RMSE_mean=("RMSE", "mean"), R2_mean=("R2", "mean")).reset_index()
    agg = agg.sort_values(["RMSE_mean", "R2_mean"], ascending=[True, False])
    return PrimaryModelSelection(source=source, model_name=str(agg.iloc[0]["model"]))


def save_fig(fig: plt.Figure, out_svg: Path, out_png: Optional[Path], dpi: int = 300) -> None:
    """Save figure with required outputs."""
    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_model_rmse_plot(comparison_test: pd.DataFrame, plots_svg: Path, plots_png: Optional[Path]) -> None:
    """Plot model comparison RMSE bars."""
    if comparison_test.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_test))
    colors = [NATURE_PALETTE[c] for c in ["blue", "orange", "green", "purple", "cyan", "red2"]]
    ax.bar(
        x,
        comparison_test["RMSE_mean"].values,
        yerr=np.nan_to_num(comparison_test["RMSE_std"].values, nan=0.0),
        color=[colors[i % len(colors)] for i in x],
        alpha=0.9,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_test["model"].tolist(), rotation=30, ha="right", fontweight="bold")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Model comparison on test set (RMSE)")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    save_fig(
        fig,
        plots_svg / "paper_model_comparison_rmse_test.svg",
        None if plots_png is None else plots_png / "paper_model_comparison_rmse_test.png",
    )


def make_parity_plot(df_test: pd.DataFrame, model_name: str, plots_svg: Path, plots_png: Optional[Path]) -> None:
    """Parity plot for selected model on test."""
    data = df_test[df_test["model"] == model_name]
    if data.empty:
        return
    y_true = data["y_true"].values
    y_pred = data["y_pred"].values
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=24, c=NATURE_PALETTE["blue"], alpha=0.7, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "--", color=NATURE_PALETTE["orange"], linewidth=2)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Parity plot (test): {model_name}")
    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontweight("bold")
    save_fig(
        fig,
        plots_svg / "paper_parity_plot_best_model_test.svg",
        None if plots_png is None else plots_png / "paper_parity_plot_best_model_test.png",
    )


def make_residuals_plot(df_test: pd.DataFrame, model_name: str, plots_svg: Path, plots_png: Optional[Path]) -> None:
    """Residual plot for selected model on test."""
    data = df_test[df_test["model"] == model_name].copy()
    if data.empty:
        return
    data["residual"] = data["y_pred"] - data["y_true"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(data["y_pred"], data["residual"], c=NATURE_PALETTE["green"], s=24, alpha=0.7, edgecolor="none")
    ax.axhline(0.0, color=NATURE_PALETTE["orange"], linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred - obs)")
    ax.set_title(f"Residuals (test): {model_name}")
    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontweight("bold")
    save_fig(
        fig,
        plots_svg / "paper_residuals_best_model_test.svg",
        None if plots_png is None else plots_png / "paper_residuals_best_model_test.png",
    )


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048) -> Optional[DataStructs.ExplicitBitVect]:
    """Build Morgan bit vector from SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def fingerprint_matrix(smiles_list: Sequence[str], radius: int = 2, nbits: int = 2048) -> Tuple[np.ndarray, List[int]]:
    """Convert SMILES to dense numpy matrix of Morgan bits."""
    fps = []
    valid_idx = []
    for i, s in enumerate(smiles_list):
        fp = morgan_fp(s, radius=radius, nbits=nbits)
        if fp is None:
            continue
        arr = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_idx.append(i)
    if not fps:
        return np.zeros((0, nbits), dtype=np.int8), []
    return np.vstack(fps), valid_idx


def compute_applicability_domain(
    pred_df: pd.DataFrame,
    primary_model: str,
    seeds: Sequence[int],
    out_csv: Path,
) -> pd.DataFrame:
    """Compute abs error vs nearest train neighbor similarity per seed."""
    rows = []
    model_df = pred_df[pred_df["model"] == primary_model].copy()
    for seed in seeds:
        train = model_df[(model_df["seed"] == seed) & (model_df["split"] == "train")]
        test = model_df[(model_df["seed"] == seed) & (model_df["split"] == "test")]
        if train.empty or test.empty:
            continue
        train_fps = [morgan_fp(s) for s in train["smiles"].tolist()]
        train_fps = [fp for fp in train_fps if fp is not None]
        if not train_fps:
            continue
        for _, r in test.iterrows():
            fp = morgan_fp(r["smiles"])
            if fp is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            nn = float(max(sims) if sims else np.nan)
            rows.append(
                {
                    "seed": seed,
                    "id": r["id"],
                    "smiles": r["smiles"],
                    "y_true": float(r["y_true"]),
                    "y_pred": float(r["y_pred"]),
                    "abs_error": abs(float(r["y_pred"] - r["y_true"])),
                    "nn_similarity": nn,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def make_error_similarity_plot(df: pd.DataFrame, plots_svg: Path, plots_png: Optional[Path]) -> None:
    """Scatter of error vs nearest-neighbor similarity."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["nn_similarity"], df["abs_error"], c=NATURE_PALETTE["purple"], alpha=0.6, s=22, edgecolor="none")
    ax.set_xlabel("Nearest-neighbor Tanimoto similarity (train reference)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Applicability domain: error vs similarity")
    save_fig(
        fig,
        plots_svg / "paper_error_vs_similarity.svg",
        None if plots_png is None else plots_png / "paper_error_vs_similarity.png",
    )


def make_y_scramble_plot(
    model_df: pd.DataFrame,
    primary_model: str,
    seed: int,
    plots_svg: Path,
    plots_png: Optional[Path],
) -> None:
    """Quick y-scramble sanity check with linear baseline on Morgan features."""
    train = model_df[(model_df["model"] == primary_model) & (model_df["seed"] == seed) & (model_df["split"] == "train")]
    test = model_df[(model_df["model"] == primary_model) & (model_df["seed"] == seed) & (model_df["split"] == "test")]
    if train.empty or test.empty:
        return

    Xtr, idx_tr = fingerprint_matrix(train["smiles"].tolist())
    Xte, idx_te = fingerprint_matrix(test["smiles"].tolist())
    if Xtr.shape[0] < 10 or Xte.shape[0] < 5:
        return
    ytr = train.iloc[idx_tr]["y_true"].values
    yte = test.iloc[idx_te]["y_true"].values

    lr = LinearRegression()
    lr.fit(Xtr, ytr)
    yhat_real = lr.predict(Xte)
    mse = mean_squared_error(yte, yhat_real)
    rmse_real = float(np.sqrt(mse))

    rng = np.random.default_rng(2027)
    scramble_rmses = []
    for _ in range(10):
        ys = ytr.copy()
        rng.shuffle(ys)
        m = LinearRegression()
        m.fit(Xtr, ys)
        yhat = m.predict(Xte)
        mse_scramble = mean_squared_error(yte, yhat)
        scramble_rmses.append(float(np.sqrt(mse_scramble)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.boxplot(scramble_rmses, positions=[0], widths=0.5)
    ax.scatter([1], [rmse_real], c=NATURE_PALETTE["orange"], s=80, label="Real labels")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Scrambled", "Real"], fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.set_title("Y-scramble sanity check")
    ax.legend()
    save_fig(
        fig,
        plots_svg / "paper_y_scramble.svg",
        None if plots_png is None else plots_png / "paper_y_scramble.png",
    )


def build_train_from_splits(
    splits_dir: Path,
    seeds: Sequence[int],
    id_col: str,
    smiles_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Build fallback train rows from seed train CSV files.

    Notes:
    - The caller should set/overwrite the `model` column to the chosen primary model.
    - `y_pred` is set to NaN by design for fallback rows.
    """
    rows: List[pd.DataFrame] = []
    for seed in seeds:
        train_path = splits_dir / f"seed_{seed}" / "train.csv"
        if not train_path.exists():
            continue
        df = pd.read_csv(train_path)
        idc = _find_col(df, ["id", "mol_id", "molecule_id", id_col])
        smc = _find_col(df, ["smiles", "canonical_smiles", smiles_col])
        ytc = _find_col(df, ["y_true", "true", "target", target_col])
        if idc is None or smc is None or ytc is None:
            continue
        part = pd.DataFrame(
            {
                "seed": int(seed),
                "model": "",
                "split": "train",
                "id": df[idc],
                "smiles": df[smc],
                "y_true": pd.to_numeric(df[ytc], errors="coerce"),
                "y_pred": np.nan,
            }
        )
        part = part.dropna(subset=["id", "smiles", "y_true"]).copy()
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["seed", "model", "split", "id", "smiles", "y_true", "y_pred"])
    out = pd.concat(rows, ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    out["model"] = out["model"].astype(str)
    out["split"] = "train"
    return out

def run_shap_analysis(
    pred_df: pd.DataFrame,
    baseline_models: set[str],
    seeds: Sequence[int],
    outdir: Path,
    plots_svg: Path,
    plots_png: Optional[Path],
    max_samples: int,
    top_k_bits: int,
) -> None:
    """Run SHAP on one strong tree baseline using Morgan features and training rows."""
    try:
        import shap  # type: ignore
    except Exception as e:
        print(f"[WARN] SHAP not available ({e}); skipping SHAP section.")
        return

    candidate = None
    names = sorted(baseline_models)
    for n in names:
        nl = n.lower()
        if "xgb" in nl or "xgboost" in nl:
            candidate = n
            break
    if candidate is None:
        for n in names:
            if "rf" in n.lower() or "random" in n.lower():
                candidate = n
                break
    if candidate is None:
        print("[WARN] No tree baseline found for SHAP; skipping SHAP section.")
        return

    train = pred_df[(pred_df["model"] == candidate) & (pred_df["split"] == "train")].copy()
    if train.empty:
        print("[WARN] No train predictions for chosen baseline SHAP model; skipping.")
        return

    # Keep one row per molecule for training surrogate.
    train = train.drop_duplicates(subset=["id", "smiles"]).reset_index(drop=True)
    X, idx = fingerprint_matrix(train["smiles"].tolist())
    if X.shape[0] < 20:
        print("[WARN] Too few valid training molecules for SHAP; skipping.")
        return
    y = train.iloc[idx]["y_true"].values
    train_use = train.iloc[idx].reset_index(drop=True)

    model = None
    try:
        import xgboost as xgb  # type: ignore

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=2027,
            n_jobs=4,
        )
        if "xgb" not in candidate.lower() and "xgboost" not in candidate.lower():
            model = RandomForestRegressor(n_estimators=500, random_state=2027, n_jobs=4)
    except Exception:
        model = RandomForestRegressor(n_estimators=500, random_state=2027, n_jobs=4)

    model.fit(X, y)

    n = min(max_samples, X.shape[0])
    rng = np.random.default_rng(2027)
    sample_idx = rng.choice(np.arange(X.shape[0]), size=n, replace=False)
    Xs = X[sample_idx]
    sample_df = train_use.iloc[sample_idx].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:top_k_bits]

    shap_dir = outdir / "shap"
    ensure_dirs(shap_dir)

    glob = pd.DataFrame(
        {
            "bit": np.arange(len(mean_abs), dtype=int),
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    glob["rank"] = np.arange(1, len(glob) + 1)
    glob.to_csv(shap_dir / "global_shap_importance.csv", index=False)

    sample_out = pd.DataFrame({"id": sample_df["id"].values, "smiles": sample_df["smiles"].values})
    for b in top_idx:
        sample_out[f"bit_{int(b)}"] = Xs[:, b]
        sample_out[f"shap_bit_{int(b)}"] = shap_values[:, b]
    sample_out.to_csv(shap_dir / "shap_values_sample.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 10))
    bar_vals = mean_abs[top_idx][::-1]
    labels = [f"bit_{int(i)}" for i in top_idx[::-1]]
    ax.barh(np.arange(len(labels)), bar_vals, color=NATURE_PALETTE["blue"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("SHAP global importance (top bits)")
    save_fig(fig, plots_svg / "shap_bar_top_bits.svg", None if plots_png is None else plots_png / "shap_bar_top_bits.png")

    # Beeswarm-like fallback with matplotlib only.
    fig, ax = plt.subplots(figsize=(9, 8))
    for yloc, b in enumerate(top_idx[: min(20, len(top_idx))]):
        xvals = shap_values[:, b]
        yjit = np.full_like(xvals, yloc, dtype=float) + np.random.default_rng(yloc).normal(0, 0.08, size=len(xvals))
        colors = np.where(Xs[:, b] > 0, NATURE_PALETTE["orange"], NATURE_PALETTE["cyan"])
        ax.scatter(xvals, yjit, c=colors, s=10, alpha=0.5)
    ax.set_yticks(np.arange(min(20, len(top_idx))))
    ax.set_yticklabels([f"bit_{int(i)}" for i in top_idx[: min(20, len(top_idx))]], fontweight="bold")
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP summary (top bits)")
    save_fig(
        fig,
        plots_svg / "shap_summary_beeswarm_top_bits.svg",
        None if plots_png is None else plots_png / "shap_summary_beeswarm_top_bits.png",
    )

    top10 = top_idx[:10]
    sub_rows = []
    for bit in top10:
        found = False
        for smi in train_use["smiles"].tolist():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            bit_info: Dict[int, List[Tuple[int, int]]] = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bit_info)
            if int(bit) not in bit_info:
                continue
            atom, rad = bit_info[int(bit)][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom)
            amap: Dict[int, int] = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            smarts = Chem.MolToSmarts(submol) if submol is not None else ""
            atom_indices = sorted(amap.keys())
            sub_rows.append(
                {
                    "bit": int(bit),
                    "example_smiles": smi,
                    "atom_indices": ",".join(map(str, atom_indices)),
                    "substructure_smarts": smarts,
                }
            )
            found = True
            break
        if not found:
            sub_rows.append({"bit": int(bit), "example_smiles": "", "atom_indices": "", "substructure_smarts": ""})
    pd.DataFrame(sub_rows).to_csv(shap_dir / "top_bits_substructures.csv", index=False)


def murcko_smiles(smiles: str) -> str:
    """Get Murcko scaffold SMILES; empty if not possible."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def _replace_core_signature(mol: Chem.Mol, core: Chem.Mol) -> str:
    """Substituent signature via ReplaceCore."""
    try:
        rep = Chem.ReplaceCore(mol, core)
        if rep is None:
            return ""
        return Chem.MolToSmiles(rep)
    except Exception:
        return ""


def run_rgroup_analysis(
    pred_df: pd.DataFrame,
    primary_model: str,
    seeds: Sequence[int],
    outdir: Path,
    plots_svg: Path,
    plots_png: Optional[Path],
    min_count: int,
    top_k_rgroups: int,
    max_cores: int,
) -> None:
    """Run tiered R-group/substituent analysis on train rows."""
    from rdkit.Chem import rdRGroupDecomposition

    rg_dir = outdir / "rgroup"
    ensure_dirs(rg_dir)

    train = pred_df[(pred_df["model"] == primary_model) & (pred_df["split"] == "train")].copy()
    if train.empty:
        print("[WARN] No train rows for R-group analysis.")
        return
    finite_pred = np.isfinite(train["y_pred"].values)
    train["residual"] = np.where(finite_pred, train["y_pred"] - train["y_true"], np.nan)
    train["scaffold"] = train["smiles"].apply(murcko_smiles)
    train = train[train["scaffold"] != ""].copy()
    if train.empty:
        print("[WARN] Could not compute scaffolds for R-group analysis.")
        return

    top_scaffolds = train["scaffold"].value_counts().head(max_cores)
    total_n = len(train)
    core_report = []
    best = {"core": None, "coverage": -1.0, "rows": None, "idx": None}

    for core_smi, count in top_scaffolds.items():
        core = Chem.MolFromSmiles(core_smi)
        if core is None:
            continue
        matched_idx = []
        params = rdRGroupDecomposition.RGroupDecompositionParameters()
        params.asSmiles = True
        rgd = rdRGroupDecomposition.RGroupDecomposition([core], params)
        for i, smi in enumerate(train["smiles"].tolist()):
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not mol.HasSubstructMatch(core):
                continue
            try:
                ok = rgd.Add(mol)
                if ok >= 0:
                    matched_idx.append(i)
            except Exception:
                continue
        if not matched_idx:
            core_report.append(
                {
                    "candidate_core_smiles": core_smi,
                    "scaffold_count": int(count),
                    "decomposed_count": 0,
                    "coverage_fraction": 0.0,
                    "chosen_core": 0,
                }
            )
            continue
        try:
            _ = rgd.Process()
            try:
                rows = rgd.GetRGroupsAsRows(asSmiles=True)
            except TypeError:
                rows = rgd.GetRGroupsAsRows()
            decomposed = len(rows)
        except Exception:
            rows = []
            decomposed = 0
        coverage = decomposed / float(total_n)
        core_report.append(
            {
                "candidate_core_smiles": core_smi,
                "scaffold_count": int(count),
                "decomposed_count": decomposed,
                "coverage_fraction": coverage,
                "chosen_core": 0,
            }
        )
        if coverage > best["coverage"]:
            best = {"core": core_smi, "coverage": coverage, "rows": rows, "idx": matched_idx}

    use_tier1 = best["core"] is not None and best["coverage"] > 0.15 and best["rows"] is not None and len(best["rows"]) > 0

    if core_report:
        rep_df = pd.DataFrame(core_report)
        if use_tier1:
            rep_df.loc[rep_df["candidate_core_smiles"] == best["core"], "chosen_core"] = 1
        rep_df.to_csv(rg_dir / "core_selection_report.csv", index=False)
    else:
        pd.DataFrame(
            columns=["candidate_core_smiles", "scaffold_count", "decomposed_count", "coverage_fraction", "chosen_core"]
        ).to_csv(rg_dir / "core_selection_report.csv", index=False)

    if use_tier1:
        t = train.reset_index(drop=True)
        rows = []
        matched_idx = best["idx"][: len(best["rows"])]
        for local_i, rg_row in enumerate(best["rows"]):
            tr = t.iloc[matched_idx[local_i]]
            row = {
                "seed": tr["seed"],
                "split": "train",
                "id": tr["id"],
                "smiles": tr["smiles"],
                "scaffold": tr["scaffold"],
                "core_smiles": best["core"],
                "y_true": tr["y_true"],
                "y_pred": tr["y_pred"],
                "residual": tr["residual"],
            }
            for k, v in rg_row.items():
                if k.lower().startswith("r"):
                    row[k] = v if v is not None else ""
            rows.append(row)
        rtable = pd.DataFrame(rows)
    else:
        rows = []
        for _, tr in train.iterrows():
            mol = Chem.MolFromSmiles(tr["smiles"])
            core = Chem.MolFromSmiles(tr["scaffold"])
            sig = ""
            if mol is not None and core is not None:
                sig = _replace_core_signature(mol, core)
            rows.append(
                {
                    "seed": tr["seed"],
                    "split": "train",
                    "id": tr["id"],
                    "smiles": tr["smiles"],
                    "scaffold": tr["scaffold"],
                    "core_smiles": tr["scaffold"],
                    "R1": sig,
                    "y_true": tr["y_true"],
                    "y_pred": tr["y_pred"],
                    "residual": tr["residual"],
                }
            )
        rtable = pd.DataFrame(rows)

    rtable.to_csv(rg_dir / "rgroup_table.csv", index=False)

    r_cols = [c for c in rtable.columns if c.upper().startswith("R")]
    if not r_cols:
        return
    long = rtable.melt(
        id_vars=["seed", "id", "smiles", "scaffold", "core_smiles", "y_true", "y_pred", "residual"],
        value_vars=r_cols,
        var_name="rgroup_position",
        value_name="substituent",
    )
    long = long[(long["substituent"].notna()) & (long["substituent"].astype(str) != "")].copy()
    if long.empty:
        return

    core_mean_true = float(rtable["y_true"].mean())
    finite_pred_n = int(np.isfinite(rtable["y_pred"].values).sum())
    core_mean_pred = float(rtable["y_pred"].mean()) if finite_pred_n >= 10 else np.nan

    stats_rows = []
    for (pos, sub), g in long.groupby(["rgroup_position", "substituent"]):
        if len(g) < min_count:
            continue
        stats_rows.append(
            {
                "rgroup_position": pos,
                "substituent": sub,
                "count": len(g),
                "mean_y_true": g["y_true"].mean(),
                "std_y_true": g["y_true"].std(ddof=1),
                "mean_y_pred": g["y_pred"].mean(),
                "std_y_pred": g["y_pred"].std(ddof=1),
                "mean_residual": g["residual"].mean(),
                "enrichment": g["y_true"].mean() - core_mean_true,
                "pred_shift": (g["y_pred"].mean() - core_mean_pred) if finite_pred_n >= 10 else np.nan,
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(rg_dir / "rgroup_summary_stats.csv", index=False)

    if stats_df.empty:
        return
    stats_df["abs_effect"] = np.abs(stats_df["enrichment"])
    top_eff = stats_df.sort_values("abs_effect", ascending=False).head(top_k_rgroups)
    top_eff.to_csv(rg_dir / "top_rgroup_effects.csv", index=False)

    # Plots
    core_sel = pd.read_csv(rg_dir / "core_selection_report.csv")
    if not core_sel.empty:
        core_sel = core_sel.sort_values("coverage_fraction", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(np.arange(len(core_sel)), core_sel["coverage_fraction"], color=NATURE_PALETTE["blue"])
        ax.set_xticks(np.arange(len(core_sel)))
        ax.set_xticklabels([f"core_{i+1}" for i in range(len(core_sel))], rotation=30, ha="right", fontweight="bold")
        ax.set_ylabel("Coverage fraction")
        ax.set_title("R-group core candidate coverage")
        save_fig(
            fig,
            plots_svg / "rgroup_core_coverage.svg",
            None if plots_png is None else plots_png / "rgroup_core_coverage.png",
        )

    fig, ax = plt.subplots(figsize=(11, 6))
    show = top_eff.head(top_k_rgroups).iloc[::-1]
    lab = [f"{r.rgroup_position}:{str(r.substituent)[:18]}" for r in show.itertuples()]
    ax.barh(np.arange(len(show)), show["enrichment"], color=NATURE_PALETTE["green"], alpha=0.8, label="mean_y_true shift")
    has_pred_shift = np.isfinite(show["pred_shift"].values).any()
    if has_pred_shift:
        ax.barh(np.arange(len(show)), show["pred_shift"], color=NATURE_PALETTE["purple"], alpha=0.5, label="mean_y_pred shift")
    ax.set_yticks(np.arange(len(show)))
    ax.set_yticklabels(lab, fontweight="bold")
    ax.set_xlabel("Shift vs core mean")
    ax.set_title("Top R-group effects")
    if has_pred_shift:
        ax.legend()
    save_fig(
        fig,
        plots_svg / "rgroup_effects_top.svg",
        None if plots_png is None else plots_png / "rgroup_effects_top.png",
    )

    # Residual distributions by top substituents
    top_pairs = set(zip(top_eff["rgroup_position"], top_eff["substituent"]))
    dist = long[long.apply(lambda r: (r["rgroup_position"], r["substituent"]) in top_pairs, axis=1)].copy()
    if not dist.empty:
        labels = []
        arrays = []
        for (pos, sub), g in dist.groupby(["rgroup_position", "substituent"]):
            labels.append(f"{pos}:{str(sub)[:12]}")
            arrays.append(g["residual"].values)
        order = np.argsort([-len(a) for a in arrays])[: min(20, len(arrays))]
        labels = [labels[i] for i in order]
        arrays = [arrays[i] for i in order]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(arrays, labels=labels, showfliers=False)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontweight="bold")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals by top substituents")
        save_fig(
            fig,
            plots_svg / "rgroup_residuals_by_substituent.svg",
            None if plots_png is None else plots_png / "rgroup_residuals_by_substituent.png",
        )


def run_mmpa_analysis(
    pred_df: pd.DataFrame,
    primary_model: str,
    outdir: Path,
    plots_svg: Path,
    plots_png: Optional[Path],
) -> None:
    """Approximate MMPA by pairing molecules within scaffold and comparing substituent signatures."""
    mmpa_dir = outdir / "mmpa"
    ensure_dirs(mmpa_dir)

    data = pred_df[(pred_df["model"] == primary_model) & (pred_df["split"] == "train")].copy()
    if data.empty:
        pd.DataFrame(columns=["id1", "id2", "scaffold", "transform_smarts", "delta_pIC50", "delta_pred", "seed"]).to_csv(
            mmpa_dir / "mmpa_pairs.csv", index=False
        )
        pd.DataFrame(columns=["transform_smarts", "count", "mean_delta_pIC50", "mean_delta_pred", "std_delta"]).to_csv(
            mmpa_dir / "mmpa_transform_summary.csv", index=False
        )
        return

    data["scaffold"] = data["smiles"].apply(murcko_smiles)
    data = data[data["scaffold"] != ""].copy()

    sigs = []
    for _, r in data.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        core = Chem.MolFromSmiles(r["scaffold"])
        sig = _replace_core_signature(mol, core) if mol is not None and core is not None else ""
        sigs.append(sig)
    data["sub_sig"] = sigs
    data = data[data["sub_sig"] != ""].copy()

    pairs = []
    max_pairs = 50000
    for (seed, scaf), g in data.groupby(["seed", "scaffold"]):
        gg = g.head(120).reset_index(drop=True)
        for i, j in combinations(range(len(gg)), 2):
            if len(pairs) >= max_pairs:
                break
            r1 = gg.iloc[i]
            r2 = gg.iloc[j]
            if r1["sub_sig"] == r2["sub_sig"]:
                continue
            pairs.append(
                {
                    "id1": r1["id"],
                    "id2": r2["id"],
                    "scaffold": scaf,
                    "transform_smarts": f"{r1['sub_sig']}>>{r2['sub_sig']}",
                    "delta_pIC50": float(r2["y_true"] - r1["y_true"]),
                    "delta_pred": float(r2["y_pred"] - r1["y_pred"])
                    if np.isfinite(r1["y_pred"]) and np.isfinite(r2["y_pred"])
                    else np.nan,
                    "seed": int(seed),
                }
            )
        if len(pairs) >= max_pairs:
            break

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(mmpa_dir / "mmpa_pairs.csv", index=False)

    if pairs_df.empty:
        pd.DataFrame(columns=["transform_smarts", "count", "mean_delta_pIC50", "mean_delta_pred", "std_delta"]).to_csv(
            mmpa_dir / "mmpa_transform_summary.csv", index=False
        )
        return

    summ = (
        pairs_df.groupby("transform_smarts")
        .agg(
            count=("transform_smarts", "size"),
            mean_delta_pIC50=("delta_pIC50", "mean"),
            mean_delta_pred=("delta_pred", "mean"),
            std_delta=("delta_pIC50", "std"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summ.to_csv(mmpa_dir / "mmpa_transform_summary.csv", index=False)

    show = summ.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(np.arange(len(show)), show["count"], color=NATURE_PALETTE["cyan"], alpha=0.8)
    ax.set_yticks(np.arange(len(show)))
    ax.set_yticklabels([str(t)[:40] for t in show["transform_smarts"]], fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_title("Top MMPA transforms")
    save_fig(fig, plots_svg / "mmpa_top_transforms.svg", None if plots_png is None else plots_png / "mmpa_top_transforms.png")


def paired_rmse_stat_test(per_seed: pd.DataFrame, baseline_models: set[str], dmpnn_models: set[str], out_csv: Path) -> None:
    """Paired significance test on test RMSE: best baseline vs dmpnn."""
    rows = []
    sub = per_seed[per_seed["split"] == "test"].copy()
    if not baseline_models or not dmpnn_models:
        pd.DataFrame(columns=["model_a", "model_b", "test_used", "p_value", "effect_direction"]).to_csv(out_csv, index=False)
        return

    bagg = sub[sub["model"].isin(baseline_models)].groupby("model").agg(rmse=("RMSE", "mean"), r2=("R2", "mean")).reset_index()
    dagg = sub[sub["model"].isin(dmpnn_models)].groupby("model").agg(rmse=("RMSE", "mean"), r2=("R2", "mean")).reset_index()
    if bagg.empty or dagg.empty:
        pd.DataFrame(columns=["model_a", "model_b", "test_used", "p_value", "effect_direction"]).to_csv(out_csv, index=False)
        return

    best_b = bagg.sort_values(["rmse", "r2"], ascending=[True, False]).iloc[0]["model"]
    best_d = dagg.sort_values(["rmse", "r2"], ascending=[True, False]).iloc[0]["model"]

    a = sub[sub["model"] == best_b][["seed", "RMSE"]].rename(columns={"RMSE": "rmse_b"})
    b = sub[sub["model"] == best_d][["seed", "RMSE"]].rename(columns={"RMSE": "rmse_d"})
    m = pd.merge(a, b, on="seed", how="inner").sort_values("seed")
    if len(m) < 2:
        rows.append(
            {
                "model_a": best_b,
                "model_b": best_d,
                "test_used": "insufficient_seeds",
                "p_value": np.nan,
                "effect_direction": "undetermined",
            }
        )
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return

    dif = m["rmse_d"].values - m["rmse_b"].values
    effect = "dmpnn_better" if np.mean(dif) < 0 else "baseline_better"

    test_used = "paired_ttest"
    pval = np.nan
    try:
        from scipy import stats  # type: ignore

        try:
            _st, pval = stats.wilcoxon(m["rmse_b"].values, m["rmse_d"].values)
            test_used = "wilcoxon"
        except Exception:
            _st, pval = stats.ttest_rel(m["rmse_b"].values, m["rmse_d"].values)
            test_used = "paired_ttest"
    except Exception:
        n = len(dif)
        sd = np.std(dif, ddof=1)
        if sd > 0 and n > 1:
            tstat = np.mean(dif) / (sd / math.sqrt(n))
            # fallback: normal approximation
            pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(tstat) / math.sqrt(2))))
            test_used = "paired_ttest_normal_approx"

    rows.append(
        {
            "model_a": best_b,
            "model_b": best_d,
            "test_used": test_used,
            "p_value": pval,
            "effect_direction": effect,
        }
    )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def collect_versions() -> Dict[str, str]:
    """Collect dependency versions."""
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": getattr(Chem, "__version__", "unknown"),
    }
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:
        versions["sklearn"] = "not_available"
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "not_available"
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:
        versions["xgboost"] = "not_available"
    try:
        import shap

        versions["shap"] = shap.__version__
    except Exception:
        versions["shap"] = "not_available"
    return versions


def write_manifest(
    outdir: Path,
    args: argparse.Namespace,
    selected: PrimaryModelSelection,
    key_files: Sequence[Path],
) -> None:
    """Write run manifest CSV and TXT."""
    versions = collect_versions()
    timestamp = datetime.now().isoformat(timespec="seconds")

    hashes = []
    for p in key_files:
        if p.exists():
            hashes.append((str(p), sha256_file(p)))

    manifest_csv = outdir / "run_manifest.csv"
    rows = [
        {
            "timestamp": timestamp,
            "splits_dir": args.splits_dir,
            "baselines_dir": args.baselines_dir,
            "dmpnn_dir": args.dmpnn_dir,
            "outdir": args.outdir,
            "seeds": args.seeds,
            "selected_primary_model_source": selected.source,
            "selected_primary_model_name": selected.model_name,
            "file_hashes_json": json.dumps(dict(hashes), ensure_ascii=False),
            "versions_json": json.dumps(versions, ensure_ascii=False),
        }
    ]
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)

    manifest_txt = outdir / "run_manifest.txt"
    with manifest_txt.open("w", encoding="utf-8") as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"splits_dir: {args.splits_dir}\n")
        f.write(f"baselines_dir: {args.baselines_dir}\n")
        f.write(f"dmpnn_dir: {args.dmpnn_dir}\n")
        f.write(f"outdir: {args.outdir}\n")
        f.write(f"seeds: {args.seeds}\n")
        f.write(f"selected_primary_model: {selected.source}:{selected.model_name}\n")
        f.write("versions:\n")
        for k, v in versions.items():
            f.write(f"  {k}: {v}\n")
        f.write("file_hashes:\n")
        for p, h in hashes:
            f.write(f"  {p}: {h}\n")


def parse_seed_list(s: str) -> List[int]:
    """Parse comma separated seeds list."""
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main() -> None:
    """Main entrypoint."""
    args = parse_args()
    configure_plot_style(args)

    seeds = parse_seed_list(args.seeds)

    outdir = Path(args.outdir)
    plots_svg = outdir / "plots" / "svg"
    plots_png = None if args.svg_only and not args.png else outdir / "plots" / "png"
    ensure_dirs(outdir, plots_svg)
    if plots_png is not None:
        ensure_dirs(plots_png)

    print("[INFO] Loading prediction tables...")
    baselines_pred_path = Path(args.baselines_dir) / "all_predictions_regression.csv"
    dmpnn_pred_path = Path(args.dmpnn_dir) / "all_predictions_regression.csv"

    bdf_raw = read_csv_if_exists(baselines_pred_path)
    ddf_raw = read_csv_if_exists(dmpnn_pred_path)

    if bdf_raw is None and ddf_raw is None:
        raise FileNotFoundError("No prediction tables found in baselines_out or dmpnn_out.")

    bdf = None
    ddf = None
    if bdf_raw is not None:
        bdf = standardize_prediction_table(
            bdf_raw,
            id_col=args.id_col,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            default_model="baseline",
        )
        bdf.to_csv(outdir / "standardized_predictions_baselines.csv", index=False)
    else:
        pd.DataFrame(columns=["seed", "model", "split", "id", "smiles", "y_true", "y_pred"]).to_csv(
            outdir / "standardized_predictions_baselines.csv", index=False
        )

    if ddf_raw is not None:
        ddf = standardize_prediction_table(
            ddf_raw,
            id_col=args.id_col,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            default_model="dmpnn",
        )
        ddf.to_csv(outdir / "standardized_predictions_dmpnn.csv", index=False)
    else:
        pd.DataFrame(columns=["seed", "model", "split", "id", "smiles", "y_true", "y_pred"]).to_csv(
            outdir / "standardized_predictions_dmpnn.csv", index=False
        )

    pred_frames = [x for x in [bdf, ddf] if x is not None]
    pred = pd.concat(pred_frames, ignore_index=True)
    pred = pred[pred["seed"].isin(seeds)].copy()
    if pred.empty:
        raise ValueError("No prediction rows after seed filtering.")

    print("[INFO] Computing per-seed metrics...")
    per_seed = compute_per_seed_metrics(pred)
    per_seed_test = per_seed[per_seed["split"] == "test"].copy()
    per_seed_val = per_seed[per_seed["split"] == "val"].copy()
    per_seed_test.to_csv(outdir / "per_seed_metrics_test.csv", index=False)
    per_seed_val.to_csv(outdir / "per_seed_metrics_val.csv", index=False)

    cmp_test = aggregate_metrics(per_seed, split="test")
    cmp_val = aggregate_metrics(per_seed, split="val")
    cmp_test.to_csv(outdir / "final_model_comparison_test.csv", index=False)
    cmp_val.to_csv(outdir / "final_model_comparison_val.csv", index=False)

    baseline_models = set() if bdf is None else set(bdf["model"].unique().tolist())
    dmpnn_models = set() if ddf is None else set(ddf["model"].unique().tolist())

    paired_rmse_stat_test(per_seed, baseline_models, dmpnn_models, outdir / "stat_tests_rmse.csv")

    print("[INFO] Selecting primary model...")
    primary = choose_primary_model(
        per_seed=per_seed,
        source=args.primary_model_source,
        primary_model_name=args.primary_model_name,
        baseline_models=baseline_models,
        dmpnn_models=dmpnn_models,
    )

    print(f"[INFO] Primary model selected: {primary.source}:{primary.model_name}")

    pred_for_analysis = pred.copy()
    if args.fallback_train_from_splits:
        existing_train = pred_for_analysis[
            (pred_for_analysis["model"] == primary.model_name) & (pred_for_analysis["split"] == "train")
        ]
        have_seeds = set(existing_train["seed"].astype(int).tolist())
        missing_seeds = [s for s in seeds if s not in have_seeds]
        if missing_seeds:
            fallback_train = build_train_from_splits(
                splits_dir=Path(args.splits_dir),
                seeds=missing_seeds,
                id_col=args.id_col,
                smiles_col=args.smiles_col,
                target_col=args.target_col,
            )
            if not fallback_train.empty:
                fallback_train["model"] = primary.model_name
                pred_for_analysis = pd.concat([pred_for_analysis, fallback_train], ignore_index=True)
                print(f"[INFO] Added fallback train rows from splits for seeds: {missing_seeds}")
        pred_for_analysis.to_csv(outdir / "pred_with_fallback_train.csv", index=False)

    print("[INFO] Generating core figures...")
    make_model_rmse_plot(cmp_test, plots_svg, plots_png)
    test_df = pred_for_analysis[pred_for_analysis["split"] == "test"].copy()
    make_parity_plot(test_df, primary.model_name, plots_svg, plots_png)
    make_residuals_plot(test_df, primary.model_name, plots_svg, plots_png)

    ad_dir = outdir / "applicability_domain"
    ensure_dirs(ad_dir)
    ad_df = compute_applicability_domain(pred_for_analysis, primary.model_name, seeds, ad_dir / "error_similarity_table.csv")
    make_error_similarity_plot(ad_df, plots_svg, plots_png)

    if seeds:
        make_y_scramble_plot(pred_for_analysis, primary.model_name, seeds[0], plots_svg, plots_png)

    if args.run_shap:
        print("[INFO] Running SHAP analysis...")
        run_shap_analysis(
            pred_df=pred,
            baseline_models=baseline_models,
            seeds=seeds,
            outdir=outdir,
            plots_svg=plots_svg,
            plots_png=plots_png,
            max_samples=args.max_shap_samples,
            top_k_bits=args.top_k_bits,
        )

    if args.run_rgroup:
        print("[INFO] Running R-group/substituent analysis...")
        run_rgroup_analysis(
            pred_df=pred_for_analysis,
            primary_model=primary.model_name,
            seeds=seeds,
            outdir=outdir,
            plots_svg=plots_svg,
            plots_png=plots_png,
            min_count=args.min_rgroup_count,
            top_k_rgroups=args.top_k_rgroups,
            max_cores=args.max_cores,
        )

    if args.run_mmpa:
        print("[INFO] Running MMPA analysis...")
        run_mmpa_analysis(pred_for_analysis, primary.model_name, outdir, plots_svg, plots_png)

    key_files = [
        baselines_pred_path,
        dmpnn_pred_path,
        Path(args.baselines_dir) / "summary_metrics_regression_aggregate.csv",
        Path(args.dmpnn_dir) / "summary_metrics_regression_aggregate.csv",
    ]
    for s in seeds:
        key_files.extend(
            [
                Path(args.splits_dir) / f"seed_{s}" / "train.csv",
                Path(args.splits_dir) / f"seed_{s}" / "val.csv",
                Path(args.splits_dir) / f"seed_{s}" / "test.csv",
            ]
        )
    write_manifest(outdir, args, primary, key_files)

    print(f"[INFO] Final tables: {outdir}")
    print(f"[INFO] Plots folder: {plots_svg}")
    print("DONE")


if __name__ == "__main__":
    main()
