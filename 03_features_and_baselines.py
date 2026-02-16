#!/usr/bin/env python3
"""Feature generation, baseline modeling, evaluation, and plotting for QSAR tasks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


NATURE_PALETTE: Dict[str, str] = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}


@dataclass
class PlotConfig:
    """Configuration for plot typography and output behavior."""

    title_size: int
    label_size: int
    tick_size: int
    legend_size: int
    dpi: int
    save_png: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate features, train baselines, evaluate, and plot.")
    parser.add_argument("--splits_dir", type=Path, required=True, help="Directory containing seed split folders.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--task", type=str, default="both", choices=["regression", "classification", "both"], help="Task type.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint nBits.")
    parser.add_argument("--title_size", type=int, default=14, help="Plot title font size.")
    parser.add_argument("--label_size", type=int, default=12, help="Axis label font size.")
    parser.add_argument("--tick_size", type=int, default=10, help="Tick label font size.")
    parser.add_argument("--legend_size", type=int, default=10, help="Legend font size.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    parser.add_argument("--png", action="store_true", help="Also save PNG versions.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def parse_seed_list(seed_text: str) -> List[int]:
    """Convert comma-separated seed string into list of ints."""
    try:
        seeds = [int(tok.strip()) for tok in seed_text.split(",") if tok.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --seeds format: {seed_text}. Expected comma-separated integers.") from exc
    if not seeds:
        raise ValueError("No valid seeds parsed from --seeds.")
    return seeds


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV table with simple extension-based separator handling."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file extension for split file: {path}")


def find_seed_split_files(seed_dir: Path) -> Dict[str, Path]:
    """Locate train/val/test files for one seed."""
    candidates = sorted([p for p in seed_dir.glob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}])
    mapping: Dict[str, Path] = {}
    for split in ["train", "val", "test"]:
        matching = [p for p in candidates if split in p.stem.lower()]
        if not matching:
            raise FileNotFoundError(
                f"Could not find a {split} split file in {seed_dir}. Expected a filename containing '{split}'."
            )
        mapping[split] = matching[0]
    return mapping


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Infer ID, smiles, and target columns from a dataframe."""
    columns = {c.lower(): c for c in df.columns}

    smiles_candidates = ["rdkit_canonical_smiles", "canonical_smiles", "smiles"]
    id_candidates = ["molecule_chembl_id", "chembl_id", "id", "compound_id"]
    target_candidates = ["y", "target", "label", "activity", "value"]

    smiles_col = next((columns[c] for c in smiles_candidates if c in columns), None)
    id_col = next((columns[c] for c in id_candidates if c in columns), None)
    target_col = next((columns[c] for c in target_candidates if c in columns), None)

    if smiles_col is None:
        raise ValueError(f"Could not infer smiles column from columns: {list(df.columns)}")
    if target_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1:
            target_col = numeric_cols[0]
        else:
            raise ValueError(
                "Could not infer target column. Add one of [y,target,label,activity,value] or ensure a single numeric target."
            )
    if id_col is None:
        id_col = "_generated_id"
        df[id_col] = np.arange(len(df))

    return id_col, smiles_col, target_col


def canonicalize_smiles(smiles: Any) -> Optional[str]:
    """Return canonical SMILES if possible."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def featurize_smiles(smiles_list: Sequence[Any], radius: int, n_bits: int) -> np.ndarray:
    """Generate Morgan fingerprints from a list of SMILES strings."""
    feats = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        can = canonicalize_smiles(smi)
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(bitvect, arr)
        feats[i] = arr
    return feats


def build_regression_models(include_xgb: bool) -> Dict[str, Any]:
    """Construct regression model dict."""
    models: Dict[str, Any] = {
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "svr_rbf": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", SVR(C=10.0, gamma="scale"))]),
        "rf": RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=0),
        "knn": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=7))]),
    }
    if include_xgb:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=-1,
        )
    return models


def build_classification_models(include_xgb: bool) -> Dict[str, Any]:
    """Construct classification model dict."""
    models: Dict[str, Any] = {
        "logreg": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, n_jobs=-1))]),
        "svc_rbf": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", SVC(probability=True, C=3.0, gamma="scale", random_state=0))]),
        "rf": RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
        "gbc": GradientBoostingClassifier(random_state=0),
        "knn": Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=11))]),
    }
    if include_xgb:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=0,
            n_jobs=-1,
        )
    return models


def get_predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    """Get positive-class probabilities from a classifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        raw = np.asarray(raw)
        if raw.ndim == 2:
            raw = raw[:, 1]
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError(f"Model {type(model)} has neither predict_proba nor decision_function.")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred_label: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred_label)),
        "f1": float(f1_score(y_true, y_pred_label, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_label, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["roc_auc"] = float("nan")
    return out


def configure_matplotlib(cfg: PlotConfig) -> None:
    """Set global matplotlib style constraints required by task."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "axes.titlesize": cfg.title_size,
            "axes.labelsize": cfg.label_size,
            "xtick.labelsize": cfg.tick_size,
            "ytick.labelsize": cfg.tick_size,
            "legend.fontsize": cfg.legend_size,
        }
    )


def save_figure(fig: plt.Figure, svg_path: Path, png_path: Optional[Path], cfg: PlotConfig) -> None:
    """Save figure to svg and optionally png."""
    fig.tight_layout()
    fig.savefig(svg_path, format="svg")
    if cfg.save_png and png_path is not None:
        fig.savefig(png_path, dpi=cfg.dpi)
    plt.close(fig)


def plot_regression_comparison(summary: pd.DataFrame, plot_dir_svg: Path, plot_dir_png: Path, cfg: PlotConfig) -> None:
    """Plot mean test RMSE by model with std bars."""
    test = summary[summary["split"] == "test"].copy()
    test = test.sort_values("rmse_mean", ascending=True)
    x = np.arange(len(test))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, test["rmse_mean"].values, yerr=test["rmse_std"].values, color=NATURE_PALETTE["blue"], edgecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(test["model"].tolist(), rotation=25, ha="right", fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.set_title("Regression model comparison (test RMSE)")
    ax.set_xlabel("Model")
    save_figure(
        fig,
        plot_dir_svg / "regression_model_comparison_rmse_test.svg",
        (plot_dir_png / "regression_model_comparison_rmse_test.png") if cfg.save_png else None,
        cfg,
    )


def plot_parity(df: pd.DataFrame, best_model: str, plot_dir_svg: Path, plot_dir_png: Path, cfg: PlotConfig) -> None:
    """Parity plot for best regression model on test split."""
    sub = df[(df["model"] == best_model) & (df["split"] == "test")].copy()
    y_true = sub["y_true"].values
    y_pred = sub["y_pred"].values
    lim_min = min(np.min(y_true), np.min(y_pred))
    lim_max = max(np.max(y_true), np.max(y_pred))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.8, color=NATURE_PALETTE["orange"], edgecolor="black", linewidth=0.3, label=best_model)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color=NATURE_PALETTE["blue"], label="y=x")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title(f"Parity plot (test) - best model: {best_model}")
    ax.legend(frameon=False)
    save_figure(
        fig,
        plot_dir_svg / "parity_plot_test_best_model.svg",
        (plot_dir_png / "parity_plot_test_best_model.png") if cfg.save_png else None,
        cfg,
    )


def plot_residuals(df: pd.DataFrame, best_model: str, plot_dir_svg: Path, plot_dir_png: Path, cfg: PlotConfig) -> None:
    """Residuals plot for best regression model on test split."""
    sub = df[(df["model"] == best_model) & (df["split"] == "test")].copy()
    residuals = sub["y_true"].values - sub["y_pred"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["y_pred"].values, residuals, alpha=0.8, color=NATURE_PALETTE["green"], edgecolor="black", linewidth=0.3)
    ax.axhline(0.0, linestyle="--", color=NATURE_PALETTE["red2"])
    ax.set_xlabel("y_pred")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title(f"Residuals plot (test) - best model: {best_model}")
    save_figure(
        fig,
        plot_dir_svg / "residuals_plot_test_best_model.svg",
        (plot_dir_png / "residuals_plot_test_best_model.png") if cfg.save_png else None,
        cfg,
    )


def plot_r2_across_seeds(by_seed: pd.DataFrame, plot_dir_svg: Path, plot_dir_png: Path, cfg: PlotConfig) -> None:
    """Plot per-seed test R2 for each model."""
    test = by_seed[by_seed["split"] == "test"].copy()
    models = sorted(test["model"].unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [NATURE_PALETTE[k] for k in ["blue", "orange", "green", "purple", "cyan", "red2"]]
    for i, model in enumerate(models):
        sub = test[test["model"] == model].sort_values("seed")
        ax.plot(sub["seed"].values, sub["r2"].values, marker="o", linestyle="-", color=colors[i % len(colors)], label=model)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test R2")
    ax.set_title("R2 across seeds")
    ax.legend(frameon=False)
    save_figure(
        fig,
        plot_dir_svg / "r2_across_seeds.svg",
        (plot_dir_png / "r2_across_seeds.png") if cfg.save_png else None,
        cfg,
    )


def plot_roc_curve(df_cls: pd.DataFrame, best_model: str, plot_dir_svg: Path, plot_dir_png: Path, cfg: PlotConfig) -> None:
    """Plot ROC curve for best classifier on test split."""
    sub = df_cls[(df_cls["model"] == best_model) & (df_cls["split"] == "test")].copy()
    y_true = sub["y_true"].values
    y_proba = sub["y_proba"].values
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=NATURE_PALETTE["purple"], label=f"ROC-AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="black")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve (test) - best model: {best_model}")
    ax.legend(frameon=False)
    save_figure(
        fig,
        plot_dir_svg / "roc_curves_test_best_model.svg",
        (plot_dir_png / "roc_curves_test_best_model.png") if cfg.save_png else None,
        cfg,
    )


def plot_pr_curve(df_cls: pd.DataFrame, best_model: str, plot_dir_svg: Path, plot_dir_png: Path, cfg: PlotConfig) -> None:
    """Plot PR curve for best classifier on test split."""
    sub = df_cls[(df_cls["model"] == best_model) & (df_cls["split"] == "test")].copy()
    y_true = sub["y_true"].values
    y_proba = sub["y_proba"].values
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color=NATURE_PALETTE["cyan"], label=f"PR-AUC={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve (test) - best model: {best_model}")
    ax.legend(frameon=False)
    save_figure(
        fig,
        plot_dir_svg / "pr_curves_test_best_model.svg",
        (plot_dir_png / "pr_curves_test_best_model.png") if cfg.save_png else None,
        cfg,
    )


def plot_confusion(df_cls: pd.DataFrame, best_model: str, plot_dir_svg: Path, plot_dir_png: Path, cfg: PlotConfig) -> None:
    """Plot confusion matrix for best classifier on test split."""
    sub = df_cls[(df_cls["model"] == best_model) & (df_cls["split"] == "test")].copy()
    cm = confusion_matrix(sub["y_true"].values, sub["y_pred_label"].values)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontweight="bold")
    ax.set_yticklabels(["True 0", "True 1"], fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontweight="bold")
    ax.set_title(f"Confusion matrix (test) - best model: {best_model}")
    save_figure(
        fig,
        plot_dir_svg / "confusion_matrix_test_best_model.svg",
        (plot_dir_png / "confusion_matrix_test_best_model.png") if cfg.save_png else None,
        cfg,
    )


def aggregate_with_std(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate by model/split as mean/std for each metric."""
    agg_map: Dict[str, List[str]] = {m: ["mean", "std"] for m in metric_cols}
    out = df.groupby(["model", "split"], as_index=False).agg(agg_map)
    out.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in out.columns]
    out = out.rename(columns={"model_": "model", "split_": "split"})
    return out


def model_params_as_json(model: Any) -> str:
    """Serialize model hyperparameters to JSON string."""
    try:
        params = model.get_params(deep=True)
    except Exception:
        params = {"repr": repr(model)}
    return json.dumps(params, sort_keys=True, default=str)


def maybe_has_classification_data(train_df: pd.DataFrame, target_col: str) -> bool:
    """Heuristic to decide if classification can be run."""
    y = train_df[target_col].dropna().values
    unique = np.unique(y)
    if len(unique) <= 1:
        return False
    if len(unique) == 2 and set(np.unique(unique)).issubset({0, 1}):
        return True
    if np.all(np.equal(np.mod(unique, 1), 0)) and len(unique) <= 10:
        return True
    return False


def run() -> None:
    """Main execution routine."""
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    ensure_dir(args.outdir)

    include_xgb = False
    try:
        import xgboost  # noqa: F401

        include_xgb = True
    except Exception:
        include_xgb = False

    plot_cfg = PlotConfig(
        title_size=args.title_size,
        label_size=args.label_size,
        tick_size=args.tick_size,
        legend_size=args.legend_size,
        dpi=args.dpi,
        save_png=args.png,
    )
    configure_matplotlib(plot_cfg)

    metrics_reg_rows: List[Dict[str, Any]] = []
    metrics_cls_rows: List[Dict[str, Any]] = []
    pred_reg_rows: List[Dict[str, Any]] = []
    pred_cls_rows: List[Dict[str, Any]] = []
    model_param_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []

    print("loading splits")
    for seed in seeds:
        seed_in = args.splits_dir / f"seed_{seed}"
        seed_out = args.outdir / f"seed_{seed}"
        ensure_dir(seed_out)
        files = find_seed_split_files(seed_in)

        train_df = read_table(files["train"]).copy()
        val_df = read_table(files["val"]).copy()
        test_df = read_table(files["test"]).copy()

        id_col, smiles_col, target_col = infer_columns(train_df)
        for df in [val_df, test_df]:
            if id_col not in df.columns:
                df[id_col] = np.arange(len(df))

        print("generating features")
        t0 = time.perf_counter()
        x_train = featurize_smiles(train_df[smiles_col].tolist(), args.radius, args.n_bits)
        x_val = featurize_smiles(val_df[smiles_col].tolist(), args.radius, args.n_bits)
        x_test = featurize_smiles(test_df[smiles_col].tolist(), args.radius, args.n_bits)
        feat_time = time.perf_counter() - t0

        y_train = train_df[target_col].astype(float).values
        y_val = val_df[target_col].astype(float).values
        y_test = test_df[target_col].astype(float).values

        run_reg = args.task in {"regression", "both"}
        run_cls = args.task in {"classification", "both"} and maybe_has_classification_data(train_df, target_col)

        if args.task in {"classification", "both"} and not run_cls:
            print(f"seed {seed}: classification skipped (target is not binary-like).")

        if run_reg:
            models_reg = build_regression_models(include_xgb)
            for model_name, model in models_reg.items():
                print(f"training each model: regression/{model_name} (seed={seed})")
                model_instance = clone(model)
                t1 = time.perf_counter()
                model_instance.fit(x_train, y_train)
                train_time = time.perf_counter() - t1

                for split_name, x_split, y_split, frame in [
                    ("val", x_val, y_val, val_df),
                    ("test", x_test, y_test, test_df),
                ]:
                    t2 = time.perf_counter()
                    y_pred = model_instance.predict(x_split)
                    pred_time = time.perf_counter() - t2
                    print("evaluation")
                    m = compute_regression_metrics(y_split, y_pred)
                    metrics_reg_rows.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "split": split_name,
                            **m,
                        }
                    )
                    timing_rows.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "task": "regression",
                            "split": split_name,
                            "feature_time_sec": feat_time,
                            "train_time_sec": train_time,
                            "predict_time_sec": pred_time,
                        }
                    )
                    for row_idx, (_, row) in enumerate(frame.iterrows()):
                        pred_reg_rows.append(
                            {
                                "seed": seed,
                                "model": model_name,
                                "split": split_name,
                                "molecule_chembl_id": row.get(id_col),
                                "rdkit_canonical_smiles": canonicalize_smiles(row.get(smiles_col)),
                                "y_true": float(y_split[row_idx]),
                                "y_pred": float(y_pred[row_idx]),
                            }
                        )

                model_param_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "task": "regression",
                        "params_json": model_params_as_json(model_instance),
                    }
                )

        if run_cls:
            y_train_cls = train_df[target_col].astype(int).values
            y_val_cls = val_df[target_col].astype(int).values
            y_test_cls = test_df[target_col].astype(int).values
            models_cls = build_classification_models(include_xgb)

            for model_name, model in models_cls.items():
                print(f"training each model: classification/{model_name} (seed={seed})")
                model_instance = clone(model)
                t1 = time.perf_counter()
                model_instance.fit(x_train, y_train_cls)
                train_time = time.perf_counter() - t1

                for split_name, x_split, y_split, frame in [
                    ("val", x_val, y_val_cls, val_df),
                    ("test", x_test, y_test_cls, test_df),
                ]:
                    t2 = time.perf_counter()
                    y_proba = get_predict_proba(model_instance, x_split)
                    y_pred_label = (y_proba >= 0.5).astype(int)
                    pred_time = time.perf_counter() - t2
                    print("evaluation")
                    m = compute_classification_metrics(y_split, y_proba, y_pred_label)
                    metrics_cls_rows.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "split": split_name,
                            **m,
                        }
                    )
                    timing_rows.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "task": "classification",
                            "split": split_name,
                            "feature_time_sec": feat_time,
                            "train_time_sec": train_time,
                            "predict_time_sec": pred_time,
                        }
                    )
                    for row_idx, (_, row) in enumerate(frame.iterrows()):
                        pred_cls_rows.append(
                            {
                                "seed": seed,
                                "model": model_name,
                                "split": split_name,
                                "id": row.get(id_col),
                                "smiles": row.get(smiles_col),
                                "y_true": int(y_split[row_idx]),
                                "y_proba": float(y_proba[row_idx]),
                                "y_pred_label": int(y_pred_label[row_idx]),
                            }
                        )

                model_param_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "task": "classification",
                        "params_json": model_params_as_json(model_instance),
                    }
                )

        seed_metrics_reg = pd.DataFrame([r for r in metrics_reg_rows if r["seed"] == seed])
        seed_metrics_cls = pd.DataFrame([r for r in metrics_cls_rows if r["seed"] == seed])
        seed_preds_reg = pd.DataFrame([r for r in pred_reg_rows if r["seed"] == seed])
        seed_preds_cls = pd.DataFrame([r for r in pred_cls_rows if r["seed"] == seed])
        seed_params = pd.DataFrame([r for r in model_param_rows if r["seed"] == seed])
        seed_timing = pd.DataFrame([r for r in timing_rows if r["seed"] == seed])

        seed_metrics_reg.to_csv(seed_out / "metrics_regression.csv", index=False)
        seed_preds_reg.to_csv(seed_out / "predictions_regression.csv", index=False)
        seed_params.to_csv(seed_out / "model_params.csv", index=False)
        seed_timing.to_csv(seed_out / "timing.csv", index=False)
        if not seed_metrics_cls.empty:
            seed_metrics_cls.to_csv(seed_out / "metrics_classification.csv", index=False)
            seed_preds_cls.to_csv(seed_out / "predictions_classification.csv", index=False)

    # Top-level outputs
    df_metrics_reg = pd.DataFrame(metrics_reg_rows)
    df_metrics_cls = pd.DataFrame(metrics_cls_rows)
    df_preds_reg = pd.DataFrame(pred_reg_rows)
    df_preds_cls = pd.DataFrame(pred_cls_rows)

    if not df_metrics_reg.empty:
        df_metrics_reg.to_csv(args.outdir / "summary_metrics_regression_by_seed.csv", index=False)
        agg_reg = aggregate_with_std(df_metrics_reg, ["rmse", "mae", "r2"])
        agg_reg.to_csv(args.outdir / "summary_metrics_regression_aggregate.csv", index=False)
        df_preds_reg.to_csv(args.outdir / "all_predictions_regression.csv", index=False)
    else:
        agg_reg = pd.DataFrame()

    if not df_metrics_cls.empty:
        df_metrics_cls.to_csv(args.outdir / "summary_metrics_classification_by_seed.csv", index=False)
        agg_cls = aggregate_with_std(df_metrics_cls, ["roc_auc", "pr_auc", "accuracy", "f1", "precision", "recall"])
        agg_cls.to_csv(args.outdir / "summary_metrics_classification_aggregate.csv", index=False)
        df_preds_cls.to_csv(args.outdir / "all_predictions_classification.csv", index=False)

    # plotting
    plot_svg = args.outdir / "plots" / "svg"
    plot_png = args.outdir / "plots" / "png"
    ensure_dir(plot_svg)
    if args.png:
        ensure_dir(plot_png)

    if not df_metrics_reg.empty:
        plot_regression_comparison(agg_reg, plot_svg, plot_png, plot_cfg)

        # deterministic best reg model: lowest mean test RMSE, tie -> higher mean R2
        reg_test = agg_reg[agg_reg["split"] == "test"].copy()
        reg_test = reg_test.sort_values(["rmse_mean", "r2_mean"], ascending=[True, False])
        best_reg_model = reg_test.iloc[0]["model"]

        plot_parity(df_preds_reg, best_reg_model, plot_svg, plot_png, plot_cfg)
        plot_residuals(df_preds_reg, best_reg_model, plot_svg, plot_png, plot_cfg)
        plot_r2_across_seeds(df_metrics_reg, plot_svg, plot_png, plot_cfg)

    if not df_metrics_cls.empty:
        agg_cls = pd.read_csv(args.outdir / "summary_metrics_classification_aggregate.csv")
        cls_test = agg_cls[agg_cls["split"] == "test"].copy()
        cls_test = cls_test.sort_values(["roc_auc_mean", "pr_auc_mean"], ascending=[False, False])
        best_cls_model = cls_test.iloc[0]["model"]

        plot_roc_curve(df_preds_cls, best_cls_model, plot_svg, plot_png, plot_cfg)
        plot_pr_curve(df_preds_cls, best_cls_model, plot_svg, plot_png, plot_cfg)
        plot_confusion(df_preds_cls, best_cls_model, plot_svg, plot_png, plot_cfg)

    manifest = {
        "splits_dir": str(args.splits_dir),
        "outdir": str(args.outdir),
        "seeds": seeds,
        "task": args.task,
        "fingerprint": {"radius": args.radius, "n_bits": args.n_bits},
        "xgboost_used": include_xgb,
    }
    with open(args.outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"summary metrics: {args.outdir / 'summary_metrics_regression_by_seed.csv'}")
    if not df_metrics_cls.empty:
        print(f"summary metrics: {args.outdir / 'summary_metrics_classification_by_seed.csv'}")
    print(f"plots folder: {plot_svg}")
    print("DONE")


def main() -> None:
    """Program entrypoint."""
    try:
        run()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
