#!/usr/bin/env python3
"""Score screening library with D-MPNN, optional XGBoost consensus, and AD analysis."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for inference.") from exc

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

NATURE = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


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


def load_dmpnn_module(repo_root: Path):
    path = repo_root / "04_train_dmpnn.py"
    spec = importlib.util.spec_from_file_location("dmpnn_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import 04_train_dmpnn.py")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def choose_smiles_col(df: pd.DataFrame, preferred: str, fallbacks: Sequence[str]) -> str:
    if preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    raise ValueError(f"No SMILES column found from: {[preferred, *fallbacks]}")


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    smi = str(smiles).strip()
    if not smi:
        return None
    return Chem.MolFromSmiles(smi)


def morgan_fp(smiles: str, radius: int, nbits: int, use_chirality: bool = True) -> Optional[DataStructs.cDataStructs.ExplicitBitVect]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=use_chirality)


def write_manifest(path: Path, args: argparse.Namespace, n_models: int, xgb_found: bool) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "rdkit": getattr(Chem, "__version__", "unknown"),
        "torch": torch.__version__,
        "args": " ".join(sys.argv[1:]),
        "n_dmpnn_models": n_models,
        "xgb_found": xgb_found,
        "umap_enabled": bool(args.umap),
        "umap_radius": int(args.umap_radius),
        "umap_nbits": int(args.umap_nbits),
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "umap_metric": str(args.umap_metric),
        "umap_random_state": int(args.umap_random_state),
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def infer_dmpnn(
    lib_df: pd.DataFrame,
    dmpnn_mod: Any,
    args: argparse.Namespace,
    smiles_col: str,
) -> Tuple[pd.DataFrame, int]:
    seeds = parse_seeds(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_inference_graphs(
        use_morgan_features: bool,
        morgan_nbits: int,
    ) -> Tuple[List[Any], List[int]]:
        graphs = []
        valid_idx = []
        for idx, row in lib_df.iterrows():
            smi = row.get(smiles_col, "")
            g = dmpnn_mod.parse_smiles_to_graph(
                str(smi),
                str(row.get(args.id_col, f"row_{idx}")),
                y_reg=None,
                y_cls=None,
                use_morgan_features=use_morgan_features,
                morgan_nbits=morgan_nbits,
            )
            if g is not None:
                graphs.append(g)
                valid_idx.append(idx)
        return graphs, valid_idx

    all_pred_map: Dict[int, List[float]] = {i: [] for i in lib_df.index}
    n_models = 0

    for seed in seeds:
        seed_dir = Path(args.dmpnn_dir) / f"seed_{seed}"
        hp_path = seed_dir / "hyperparams.json"
        tscaler_path = seed_dir / "target_scaler.json"

        if hp_path.exists():
            hp = json.loads(hp_path.read_text(encoding="utf-8"))
        else:
            hp = {
                "hidden_size": 512,
                "depth": 4,
                "dropout": 0.05,
                "ffn_num_layers": 2,
                "ffn_hidden_size": 512,
                "use_morgan_features": False,
                "morgan_nbits": 2048,
                "morgan_project_dim": 0,
            }

        target_scaler = None
        if tscaler_path.exists():
            target_scaler = json.loads(tscaler_path.read_text(encoding="utf-8"))

        for k in range(args.ensemble_size):
            ckpt = seed_dir / "checkpoints" / f"ens_{k}_best_regression.pt"
            if not ckpt.exists():
                continue

            use_morgan_features = bool(hp.get("use_morgan_features", False))
            morgan_nbits = int(hp.get("morgan_nbits", 2048))
            graphs, valid_idx = build_inference_graphs(use_morgan_features=use_morgan_features, morgan_nbits=morgan_nbits)
            if not graphs:
                raise RuntimeError("No valid molecules for DMPNN inference.")

            loader = DataLoader(
                dmpnn_mod.GraphDataset(graphs),
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=dmpnn_mod.collate_graphs,
            )

            model = dmpnn_mod.DMPNNModel(
                atom_dim=graphs[0].atom_features.shape[1],
                bond_dim=graphs[0].bond_features.shape[1] if graphs[0].bond_features.size else dmpnn_mod.BOND_FEAT_DIM,
                hidden_size=int(hp.get("hidden_size", 512)),
                depth=int(hp.get("depth", 4)),
                dropout=float(hp.get("dropout", 0.05)),
                ffn_num_layers=int(hp.get("ffn_num_layers", 2)),
                ffn_hidden_size=int(hp.get("ffn_hidden_size", 512)),
                task="regression",
                use_morgan_features=use_morgan_features,
                morgan_nbits=morgan_nbits,
                morgan_project_dim=int(hp.get("morgan_project_dim", 0)),
            ).to(device)

            try:
                obj = torch.load(ckpt, map_location=device, weights_only=True)
            except TypeError:
                obj = torch.load(ckpt, map_location=device)
            if "model_state_dict" not in obj:
                print(f"[WARN] Unexpected checkpoint format: {ckpt}")
                continue
            model.load_state_dict(obj["model_state_dict"])

            out = dmpnn_mod.predict_torch(model, loader, device=device, task="regression", target_scaler=target_scaler)
            ypred = out["y_pred"].astype(float)
            for pos, ridx in enumerate(valid_idx):
                all_pred_map[ridx].append(float(ypred[pos]))
            n_models += 1

    pred_mean = []
    pred_std = []
    for ridx in lib_df.index:
        vals = all_pred_map.get(ridx, [])
        if vals:
            pred_mean.append(float(np.mean(vals)))
            pred_std.append(float(np.std(vals, ddof=0)))
        else:
            pred_mean.append(np.nan)
            pred_std.append(np.nan)

    out_df = lib_df.copy()
    out_df["pred_pIC50_dmpnn"] = pred_mean
    out_df["pred_pIC50_dmpnn_std"] = pred_std
    out_df["n_models_used"] = int(n_models)
    out_df["dmpnn_backend"] = "torch_checkpoint"
    return out_df, n_models


def build_ad_reference(splits_dir: Path, seeds: Sequence[int]) -> List[str]:
    out = []
    for s in seeds:
        fp = splits_dir / f"seed_{s}" / "train.csv"
        if not fp.exists():
            print(f"[WARN] Missing split file for AD: {fp}")
            continue
        df = pd.read_csv(fp)
        col = choose_smiles_col(df, "rdkit_canonical_smiles", ["canonical_smiles", "smiles", "SMILES"])
        for smi in df[col].astype(str):
            mol = safe_mol_from_smiles(smi)
            if mol is None:
                continue
            out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return sorted(set(out))


def assign_ad_buckets(nn_sim: float, low: float, high: float) -> str:
    if pd.isna(nn_sim):
        return "out_of_domain"
    if nn_sim >= high:
        return "in_domain"
    if nn_sim >= low:
        return "novel"
    return "out_of_domain"


def maybe_predict_xgb(lib_df: pd.DataFrame, args: argparse.Namespace, smiles_col: str) -> Tuple[np.ndarray, bool]:
    models_dir = Path(args.baselines_dir) / "models"
    if not models_dir.exists() or joblib is None:
        return np.full((len(lib_df),), np.nan), False

    feature_config_path = models_dir / "feature_config.json"
    if feature_config_path.exists():
        feature_config = json.loads(feature_config_path.read_text(encoding="utf-8"))
        fp_radius = int(feature_config["radius"])
        fp_nbits = int(feature_config["n_bits"])
        fp_use_chirality = bool(feature_config.get("use_chirality", True))
    else:
        fp_radius = int(args.ad_radius)
        fp_nbits = int(args.ad_nbits)
        fp_use_chirality = True

    seeds = parse_seeds(args.seeds)
    model_files: List[Path] = []
    for s in seeds:
        for ext in ["pkl", "joblib"]:
            cand = models_dir / f"xgboost_seed{s}.{ext}"
            if cand.exists():
                model_files.append(cand)
                break

    if not model_files:
        return np.full((len(lib_df),), np.nan), False

    fps = np.zeros((len(lib_df), fp_nbits), dtype=np.float32)
    for i, smi in enumerate(lib_df[smiles_col].astype(str)):
        fp = morgan_fp(smi, radius=fp_radius, nbits=fp_nbits, use_chirality=fp_use_chirality)
        if fp is None:
            continue
        arr = np.zeros((fp_nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr

    preds = []
    for mf in model_files:
        mdl = joblib.load(mf)
        preds.append(np.asarray(mdl.predict(fps), dtype=float))

    return np.mean(np.vstack(preds), axis=0), True


def make_plots(scored_df: pd.DataFrame, out_svg_dir: Path, xgb_avail: bool) -> None:
    out_svg_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    color_map = {"in_domain": NATURE["blue"], "novel": NATURE["green"], "out_of_domain": NATURE["orange"]}
    for bucket, c in color_map.items():
        sub = scored_df[scored_df["ad_bucket"] == bucket]
        ax.scatter(sub["nn_similarity"], sub["pred_pIC50_dmpnn"], s=18, alpha=0.8, color=c, label=bucket)
    ax.set_xlabel("Nearest-neighbor Tanimoto similarity")
    ax.set_ylabel("Predicted pIC50 (DMPNN)")
    ax.set_title("Predictions vs Similarity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_svg_dir / "preds_vs_similarity.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(scored_df["pred_pIC50_dmpnn"].dropna(), bins=40, color=NATURE["purple"], alpha=0.7, label="DMPNN")
    if xgb_avail:
        ax.hist(scored_df["pred_pIC50_xgb"].dropna(), bins=40, color=NATURE["cyan"], alpha=0.5, label="XGB")
    ax.set_xlabel("Predicted pIC50")
    ax.set_ylabel("Count")
    ax.set_title("Score Histograms")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_svg_dir / "score_histograms.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    top10 = scored_df.sort_values("pred_pIC50_dmpnn", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    lines = ["Top 10 by DMPNN prediction"]
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        lines.append(f"{i:2d}. {r.get('Catalog_NO', '')} | {r.get('Name', '')} | pIC50={r.get('pred_pIC50_dmpnn', np.nan):.3f} | sim={r.get('nn_similarity', np.nan):.3f}")
    ax.text(0.01, 0.98, "\n".join(lines), va="top", ha="left")
    fig.tight_layout()
    fig.savefig(out_svg_dir / "top10_table.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def to_canonical_smiles(smiles: str) -> Optional[str]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def run_umap_and_make_plots(
    ranked_df: pd.DataFrame,
    outdir: Path,
    args: argparse.Namespace,
    id_col: str,
    name_col: str,
    smiles_col: str,
) -> None:
    if not args.umap:
        print("[INFO] UMAP plotting disabled (--umap).")
        return

    try:
        import umap
    except Exception as exc:
        raise RuntimeError("UMAP is not installed. Please run: pip install umap-learn") from exc

    work_df = ranked_df.copy()
    if "canonical_smiles" in work_df.columns:
        work_df["canonical_smiles"] = work_df["canonical_smiles"].astype(str)
    else:
        work_df["canonical_smiles"] = work_df[smiles_col].astype(str)

    work_df["canonical_smiles"] = [to_canonical_smiles(smi) for smi in work_df["canonical_smiles"].astype(str)]

    fp_matrix = np.zeros((len(work_df), args.umap_nbits), dtype=np.float32)
    for i, smi in enumerate(work_df["canonical_smiles"].astype(str)):
        fp = morgan_fp(smi, radius=args.umap_radius, nbits=args.umap_nbits, use_chirality=True)
        if fp is None:
            continue
        arr = np.zeros((args.umap_nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_matrix[i] = arr.astype(np.float32)

    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_random_state,
    )
    embedding = reducer.fit_transform(fp_matrix)
    work_df["umap1"] = embedding[:, 0]
    work_df["umap2"] = embedding[:, 1]

    top_ids = set()
    top_path = outdir / "top_for_docking.csv"
    if top_path.exists():
        top_df = pd.read_csv(top_path)
        if id_col in top_df.columns:
            top_ids = set(top_df[id_col].astype(str))
    work_df["is_top_for_docking"] = work_df[id_col].astype(str).isin(top_ids)

    out_cols = [
        id_col,
        name_col,
        "canonical_smiles",
        "pred_pIC50_dmpnn",
        "pred_pIC50_xgb",
        "pred_pIC50_consensus",
        "nn_similarity",
        "ad_bucket",
        "umap1",
        "umap2",
    ]
    out_cols = [c for c in out_cols if c in work_df.columns]
    work_df[out_cols].to_csv(outdir / "umap_library.csv", index=False)

    plots_data_dir = outdir / "plots_data"
    plots_data_dir.mkdir(parents=True, exist_ok=True)
    work_df.to_csv(plots_data_dir / "umap_points.csv", index=False)

    plot_dir = outdir / "plots" / "svg"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ad_colors = {
        "in_domain": NATURE["blue"],
        "novel": NATURE["green"],
        "out_of_domain": NATURE["orange"],
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for bucket, color in ad_colors.items():
        sub = work_df[work_df["ad_bucket"] == bucket]
        ax.scatter(sub["umap1"], sub["umap2"], s=14, alpha=0.85, color=color, label=bucket)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP by AD Bucket")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "umap_ad_bucket.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(work_df["umap1"], work_df["umap2"], c=work_df["nn_similarity"], s=14, alpha=0.9, cmap="viridis")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP by NN Similarity")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("nn_similarity")
    fig.tight_layout()
    fig.savefig(plot_dir / "umap_similarity.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(work_df["umap1"], work_df["umap2"], c=work_df["pred_pIC50_dmpnn"], s=14, alpha=0.9, cmap="plasma")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP by Predicted pIC50 (DMPNN)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pred_pIC50_dmpnn")
    fig.tight_layout()
    fig.savefig(plot_dir / "umap_pred_dmpnn.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(work_df["umap1"], work_df["umap2"], s=12, alpha=0.7, color="#BBBBBB", label="library")
    top_sub = work_df[work_df["is_top_for_docking"]]
    if not top_sub.empty:
        ax.scatter(top_sub["umap1"], top_sub["umap2"], s=20, alpha=0.95, color=NATURE["red2"], label="top_for_docking")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP Top Hits")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "umap_top_hits.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen library with trained QSAR models")
    p.add_argument("--library_csv", default="screening/library_clean.csv")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--dmpnn_dir", default="dmpnn_out")
    p.add_argument("--baselines_dir", default="baselines_out")
    p.add_argument("--outdir", default="screening")
    p.add_argument("--task", choices=["regression"], default="regression")
    p.add_argument("--id_col", default="Catalog_NO")
    p.add_argument("--name_col", default="Name")
    p.add_argument("--smiles_col", default="canonical_smiles")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--use_dmpnn_ens_only", type=str2bool, default=True)
    p.add_argument("--ad_radius", type=int, default=2)
    p.add_argument("--ad_nbits", type=int, default=2048)
    p.add_argument("--ad_threshold", type=float, default=0.35)
    p.add_argument("--ad_novel_low", type=float, default=0.20)
    p.add_argument("--consensus", type=str2bool, default=True)
    p.add_argument("--top_consensus_pct", type=float, default=0.5)
    p.add_argument("--top_novel_pct", type=float, default=0.1)

    p.add_argument("--title_size", type=float, default=18)
    p.add_argument("--label_size", type=float, default=16)
    p.add_argument("--tick_size", type=float, default=14)
    p.add_argument("--legend_size", type=float, default=14)
    p.add_argument("--base_font", type=float, default=14)

    p.add_argument("--umap", action="store_true", default=True)
    p.add_argument("--umap_radius", type=int, default=2)
    p.add_argument("--umap_nbits", type=int, default=2048)
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_metric", type=str, default="jaccard")
    p.add_argument("--umap_random_state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots" / "svg").mkdir(parents=True, exist_ok=True)
    setup_plot_style(args)

    lib_df = pd.read_csv(args.library_csv)
    for col in [args.id_col, args.name_col]:
        if col not in lib_df.columns:
            raise ValueError(f"Required column missing: {col}")

    smiles_col = choose_smiles_col(lib_df, args.smiles_col, ["rdkit_canonical_smiles", "canonical_smiles", "SMILES", "smiles"])
    print(f"[INFO] Using smiles column: {smiles_col}")

    dmpnn_mod = load_dmpnn_module(Path(__file__).resolve().parent)
    preds_df, n_models = infer_dmpnn(lib_df, dmpnn_mod, args, smiles_col)
    if n_models == 0:
        raise RuntimeError("No DMPNN checkpoints were loaded for inference.")

    preds_df.to_csv(outdir / "preds_dmpnn.csv", index=False)

    seeds = parse_seeds(args.seeds)
    ref_smiles = build_ad_reference(Path(args.splits_dir), seeds)
    ref_fps = [morgan_fp(s, args.ad_radius, args.ad_nbits) for s in ref_smiles]
    ref_fps = [fp for fp in ref_fps if fp is not None]
    if not ref_fps:
        raise RuntimeError("No valid reference fingerprints for AD.")

    nn_sims: List[float] = []
    for smi in preds_df[smiles_col].astype(str):
        q = morgan_fp(smi, args.ad_radius, args.ad_nbits)
        if q is None:
            nn_sims.append(np.nan)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        nn_sims.append(float(max(sims)) if sims else np.nan)

    scored_df = preds_df.copy()
    scored_df["nn_similarity"] = nn_sims
    scored_df["ad_bucket"] = [assign_ad_buckets(v, args.ad_novel_low, args.ad_threshold) for v in scored_df["nn_similarity"]]

    xgb_pred = np.full((len(scored_df),), np.nan)
    xgb_avail = False
    if args.consensus:
        xgb_pred, xgb_avail = maybe_predict_xgb(scored_df, args, smiles_col)
        if not xgb_avail:
            print("[WARN] XGB models not found. Consensus falls back to DMPNN.")

    scored_df["pred_pIC50_xgb"] = xgb_pred
    if xgb_avail:
        scored_df["pred_pIC50_consensus"] = 0.5 * scored_df["pred_pIC50_dmpnn"] + 0.5 * scored_df["pred_pIC50_xgb"]
    else:
        scored_df["pred_pIC50_consensus"] = scored_df["pred_pIC50_dmpnn"]

    scored_df.to_csv(outdir / "library_scored_with_ad.csv", index=False)

    ranked_df = scored_df.copy()
    ranked_df["rank_dmpnn"] = ranked_df["pred_pIC50_dmpnn"].rank(method="first", ascending=False).astype("Int64")
    if xgb_avail:
        ranked_df["rank_consensus"] = ranked_df["pred_pIC50_consensus"].rank(method="first", ascending=False).astype("Int64")
    else:
        ranked_df["rank_consensus"] = ranked_df["rank_dmpnn"]

    ranked_df.to_csv(outdir / "library_ranked.csv", index=False)

    in_domain = ranked_df[ranked_df["ad_bucket"] == "in_domain"].copy()
    novel = ranked_df[ranked_df["ad_bucket"] == "novel"].copy()

    n_cons = max(1, int(np.ceil(len(in_domain) * (args.top_consensus_pct / 100.0)))) if len(in_domain) > 0 else 0
    n_nov = max(1, int(np.ceil(len(novel) * (args.top_novel_pct / 100.0)))) if len(novel) > 0 else 0

    cons_sort_col = "pred_pIC50_consensus" if xgb_avail else "pred_pIC50_dmpnn"
    top_cons = in_domain.sort_values(cons_sort_col, ascending=False).head(n_cons)
    top_nov = novel.sort_values("pred_pIC50_dmpnn", ascending=False).head(n_nov)

    top = pd.concat([top_cons, top_nov], ignore_index=True)
    if not top.empty:
        top = top.drop_duplicates(subset=[args.id_col], keep="first")
    top.to_csv(outdir / "top_for_docking.csv", index=False)

    make_plots(scored_df, outdir / "plots" / "svg", xgb_avail)
    run_umap_and_make_plots(
        ranked_df=ranked_df,
        outdir=outdir,
        args=args,
        id_col=args.id_col,
        name_col=args.name_col,
        smiles_col=smiles_col,
    )
    write_manifest(outdir / "run_manifest_screen.csv", args, n_models=n_models, xgb_found=xgb_avail)

    print(f"[DONE] Wrote: {outdir / 'preds_dmpnn.csv'}")
    print(f"[DONE] Wrote: {outdir / 'library_scored_with_ad.csv'}")
    print(f"[DONE] Wrote: {outdir / 'library_ranked.csv'}")
    print(f"[DONE] Wrote: {outdir / 'top_for_docking.csv'}")


if __name__ == "__main__":
    main()
