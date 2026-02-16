#!/usr/bin/env python3
"""Train D-MPNN-style QSAR models with Chemprop backend or a PyTorch fallback.

This script supports reproducible multi-seed training for:
- regression (pIC50)
- classification (Active)
- both tasks (trained as separate models)

Outputs include per-seed checkpoints, predictions, metrics, training logs,
aggregate summaries, manifests, and publication-style SVG plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for this script.") from exc


NATURE_PALETTE = {
    "blue": "#3B4992",
    "orange": "#EE0000",
    "green": "#008B45",
    "purple": "#631879",
    "cyan": "#008280",
    "red2": "#BB0021",
}
SPLIT_COLORS = {"train": NATURE_PALETTE["blue"], "val": NATURE_PALETTE["green"], "test": NATURE_PALETTE["orange"]}


@dataclass
class GraphDatum:
    """Container for one molecule graph and targets."""

    smiles: str
    mol_id: str
    atom_features: np.ndarray
    edge_index: np.ndarray
    bond_features: np.ndarray
    rev_edge_index: np.ndarray
    y_reg: Optional[float]
    y_cls: Optional[float]


class GraphDataset(Dataset):
    """PyTorch dataset for graph datums."""

    def __init__(self, items: Sequence[GraphDatum]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GraphDatum:
        return self.items[idx]


COMMON_ATOMIC_NUMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
STEREO_TYPES = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]
BOND_FEAT_DIM = len(BOND_TYPES) + 1 + 2 + len(STEREO_TYPES) + 1


def one_hot_with_other(value: Any, choices: Sequence[Any]) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1.0
    return vec


def atom_feature(atom: rdchem.Atom) -> np.ndarray:
    """Build atom feature vector."""
    feats: List[float] = []
    feats += one_hot_with_other(atom.GetAtomicNum(), COMMON_ATOMIC_NUMS)
    feats += one_hot_with_other(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    feats += one_hot_with_other(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += [float(atom.GetIsAromatic())]
    feats += one_hot_with_other(atom.GetHybridization(), HYBRIDIZATIONS)
    feats += one_hot_with_other(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.asarray(feats, dtype=np.float32)


def bond_feature(bond: rdchem.Bond) -> np.ndarray:
    """Build bond feature vector."""
    feats: List[float] = []
    feats += one_hot_with_other(bond.GetBondType(), BOND_TYPES)
    feats += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    feats += one_hot_with_other(bond.GetStereo(), STEREO_TYPES)
    return np.asarray(feats, dtype=np.float32)


def parse_smiles_to_graph(smiles: str, mol_id: str, y_reg: Optional[float], y_cls: Optional[float]) -> Optional[GraphDatum]:
    """Convert one SMILES to directed graph representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = np.vstack([atom_feature(a) for a in mol.GetAtoms()]).astype(np.float32)

    edges: List[Tuple[int, int]] = []
    e_feats: List[np.ndarray] = []
    pair_to_eidx: Dict[Tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = bond_feature(bond)
        pair_to_eidx[(u, v)] = len(edges)
        edges.append((u, v))
        e_feats.append(bf)
        pair_to_eidx[(v, u)] = len(edges)
        edges.append((v, u))
        e_feats.append(bf)

    if len(edges) == 0:
        edge_index = np.zeros((0, 2), dtype=np.int64)
        bond_feats = np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)
        rev = np.zeros((0,), dtype=np.int64)
    else:
        edge_index = np.asarray(edges, dtype=np.int64)
        bond_feats = np.vstack(e_feats).astype(np.float32)
        rev = np.asarray([pair_to_eidx[(v, u)] for u, v in edges], dtype=np.int64)

    return GraphDatum(
        smiles=smiles,
        mol_id=mol_id,
        atom_features=atom_feats,
        edge_index=edge_index,
        bond_features=bond_feats,
        rev_edge_index=rev,
        y_reg=y_reg,
        y_cls=y_cls,
    )


def collate_graphs(items: Sequence[GraphDatum]) -> Dict[str, Any]:
    """Collate variable-sized molecule graphs."""
    atom_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    bond_parts: List[torch.Tensor] = []
    rev_parts: List[torch.Tensor] = []

    y_reg: List[float] = []
    y_cls: List[float] = []
    ids: List[str] = []
    smiles: List[str] = []

    atom_scope: List[Tuple[int, int]] = []
    edge_scope: List[Tuple[int, int]] = []

    atom_offset = 0
    edge_offset = 0

    for g in items:
        n_atoms = g.atom_features.shape[0]
        n_edges = g.edge_index.shape[0]

        atom_parts.append(torch.from_numpy(g.atom_features))
        if n_edges > 0:
            edge_parts.append(torch.from_numpy(g.edge_index + atom_offset))
            bond_parts.append(torch.from_numpy(g.bond_features))
            rev_parts.append(torch.from_numpy(g.rev_edge_index + edge_offset))

        atom_scope.append((atom_offset, n_atoms))
        edge_scope.append((edge_offset, n_edges))
        atom_offset += n_atoms
        edge_offset += n_edges

        y_reg.append(np.nan if g.y_reg is None else float(g.y_reg))
        y_cls.append(np.nan if g.y_cls is None else float(g.y_cls))
        ids.append(g.mol_id)
        smiles.append(g.smiles)

    atom_f = torch.cat(atom_parts, dim=0)
    if len(edge_parts) > 0:
        edge_index = torch.cat(edge_parts, dim=0).long()
        bond_f = torch.cat(bond_parts, dim=0).float()
        rev_edge = torch.cat(rev_parts, dim=0).long()
    else:
        edge_index = torch.zeros((0, 2), dtype=torch.long)
        bond_f = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)
        rev_edge = torch.zeros((0,), dtype=torch.long)

    return {
        "atom_features": atom_f.float(),
        "edge_index": edge_index,
        "bond_features": bond_f,
        "rev_edge_index": rev_edge,
        "atom_scope": atom_scope,
        "edge_scope": edge_scope,
        "y_reg": torch.tensor(y_reg, dtype=torch.float32),
        "y_cls": torch.tensor(y_cls, dtype=torch.float32),
        "ids": ids,
        "smiles": smiles,
    }


class DMPNNModel(nn.Module):
    """A compact D-MPNN-like model with directed bond messages."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_size: int,
        depth: int,
        dropout: float,
        ffn_num_layers: int,
        ffn_hidden_size: int,
        task: str,
    ) -> None:
        super().__init__()
        self.task = task
        self.depth = max(1, depth)
        self.dropout = nn.Dropout(dropout)

        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_size)
        self.W_m = nn.Linear(hidden_size, hidden_size)
        self.W_a = nn.Linear(atom_dim + hidden_size, hidden_size)

        head_layers: List[nn.Module] = []
        in_dim = hidden_size
        for _ in range(max(1, ffn_num_layers) - 1):
            head_layers.extend([nn.Linear(in_dim, ffn_hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = ffn_hidden_size
        out_dim = 1
        head_layers.append(nn.Linear(in_dim, out_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(self, batch: Dict[str, Any], return_embeddings: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        atom_f = batch["atom_features"]
        edge_index = batch["edge_index"]
        bond_f = batch["bond_features"]
        rev_edge = batch["rev_edge_index"]
        atom_scope = batch["atom_scope"]

        n_atoms = atom_f.shape[0]
        n_edges = edge_index.shape[0]

        if n_edges == 0:
            atom_state = F.relu(self.W_a(torch.cat([atom_f, torch.zeros((n_atoms, self.W_m.out_features), device=atom_f.device)], dim=1)))
        else:
            src = edge_index[:, 0]
            dst = edge_index[:, 1]
            h0 = F.relu(self.W_i(torch.cat([atom_f[src], bond_f], dim=1)))
            h = h0
            for _ in range(self.depth - 1):
                atom_in = torch.zeros((n_atoms, h.shape[1]), device=h.device)
                atom_in.index_add_(0, dst, h)
                m = atom_in[src] - h[rev_edge]
                h = F.relu(h0 + self.W_m(m))
                h = self.dropout(h)

            atom_in = torch.zeros((n_atoms, h.shape[1]), device=h.device)
            atom_in.index_add_(0, dst, h)
            atom_state = F.relu(self.W_a(torch.cat([atom_f, atom_in], dim=1)))

        mol_vecs: List[torch.Tensor] = []
        for start, length in atom_scope:
            if length <= 0:
                mol_vecs.append(torch.zeros((atom_state.shape[1],), device=atom_state.device))
            else:
                mol_vecs.append(atom_state[start : start + length].mean(dim=0))
        mol_emb = torch.stack(mol_vecs, dim=0)
        pred = self.head(self.dropout(mol_emb)).squeeze(1)
        if return_embeddings:
            return pred, mol_emb
        return pred


class EarlyStopper:
    """Simple early stopping helper."""

    def __init__(self, patience: int, mode: str) -> None:
        self.patience = patience
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = -1
        self.count = 0

    def step(self, value: float, epoch: int) -> bool:
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _binary_curves(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y = y_true[order]
    score = y_score[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    tpr = tp / pos
    fpr = fp / neg
    precision = tp / np.maximum(1, tp + fp)
    recall = tpr

    distinct = np.where(np.diff(score) != 0)[0]
    keep = np.r_[distinct, len(score) - 1]
    return fpr[keep], tpr[keep], np.stack([precision[keep], recall[keep]], axis=1)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = _binary_curves(y, y_score)
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    return float(np.trapz(tpr, fpr))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    _, _, pr = _binary_curves(y, y_score)
    precision = pr[:, 0]
    recall = pr[:, 1]
    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[precision[0] if len(precision) else 1.0, precision, np.mean(y)]
    order = np.argsort(recall)
    return float(np.trapz(precision[order], recall[order]))


def classification_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y = y_true.astype(int)
    yp = (y_score >= threshold).astype(int)
    tp = int(np.sum((y == 1) & (yp == 1)))
    tn = int(np.sum((y == 0) & (yp == 0)))
    fp = int(np.sum((y == 0) & (yp == 1)))
    fn = int(np.sum((y == 1) & (yp == 0)))
    acc = (tp + tn) / max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def apply_plot_style(args: argparse.Namespace) -> None:
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": args.title_size,
            "axes.labelsize": args.label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
            "font.size": args.base_font,
        }
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def has_chemprop() -> bool:
    try:
        import chemprop  # noqa: F401

        return True
    except Exception:
        return False


def run_chemprop_train_predict(
    args: argparse.Namespace,
    task: str,
    seed_dir: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    checkpoint_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Try Chemprop CLI flow; raises on failure."""
    save_dir = seed_dir / f"chemprop_{task}"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = "regression" if task == "regression" else "classification"
    metric = "rmse" if task == "regression" else "auc"
    target_col = args.target_col if task == "regression" else args.label_col

    cmd_train = [
        sys.executable,
        "-m",
        "chemprop.train",
        "--data_path",
        str(train_csv),
        "--separate_val_path",
        str(val_csv),
        "--separate_test_path",
        str(test_csv),
        "--dataset_type",
        dataset_type,
        "--save_dir",
        str(save_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--depth",
        str(args.depth),
        "--hidden_size",
        str(args.hidden_size),
        "--dropout",
        str(args.dropout),
        "--ffn_num_layers",
        str(args.ffn_num_layers),
        "--ffn_hidden_size",
        str(args.ffn_hidden_size),
        "--smiles_columns",
        args.smiles_col,
        "--target_columns",
        target_col,
        "--init_lr",
        str(args.lr),
        "--final_lr",
        str(args.lr),
        "--max_lr",
        str(args.lr),
        "--seed",
        str(args.current_seed),
        "--pytorch_seed",
        str(args.current_seed),
        "--metric",
        metric,
        "--quiet",
    ]

    subprocess.run(cmd_train, check=True, capture_output=True, text=True)

    ckpt_candidates = list(save_dir.rglob("*.pt"))
    if not ckpt_candidates:
        raise RuntimeError("Chemprop training did not produce checkpoints.")
    best_ckpt = ckpt_candidates[0]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, checkpoint_path)

    def predict(split_csv: Path, out_csv: Path) -> pd.DataFrame:
        cmd_pred = [
            sys.executable,
            "-m",
            "chemprop.predict",
            "--test_path",
            str(split_csv),
            "--preds_path",
            str(out_csv),
            "--checkpoint_path",
            str(best_ckpt),
            "--smiles_columns",
            args.smiles_col,
        ]
        subprocess.run(cmd_pred, check=True, capture_output=True, text=True)
        return pd.read_csv(out_csv)

    val_pred = predict(val_csv, seed_dir / f"chemprop_preds_{task}_val.csv")
    test_pred = predict(test_csv, seed_dir / f"chemprop_preds_{task}_test.csv")
    train_pred = predict(train_csv, seed_dir / f"chemprop_preds_{task}_train.csv")
    return train_pred, val_pred, test_pred


def make_datasets_for_seed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    seed_dir: Path,
) -> Tuple[List[GraphDatum], List[GraphDatum], List[GraphDatum], pd.DataFrame]:
    """Featurize and validate split dataframes."""

    invalid_rows: List[Dict[str, Any]] = []

    def build(df: pd.DataFrame, split_name: str) -> List[GraphDatum]:
        out: List[GraphDatum] = []
        for i, row in df.iterrows():
            smi = str(row.get(args.smiles_col, "")).strip()
            mol_id = str(row.get(args.id_col, f"row_{i}"))
            if not smi:
                invalid_rows.append({"split": split_name, "row_index": int(i), "id": mol_id, "smiles": smi, "reason": "empty_smiles"})
                continue
            y_reg = None
            y_cls = None
            if args.task in {"regression", "both"}:
                y_reg = safe_float(row.get(args.target_col))
                if math.isnan(y_reg):
                    invalid_rows.append({"split": split_name, "row_index": int(i), "id": mol_id, "smiles": smi, "reason": "invalid_regression_target"})
                    continue
            if args.task in {"classification", "both"}:
                y_cls = safe_float(row.get(args.label_col))
                if math.isnan(y_cls):
                    invalid_rows.append({"split": split_name, "row_index": int(i), "id": mol_id, "smiles": smi, "reason": "invalid_classification_label"})
                    continue
            g = parse_smiles_to_graph(smi, mol_id, y_reg, y_cls)
            if g is None:
                invalid_rows.append({"split": split_name, "row_index": int(i), "id": mol_id, "smiles": smi, "reason": "rdkit_parse_failed"})
                continue
            out.append(g)
        return out

    train = build(train_df, "train")
    val = build(val_df, "val")
    test = build(test_df, "test")

    invalid_df = pd.DataFrame(invalid_rows)
    invalid_df.to_csv(seed_dir / "invalid_smiles.csv", index=False)
    return train, val, test, invalid_df


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson_r(y_true, y_pred),
    }


def evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
    }
    out.update(classification_at_threshold(y_true, y_prob))
    return out


def predict_torch(
    model: DMPNNModel,
    loader: DataLoader,
    device: torch.device,
    task: str,
    export_embeddings: bool = False,
    target_scaler: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    model.eval()
    ys: List[float] = []
    preds: List[float] = []
    logits: List[float] = []
    ids: List[str] = []
    smiles: List[str] = []
    embeds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if export_embeddings:
                out, emb = model(batch_t, return_embeddings=True)
                embeds.append(emb.detach().cpu().numpy())
            else:
                out = model(batch_t)
            if task == "regression":
                y = batch_t["y_reg"].detach().cpu().numpy()
                p = out.detach().cpu().numpy()
                if target_scaler is not None:
                    p = p * target_scaler["std"] + target_scaler["mean"]
                ys.extend(y.tolist())
                preds.extend(p.tolist())
            else:
                y = batch_t["y_cls"].detach().cpu().numpy()
                logit = out.detach().cpu().numpy()
                prob = 1 / (1 + np.exp(-logit))
                ys.extend(y.tolist())
                logits.extend(logit.tolist())
                preds.extend(prob.tolist())
            ids.extend(batch["ids"])
            smiles.extend(batch["smiles"])

    out_d: Dict[str, Any] = {"y_true": np.asarray(ys), "ids": ids, "smiles": smiles}
    if task == "regression":
        out_d["y_pred"] = np.asarray(preds)
    else:
        out_d["y_logit"] = np.asarray(logits)
        out_d["y_proba"] = np.asarray(preds)
        out_d["y_pred_label"] = (np.asarray(preds) >= 0.5).astype(int)
    if export_embeddings:
        out_d["embeddings"] = np.vstack(embeds) if embeds else np.zeros((0,))
    return out_d


def train_torch_task(
    task: str,
    args: argparse.Namespace,
    seed: int,
    ensemble_id: int,
    seed_dir: Path,
    train_items: List[GraphDatum],
    val_items: List[GraphDatum],
    test_items: List[GraphDatum],
    device: torch.device,
    target_scaler: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Train torch fallback model for one task/seed/ensemble member."""
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / (f"ens_{ensemble_id}_best_regression.pt" if task == "regression" else f"ens_{ensemble_id}_best_classification.pt")

    train_loader = DataLoader(GraphDataset(train_items), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_graphs)
    val_loader = DataLoader(GraphDataset(val_items), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_graphs)
    test_loader = DataLoader(GraphDataset(test_items), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_graphs)

    atom_dim = train_items[0].atom_features.shape[1]
    bond_dim = train_items[0].bond_features.shape[1] if train_items[0].bond_features.size else BOND_FEAT_DIM

    model = DMPNNModel(atom_dim, bond_dim, args.hidden_size, args.depth, args.dropout, args.ffn_num_layers, args.ffn_hidden_size, task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min" if task == "regression" else "max", patience=5, factor=0.7)

    criterion: nn.Module = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    stopper = EarlyStopper(args.early_stopping_patience, mode="min" if task == "regression" else "max")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    history: List[Dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                out = model(batch_t)
                target = batch_t["y_reg"] if task == "regression" else batch_t["y_cls"]
                if task == "regression" and target_scaler is not None:
                    target = (target - target_scaler["mean"]) / target_scaler["std"]
                loss = criterion(out, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        val_pred = predict_torch(model, val_loader, device, task, target_scaler=target_scaler)
        val_true = val_pred["y_true"]
        if task == "regression":
            vm = evaluate_regression(val_true, val_pred["y_pred"])
            monitor = vm["rmse"]
            scheduler.step(monitor)
            row = {
                "epoch": epoch,
                "ensemble_id": ensemble_id,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_rmse": vm["rmse"],
                "val_mae": vm["mae"],
                "val_r2": vm["r2"],
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": time.time() - ep_start,
            }
        else:
            vm = evaluate_classification(val_true, val_pred["y_proba"])
            monitor = vm["roc_auc"]
            scheduler.step(monitor if not math.isnan(monitor) else 0.0)
            row = {
                "epoch": epoch,
                "ensemble_id": ensemble_id,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_roc_auc": vm["roc_auc"],
                "val_pr_auc": vm["pr_auc"],
                "val_f1": vm["f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": time.time() - ep_start,
            }
        history.append(row)

        should_stop = stopper.step(-1e12 if math.isnan(monitor) and task == "classification" else monitor, epoch)
        if stopper.best_epoch == epoch:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_metric": stopper.best}, best_ckpt)
        if should_stop:
            break

    train_time = time.time() - t0
    best_obj = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_obj["model_state_dict"])

    pred_start = time.time()
    train_out = predict_torch(model, train_loader, device, task, export_embeddings=False, target_scaler=target_scaler)
    val_out = predict_torch(model, val_loader, device, task, export_embeddings=args.export_embeddings, target_scaler=target_scaler)
    test_out = predict_torch(model, test_loader, device, task, export_embeddings=args.export_embeddings, target_scaler=target_scaler)
    pred_time = time.time() - pred_start

    if task == "regression":
        val_metrics = evaluate_regression(val_out["y_true"], val_out["y_pred"])
        test_metrics = evaluate_regression(test_out["y_true"], test_out["y_pred"])
    else:
        val_metrics = evaluate_classification(val_out["y_true"], val_out["y_proba"])
        test_metrics = evaluate_classification(test_out["y_true"], test_out["y_proba"])

    return {
        "history": pd.DataFrame(history),
        "train_pred": train_out,
        "val_pred": val_out,
        "test_pred": test_out,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_epoch": best_obj.get("epoch", stopper.best_epoch),
        "train_time_sec": train_time,
        "predict_time_sec": pred_time,
    }


def ensure_columns(df: pd.DataFrame, required: Sequence[str], file_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")


def maybe_save(fig: plt.Figure, svg_path: Path, png_path: Optional[Path], args: argparse.Namespace) -> None:
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    if args.png and png_path is not None:
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plots(args: argparse.Namespace, outdir: Path, seeds: List[int], task_modes: List[str]) -> None:
    apply_plot_style(args)
    svg_dir = outdir / "plots" / "svg"
    png_dir = outdir / "plots" / "png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    if args.png:
        png_dir.mkdir(parents=True, exist_ok=True)

    seeds_to_plot = seeds if args.plots_all_seeds else [seeds[0]]

    if "regression" in task_modes:
        for seed in seeds_to_plot:
            path = outdir / f"seed_{seed}" / "training_log_regression.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(df["epoch"], df["val_rmse"], color=NATURE_PALETTE["blue"], linewidth=2.2)
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Validation RMSE", fontweight="bold")
            ax.set_title(f"Learning Curve (Regression) Seed {seed}", fontweight="bold")
            maybe_save(fig, svg_dir / f"learning_curve_regression_seed_{seed}.svg", (png_dir / f"learning_curve_regression_seed_{seed}.png") if args.png else None, args)

        all_pred_path = outdir / "all_predictions_regression.csv"
        if all_pred_path.exists():
            ap = pd.read_csv(all_pred_path)
            if "ensemble_id" in ap.columns:
                ap = ap[ap["ensemble_id"].astype(str) == "ens"].copy()
            test_df = ap[ap["split"] == "test"].copy()
            if not test_df.empty:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(test_df["y_true"], test_df["y_pred"], s=25, alpha=0.7, color=NATURE_PALETTE["purple"], edgecolors="none")
                mn = float(min(test_df["y_true"].min(), test_df["y_pred"].min()))
                mx = float(max(test_df["y_true"].max(), test_df["y_pred"].max()))
                ax.plot([mn, mx], [mn, mx], linestyle="--", color=NATURE_PALETTE["orange"], linewidth=2)
                ax.set_xlabel("Observed pIC50", fontweight="bold")
                ax.set_ylabel("Predicted pIC50", fontweight="bold")
                ax.set_title("Parity Plot (Test, All Seeds)", fontweight="bold")
                maybe_save(fig, svg_dir / "parity_plot_test_regression_all_seeds.svg", (png_dir / "parity_plot_test_regression_all_seeds.png") if args.png else None, args)

                fig, ax = plt.subplots(figsize=(7, 5))
                residual = test_df["y_true"] - test_df["y_pred"]
                ax.scatter(test_df["y_pred"], residual, s=25, alpha=0.7, color=NATURE_PALETTE["cyan"], edgecolors="none")
                ax.axhline(0, linestyle="--", color=NATURE_PALETTE["red2"], linewidth=2)
                ax.set_xlabel("Predicted pIC50", fontweight="bold")
                ax.set_ylabel("Residual (True - Pred)", fontweight="bold")
                ax.set_title("Residuals Plot (Test, All Seeds)", fontweight="bold")
                maybe_save(fig, svg_dir / "residuals_plot_test_regression_all_seeds.svg", (png_dir / "residuals_plot_test_regression_all_seeds.png") if args.png else None, args)

        by_seed = outdir / "summary_metrics_regression_by_seed.csv"
        if by_seed.exists():
            ms = pd.read_csv(by_seed)
            ms = ms[ms["split"] == "test"]
            if not ms.empty:
                metrics = ["rmse", "mae", "r2"]
                means = [ms[m].mean() for m in metrics]
                stds = [ms[m].std(ddof=1) for m in metrics]
                fig, ax = plt.subplots(figsize=(7, 5))
                x = np.arange(len(metrics))
                ax.bar(x, means, yerr=stds, color=[NATURE_PALETTE["blue"], NATURE_PALETTE["green"], NATURE_PALETTE["orange"]], capsize=6)
                ax.set_xticks(x)
                ax.set_xticklabels([m.upper() for m in metrics], fontweight="bold")
                ax.set_ylabel("Metric Value", fontweight="bold")
                ax.set_title("Regression Metrics Across Seeds (Test)", fontweight="bold")
                maybe_save(fig, svg_dir / "metrics_regression_across_seeds.svg", (png_dir / "metrics_regression_across_seeds.png") if args.png else None, args)

    if "classification" in task_modes:
        for seed in seeds_to_plot:
            path = outdir / f"seed_{seed}" / "training_log_classification.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(df["epoch"], df["val_roc_auc"], color=NATURE_PALETTE["blue"], linewidth=2.2)
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Validation ROC-AUC", fontweight="bold")
            ax.set_title(f"Learning Curve (Classification) Seed {seed}", fontweight="bold")
            maybe_save(fig, svg_dir / f"learning_curve_classification_seed_{seed}.svg", (png_dir / f"learning_curve_classification_seed_{seed}.png") if args.png else None, args)

        by_seed = outdir / "summary_metrics_classification_by_seed.csv"
        all_pred = outdir / "all_predictions_classification.csv"
        if by_seed.exists() and all_pred.exists():
            ms = pd.read_csv(by_seed)
            tp = pd.read_csv(all_pred)
            if "ensemble_id" in tp.columns:
                tp = tp[tp["ensemble_id"].astype(str) == "ens"].copy()
            test_ms = ms[ms["split"] == "test"].copy()
            if not test_ms.empty:
                best = test_ms.sort_values(["roc_auc", "pr_auc"], ascending=[False, False]).iloc[0]
                best_seed = int(best["seed"])
                best_pred = tp[(tp["seed"] == best_seed) & (tp["split"] == "test")].copy()
                if not best_pred.empty and best_pred["y_true"].nunique() == 2:
                    y_true = best_pred["y_true"].to_numpy().astype(int)
                    y_prob = best_pred["y_proba"].to_numpy().astype(float)
                    fpr, tpr, pr = _binary_curves(y_true, y_prob)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.plot(np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0], color=NATURE_PALETTE["purple"], linewidth=2.4)
                    ax.plot([0, 1], [0, 1], "--", color=NATURE_PALETTE["orange"], linewidth=2)
                    ax.set_xlabel("False Positive Rate", fontweight="bold")
                    ax.set_ylabel("True Positive Rate", fontweight="bold")
                    ax.set_title(f"ROC Curve (Test, Best Seed={best_seed})", fontweight="bold")
                    maybe_save(fig, svg_dir / "roc_curve_test_best_seed.svg", (png_dir / "roc_curve_test_best_seed.png") if args.png else None, args)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    rec = np.r_[0.0, pr[:, 1], 1.0]
                    prec = np.r_[pr[0, 0] if len(pr) else 1.0, pr[:, 0], np.mean(y_true)]
                    ax.plot(rec, prec, color=NATURE_PALETTE["cyan"], linewidth=2.4)
                    ax.set_xlabel("Recall", fontweight="bold")
                    ax.set_ylabel("Precision", fontweight="bold")
                    ax.set_title(f"PR Curve (Test, Best Seed={best_seed})", fontweight="bold")
                    maybe_save(fig, svg_dir / "pr_curve_test_best_seed.svg", (png_dir / "pr_curve_test_best_seed.png") if args.png else None, args)

                    cls = classification_at_threshold(y_true, y_prob)
                    cm = np.array([[cls["tn"], cls["fp"]], [cls["fn"], cls["tp"]]], dtype=float)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(cm, cmap="Blues")
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontweight="bold", fontsize=args.label_size)
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(["Pred 0", "Pred 1"], fontweight="bold")
                    ax.set_yticklabels(["True 0", "True 1"], fontweight="bold")
                    ax.set_title(f"Confusion Matrix (Test, Best Seed={best_seed})", fontweight="bold")
                    fig.colorbar(im, ax=ax)
                    maybe_save(fig, svg_dir / "confusion_matrix_test_best_seed.svg", (png_dir / "confusion_matrix_test_best_seed.png") if args.png else None, args)


def prediction_dict_to_rows(task: str, seed: int, ensemble_id: Any, model_name: str, split: str, pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = len(pred["ids"])
    for i in range(n):
        row: Dict[str, Any] = {
            "seed": seed,
            "ensemble_id": ensemble_id,
            "model_name": model_name,
            "split": split,
            "id": pred["ids"][i],
            "smiles": pred["smiles"][i],
            "y_true": pred["y_true"][i],
        }
        if task == "regression":
            row["y_pred"] = pred["y_pred"][i]
        else:
            row["y_logit"] = pred["y_logit"][i]
            row["y_proba"] = pred["y_proba"][i]
            row["y_pred_label"] = pred["y_pred_label"][i]
        rows.append(row)
    return rows


def aggregate_predictions(pred_df: pd.DataFrame, task: str, mode: str) -> pd.DataFrame:
    value_col = "y_pred" if task == "regression" else "y_proba"
    agg_fn = np.mean if mode == "mean" else np.median
    key_cols = ["seed", "split", "id", "smiles"]

    grouped = pred_df.groupby(key_cols, sort=False)
    out = grouped.agg(y_true=("y_true", "first"), y_agg=(value_col, agg_fn)).reset_index()
    out["ensemble_id"] = "ens"
    out["model_name"] = "dmpnn_ens"
    if task == "regression":
        out = out.rename(columns={"y_agg": "y_pred"})
    else:
        out = out.rename(columns={"y_agg": "y_proba"})
        out["y_proba"] = out["y_proba"].astype(float)
        out["y_logit"] = np.log(np.clip(out["y_proba"], 1e-7, 1 - 1e-7) / np.clip(1 - out["y_proba"], 1e-7, 1 - 1e-7))
        out["y_pred_label"] = (out["y_proba"] >= 0.5).astype(int)
    return out


def evaluate_ensemble_metrics(pred_df: pd.DataFrame, task: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split in ["val", "test"]:
        sdf = pred_df[pred_df["split"] == split]
        if sdf.empty:
            continue
        y_true = sdf["y_true"].to_numpy(dtype=float)
        if task == "regression":
            y_pred = sdf["y_pred"].to_numpy(dtype=float)
            rows.append({"model_name": "dmpnn_ens", "split": split, **evaluate_regression(y_true, y_pred), "seed": seed})
        else:
            y_prob = sdf["y_proba"].to_numpy(dtype=float)
            rows.append({"model_name": "dmpnn_ens", "split": split, **evaluate_classification(y_true, y_prob), "seed": seed})
    return pd.DataFrame(rows)


def to_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train D-MPNN QSAR models with reproducible multi-seed execution.")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--outdir", default="dmpnn_out")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--task", default="regression", choices=["regression", "classification", "both"])

    p.add_argument("--smiles_col", default="rdkit_canonical_smiles")
    p.add_argument("--id_col", default="molecule_chembl_id")
    p.add_argument("--target_col", default="pIC50")
    p.add_argument("--label_col", default="Active")

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--ffn_num_layers", type=int, default=2)
    p.add_argument("--ffn_hidden_size", type=int, default=512)
    p.add_argument("--message_passing", default="directed")
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--ensemble_mode", choices=["mean", "median"], default="mean")
    p.add_argument("--target_standardize", action="store_true", default=True)
    p.add_argument("--metric_for_early_stop", default=None)

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--save_attention", action="store_true", default=False)
    p.add_argument("--export_embeddings", action="store_true", default=False)

    p.add_argument("--title_size", type=int, default=18)
    p.add_argument("--label_size", type=int, default=16)
    p.add_argument("--tick_size", type=int, default=14)
    p.add_argument("--legend_size", type=int, default=14)
    p.add_argument("--base_font", type=int, default=14)
    p.add_argument("--svg_only", action="store_true", default=True)
    p.add_argument("--png", action="store_true", default=False)
    p.add_argument("--plots_all_seeds", action="store_true", default=False)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    seeds = parse_seeds(args.seeds)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    backend = "chemprop" if has_chemprop() else "torch_fallback"
    print(f"[INFO] Selected backend: {backend}")
    print(f"[INFO] Device: {device}")

    tasks = ["regression", "classification"] if args.task == "both" else [args.task]
    if args.metric_for_early_stop is None:
        args.metric_for_early_stop = "rmse" if args.task == "regression" else "roc_auc"

    manifest_rows: List[Dict[str, Any]] = []
    all_reg_metrics: List[Dict[str, Any]] = []
    all_cls_metrics: List[Dict[str, Any]] = []
    all_reg_preds: List[pd.DataFrame] = []
    all_cls_preds: List[pd.DataFrame] = []

    for seed in seeds:
        args.current_seed = seed
        print(f"\n[INFO] ==== Seed {seed} ====")
        seed_dir = outdir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        split_dir = Path(args.splits_dir) / f"seed_{seed}"
        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        test_csv = split_dir / "test.csv"
        for fp in [train_csv, val_csv, test_csv]:
            if not fp.exists():
                raise FileNotFoundError(f"Missing split file: {fp}")

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

        ensure_columns(train_df, [args.smiles_col], train_csv)
        ensure_columns(val_df, [args.smiles_col], val_csv)
        ensure_columns(test_df, [args.smiles_col], test_csv)

        seed_tasks = list(tasks)
        if "regression" in seed_tasks:
            ensure_columns(train_df, [args.target_col], train_csv)
            ensure_columns(val_df, [args.target_col], val_csv)
            ensure_columns(test_df, [args.target_col], test_csv)
        if "classification" in seed_tasks:
            if args.label_col not in train_df.columns:
                print(f"[WARN] Label column {args.label_col} not found in train split for seed {seed}; skipping classification.")
                seed_tasks = [t for t in seed_tasks if t != "classification"]
            else:
                cls_values = pd.concat([train_df[args.label_col], val_df[args.label_col], test_df[args.label_col]], axis=0).dropna().unique()
                if len(np.unique(cls_values)) < 2:
                    print(f"[WARN] Only one class present for seed {seed}; skipping classification.")
                    seed_tasks = [t for t in seed_tasks if t != "classification"]

        use_cuda = device.type == "cuda"
        set_global_seed(seed, use_cuda=use_cuda)

        feat_start = time.time()
        train_items, val_items, test_items, invalid_df = make_datasets_for_seed(train_df, val_df, test_df, args, seed_dir)
        feat_time = time.time() - feat_start
        if len(train_items) == 0 or len(val_items) == 0 or len(test_items) == 0:
            raise RuntimeError(f"No valid molecules after featurization for seed {seed}.")

        hyperparams = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "dropout": args.dropout,
            "ffn_num_layers": args.ffn_num_layers,
            "ffn_hidden_size": args.ffn_hidden_size,
            "early_stopping_patience": args.early_stopping_patience,
            "message_passing": args.message_passing,
            "ensemble_size": args.ensemble_size if backend == "torch_fallback" else 1,
            "ensemble_mode": args.ensemble_mode,
            "target_standardize": bool(args.target_standardize),
        }
        with (seed_dir / "hyperparams.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(hyperparams), f, indent=2)

        seed_train_time = 0.0
        seed_predict_time = 0.0

        for task in list(seed_tasks):
            print(f"[INFO] Seed {seed} task={task} training started")
            task_backend = backend
            ens_size = args.ensemble_size if backend == "torch_fallback" else 1

            target_scaler: Optional[Dict[str, float]] = None
            if task == "regression" and backend == "torch_fallback" and args.target_standardize:
                y_train = np.asarray([x.y_reg for x in train_items if x.y_reg is not None], dtype=float)
                scaler_std = float(np.std(y_train))
                if scaler_std <= 0:
                    scaler_std = 1.0
                target_scaler = {"mean": float(np.mean(y_train)), "std": scaler_std}
                with (seed_dir / "target_scaler.json").open("w", encoding="utf-8") as f:
                    json.dump(target_scaler, f, indent=2)

            member_results: List[Dict[str, Any]] = []
            for ensemble_id in range(ens_size):
                internal_seed = seed * 1000 + ensemble_id
                set_global_seed(internal_seed, use_cuda=use_cuda)
                if backend == "chemprop":
                    try:
                        ckpt = seed_dir / "checkpoints" / (f"ens_{ensemble_id}_best_regression.pt" if task == "regression" else f"ens_{ensemble_id}_best_classification.pt")
                        tr_pred, v_pred, te_pred = run_chemprop_train_predict(args, task, seed_dir, train_csv, val_csv, test_csv, ckpt)
                        result: Dict[str, Any] = {"history": pd.DataFrame(), "train_time_sec": float("nan"), "predict_time_sec": float("nan")}
                        if task == "regression":
                            pred_col = tr_pred.columns[-1]
                            result["train_pred"] = {"ids": train_df.get(args.id_col, pd.Series(np.arange(len(train_df)))).astype(str).tolist(), "smiles": train_df[args.smiles_col].astype(str).tolist(), "y_true": train_df[args.target_col].to_numpy(dtype=float), "y_pred": tr_pred[pred_col].to_numpy(dtype=float)}
                            result["val_pred"] = {"ids": val_df.get(args.id_col, pd.Series(np.arange(len(val_df)))).astype(str).tolist(), "smiles": val_df[args.smiles_col].astype(str).tolist(), "y_true": val_df[args.target_col].to_numpy(dtype=float), "y_pred": v_pred[v_pred.columns[-1]].to_numpy(dtype=float)}
                            result["test_pred"] = {"ids": test_df.get(args.id_col, pd.Series(np.arange(len(test_df)))).astype(str).tolist(), "smiles": test_df[args.smiles_col].astype(str).tolist(), "y_true": test_df[args.target_col].to_numpy(dtype=float), "y_pred": te_pred[te_pred.columns[-1]].to_numpy(dtype=float)}
                            result["val_metrics"] = evaluate_regression(result["val_pred"]["y_true"], result["val_pred"]["y_pred"])
                            result["test_metrics"] = evaluate_regression(result["test_pred"]["y_true"], result["test_pred"]["y_pred"])
                        else:
                            pred_col = tr_pred.columns[-1]
                            tr_prob = tr_pred[pred_col].to_numpy(dtype=float)
                            va_prob = v_pred[v_pred.columns[-1]].to_numpy(dtype=float)
                            te_prob = te_pred[te_pred.columns[-1]].to_numpy(dtype=float)
                            result["train_pred"] = {"ids": train_df.get(args.id_col, pd.Series(np.arange(len(train_df)))).astype(str).tolist(), "smiles": train_df[args.smiles_col].astype(str).tolist(), "y_true": train_df[args.label_col].to_numpy(dtype=float), "y_logit": np.log(np.clip(tr_prob, 1e-7, 1 - 1e-7) / np.clip(1 - tr_prob, 1e-7, 1 - 1e-7)), "y_proba": tr_prob, "y_pred_label": (tr_prob >= 0.5).astype(int)}
                            result["val_pred"] = {"ids": val_df.get(args.id_col, pd.Series(np.arange(len(val_df)))).astype(str).tolist(), "smiles": val_df[args.smiles_col].astype(str).tolist(), "y_true": val_df[args.label_col].to_numpy(dtype=float), "y_logit": np.log(np.clip(va_prob, 1e-7, 1 - 1e-7) / np.clip(1 - va_prob, 1e-7, 1 - 1e-7)), "y_proba": va_prob, "y_pred_label": (va_prob >= 0.5).astype(int)}
                            result["test_pred"] = {"ids": test_df.get(args.id_col, pd.Series(np.arange(len(test_df)))).astype(str).tolist(), "smiles": test_df[args.smiles_col].astype(str).tolist(), "y_true": test_df[args.label_col].to_numpy(dtype=float), "y_logit": np.log(np.clip(te_prob, 1e-7, 1 - 1e-7) / np.clip(1 - te_prob, 1e-7, 1 - 1e-7)), "y_proba": te_prob, "y_pred_label": (te_prob >= 0.5).astype(int)}
                            result["val_metrics"] = evaluate_classification(result["val_pred"]["y_true"], result["val_pred"]["y_proba"])
                            result["test_metrics"] = evaluate_classification(result["test_pred"]["y_true"], result["test_pred"]["y_proba"])
                    except Exception as exc:
                        print(f"[WARN] Chemprop failed for seed={seed} task={task} with error: {exc}. Falling back to torch_fallback.")
                        task_backend = "torch_fallback"
                        result = train_torch_task(task, args, seed, ensemble_id, seed_dir, train_items, val_items, test_items, device, target_scaler=target_scaler)
                else:
                    result = train_torch_task(task, args, seed, ensemble_id, seed_dir, train_items, val_items, test_items, device, target_scaler=target_scaler)
                member_results.append(result)
                seed_train_time += safe_float(result.get("train_time_sec", 0.0))
                seed_predict_time += safe_float(result.get("predict_time_sec", 0.0))

            if task == "regression":
                hist_df = pd.concat([x["history"] for x in member_results if not x["history"].empty], ignore_index=True) if member_results else pd.DataFrame()
                hist_df.to_csv(seed_dir / "training_log_regression.csv", index=False)

                member_rows: List[Dict[str, Any]] = []
                member_metrics: List[Dict[str, Any]] = []
                for ens_id, result in enumerate(member_results):
                    for split_name in ["train", "val", "test"]:
                        member_rows.extend(prediction_dict_to_rows("regression", seed, ens_id, "dmpnn_member", split_name, result[f"{split_name}_pred"]))
                    member_metrics.append({"seed": seed, "ensemble_id": ens_id, "model_name": "dmpnn_member", "split": "val", **result["val_metrics"]})
                    member_metrics.append({"seed": seed, "ensemble_id": ens_id, "model_name": "dmpnn_member", "split": "test", **result["test_metrics"]})

                member_pred_df = pd.DataFrame(member_rows)
                ens_pred_df = aggregate_predictions(member_pred_df, "regression", args.ensemble_mode)
                pred_df = pd.concat([member_pred_df, ens_pred_df], ignore_index=True)
                pred_df.to_csv(seed_dir / "predictions_regression.csv", index=False)

                metrics_member_df = pd.DataFrame(member_metrics)
                metrics_member_df.to_csv(seed_dir / "metrics_regression_members.csv", index=False)
                metrics_df = evaluate_ensemble_metrics(ens_pred_df, "regression", seed)
                metrics_df.to_csv(seed_dir / "metrics_regression.csv", index=False)

                all_reg_preds.append(pred_df)
                all_reg_metrics.extend(metrics_df.to_dict(orient="records"))

                first_result = member_results[0]
                if args.export_embeddings and "embeddings" in first_result["test_pred"]:
                    emb = first_result["test_pred"]["embeddings"]
                    if emb.ndim == 2 and emb.shape[0] == len(first_result["test_pred"]["ids"]):
                        emb_df = pd.DataFrame(emb)
                        emb_df.insert(0, "id", first_result["test_pred"]["ids"])
                        emb_df.insert(1, "smiles", first_result["test_pred"]["smiles"])
                        emb_df.to_csv(seed_dir / "test_embeddings_regression.csv", index=False)

            if task == "classification":
                hist_df = pd.concat([x["history"] for x in member_results if not x["history"].empty], ignore_index=True) if member_results else pd.DataFrame()
                hist_df.to_csv(seed_dir / "training_log_classification.csv", index=False)

                member_rows = []
                for ens_id, result in enumerate(member_results):
                    for split_name in ["train", "val", "test"]:
                        member_rows.extend(prediction_dict_to_rows("classification", seed, ens_id, "dmpnn_member", split_name, result[f"{split_name}_pred"]))
                member_pred_df = pd.DataFrame(member_rows)
                ens_pred_df = aggregate_predictions(member_pred_df, "classification", args.ensemble_mode)
                pred_df = pd.concat([member_pred_df, ens_pred_df], ignore_index=True)
                pred_df.to_csv(seed_dir / "predictions_classification.csv", index=False)

                metrics_df = evaluate_ensemble_metrics(ens_pred_df, "classification", seed)
                metrics_df.to_csv(seed_dir / "metrics_classification.csv", index=False)
                cm_test = classification_at_threshold(
                    ens_pred_df[ens_pred_df["split"] == "test"]["y_true"].to_numpy(dtype=float),
                    ens_pred_df[ens_pred_df["split"] == "test"]["y_proba"].to_numpy(dtype=float),
                )
                pd.DataFrame([{"split": "test", **cm_test, "seed": seed}]).to_csv(seed_dir / "confusion_matrix_test.csv", index=False)

                all_cls_preds.append(pred_df)
                all_cls_metrics.extend(metrics_df.to_dict(orient="records"))

                first_result = member_results[0]
                if args.export_embeddings and "embeddings" in first_result["test_pred"]:
                    emb = first_result["test_pred"]["embeddings"]
                    if emb.ndim == 2 and emb.shape[0] == len(first_result["test_pred"]["ids"]):
                        emb_df = pd.DataFrame(emb)
                        emb_df.insert(0, "id", first_result["test_pred"]["ids"])
                        emb_df.insert(1, "smiles", first_result["test_pred"]["smiles"])
                        emb_df.to_csv(seed_dir / "test_embeddings_classification.csv", index=False)

            print(f"[INFO] Seed {seed} task={task} complete. backend={task_backend}")

        pd.DataFrame([{"seed": seed, "featurize_time_sec": feat_time, "train_time_sec": seed_train_time, "predict_time_sec": seed_predict_time}]).to_csv(seed_dir / "timing.csv", index=False)

        manifest_rows.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "backend": backend,
                "splits_dir": str(args.splits_dir),
                "seed": seed,
                "task": args.task,
                "hyperparams": json.dumps(to_jsonable(hyperparams)),
                "train_count": len(train_items),
                "val_count": len(val_items),
                "test_count": len(test_items),
                "python_version": sys.version.replace("\n", " "),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "torch_version": torch.__version__,
                "rdkit_version": getattr(Chem, "__version__", "unknown"),
                "chemprop_version": (subprocess.run([sys.executable, "-c", "import chemprop; print(chemprop.__version__)"], capture_output=True, text=True).stdout.strip() if backend == "chemprop" else "not_installed"),
                "device": str(device),
                "train_sha256": sha256_file(train_csv),
                "val_sha256": sha256_file(val_csv),
                "test_sha256": sha256_file(test_csv),
                "deterministic_cudnn": bool(device.type == "cuda"),
                "cudnn_benchmark": False if device.type == "cuda" else "na",
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(outdir / "run_manifest.csv", index=False)
    with (outdir / "run_manifest.txt").open("w", encoding="utf-8") as f:
        f.write("D-MPNN Training Run Manifest\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
        f.write(f"Backend selected: {backend}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Task: {args.task}\n")
        f.write("Determinism: python/numpy/torch seeded; cudnn deterministic=True and benchmark=False on CUDA.\n")
        f.write(f"Manifest CSV: {outdir / 'run_manifest.csv'}\n")

    if all_reg_metrics:
        reg_ms = pd.DataFrame(all_reg_metrics)
        reg_ms.to_csv(outdir / "summary_metrics_regression_by_seed.csv", index=False)
        reg_agg = reg_ms.groupby("split")[["rmse", "mae", "r2", "pearson"]].agg(["mean", "std"]).reset_index()
        reg_agg.columns = ["_".join(col).strip("_") for col in reg_agg.columns.values]
        reg_agg.to_csv(outdir / "summary_metrics_regression_aggregate.csv", index=False)
    if all_reg_preds:
        pd.concat(all_reg_preds, axis=0, ignore_index=True).to_csv(outdir / "all_predictions_regression.csv", index=False)

    if all_cls_metrics:
        cls_ms = pd.DataFrame(all_cls_metrics)
        cls_ms.to_csv(outdir / "summary_metrics_classification_by_seed.csv", index=False)
        cls_agg = cls_ms.groupby("split")[["roc_auc", "pr_auc", "acc", "f1", "precision", "recall"]].agg(["mean", "std"]).reset_index()
        cls_agg.columns = ["_".join(col).strip("_") for col in cls_agg.columns.values]
        cls_agg.to_csv(outdir / "summary_metrics_classification_aggregate.csv", index=False)
    if all_cls_preds:
        pd.concat(all_cls_preds, axis=0, ignore_index=True).to_csv(outdir / "all_predictions_classification.csv", index=False)

    make_plots(args, outdir, seeds, tasks)

    print(f"[INFO] Summary metrics path: {outdir}")
    print(f"[INFO] Plot directory: {outdir / 'plots' / 'svg'}")
    print("DONE")


if __name__ == "__main__":
    main()
