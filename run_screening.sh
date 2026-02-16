#!/usr/bin/env bash
set -euo pipefail

mkdir -p screening/plots/svg

echo "[INFO] Step 1/3: Preparing library"
python 07_prepare_library.py \
  --in_csv library.csv \
  --outdir screening \
  --id_col Catalog_NO \
  --name_col Name

echo "[INFO] Step 2/3: Screening with QSAR"
python 08_screen_qsar.py \
  --library_csv screening/library_clean.csv \
  --splits_dir splits \
  --dmpnn_dir dmpnn_out \
  --baselines_dir baselines_out \
  --outdir screening \
  --task regression \
  --id_col Catalog_NO \
  --name_col Name \
  --smiles_col canonical_smiles

echo "[INFO] Step 3/3: Building docking SDF"
python 09_make_sdf_for_docking.py \
  --in_csv screening/top_for_docking.csv \
  --out_sdf screening/top_for_docking.sdf \
  --id_col Catalog_NO \
  --name_col Name \
  --smiles_col canonical_smiles

echo "[DONE] Screening pipeline complete. Key outputs:"
echo "  - screening/library_clean.csv"
echo "  - screening/library_failures.csv"
echo "  - screening/library_dedup_removed.csv"
echo "  - screening/preds_dmpnn.csv"
echo "  - screening/library_scored_with_ad.csv"
echo "  - screening/library_ranked.csv"
echo "  - screening/top_for_docking.csv"
echo "  - screening/top_for_docking.sdf"
echo "  - screening/plots/svg/*.svg"
