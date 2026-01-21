#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG --------
DATASET_NAME="yourproject-code"   # only used for logging
EXPORT="/tmp/kaggle_code_export"
MESSAGE="${1:-code update}"

# -------- CLEAN EXPORT DIR --------
echo "▶ Exporting code to $EXPORT"
rm -rf "$EXPORT"
mkdir -p "$EXPORT"

# -------- COPY FILES --------
rsync -av \
  --exclude '__pycache__' \
  --exclude 'wandb' \
  src notebooks outputs requirements.txt README.md dataset-metadata.json \
  "$EXPORT/"

# -------- UPLOAD --------
echo "▶ Uploading Kaggle dataset ($DATASET_NAME)"
kaggle datasets version \
  -p "$EXPORT" \
  --dir-mode zip \
  -m "$MESSAGE"

echo "✅ Done."
