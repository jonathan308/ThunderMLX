#!/bin/zsh
# Upload the mixed-4.5 quant to Hugging Face.
#
# Prereqs (one-time):
#   pip install -U "huggingface_hub[cli]" hf_transfer
#   huggingface-cli login          # paste a WRITE token
#
# Usage:
#   HF_NAMESPACE=<your-hf-username> ./upload.sh
#
# Uploads from the external-drive copy so the serving copies are untouched.
# 270 GB — expect this to run for hours depending on uplink; it is resumable
# (re-running skips already-uploaded files).
set -euo pipefail

NS="${HF_NAMESPACE:?set HF_NAMESPACE to your Hugging Face username or org}"
REPO="$NS/MiniMax-M3-Mixed-4.5bit-MLX"
SRC="${MIXED45_SRC:-/Volumes/Models/MiniMax-M3-mixed-4.5}"
HERE="${0:A:h}"

[[ -f "$SRC/model.safetensors.index.json" ]] || { echo "missing index in $SRC"; exit 1 }

# Ship the curated model card, replacing the converter's auto-generated one.
cp "$HERE/README.md" "$SRC/README.md"

# AppleDouble junk must never reach the Hub.
find "$SRC" -name '._*' -delete

export HF_HUB_ENABLE_HF_TRANSFER=1
hf repos create "$REPO" --repo-type model 2>/dev/null || true
# upload-large-folder: resumable, chunked — re-running resumes where it left off.
hf upload-large-folder "$REPO" "$SRC" --repo-type model

echo "LIVE: https://huggingface.co/$REPO"
