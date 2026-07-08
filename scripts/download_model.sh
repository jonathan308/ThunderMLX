#!/bin/zsh
#
# Download the configured MiniMax-M3 model to a local folder on rank 0, and
# optionally to rank 1 at the same path. The dashboard calls this in a screen
# session; it is also useful as a terminal fallback.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
CLUSTER_DEFAULT="${SCRIPT_DIR:h}"
CLUSTER="${M3_CLUSTER_DIR:-$CLUSTER_DEFAULT}"
if [[ -f "$CLUSTER/.env.local" ]]; then
  source "$CLUSTER/.env.local"
elif [[ -f "$CLUSTER/m3_cluster.env" ]]; then
  source "$CLUSTER/m3_cluster.env"
elif [[ -f "$CLUSTER/.env" ]]; then
  source "$CLUSTER/.env"
fi

REPO_ID="${1:-${MLX_M3_MODEL_ID:-mlx-community/MiniMax-M3-4bit}}"
TARGET="${2:-${MLX_M3_MODEL:-}}"
if [[ -z "$TARGET" || "$TARGET" != /* && "$TARGET" != "~"* && "$TARGET" != "."* ]]; then
  SAFE="${REPO_ID//\//--}"
  TARGET="$HOME/.cache/m3-models/$SAFE"
fi

download_python='
import os
import sys
from pathlib import Path

repo_id = os.environ["M3_DOWNLOAD_REPO_ID"]
target = Path(os.environ["M3_DOWNLOAD_TARGET"]).expanduser()
target.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("huggingface_hub is required for dashboard downloads.", file=sys.stderr)
    print("Install it with: python3 -m pip install -U huggingface_hub", file=sys.stderr)
    raise

print(f"[download] repo={repo_id}")
print(f"[download] target={target}")
snapshot_download(
    repo_id=repo_id,
    local_dir=str(target),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("[download] complete")
'

echo "[download] rank 0 starting: $REPO_ID -> $TARGET"
M3_DOWNLOAD_REPO_ID="$REPO_ID" M3_DOWNLOAD_TARGET="$TARGET" python3 -c "$download_python"

if [[ "${M3_DOWNLOAD_ON_WORKER:-0}" == "1" ]]; then
  PEER="${M3_PEER:-${M3_RANK1_DIRECT_SSH:-}}"
  if [[ -z "$PEER" ]]; then
    PEER="${M3_RANK1_FALLBACK_SSH:-}"
  fi
  if [[ -z "$PEER" ]]; then
    echo "[download] worker requested but no M3_PEER/M3_RANK1_DIRECT_SSH is configured" >&2
    exit 2
  fi
  echo "[download] rank 1 starting over SSH: $PEER"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "M3_DOWNLOAD_REPO_ID=$(printf %q "$REPO_ID") M3_DOWNLOAD_TARGET=$(printf %q "$TARGET") python3 -c $(printf %q "$download_python")"
fi

echo "[download] done. Restart the cluster so MLX_M3_MODEL is loaded from: $TARGET"
