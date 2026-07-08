#!/bin/zsh
#
# run_server.sh — launch the sharded mlx_vlm OpenAI server for MiniMax-M3
# Single-node fallback for local testing. Designed to be called by launchd or by hand.
#
# Endpoint: http://<STUDIO_TAILSCALE>:8080/v1   (see /v1/models, /v1/chat/completions)
#
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
DIR="${M3_CLUSTER_DIR:-${SCRIPT_DIR:h}}"
if [[ -f "$DIR/.env.local" ]]; then
  source "$DIR/.env.local"
elif [[ -f "$DIR/m3_cluster.env" ]]; then
  source "$DIR/m3_cluster.env"
elif [[ -f "$DIR/.env" ]]; then
  source "$DIR/.env"
fi

export MLX_M3_MODEL="${MLX_M3_MODEL:-mlx-community/MiniMax-M3-4bit}"
export MLX_M3_HOST="${MLX_M3_HOST:-0.0.0.0}"
export MLX_M3_PORT="${MLX_M3_PORT:-8080}"
export MLX_TRUST_REMOTE_CODE="true"   # M3 ships remote config code
export MLX_METAL_FAST_SYNCH="1"       # faster metal syncs (matches MLX docs)
# KV cache quantization — lets 225GB model + long contexts fit comfortably in 256GB
export KV_BITS="4"
export KV_QUANT_SCHEME="turboquant"

cd "$DIR"
exec "${M3_PYTHON:-python3}" sharded_server.py
