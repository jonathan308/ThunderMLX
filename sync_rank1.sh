#!/bin/zsh
#
# Sync cluster source files to rank 1 before any lazy M3 launch.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
if [[ -f "$SCRIPT_DIR/.env.local" ]]; then
  source "$SCRIPT_DIR/.env.local"
elif [[ -f "$SCRIPT_DIR/m3_cluster.env" ]]; then
  source "$SCRIPT_DIR/m3_cluster.env"
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
  source "$SCRIPT_DIR/.env"
fi

CLUSTER="${M3_CLUSTER_DIR:-$SCRIPT_DIR}"
DIRECT_PEER="${M3_DIRECT_PEER:-${M3_RANK1_DIRECT_SSH:-}}"
FALLBACK_PEER="${M3_RANK1_FALLBACK_SSH:-${M3_TAILSCALE_PEER:-}}"
PEER="${M3_PEER:-}"

if [[ -z "$PEER" ]]; then
  if [[ -n "$DIRECT_PEER" ]] && ssh -o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1 \
      "$DIRECT_PEER" 'true' >/dev/null 2>&1; then
    PEER="$DIRECT_PEER"
  elif [[ -n "$FALLBACK_PEER" ]]; then
    PEER="$FALLBACK_PEER"
    echo "[sync] direct SSH unavailable; using fallback SSH for source sync." >&2
  else
    echo "[sync] ERROR: set M3_PEER or M3_RANK1_DIRECT_SSH in .env.local." >&2
    exit 2
  fi
fi

FILES=(
  .env.example
  bin/mlx-python
  LICENSE
  README.md
  M3_Start.command
  M3_Stop.command
  auto_restart.sh
  cluster_gui.py
  constrained_tools.py
  dashboard.html
  launch_cluster.sh
  m3_batch_cancel.py
  m3_capture.py
  m3_eagle3.py
  m3_multimodal_cache.py
  m3_pipeline_patch.py
  m3_warmup.py
  run_with_watchdog.py
  sharded_server.py
  stop_cluster.sh
  sync_rank1.sh
)

ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
  "mkdir -p '$CLUSTER' '$CLUSTER/bin' '$CLUSTER/ops'" >/dev/null
for f in "${FILES[@]}"; do
  if [[ -f "$CLUSTER/$f" ]]; then
    scp -o BatchMode=yes -o ConnectTimeout=10 \
      "$CLUSTER/$f" "$PEER:$CLUSTER/$f" >/dev/null
  fi
done

for f in ops/known_answer.py ops/check_runtime_compat.py; do
  if [[ -f "$CLUSTER/$f" ]]; then
    scp -o BatchMode=yes -o ConnectTimeout=10 \
      "$CLUSTER/$f" "$PEER:$CLUSTER/$f" >/dev/null
  fi
done

if [[ -d "$CLUSTER/docs" ]]; then
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "rm -rf '$CLUSTER/docs'; mkdir -p '$CLUSTER'" >/dev/null
  scp -r -o BatchMode=yes -o ConnectTimeout=10 \
    "$CLUSTER/docs" "$PEER:$CLUSTER/docs" >/dev/null
fi

for d in probes tools scripts runtime_patches; do
  if [[ -d "$CLUSTER/$d" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
      "rm -rf '$CLUSTER/$d'; mkdir -p '$CLUSTER'" >/dev/null
    scp -r -o BatchMode=yes -o ConnectTimeout=10 \
      "$CLUSTER/$d" "$PEER:$CLUSTER/$d" >/dev/null
  fi
done

if [[ -d "$CLUSTER/MSA Support" ]]; then
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "rm -rf '$CLUSTER/MSA Support'; mkdir -p '$CLUSTER'" >/dev/null
  scp -r -o BatchMode=yes -o ConnectTimeout=10 \
    "$CLUSTER/MSA Support" "$PEER:$CLUSTER/" >/dev/null
fi

ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
  "chmod +x '$CLUSTER/bin/mlx-python' '$CLUSTER/M3_Start.command' '$CLUSTER/M3_Stop.command' '$CLUSTER/launch_cluster.sh' '$CLUSTER/stop_cluster.sh' '$CLUSTER/auto_restart.sh' '$CLUSTER/sync_rank1.sh' '$CLUSTER/cluster_gui.py' '$CLUSTER/m3_warmup.py'; find '$CLUSTER/probes' '$CLUSTER/tools' '$CLUSTER/scripts' -type f \\( -name '*.py' -o -name '*.sh' \\) -exec chmod +x {} + 2>/dev/null || true; rm -rf '$CLUSTER/__pycache__' '$CLUSTER/probes/__pycache__' '$CLUSTER/tools/__pycache__'" \
  >/dev/null
