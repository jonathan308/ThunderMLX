#!/bin/zsh
#
# M3_Stop.command - double-click graceful stop for the MiniMax-M3 cluster.
#
# Safe to copy to Desktop. It resolves the real cluster folder, then delegates
# to stop_cluster.sh, which stops generation/ranks, auto-restart, warmup, and
# the dashboard, then checks for orphaned Metal wired memory.
set -euo pipefail

clear 2>/dev/null || true
echo "=================================================="
echo "  MiniMax-M3 Cluster - STOP"
echo "=================================================="
echo ""

SCRIPT_DIR="${0:A:h}"
if [[ -f "$SCRIPT_DIR/stop_cluster.sh" ]]; then
  CLUSTER_DEFAULT="$SCRIPT_DIR"
else
  CLUSTER_DEFAULT="$HOME/minimax-m3-cluster"
fi
CLUSTER="${M3_CLUSTER_DIR:-$CLUSTER_DEFAULT}"

if [[ ! -d "$CLUSTER" || ! -f "$CLUSTER/stop_cluster.sh" ]]; then
  echo "Cluster folder not found: $CLUSTER"
  echo "Set M3_CLUSTER_DIR or place this script inside the minimax-m3-cluster repo."
  echo ""
  read -k1 "?Press any key to close..."
  exit 2
fi

cd "$CLUSTER"

if /bin/zsh "$CLUSTER/stop_cluster.sh"; then
  EXIT_CODE=0
else
  EXIT_CODE=$?
fi

echo ""
if [[ "$EXIT_CODE" == "0" ]]; then
  echo "Stop sequence finished."
else
  echo "Stop sequence exited with code $EXIT_CODE."
fi
echo ""
if [[ -t 0 ]]; then
  read -k1 "?Press any key to close..."
fi
exit "$EXIT_CODE"
