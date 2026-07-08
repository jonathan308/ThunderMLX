#!/bin/zsh
# M3_Restore_Gold.command — revert the cluster to the certified
# gold-endgame-v1 restore point (2026-07-07), including .env.local.
set -euo pipefail
clear 2>/dev/null || true
CLUSTER="${M3_CLUSTER_DIR:-$HOME/minimax-m3-cluster}"
RP="$CLUSTER/ops/restore-points/gold-endgame-v1"
echo "=================================================="
echo "  MiniMax-M3 — RESTORE gold-endgame-v1"
echo "=================================================="
echo ""
echo "This will STOP the cluster, revert all cluster code to the"
echo "gold-endgame-v1 tag, restore the matching .env.local, and"
echo "sync rank 1. Your prompt caches and lifetime stats are kept."
echo ""
read -q "?Proceed? [y/N] " || { echo ""; echo "Cancelled."; exit 0; }
echo ""
cd "$CLUSTER"
/bin/zsh "$CLUSTER/stop_cluster.sh" || true
pkill -f auto_restart 2>/dev/null || true
git checkout gold-endgame-v1 -- .
cp "$RP/env.local.snapshot" "$CLUSTER/.env.local"
/bin/zsh "$CLUSTER/sync_rank1.sh"
echo ""
echo "Restored to gold-endgame-v1. Double-click M3_Start.command to boot."
read -k1 "?Press any key to close..."
