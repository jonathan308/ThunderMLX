#!/bin/zsh
# EASY BUTTON: revert to the certified-stable point (SSD cache + keepwarm
# OFF, stop-certified build). Double-click and wait for "DONE".
# RAM prompt cache stays ON — only SSD persistence/keepwarm are disabled.
CLUSTER="${M3_CLUSTER_DIR:-$HOME/minimax-m3-cluster}"
exec /bin/zsh "$CLUSTER/ops/ssd_cache_toggle.sh" off
