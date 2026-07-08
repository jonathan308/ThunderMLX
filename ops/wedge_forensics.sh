#!/bin/zsh
# Post-wedge forensics: capture kernel/Thunderbolt/RDMA unified-log activity
# around the wedge window before the evidence is lost to the reboot.
# Usage: wedge_forensics.sh <output_prefix> [lookback_minutes]
OUT_PREFIX=${1:-wedge}
LOOKBACK=${2:-12}
OPS=$HOME/minimax-m3-cluster/ops
mkdir -p $OPS/logs

# Thunderbolt / IOKit / RDMA-adjacent subsystem traffic (bounded, compact)
log show --last ${LOOKBACK}m --style compact \
  --predicate '(subsystem CONTAINS[c] "thunderbolt") OR (subsystem CONTAINS[c] "iokit") OR (eventMessage CONTAINS[c] "rdma") OR (eventMessage CONTAINS[c] "jaccl") OR (eventMessage CONTAINS[c] "en7")' \
  > $OPS/logs/${OUT_PREFIX}_logshow_tb.txt 2>&1

# Kernel-process errors/faults only (keeps size sane)
log show --last ${LOOKBACK}m --style compact --process kernel --level error \
  > $OPS/logs/${OUT_PREFIX}_logshow_kernel_err.txt 2>&1

# Link + interface state snapshots
{ ifconfig en7; echo "---"; netstat -i | head -20 } > $OPS/logs/${OUT_PREFIX}_link.txt 2>&1

echo "forensics written: $OPS/logs/${OUT_PREFIX}_*"
