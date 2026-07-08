#!/bin/zsh
# Wait for cluster API up + warmup complete. Usage: wait_ready.sh [timeout_seconds]
TIMEOUT=${1:-1500}
deadline=$(( $(date +%s) + TIMEOUT ))
until curl -s -m 3 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; do
  (( $(date +%s) > deadline )) && { echo "wait_ready: API timeout after ${TIMEOUT}s"; exit 1 }
  sleep 10
done
echo "wait_ready: API up at $(date '+%T')"
# Phase 2: wait for warmup to START (2026-07-05 19:11 lesson: checking for the
# warmup process before it had spawned raced straight through and the soak hit
# a cold server — recorded as a false wedge and reboot-looped the machines).
start_deadline=$(( $(date +%s) + 180 ))
seen=0
while (( $(date +%s) < start_deadline )); do
  if pgrep -f m3_warmup.py >/dev/null 2>&1; then seen=1; break; fi
  sleep 5
done
if (( seen )); then
  echo "wait_ready: warmup started, waiting for completion"
else
  echo "wait_ready: WARNING warmup never seen within 180s (disabled? proceed to coherence gate)"
fi
# Phase 3: wait for warmup to finish (2 consecutive empty checks)
empty=0
while (( empty < 2 )); do
  (( $(date +%s) > deadline )) && { echo "wait_ready: warmup still running at timeout"; exit 1 }
  if pgrep -f m3_warmup.py >/dev/null 2>&1; then empty=0; else empty=$(( empty + 1 )); fi
  sleep 15
done
echo "wait_ready: warmup complete at $(date '+%T')"
