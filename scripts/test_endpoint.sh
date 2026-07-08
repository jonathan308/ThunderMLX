#!/bin/zsh
#
# test_endpoint.sh — self-contained test for the M3 cluster.
# Run AFTER a fresh reboot (both machines clean). Does everything:
#   1. Launches the cluster
#   2. Waits for it to come up
#   3. Fires 4 test requests WITH per-request timeout (no orphan on hang)
#   4. Monitors memory before/after each
#   5. Reports PASS/FAIL clearly
#   6. If a request hangs, kills ONLY that request (not the server) so memory isn't orphaned
#
# Usage:  ~/minimax-m3-cluster/scripts/test_endpoint.sh
#
set -uo pipefail

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

ENDPOINT="${M3_TEST_ENDPOINT:-${M3_PUBLIC_BASE_URL%/v1}}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8080}"
LOG="$CLUSTER/test_run.log"
PEER="${M3_PEER:-${M3_RANK1_DIRECT_SSH:-${M3_RANK1_FALLBACK_SSH:-}}}"
KEY="${M3_SSH_KEY:-}"

mem() { memory_pressure 2>/dev/null | grep "free percentage" | head -1 | sed 's/.*: //'; }
ssh_cmd() {
  if [[ -n "$KEY" ]]; then
    ssh -i "$KEY" -o BatchMode=yes "$@"
  else
    ssh -o BatchMode=yes "$@"
  fi
}
mem_mb() { ssh_cmd -o ConnectTimeout=10 "$PEER" 'memory_pressure 2>/dev/null | grep "free percentage" | head -1 | sed "s/.*: //"' 2>/dev/null; }

echo "===== M3 CLUSTER TEST (self-contained) =====" | tee "$LOG"
echo "" | tee -a "$LOG"

# --- 0. Clean slate check ---
echo "[0] Checking both machines are clean..." | tee -a "$LOG"
S=$(mem); MB=$(mem_mb)
echo "    Rank 0: $S free | Rank 1: $MB free" | tee -a "$LOG"
if [[ "$S" == *"9"*% ]] && [[ "$MB" == *"9"*% ]]; then
  echo "    OK - both clean (90%+ free)" | tee -a "$LOG"
else
  echo "    WARNING - memory not clean. Reboot recommended first." | tee -a "$LOG"
  echo "    Continuing anyway..." | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

# --- 1. Launch cluster ---
echo "[1] Launching cluster..." | tee -a "$LOG"
pkill -f sharded_server 2>/dev/null
pkill -f mlx._distributed 2>/dev/null
ssh_cmd "$PEER" 'pkill -f sharded_server 2>/dev/null; pkill -f mlx-python 2>/dev/null' 2>/dev/null
sleep 3
cd "$CLUSTER"
nohup /bin/zsh "$CLUSTER/launch_cluster.sh" > "$CLUSTER/test_server.log" 2>&1 &
LAUNCH_PID=$!
echo "    Launched (launcher PID $LAUNCH_PID). Waiting for startup (~40s)..." | tee -a "$LOG"
sleep 45

# --- 2. Wait for port ---
echo "[2] Waiting for server to come up..." | tee -a "$LOG"
READY=0
for i in {1..30}; do
  if curl -s --max-time 3 "$ENDPOINT/health" >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 2
done
if [[ $READY -ne 1 ]]; then
  echo "    FAIL - server didn't come up in 60s. Check $CLUSTER/test_server.log" | tee -a "$LOG"
  echo "    Stopping cluster to avoid orphan..." | tee -a "$LOG"
  pkill -f sharded_server 2>/dev/null; pkill -f mlx._distributed 2>/dev/null
  ssh_cmd "$PEER" 'pkill -f sharded_server 2>/dev/null; pkill -f mlx-python 2>/dev/null' 2>/dev/null
  exit 1
fi
echo "    OK - server up" | tee -a "$LOG"
echo "    baseline: Rank 0 $(mem) | Rank 1 $(mem_mb)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- 3. Test requests (with per-request timeout) ---
PASS=0; FAIL=0
TESTS=(
  '{"model":"x","messages":[{"role":"user","content":"What is 2+2? Just the number."}],"max_tokens":10}'
  '{"model":"x","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}'
  '{"model":"x","messages":[{"role":"user","content":"Name a color."}],"max_tokens":10}'
  '{"model":"x","messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":15}'
)

for i in {1..4}; do
  echo "[req $i] Sending..." | tee -a "$LOG"
  RESP=$(curl -s --max-time 60 "$ENDPOINT/v1/chat/completions" -H "Content-Type: application/json" -d "${TESTS[$i]}" 2>&1)
  RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "    FAIL (curl exit $RC - likely hung/timeout). Request abandoned." | tee -a "$LOG"
    FAIL=$((FAIL+1))
    echo "    !!! HANG DETECTED - server likely deadlocked. Stop it to avoid orphan:" | tee -a "$LOG"
    echo "        ~/minimax-m3-cluster/stop_cluster.sh" | tee -a "$LOG"
    break
  fi
  # Extract content
  CONTENT=$(echo "$RESP" | python3.14 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "(parse error)")
  echo "    response: $CONTENT" | tee -a "$LOG"
  echo "    mem: Rank 0 $(mem) | Rank 1 $(mem_mb)" | tee -a "$LOG"
  if [[ "$CONTENT" == *"parse error"* ]] || [[ -z "$CONTENT" ]]; then
    FAIL=$((FAIL+1))
  else
    PASS=$((PASS+1))
  fi
  sleep 3
  echo "" | tee -a "$LOG"
done

# --- 4. Report ---
echo "===== RESULT: $PASS passed, $FAIL failed =====" | tee -a "$LOG"
if [[ $FAIL -eq 0 ]] && [[ $PASS -ge 3 ]]; then
  echo "PASS - cluster stable for multiple requests!" | tee -a "$LOG"
  echo "Endpoint: $ENDPOINT/v1" | tee -a "$LOG"
else
  echo "FAIL - check logs. Stop server with ~/minimax-m3-cluster/stop_cluster.sh" | tee -a "$LOG"
fi
