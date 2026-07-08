#!/bin/zsh
# Offline acceptance rig for the nonce-coordinated decode stop.
# RUN ONLY IN A MAINTENANCE WINDOW (it owns the cluster for ~15 min).
#
# Prereqs: cluster UP with MLX_M3_SAFE_DECODE_STOP=1 live in the SERVER env
# (verify via: ps eww <rank0 pid> | grep SAFE_DECODE_STOP). Both ranks must
# run identical code (auto-relaunch now syncs, but verify after manual edits).
#
# Scenarios: (1) client disconnect mid-decode, (2) POST /v1/stop mid-decode,
# (3) disconnect during a LONG prefill, (4) stop during a buffered tool turn.
# PASS per scenario: slot free ≤20s AND rank-0 'decode stop honored' (EOS-injection design: one
# line per stop; rank 1 follows the synced EOS) AND a follow-up succeeds.
set -u
API=http://127.0.0.1:8080
LOG=/private/tmp/minimax-m3-cluster-logs/startup.log
PASS=0; FAIL=0

say() { echo "[rig $(date '+%T')] $*" }

slot_state() {
  curl -s -m 4 $API/health 2>/dev/null | python3 -c "
import json,sys
try: ar=json.load(sys.stdin).get('active_request')
except Exception: print('unknown'); raise SystemExit
print('busy' if ar else 'free')" 2>/dev/null
}

wait_slot_free() {  # $1 = seconds budget
  local budget=$1 t=0
  while (( t < budget )); do
    [[ "$(slot_state)" == "free" ]] && return 0
    sleep 2; t=$((t+2))
  done
  return 1
}

honored_since() {  # $1 = marker line count before the scenario
  local before=$1
  local now=$(grep -c "decode stop honored" $LOG 2>/dev/null || echo 0)
  echo $(( now - before ))
}

fire_generation() {  # $1 = max_tokens, prints curl pid
  curl -s -N -m 300 $API/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"Minimax-M3-No-Think\",\"messages\":[{\"role\":\"user\",\"content\":\"Write an extremely long detailed essay about bridge maintenance procedures. $(date +%s%N)\"}],\"max_tokens\":$1,\"stream\":true}" \
    >/dev/null 2>&1 &
  echo $!
}

verdict() {  # $1 name, $2 ok(0/1)
  if [[ "$2" == "0" ]]; then say "✓ PASS: $1"; PASS=$((PASS+1))
  else say "✗ FAIL: $1"; FAIL=$((FAIL+1)); fi
}

say "=== STOP ACCEPTANCE RIG ==="
[[ "$(slot_state)" == "free" ]] || { say "cluster busy — aborting rig"; exit 2 }

# --- Scenario 1: client disconnect mid-decode ---
say "S1: disconnect mid-decode"
H0=$(grep -c "decode stop honored" $LOG 2>/dev/null || echo 0)
CP=$(fire_generation 8000); sleep 15; kill $CP 2>/dev/null
S1=1; wait_slot_free 20 && (( $(honored_since $H0) >= 1 )) && S1=0
verdict "disconnect mid-decode (slot ≤20s + both ranks honored)" $S1
sleep 5

# --- Scenario 2: /v1/stop mid-decode ---
say "S2: /v1/stop mid-decode"
H0=$(grep -c "decode stop honored" $LOG 2>/dev/null || echo 0)
CP=$(fire_generation 8000); sleep 15
curl -s -m 8 -X POST $API/v1/stop -d '{}' -H "Content-Type: application/json" >/dev/null
S2=1; wait_slot_free 20 && (( $(honored_since $H0) >= 1 )) && S2=0
kill $CP 2>/dev/null
verdict "/v1/stop mid-decode" $S2
sleep 5

# --- Scenario 3: disconnect during long prefill ---
say "S3: disconnect during 14k prefill"
python3 - <<'EOF' > /tmp/rig_bigprompt.json
import json, uuid
filler = ("Survey entry %04d: instruments nominal, calibration pass complete. " % i for i in range(1400))
body = {"model": "Minimax-M3-No-Think",
        "messages": [{"role": "user", "content": f"Archive {uuid.uuid4()}: " + "".join(filler) + " Summarize."}],
        "max_tokens": 64, "stream": True}
print(json.dumps(body))
EOF
curl -s -N -m 300 $API/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/rig_bigprompt.json >/dev/null 2>&1 & CP=$!
sleep 8; kill $CP 2>/dev/null   # mid-prefill (14k @ ~350tps ≈ 40s)
S3=1; wait_slot_free 30 && S3=0
verdict "disconnect mid-prefill (slot ≤30s)" $S3
sleep 5

# --- Scenario 4: stop during buffered tool turn ---
say "S4: /v1/stop during tool turn"
H0=$(grep -c "decode stop honored" $LOG 2>/dev/null || echo 0)
curl -s -N -m 300 $API/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"Minimax-M3-No-Think",
  "messages":[{"role":"user","content":"Write a complete 2000-line HTML game using the write_file tool."}],
  "tools":[{"type":"function","function":{"name":"write_file","description":"write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"code":{"type":"string"}},"required":["path","code"]}}}],
  "max_tokens":8000,"stream":true}' >/dev/null 2>&1 & CP=$!
sleep 15
curl -s -m 8 -X POST $API/v1/stop -d '{}' -H "Content-Type: application/json" >/dev/null
S4=1; wait_slot_free 20 && (( $(honored_since $H0) >= 1 )) && S4=0
kill $CP 2>/dev/null
verdict "/v1/stop during buffered tool turn" $S4

# --- follow-up health ---
sleep 3
OK=$(curl -s -m 30 $API/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Minimax-M3-No-Think","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}' \
  | python3 -c "import json,sys; print('yes' if json.load(sys.stdin).get('choices') else 'no')" 2>/dev/null)
verdict "post-rig follow-up request" $([[ "$OK" == "yes" ]] && echo 0 || echo 1)

say "=== RESULT: $PASS pass / $FAIL fail ==="
(( FAIL == 0 )) && say "STOP FEATURE CERTIFIED — safe to enable for clients" || say "NOT certified — keep gated"
exit $FAIL
