#!/bin/zsh
# FULL-FEATURE STRESS BATTERY — pre-handover certification for the 0.6.4
# stack: batch-cancel stop + SSD persistence + keepwarm + rank-aware budget.
# Sequential, single-owner. All log greps are anchored to phase-start line
# counts (stale-log immunity, learned from run 3); a failed boot ABORTS.
set -u
C=~/minimax-m3-cluster
API=http://127.0.0.1:8080
L=/private/tmp/minimax-m3-cluster-logs/startup.log
PASS=0; FAIL=0
say() { echo "[stress $(date '+%T')] $*" }
verdict() { if [[ "$2" == "0" ]]; then say "✓ PASS: $1"; PASS=$((PASS+1)); else say "✗ FAIL: $1"; FAIL=$((FAIL+1)); fi }
mark() { grep -c "" $L 2>/dev/null | tr -d ' ' }
since() { tail -n "+${1:-1}" $L 2>/dev/null }

slot_free() { curl -s -m 4 $API/health | python3 -c "import json,sys; print('yes' if not json.load(sys.stdin).get('active_request') else 'no')" 2>/dev/null }
wait_free() { local t=0; while (( t < $1 )); do [[ "$(slot_free)" == "yes" ]] && return 0; sleep 2; t=$((t+2)); done; return 1 }

bigreq() {  # $1 session, $2 text, $3 max_tokens, $4 repeats-of-filler
  python3 - "$1" "$2" "$3" "$4" <<'EOF' > /tmp/sb_req.json
import json, sys
sid, text, mt, rep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sysmsg = "You are a precise assistant. " + "Persistent working context for the agent session. "*rep
print(json.dumps({"model":"Minimax-M3-No-Think","metadata":{"session_id":sid},
 "messages":[{"role":"system","content":sysmsg},{"role":"user","content":text}],"max_tokens":mt}))
EOF
  curl -s -m 600 $API/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/sb_req.json
}

say "=========== FULL-FEATURE STRESS BATTERY ==========="
curl -s -m 4 $API/health >/dev/null 2>&1 || { say "ABORT: API not up at battery start"; exit 90 }

# ---- Phase 0: build an 11k session so the persistence loop has a subject ----
say "P0: build 11k session (SSD save subject)"
touch /tmp/sb_p0_marker
bigreq stress3-persist 'Reply SEED-1.' 12 1400 >/dev/null 2>&1
sleep 10
NEW_SAVES=$(find ~/.cache/thundermlx/prompt-kv -maxdepth 1 -type d -newer /tmp/sb_p0_marker 2>/dev/null | wc -l | tr -d ' ')
say "P0: NEW save dirs since marker: $NEW_SAVES"
verdict "P0: session saved to SSD (fresh artifact)" $([[ "$NEW_SAVES" -ge 1 ]] && echo 0 || echo 1)

# ---- Phase 1: persistence loop closure (save -> restart -> restore) ----
say "P1: SSD persistence loop (restart + restore of the 11k session)"
touch /private/tmp/minimax_m3_stop_requested
cd $C && /bin/zsh stop_cluster.sh >/dev/null 2>&1
for i in $(seq 1 40); do sleep 1; pgrep -f "run_with_watchdog|auto_restart.sh|mlx.launch" >/dev/null 2>&1 || break; done
pkill -9 -f "run_with_watchdog|m3_warmup" 2>/dev/null
rm -f /private/tmp/minimax_m3_stop_requested $C/.stop_requested; rm -rf /private/tmp/minimax_m3_start.lock
/bin/zsh $C/M3_Start.command >/dev/null 2>&1
BOOTOK=1
for i in $(seq 1 70); do sleep 10; curl -s -m 3 $API/health >/dev/null 2>&1 && { BOOTOK=0; break; }; done
verdict "P1a: reboot into full suite" $BOOTOK
if [[ "$BOOTOK" != "0" ]]; then
  say "ABORT: boot failed — remaining phases would test a corpse"
  say "=========== RESULT: $PASS pass / $((FAIL)) fail (ABORTED at P1a) ==========="
  exit 91
fi
sleep 5
B=$(mark)
bigreq stress3-persist 'Reply SEED-2.' 12 1400 >/dev/null 2>&1
TT=$(since $B | grep "released distributed" | tail -1 | grep -oE "first_token=[0-9.]+" | cut -d= -f2)
RESTORE_HIT=$(since $B | grep -icE "ssd.*restor")
say "P1b: post-restart turn TTFT=${TT:-none}s, ssd-restore log lines=$RESTORE_HIT"
P1B=1; python3 -c "import sys; sys.exit(0 if float('${TT:-99}') < 15.0 else 1)" 2>/dev/null && [ "${RESTORE_HIT:-0}" -ge 1 ] && P1B=0
verdict "P1b: SSD restore beat cold prefill (ttft<15s + restore evidence)" $P1B

# ---- Phase 2: stop hammer (12 mixed cycles) ----
say "P2: stop hammer — 12 cycles, alternating /v1/stop and disconnect"
H_FAILS=0
for i in $(seq 1 12); do
  curl -s -N -m 120 $API/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"Minimax-M3-No-Think\",\"messages\":[{\"role\":\"user\",\"content\":\"Write an endless numbered story. run $i $(date +%s)\"}],\"max_tokens\":4000,\"stream\":true}" >/dev/null 2>&1 &
  CP=$!
  sleep $((4 + i % 7))
  if (( i % 2 == 0 )); then curl -s -m 8 -X POST $API/v1/stop -d '{}' >/dev/null; else kill $CP 2>/dev/null; fi
  if ! wait_free 15; then H_FAILS=$((H_FAILS+1)); say "P2: cycle $i slot NOT free in 15s"; fi
  kill $CP 2>/dev/null; wait $CP 2>/dev/null
done
verdict "P2: 12/12 stop cycles freed the slot <=15s (fails=$H_FAILS)" $(( H_FAILS > 0 ))

# ---- Phase 3: 10k decode marathon ----
say "P3: 10k decode (wedge check, full suite active)"
B=$(mark)
curl -s -N -m 560 $API/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"Minimax-M3-No-Think\",\"messages\":[{\"role\":\"user\",\"content\":\"Write an endless glossary of invented terms with definitions. Never stop. $(date +%s)\"}],\"max_tokens\":10000,\"stream\":true}" >/dev/null 2>&1
M=$(since $B | grep "released distributed" | tail -1)
TOK=$(echo "$M" | grep -oE "tokens=[0-9]+" | cut -d= -f2)
TPS=$(echo "$M" | grep -oE "decode_tps=[0-9.]+" | cut -d= -f2)
say "P3: tokens=${TOK:-none} decode_tps=${TPS:-none}"
P3=1; [ "${TOK:-0}" -ge 10000 ] && python3 -c "import sys; sys.exit(0 if float('${TPS:-0}') > 20 else 1)" && P3=0
verdict "P3: 10k tokens complete at >20 t/s" $P3

# ---- Phase 4: agent tool-cycles (the phase that found the warmer race) ----
say "P4: agent tool-cycle sim (10 cycles)"
A_OUT=$(/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 $C/ops/agent_traffic_test.py 2>&1 | tail -1)
say "P4: $A_OUT"
echo "$A_OUT" | grep -q "10/10 cycles clean" ; verdict "P4: 10/10 agent cycles clean" $?

# ---- Phase 5: 20k prefill + mid-prefill stop, then full completion ----
say "P5: 20k prefill with mid-prefill stop, then full run"
python3 - <<'EOF' > /tmp/sb_20k.json
import json, uuid
f = "".join("Record %05d: subsystem nominal, checksum verified, latency in budget. " % i for i in range(2000))
print(json.dumps({"model":"Minimax-M3-No-Think","messages":[{"role":"user","content":f"Archive {uuid.uuid4()}: "+f+" Summarize in one sentence."}],"max_tokens":80,"stream":True}))
EOF
curl -s -N -m 300 $API/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/sb_20k.json >/dev/null 2>&1 &
CP=$!
sleep 12
curl -s -m 8 -X POST $API/v1/stop -d '{}' >/dev/null
P5A=1; wait_free 30 && P5A=0
kill $CP 2>/dev/null
verdict "P5a: mid-prefill stop freed slot <=30s" $P5A
python3 - <<'EOF' > /tmp/sb_20kb.json
import json, uuid
f = "".join("Record %05d: subsystem nominal, checksum verified, latency in budget. " % i for i in range(2000))
print(json.dumps({"model":"Minimax-M3-No-Think","messages":[{"role":"user","content":f"Archive {uuid.uuid4()}: "+f+" Summarize in one sentence."}],"max_tokens":80}))
EOF
R=$(curl -s -m 400 $API/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/sb_20kb.json | python3 -c "import json,sys; print('ok' if json.load(sys.stdin).get('choices') else 'bad')" 2>/dev/null)
verdict "P5b: fresh 20k prefill completes (${R:-noresp})" $([[ "$R" == "ok" ]] && echo 0 || echo 1)

# ---- Phase 6: queue pressure (3 concurrent) ----
say "P6: 3 concurrent requests (queueing)"
for i in 1 2 3; do
  curl -s -m 240 $API/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"Minimax-M3-No-Think\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the word QUEUE-$i only.\"}],\"max_tokens\":8}" > /tmp/sb_q$i.out 2>&1 &
done
wait
QOK=0
for i in 1 2 3; do grep -q "QUEUE-$i" /tmp/sb_q$i.out || QOK=1; done
verdict "P6: all 3 queued requests answered correctly" $QOK
P6B=1; wait_free 20 && P6B=0
verdict "P6b: slot clean after queue drain" $P6B

# ---- Phase 7: idle survival (3 min) ----
say "P7: 180s idle survival"
P7=0
for i in $(seq 1 30); do sleep 6; curl -s -m 3 $API/health >/dev/null 2>&1 || { P7=1; say "P7: DIED at +$((i*6))s"; break; }; done
verdict "P7: survived 180s idle" $P7

# ---- Phase 8: rank1 memory audit (paging-collapse regression check) ----
say "P8: rank1 memory after full battery"
R1=$(ssh -o BatchMode=yes -o ConnectTimeout=8 ${M3_PEER:-jonathan@10.0.0.2} 'vm_stat | awk "/^Pages wired down:/ {printf \"%.0f\", \$4*16384/1073741824; exit}"' 2>/dev/null)
say "P8: rank1 wired=${R1:-unknown}GB (budget: weights ~77 + slots <=11 + margin)"
P8=1; [ "${R1:-999}" -le 105 ] && P8=0
verdict "P8: rank1 wired within budget (<=105GB)" $P8

# ---- Phase 9: scorecard ----
say "P9: final scorecard"
H=$(curl -s -m 5 $API/health 2>/dev/null)
if [ -n "$H" ]; then
  echo "$H" | python3 -c "
import json,sys
h=json.load(sys.stdin); s=h.get('recent_request_stats') or {}
print(f\"  requests ok {s.get('ok_count')}/{s.get('count')} | decode {s.get('avg_decode_tps')} t/s | reuse {s.get('avg_cache_reuse_ratio')} | ttft {s.get('avg_ttft_s')}s\")" 2>/dev/null
else
  echo "  (health unavailable)"
fi
echo "  honored stops this boot: $(since 1 | grep -c 'decode stop honored')"
echo "  ttft breakdowns (last 2): $(since 1 | grep 'ttft breakdown' | tail -2 | grep -oE 'prepare=[0-9.]+s lock_wait=[0-9.]+s generator=[0-9.]+s' | tr '\n' ' ')"
say "=========== RESULT: $PASS pass / $FAIL fail ==========="
(( FAIL == 0 )) && say "FULL SUITE STRESS-CERTIFIED — ready for user testing" || say "NOT ready — fix and re-run"
exit $FAIL
