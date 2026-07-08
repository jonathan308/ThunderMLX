#!/bin/zsh
# SOFT overnight trials — ZERO machine reboots. Process restarts only.
# On wedge (rc75): capture spindumps FIRST, then HALT machines-up.
# On any infra anomaly: HALT machines-up. Never reboot anything.
set -u
REPO=$HOME/minimax-m3-cluster
OPS=$REPO/ops
export MATRIX_FILE=$OPS/soft_matrix.json
export LC_ALL=en_US.UTF-8
source $OPS/priv.sh

say() { echo "[$(date '+%F %T')] $*" }
attention() {
  { echo "=== MORNING ATTENTION $(date '+%F %T') ==="; echo "$1"; echo;
    python3 $OPS/matrix_state.py status 2>/dev/null } >> $OPS/MORNING_ATTENTION.txt
}
halt() { say "HALT: $1"; attention "$1"; exit 1 }

restart_cluster() {
  say "process restart (env: $1)"
  cd $REPO
  /bin/zsh M3_Stop.command </dev/null >/dev/null 2>&1
  sleep 5
  /bin/zsh M3_Start.command </dev/null >/dev/null 2>&1 &
  sleep 5
  $OPS/wait_ready.sh 1500 || return 1
  # link-degradation signal: RTR failures on this init
  rtr=$(grep -c "RTR failed" /private/tmp/minimax-m3-cluster-logs/startup.log 2>/dev/null || echo 0)
  say "restart complete (cumulative RTR failures this boot-log: $rtr)"
  $OPS/coherence_check.sh || return 2
  return 0
}

POLLER_PID=""
poller_on()  { ( while :; do curl -s -m 2 http://127.0.0.1:8090/api/status >/dev/null 2>&1; sleep 2.5; done ) & POLLER_PID=$!; say "poller ON pid=$POLLER_PID" }
poller_off() { [[ -n "$POLLER_PID" ]] && { kill $POLLER_PID 2>/dev/null; say "poller OFF" }; POLLER_PID="" }
trap 'poller_off' EXIT

say "=== SOFT TRIALS START (no-reboot mode) ==="
while :; do
  python3 $OPS/matrix_state.py next >/dev/null
  rc=$?
  if (( rc == 3 )); then
    say "soft matrix exhausted"
    break
  fi
  idx=$(python3 $OPS/matrix_state.py field index)
  name=$(python3 $OPS/matrix_state.py field name)
  restart=$(python3 $OPS/matrix_state.py field restart false)
  poller=$(python3 $OPS/matrix_state.py field poller off)
  soak_cmd=$(python3 $OPS/matrix_state.py field soak_cmd)
  say "=== soft trial $idx: $name ==="

  eval "$(python3 $OPS/apply_trial_env.py $idx)"   # writes .env.local (pristine+overrides), prints SOAK_*/IOGPU*
  # ALWAYS set iogpu explicitly — sysctls persist until reboot and we never
  # reboot, so a tuned trial would otherwise leak into every later trial.
  # 0 = restore OS default.
  priv_run /usr/sbin/sysctl iogpu.wired_limit_mb=$IOGPU0 >/dev/null && say "rank0 iogpu=$IOGPU0"
  priv_run_rank1 "/usr/sbin/sysctl iogpu.wired_limit_mb=$IOGPU1" >/dev/null && say "rank1 iogpu=$IOGPU1"

  if [[ "$restart" == "true" || "$IOGPU0" != "0" ]]; then
    restart_cluster "$name"
    rrc=$?
    if (( rrc == 1 )); then python3 $OPS/matrix_state.py record $idx infra_error --note "restart never ready"; halt "trial $idx restart never became ready"; fi
    if (( rrc == 2 )); then python3 $OPS/matrix_state.py record $idx incoherent --note "coherence gate failed after restart"; halt "trial $idx incoherent after restart"; fi
  fi

  # prefill probe (16k, cache-busting)
  ptps=""
  python3 $OPS/prefill_bench.py --sizes 16384 --json > /tmp/soft_prefill.json 2>/dev/null
  prc=$?
  if (( prc == 75 )); then
    $OPS/live_wedge_capture.sh soft${idx}_prefill
    python3 $OPS/matrix_state.py record $idx wedge --note "WEDGE during prefill probe — spindumps captured, machines LEFT UP" --data regime=prefill
    halt "trial $idx wedged during prefill probe (spindumps in ops/logs)"
  elif (( prc == 0 )); then
    ptps=$(python3 -c 'import json; print(json.load(open("/tmp/soft_prefill.json")).get("prefill_tps_best",""))' 2>/dev/null)
    say "prefill: $ptps tok/s"
  fi

  [[ "$poller" == "on" ]] && poller_on
  cd $REPO
  if [[ -n "$soak_cmd" ]]; then
    eval "$soak_cmd --out-prefix soft${idx}"
  else
    python3 long_decode_soak8.py --rounds $SOAK_ROUNDS --gap $SOAK_GAP --max-tokens $SOAK_MAX_TOKENS --out-prefix "soft${idx}"
  fi
  soak_rc=$?
  poller_off

  jsonl=$(ls -t $REPO/soak8_soft${idx}_*.jsonl 2>/dev/null | head -1)
  clean=$(grep -c '"status": "ok"' "$jsonl" 2>/dev/null || echo 0)
  if (( soak_rc == 0 )); then
    python3 $OPS/matrix_state.py record $idx clean --data clean_rounds=$clean jsonl=${jsonl:t} prefill_tps=$ptps
    say "trial $idx CLEAN ($clean units, prefill $ptps)"
  elif (( soak_rc == 75 )); then
    $OPS/live_wedge_capture.sh soft${idx}_soak
    python3 $OPS/matrix_state.py record $idx wedge --note "WEDGE — spindumps captured BEFORE any recovery, machines LEFT UP" --data clean_rounds=$clean jsonl=${jsonl:t} prefill_tps=$ptps
    halt "trial $idx WEDGED (this is the forensic jackpot — spindumps in ops/logs; decide recovery in the morning)"
  else
    python3 $OPS/matrix_state.py record $idx infra_error --note "soak rc=$soak_rc" --data jsonl=${jsonl:t}
    halt "trial $idx infra error rc=$soak_rc"
  fi
done

say "writing SOFT_REPORT"
python3 $OPS/matrix_state.py status > $OPS/SOFT_REPORT.txt
say "=== SOFT TRIALS COMPLETE — cluster left on pristine config (final trial) ==="
