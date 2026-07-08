#!/bin/zsh
# Overnight fresh-boot trial orchestrator. Runs at login via LaunchAgent
# com.thundermlx.overnight (RunAtLoad). Does NOTHING unless armed.
#
# Arm:    touch ~/.config/thundermlx-ops/overnight_armed   (after priv_check passes!)
# Abort:  touch ~/.config/thundermlx-ops/ABORT   (or delete the armed flag)
#
# Per trial: apply env overrides -> iogpu tune -> boot cluster -> wait ready ->
# coherence gate -> soak -> record verdict + diagnostics -> dual reboot -> next
# trial resumes here after auto-login. Matrix exhausted: restore pristine env,
# disarm, boot cluster in the pristine config, write MORNING_REPORT.

set -u
REPO=$HOME/minimax-m3-cluster
OPS=$REPO/ops
STATE=$HOME/.config/thundermlx-ops
ARMED=$STATE/overnight_armed
mkdir -p $OPS/logs
LOG=$OPS/logs/orchestrator_$(date +%Y%m%d_%H%M%S).log
exec >>$LOG 2>&1
source $OPS/priv.sh

say() { echo "[$(date '+%F %T')] $*" }

# anchor to the exact line: newer vm_stat also has "Pages tag-storage non-tag
# wired:" which double-matched /wired/ and concatenated (84GB read as "840")
wired_gb_rank0() { vm_stat | awk '/^Pages wired down:/ {printf "%d", $4*16384/1e9; exit}' }
wired_gb_rank1() { "${RANK1_SSH[@]}" 'vm_stat' 2>/dev/null | awk '/^Pages wired down:/ {printf "%d", $4*16384/1e9; exit}' }

disarm_halt() { say "DISARM: $1"; rm -f $ARMED; exit ${2:-1} }

dual_reboot() {
  say "dual reboot: rank1 first, then self"
  sync
  priv_run_rank1 /sbin/reboot || say "WARN: rank1 reboot command failed"
  sleep 8
  priv_run /sbin/shutdown -r now || disarm_halt "self-reboot failed — halting to avoid split state"
  exit 0  # unreached
}

[[ -f $ARMED ]] || { say "not armed; normal boot, nothing to do"; exit 0 }
[[ -f $STATE/ABORT ]] && { rm -f $STATE/ABORT; disarm_halt "ABORT file honored" 0 }

say "=== orchestrator boot (armed) ==="

# 1. Wait for the TB link + rank1 ssh (it rebooted alongside us)
deadline=$(( $(date +%s) + 600 ))
until ping -c1 -t2 10.0.0.2 >/dev/null 2>&1 && "${RANK1_SSH[@]}" true 2>/dev/null; do
  (( $(date +%s) > deadline )) && disarm_halt "rank1 unreachable for 10min after boot"
  sleep 10
done
say "rank1 reachable"

# 2. Privileges must work on both ranks or the loop would strand mid-cycle
priv_check || disarm_halt "rank0 privilege check failed"
priv_check_rank1 || disarm_halt "rank1 privilege check failed"

# 3. Orphan guard on a supposedly-fresh boot (belt & braces)
w0=$(wired_gb_rank0); w1=$(wired_gb_rank1)
say "wired: rank0=${w0}GB rank1=${w1}GB"
if ! pgrep -f sharded_server >/dev/null && { (( w0 > 150 )) || (( ${w1:-0} > 75 )) }; then
  say "ORPHAN detected on fresh boot?! rebooting again"
  dual_reboot
fi

# 4. Crash-sweep, then claim next trial (file handoff via current_trial.json —
#    shell-piping the JSON mangles non-ASCII under the LaunchAgent's C locale)
export LC_ALL=en_US.UTF-8
python3 $OPS/matrix_state.py sweep
python3 $OPS/matrix_state.py next >/dev/null
rc=$?
if (( rc == 3 )); then
  say "matrix exhausted — restoring pristine env + stock mlx, booting production config"
  cp $REPO/.env.local.pristine $REPO/.env.local
  # ensure a known-good morning endpoint regardless of what the last trial installed
  if ! python3 -c "import mlx.core as mx; assert mx.__version__=='0.31.2'" 2>/dev/null; then
    $OPS/install_mlx_variant.sh stock || say "WARN: stock restore failed — morning boot may be on a variant"
  fi
  rm -f $ARMED
  cd $REPO && /bin/zsh M3_Start.command </dev/null
  $OPS/wait_ready.sh 1500 && $OPS/coherence_check.sh
  python3 $OPS/matrix_state.py status > $OPS/MORNING_REPORT.txt
  say "MORNING_REPORT written"
  exit 0
fi
idx=$(python3 $OPS/matrix_state.py field index)
name=$(python3 $OPS/matrix_state.py field name)
say "=== trial $idx: $name ==="

# 5. Apply env + tunes
eval "$(python3 $OPS/apply_trial_env.py $idx)"
say "soak: rounds=$SOAK_ROUNDS gap=$SOAK_GAP max_tokens=$SOAK_MAX_TOKENS iogpu0=$IOGPU0 iogpu1=$IOGPU1"
if [[ "$IOGPU0" != "0" ]]; then
  priv_run /usr/sbin/sysctl iogpu.wired_limit_mb=$IOGPU0 && say "rank0 iogpu tuned: $IOGPU0"
fi
if [[ "$IOGPU1" != "0" ]]; then
  priv_run_rank1 "/usr/sbin/sysctl iogpu.wired_limit_mb=$IOGPU1" && say "rank1 iogpu tuned: $IOGPU1"
fi

# 5b. Optional mlx build swap (trial field "mlx_variant": label under
#     runtime_patches/variants, or "stock"). Kernel gate on BOTH ranks before
#     boot — a corrupt build must never reach a soak (it wastes a whole trial
#     slot and its token salad can't produce wedge-relevant data).
variant=$(python3 $OPS/matrix_state.py field mlx_variant)
if [[ -n "$variant" ]]; then
  say "installing mlx variant: $variant"
  if ! $OPS/install_mlx_variant.sh "$variant"; then
    python3 $OPS/matrix_state.py record $idx install_failed --note "variant $variant install failed"
    $OPS/install_mlx_variant.sh stock || say "WARN: stock restore also failed"
    dual_reboot
  fi
  scp -q -o BatchMode=yes -i "${M3_SSH_KEY:-$HOME/.ssh/id_ed25519_thundermlx}" $OPS/known_answer.py ${M3_PEER:-jonathan@10.0.0.2}:ops-tools/ 2>/dev/null
  if ! python3 $OPS/known_answer.py || ! "${RANK1_SSH[@]}" '~/mlx-env/bin/python ~/ops-tools/known_answer.py'; then
    python3 $OPS/matrix_state.py record $idx kernel_gate_failed --note "known-answer FAILED on $variant — corrupt build, restored stock"
    $OPS/install_mlx_variant.sh stock || say "WARN: stock restore failed"
    dual_reboot
  fi
  say "variant $variant passed kernel gates on both ranks"
fi

# 6. Boot cluster
cd $REPO && /bin/zsh M3_Start.command </dev/null
if ! $OPS/wait_ready.sh 1500; then
  python3 $OPS/matrix_state.py record $idx boot_timeout --note "API/warmup never ready"
  dual_reboot
fi

# 7. Coherence gate (guard rule: every boot, every build)
if ! $OPS/coherence_check.sh; then
  python3 $OPS/matrix_state.py record $idx incoherent --note "17*23 gate failed — build/config bad, not a wedge datum"
  dual_reboot
fi

# 7b. Prefill probe — 16k cache-busting prompt (short-prompt prompt_tps is
#     overhead-diluted; baseline expectation high-300s tok/s). A wedge here is
#     a first-class wedge datum (prefill regime, not decode).
ptps=""
python3 $OPS/prefill_bench.py --sizes 16384 --json > /tmp/prefill_probe.json 2>>$LOG
prc=$?
if (( prc == 75 )); then
  python3 $OPS/matrix_state.py record $idx wedge --note "wedged during PREFILL probe (16k)" --data regime=prefill
  $OPS/wedge_forensics.sh trial${idx}_prefill 12 || true
  dual_reboot
elif (( prc == 0 )); then
  ptps=$(python3 -c 'import json; print(json.load(open("/tmp/prefill_probe.json")).get("prefill_tps_best",""))' 2>/dev/null)
  say "prefill probe: ${ptps} tok/s (16k)"
else
  say "WARN: prefill probe failed rc=$prc (continuing to soak without prefill datum)"
fi

# 8. Soak (default: long-decode essays; trials may override with soak_cmd —
#    same contract: exit 0 clean, 75 wedge, writes soak8_*trial{idx}*.jsonl)
cd $REPO
soak_cmd=$(python3 $OPS/matrix_state.py field soak_cmd)
if [[ -n "$soak_cmd" ]]; then
  eval "$soak_cmd --out-prefix trial${idx}"
else
  python3 long_decode_soak8.py --rounds $SOAK_ROUNDS --gap $SOAK_GAP --max-tokens $SOAK_MAX_TOKENS \
    --out-prefix "trial${idx}"
fi
soak_rc=$?
jsonl=$(ls -t $REPO/soak8_*trial${idx}*.jsonl 2>/dev/null | head -1)
clean=$(grep -c '"status": "ok"' "$jsonl" 2>/dev/null || echo 0)
w0=$(wired_gb_rank0); w1=$(wired_gb_rank1)
if (( soak_rc == 0 )); then
  python3 $OPS/matrix_state.py record $idx clean --data clean_rounds=$clean jsonl=$jsonl wired0=$w0 wired1=$w1 prefill_tps=$ptps
  say "trial $idx CLEAN ($clean rounds)"
elif (( soak_rc != 75 )); then
  # NOT a wedge: soak infrastructure failed (exit 75 is the only wedge signal).
  # Do not fabricate a wedge datum and do not blind-reboot — halt for inspection.
  # (2026-07-05 19:11 lesson: rc=1 against a still-warming server was recorded
  # as "wedge" and reboot-looped both machines three times.)
  python3 $OPS/matrix_state.py record $idx infra_error --note "soak rc=$soak_rc (not 75) — infra failure, machines left up" --data jsonl=$jsonl wired0=$w0 wired1=$w1
  disarm_halt "soak infra error rc=$soak_rc on trial $idx — halting matrix, machines left up for inspection"
else
  python3 $OPS/matrix_state.py record $idx wedge --data clean_rounds=$clean jsonl=$jsonl wired0=$w0 wired1=$w1 soak_rc=$soak_rc prefill_tps=$ptps
  say "trial $idx WEDGE after $clean clean rounds (wired now: $w0/$w1 GB)"
  # post-wedge diagnostics while the corpse is warm (cluster is dead anyway)
  tail -40 /private/tmp/minimax-m3-cluster-logs/startup.log > $OPS/logs/trial${idx}_wedge_startuplog_tail.txt 2>/dev/null
  ps aux | grep -E "sharded|mlx" | grep -v grep > $OPS/logs/trial${idx}_wedge_ps.txt 2>/dev/null
  $OPS/wedge_forensics.sh trial${idx} 12 || say "WARN: forensics capture failed"
fi

# 9. Fresh boots for the next trial
dual_reboot
