#!/bin/zsh
# Orphan kill-test: prove a TERM'd fully-loaded server clears wired Metal.
# Discipline (learned 2026-07-07): (1) DISARM the supervisor first — its
# hard sweep SIGKILLs, and SIGKILL-mid-load is the guaranteed orphan mint;
# (2) target the real server python (owner of :8080), never the watchdog;
# (3) only fire when fully wired (>150GB).
set -u
C=~/minimax-m3-cluster
API=http://127.0.0.1:8080
say() { echo "[killtest $(date '+%T')] $*" }
wired() { echo $(( $(vm_stat | awk '/^Pages wired down:/ {print $4; exit}' | tr -d '.') * 16384 / 1073741824 )) }

W=$(wired)
[ $W -ge 150 ] || { say "not fully wired (${W}GB) — run only against a loaded server"; exit 2 }
curl -s -m 5 $API/health >/dev/null || { say "API down — need a serving cluster"; exit 2 }

SRV=$(lsof -ti tcp:8080 -s tcp:LISTEN 2>/dev/null | head -1)
[ -n "$SRV" ] || { say "no :8080 owner found"; exit 2 }
say "server pid $SRV, wired ${W}GB — disarming supervisor and firing TERM"

# disarm: stop flag makes the supervisor exit instead of sweep/relaunch
touch /private/tmp/minimax_m3_stop_requested
kill -TERM $SRV
T0=$(date +%s)
CLEARED=""
for i in $(seq 1 60); do
  sleep 3
  W2=$(wired)
  if [ $W2 -lt 60 ]; then CLEARED=$(( $(date +%s) - T0 )); say "PASS: ${W}GB -> ${W2}GB in ${CLEARED}s — TERM cleanup works"; break; fi
done
if [ -z "$CLEARED" ]; then say "FAIL: still $(wired)GB after 180s — TERM cleanup insufficient"; RC=1; else RC=0; fi

# restart cleanly regardless — sweep BOTH ranks (an unswept rank1 zombie
# holds jaccl resources and can block or double-load the next boot)
sleep 3
pkill -TERM -f "run_with_watchdog" 2>/dev/null
ssh -o BatchMode=yes -o ConnectTimeout=8 ${M3_PEER:-jonathan@10.0.0.2} 'pkill -TERM -f "run_with_watchdog|sharded_server|mlx-vlm064-env/bin/python" 2>/dev/null' 2>/dev/null
sleep 3
rm -f /private/tmp/minimax_m3_stop_requested $C/.stop_requested; rm -rf /private/tmp/minimax_m3_start.lock
/bin/zsh $C/M3_Start.command >/dev/null 2>&1
say "restart initiated; exiting $RC"
exit $RC
