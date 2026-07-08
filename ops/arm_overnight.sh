#!/bin/zsh
# Arm the overnight matrix and kick off the first fresh-boot cycle.
# THE LAST COMMAND OF THE EVENING: it reboots BOTH machines.
# Usage: arm_overnight.sh [--dry-run]
set -u
REPO=$HOME/minimax-m3-cluster
OPS=$REPO/ops
STATE=$HOME/.config/thundermlx-ops
source $OPS/priv.sh
DRY=${1:-}

fail() { echo "PREFLIGHT FAIL: $1"; exit 1 }

echo "== overnight arm preflight =="
priv_check                || fail "rank0 privileges"
echo "  rank0 priv OK"
priv_check_rank1          || fail "rank1 privileges"
echo "  rank1 priv OK"
launchctl list 2>/dev/null | grep -q com.thundermlx.overnight || fail "LaunchAgent not loaded (install: cp ops/com.thundermlx.overnight.plist ~/Library/LaunchAgents/ && launchctl load ...)"
echo "  LaunchAgent loaded"
[[ -f $REPO/.env.local.pristine ]] || fail ".env.local.pristine snapshot missing"
echo "  pristine env snapshot OK"
ping -c1 -t2 10.0.0.2 >/dev/null 2>&1 || fail "rank1 link down"
echo "  TB link OK"

pending=$(python3 $OPS/matrix_state.py | grep -c "pending" || true)
(( pending > 0 )) || fail "no pending trials in matrix"
echo "  $pending pending trial(s)"

# any variant trials must have wheels on disk
for v in $(python3 -c "
import json
m = json.load(open('$OPS/trial_matrix.json'))
print('\n'.join(sorted({t['mlx_variant'] for t in m['trials'] if t.get('mlx_variant') and t.get('status','pending')=='pending' and t['mlx_variant']!='stock'})))
"); do
  ls $REPO/runtime_patches/variants/$v/*.whl >/dev/null 2>&1 || fail "variant '$v' has no wheels"
  n=$(ls $REPO/runtime_patches/variants/$v/*.whl | wc -l | tr -d ' ')
  [[ "$n" == "2" ]] || fail "variant '$v' needs exactly 2 wheels (mlx + mlx-metal), found $n"
  echo "  variant '$v' wheels OK"
done

python3 $OPS/matrix_state.py
if [[ "$DRY" == "--dry-run" ]]; then
  echo "== dry run: all preflight checks passed. NOT arming. =="
  exit 0
fi

echo "== ARMING — both machines reboot in 15s (ctrl-c to abort) =="
sleep 15
touch $STATE/overnight_armed
echo "armed. rebooting rank1..."
priv_run_rank1 /sbin/reboot
sleep 8
echo "rebooting rank0 (bye — see you on the other side)"
priv_run /sbin/shutdown -r now
