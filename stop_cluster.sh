#!/bin/zsh
#
# stop_cluster.sh — graceful stop helper for the MiniMax-M3 cluster.
#
# Give the Python ranks time to finish any active distributed generation and
# then shut down together. Do not SIGKILL by default: killing one rank while the
# other is inside MLX/JACCL collectives is the known orphaned-Metal-memory path.
#
set -uo pipefail

SCRIPT_DIR="${0:A:h}"
if [[ -f "$SCRIPT_DIR/.env.local" ]]; then
  source "$SCRIPT_DIR/.env.local"
elif [[ -f "$SCRIPT_DIR/m3_cluster.env" ]]; then
  source "$SCRIPT_DIR/m3_cluster.env"
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
  source "$SCRIPT_DIR/.env"
fi

DIRECT_PEER="${M3_DIRECT_PEER:-${M3_RANK1_DIRECT_SSH:-}}"
FALLBACK_PEER="${M3_RANK1_FALLBACK_SSH:-${M3_TAILSCALE_PEER:-}}"
PEER="${M3_PEER:-}"
CLUSTER="${M3_CLUSTER_DIR:-$SCRIPT_DIR}"
STOP_DRAIN_SECONDS=${M3_STOP_DRAIN_SECONDS:-960}
PORT_DOWN_TERM_SECONDS=${M3_STOP_PORT_DOWN_TERM_SECONDS:-30}
STOP_FILE="${M3_STOP_FILE:-/private/tmp/minimax_m3_stop_requested}"
API_PORT="${MLX_M3_PORT:-8080}"
API_BASE="http://127.0.0.1:${API_PORT}"
GUI_PORT="${M3_GUI_PORT:-8090}"
KEEP_DASHBOARD="${M3_STOP_KEEP_DASHBOARD:-0}"
KEEP_GATEWAY="${M3_STOP_KEEP_GATEWAY:-0}"
# Bracket one character in each process name so remote pgrep shells do not
# match their own command line while still matching the real process.
CLUSTER_REGEX=$(
  printf '%s\n' "$CLUSTER" | sed 's/[][\\.^$*+?(){}|]/\\&/g'
)
CLUSTER_PROCESS_PATTERN="$CLUSTER_REGEX/[r]un_with_watchdog.py|$CLUSTER_REGEX/[s]harded_server.py|$CLUSTER_REGEX/bin/[m]lx-python|[m]lx.launch.*$CLUSTER_REGEX|$CLUSTER_REGEX/[l]aunch_cluster.sh"
REMOTE_PROCESS_PATTERN="${(q)CLUSTER_PROCESS_PATTERN}"

if [[ -z "$PEER" ]]; then
  if [[ -n "$DIRECT_PEER" ]] && ssh -o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1 \
      "$DIRECT_PEER" 'true' >/dev/null 2>&1; then
    PEER="$DIRECT_PEER"
  elif [[ -n "$FALLBACK_PEER" ]]; then
    PEER="$FALLBACK_PEER"
    echo "Direct SSH unavailable; using fallback SSH for control/memory checks."
  else
    echo "Set M3_PEER or M3_RANK1_DIRECT_SSH in .env.local before stopping remote ranks."
    exit 2
  fi
fi

touch "$STOP_FILE"

count_alive() {
  (pgrep -fl "$CLUSTER_PROCESS_PATTERN" 2>/dev/null || true) \
    | grep -viE "[s]sh -tt|-o LogLevel=QUIET" \
    | wc -l | tr -d ' '
}

count_alive_mb() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "(pgrep -fl $REMOTE_PROCESS_PATTERN 2>/dev/null || true) | wc -l | tr -d ' '" 2>/dev/null
}

send_signal() {
  local SIG="$1"
  pkill -$SIG -f "$CLUSTER_PROCESS_PATTERN" 2>/dev/null
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "pkill -$SIG -f $REMOTE_PROCESS_PATTERN 2>/dev/null; true" 2>/dev/null
}

stop_relative_gui_from_cluster() {
  local PID PROC_CWD
  while IFS= read -r PID; do
    [[ "$PID" == <-> ]] || continue
    PROC_CWD=$(
      /usr/sbin/lsof -a -p "$PID" -d cwd -Fn 2>/dev/null \
        | sed -n 's/^n//p' | head -1
    )
    if [[ "$PROC_CWD" == "$CLUSTER" ]]; then
      kill -TERM "$PID" >/dev/null 2>&1 || true
    fi
  done < <(pgrep -f "[c]luster_gui.py" 2>/dev/null || true)
}

echo "Stopping M3 cluster gracefully (admin shutdown, drain active generation up to ${STOP_DRAIN_SECONDS}s)..."
ADMIN_SHUTDOWN_REQUESTED=0
if ! curl -s --max-time 5 -X POST "$API_BASE/admin/shutdown" >/dev/null 2>&1; then
  echo "Admin shutdown endpoint is not reachable; sending SIGTERM to any leftover ranks."
  send_signal TERM
else
  ADMIN_SHUTDOWN_REQUESTED=1
fi

STOPPED=0
PORT_DOWN_TERM_SENT=0
ITERATIONS=$(( STOP_DRAIN_SECONDS / 5 ))
if [[ "$ITERATIONS" -lt 1 ]]; then
  ITERATIONS=1
fi
for i in $(seq 1 "$ITERATIONS"); do
  sleep 5
  ALIVE=$(count_alive)
  if ALIVE_MB=$(count_alive_mb); then
    [[ "$ALIVE_MB" == <-> ]] || ALIVE_MB=unknown
  else
    ALIVE_MB=unknown
  fi
  echo "  Rank 0 procs: $ALIVE | Rank 1 procs: $ALIVE_MB ($((i*5))s)"
  if [[ "$ALIVE" == "0" ]] && [[ "$ALIVE_MB" == "0" ]]; then
    STOPPED=1
    break
  fi
  if [[ "$ADMIN_SHUTDOWN_REQUESTED" == "1" ]] && [[ "$PORT_DOWN_TERM_SENT" == "0" ]] && \
     [[ $((i*5)) -ge "$PORT_DOWN_TERM_SECONDS" ]] && \
     ! curl -s --max-time 2 "$API_BASE/health" >/dev/null 2>&1; then
    echo "  M3 API port is down but model ranks still linger; sending SIGTERM to finish clean shutdown."
    send_signal TERM
    PORT_DOWN_TERM_SENT=1
  fi
done

if [[ "$STOPPED" != "1" ]]; then
  echo "Model ranks are still alive or could not be verified after ${STOP_DRAIN_SECONDS}s."
  echo "Not sending SIGKILL by default because that can orphan Metal memory."
  echo "If you knowingly accept the orphan/reboot risk, rerun with M3_FORCE_KILL=1."
  if [[ "${M3_FORCE_KILL:-0}" == "1" ]]; then
    echo "M3_FORCE_KILL=1 set; sending SIGKILL fallback."
    send_signal KILL
    sleep 8
  fi
fi

pkill -TERM -f "[m]lx.launch.*$CLUSTER_REGEX" 2>/dev/null
pkill -TERM -f "$CLUSTER_REGEX/[l]aunch_cluster.sh" 2>/dev/null
pkill -TERM -f "$CLUSTER_REGEX/[a]uto_restart.sh" 2>/dev/null
/usr/bin/screen -S minimax_m3 -X quit >/dev/null 2>&1 || true
for pid_file in "$CLUSTER/auto_restart.pid" "$CLUSTER/m3_warmup.pid"; do
  if [[ -f "$pid_file" ]]; then
    PID=$(cat "$pid_file" 2>/dev/null)
    if [[ -n "$PID" ]]; then
      kill "$PID" >/dev/null 2>&1 || true
    fi
    rm -f "$pid_file"
  fi
done
pkill -TERM -f "$CLUSTER_REGEX/[m]3_warmup.py" 2>/dev/null || true
/usr/bin/screen -S m3_warmup -X quit >/dev/null 2>&1 || true

if [[ "$KEEP_GATEWAY" != "1" ]]; then
  if [[ -x "$CLUSTER/stop_gateway.sh" ]]; then
    /bin/zsh "$CLUSTER/stop_gateway.sh" >/dev/null 2>&1 || true
  fi
else
  echo "Keeping gateway alive for backend switch."
fi

if [[ "$KEEP_DASHBOARD" != "1" ]]; then
  echo "Stopping dashboard on port ${GUI_PORT}..."
  /usr/bin/screen -S m3_gui -X quit >/dev/null 2>&1 || true
  if [[ -f "$CLUSTER/cluster_gui.pid" ]]; then
    GUI_PID=$(cat "$CLUSTER/cluster_gui.pid" 2>/dev/null)
    if [[ -n "$GUI_PID" ]]; then
      kill "$GUI_PID" >/dev/null 2>&1 || true
    fi
    rm -f "$CLUSTER/cluster_gui.pid"
  fi
  pkill -TERM -f "$CLUSTER_REGEX/[c]luster_gui.py" 2>/dev/null || true
  # Also catch a relative-path GUI, but only when its cwd is this cluster.
  stop_relative_gui_from_cluster
else
  echo "Keeping dashboard on port ${GUI_PORT} alive for backend switch."
fi

sleep 5
echo ""
echo "Memory after stop:"
echo "  Rank 0: $(memory_pressure 2>/dev/null | grep 'free percentage' | sed 's/.*: //')"
if RANK1_FREE=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    'memory_pressure 2>/dev/null | grep "free percentage" | sed "s/.*: //"' 2>/dev/null); then
  [[ -n "$RANK1_FREE" ]] || RANK1_FREE=unknown
else
  RANK1_FREE=unknown
fi
echo "  Rank 1: $RANK1_FREE"
echo ""
echo "Checking for orphaned Metal memory..."
STOP_EXIT_STATUS=0
S_WIRED=$(memory_pressure 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc '0-9')
if [[ -z "$S_WIRED" ]]; then
  S_WIRED=$(vm_stat 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc '0-9')
fi
if [[ -n "$S_WIRED" ]]; then
  S_WIRED_GB=$(( S_WIRED * 16384 / 1024 / 1024 / 1024 ))
else
  S_WIRED_GB=unknown
  STOP_EXIT_STATUS=1
fi
REMOTE_WIRED_COMMAND='pages=$(memory_pressure 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc "0-9"); if [ -z "$pages" ]; then pages=$(vm_stat 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc "0-9"); fi; [ -n "$pages" ] || exit 1; printf "%s\n" "$pages"'
if MB_WIRED=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "$REMOTE_WIRED_COMMAND" 2>/dev/null); then
  MB_WIRED_GB=$(( MB_WIRED * 16384 / 1024 / 1024 / 1024 ))
else
  MB_WIRED_GB=unknown
  STOP_EXIT_STATUS=1
fi
echo "  Rank 0 wired: ${S_WIRED_GB}${${S_WIRED_GB:#unknown}:+GB}"
echo "  Rank 1 wired: ${MB_WIRED_GB}${${MB_WIRED_GB:#unknown}:+GB}"
echo ""
RANK0_LIMIT="${M3_ORPHAN_RANK0_WIRED_GB:-${M3_ORPHAN_STUDIO_WIRED_GB:-0}}"
RANK1_LIMIT="${M3_ORPHAN_RANK1_WIRED_GB:-${M3_ORPHAN_MACBOOK_WIRED_GB:-0}}"
if [[ "$S_WIRED_GB" != "unknown" && "$RANK0_LIMIT" != "0" && "$S_WIRED_GB" -gt "$RANK0_LIMIT" ]] || \
   [[ "$MB_WIRED_GB" != "unknown" && "$RANK1_LIMIT" != "0" && "$MB_WIRED_GB" -gt "$RANK1_LIMIT" ]]; then
  echo "ORPHANED METAL MEMORY DETECTED."
  echo "Reboot affected machines before relaunching M3."
  STOP_EXIT_STATUS=1
elif [[ "$S_WIRED_GB" == "unknown" ]] || [[ "$MB_WIRED_GB" == "unknown" ]]; then
  echo "Memory recovery could not be verified on both ranks."
else
  echo "Memory recovered cleanly. Safe to relaunch."
fi

# Final verification: report anything from THIS cluster's tree that survived.
# Escalate only when M3_FORCE_KILL=1 because SIGKILL can orphan Metal memory.
sleep 3
LOCAL_STRAGGLERS=$(pgrep -f "$CLUSTER_PROCESS_PATTERN" 2>/dev/null || true)
if ! REMOTE_STRAGGLERS=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "pgrep -f $REMOTE_PROCESS_PATTERN 2>/dev/null || true" 2>/dev/null); then
  echo "  stop: rank 1 final process verification failed."
  exit 1
fi
if [[ -n "$LOCAL_STRAGGLERS" ]] || [[ -n "$REMOTE_STRAGGLERS" ]]; then
  echo "  stop: stragglers survived graceful stop (rank 0: ${LOCAL_STRAGGLERS:-none}; rank 1: ${REMOTE_STRAGGLERS:-none})"
  if [[ "${M3_FORCE_KILL:-0}" == "1" ]]; then
    echo "  M3_FORCE_KILL=1 set; escalating remaining cluster processes to SIGKILL."
    send_signal KILL
    sleep 3
    LOCAL_STRAGGLERS=$(pgrep -f "$CLUSTER_PROCESS_PATTERN" 2>/dev/null || true)
    if ! REMOTE_STRAGGLERS=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
        "pgrep -f $REMOTE_PROCESS_PATTERN 2>/dev/null || true" 2>/dev/null); then
      echo "  stop: rank 1 post-SIGKILL verification failed."
      exit 1
    fi
    if [[ -n "$LOCAL_STRAGGLERS" ]] || [[ -n "$REMOTE_STRAGGLERS" ]]; then
      echo "  stop: cluster processes still remain after explicit SIGKILL."
      exit 1
    fi
  else
    echo "  Leaving stragglers running to avoid orphaned Metal memory."
    echo "  Rerun with M3_FORCE_KILL=1 only if you accept the orphan/reboot risk."
    exit 1
  fi
fi

exit "$STOP_EXIT_STATUS"
