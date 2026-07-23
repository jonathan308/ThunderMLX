#!/bin/zsh
#
# M3_Start.command - double-click launcher for the MiniMax-M3 cluster.
#
# This script is intentionally safe to copy to Desktop. If it is not inside the
# project directory, it resolves the real cluster folder from M3_CLUSTER_DIR or
# ~/minimax-m3-cluster, then runs all start steps from there.
set -euo pipefail

clear 2>/dev/null || true
echo "=================================================="
echo "  MiniMax-M3 Cluster - START"
echo "=================================================="
echo ""

SCRIPT_DIR="${0:A:h}"
if [[ -f "$SCRIPT_DIR/launch_cluster.sh" && -f "$SCRIPT_DIR/cluster_gui.py" ]]; then
  CLUSTER_DEFAULT="$SCRIPT_DIR"
else
  CLUSTER_DEFAULT="$HOME/minimax-m3-cluster"
fi
CLUSTER="${M3_CLUSTER_DIR:-$CLUSTER_DEFAULT}"

if [[ ! -d "$CLUSTER" || ! -f "$CLUSTER/launch_cluster.sh" ]]; then
  echo "Cluster folder not found: $CLUSTER"
  echo "Set M3_CLUSTER_DIR or place this script inside the minimax-m3-cluster repo."
  echo ""
  read -k1 "?Press any key to close..."
  exit 2
fi

cd "$CLUSTER"

if [[ -f "$CLUSTER/.env.local" ]]; then
  source "$CLUSTER/.env.local"
elif [[ -f "$CLUSTER/m3_cluster.env" ]]; then
  source "$CLUSTER/m3_cluster.env"
elif [[ -f "$CLUSTER/.env" ]]; then
  source "$CLUSTER/.env"
else
  echo "No local config found."
  echo "Copy .env.example to .env.local and fill in your cluster values first."
  echo ""
  read -k1 "?Press any key to close..."
  exit 2
fi

# Reuse the project's unattended privilege helper when available. This keeps
# double-click and remote starts from blocking on an interactive sudo prompt;
# the helper prefers NOPASSWD and otherwise reads the private 0600 password
# file through stdin without printing it.
if [[ -f "$CLUSTER/ops/priv.sh" ]]; then
  source "$CLUSTER/ops/priv.sh"
fi

API_HOST="${MLX_M3_HOST:-0.0.0.0}"
API_PORT="${MLX_M3_PORT:-8080}"
GUI_HOST="${M3_GUI_HOST:-0.0.0.0}"
GUI_PORT="${M3_GUI_PORT:-8090}"
GATEWAY_HOST="${M3_GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${M3_GATEWAY_PORT:-8010}"
OMLX_PORT="${M3_OMLX_PORT:-8000}"
GUI_PYTHON="${M3_GUI_PYTHON:-$(command -v python3)}"
LOG_DIR="${M3_LOG_DIR:-/private/tmp/minimax-m3-cluster-logs}"
HOSTFILE="${M3_HOSTFILE:-/private/tmp/m3_mlx_hosts.json}"
STOP_FILE="${M3_STOP_FILE:-/private/tmp/minimax_m3_stop_requested}"
LOCK_DIR="${M3_LOCK_DIR:-/private/tmp/minimax_m3_start.lock}"

LOCAL_API_URL="http://127.0.0.1:${API_PORT}"
LOCAL_DASHBOARD_URL="http://127.0.0.1:${GUI_PORT}"
LOCAL_GATEWAY_URL="http://127.0.0.1:${GATEWAY_PORT}"
GUI_PUBLIC_HOST="${M3_GUI_PUBLIC_HOST:-}"
if [[ -z "$GUI_PUBLIC_HOST" && -n "${M3_PUBLIC_BASE_URL:-}" ]]; then
  GUI_PUBLIC_HOST="$(python3 - <<'PY'
import os
from urllib.parse import urlparse
print(urlparse(os.environ.get("M3_PUBLIC_BASE_URL", "")).hostname or "")
PY
)"
fi
GUI_PUBLIC_URL="${M3_GUI_PUBLIC_URL:-}"
if [[ -z "$GUI_PUBLIC_URL" && -n "$GUI_PUBLIC_HOST" ]]; then
  GUI_PUBLIC_URL="http://${GUI_PUBLIC_HOST}:${GUI_PORT}"
fi
DISPLAY_ENDPOINT="${M3_PUBLIC_BASE_URL:-${LOCAL_API_URL}/v1}"
DISPLAY_DASHBOARD="${GUI_PUBLIC_URL:-$LOCAL_DASHBOARD_URL}"
DISPLAY_GATEWAY="${M3_GATEWAY_PUBLIC_BASE_URL:-}"
if [[ -z "$DISPLAY_GATEWAY" && -n "$GUI_PUBLIC_HOST" ]]; then
  DISPLAY_GATEWAY="http://${GUI_PUBLIC_HOST}:${GATEWAY_PORT}/v1"
fi
DISPLAY_GATEWAY="${DISPLAY_GATEWAY:-${LOCAL_GATEWAY_URL}/v1}"

DIRECT_PEER="${M3_DIRECT_PEER:-${M3_RANK1_DIRECT_SSH:-}}"
FALLBACK_PEER="${M3_RANK1_FALLBACK_SSH:-${M3_TAILSCALE_PEER:-}}"
PEER="${M3_PEER:-}"
if [[ -z "$PEER" ]]; then
  if [[ -n "$DIRECT_PEER" ]]; then
    PEER="$DIRECT_PEER"
  elif [[ -n "$FALLBACK_PEER" ]]; then
    PEER="$FALLBACK_PEER"
  fi
fi

pause_and_exit() {
  local code="${1:-0}"
  echo ""
  if [[ -t 0 ]]; then
    read -k1 "?Press any key to close..."
  fi
  exit "$code"
}

ensure_tailscale_access() {
  local TS_CLI="${TAILSCALE_CLI:-}"
  if [[ -z "$GUI_PUBLIC_HOST" ]]; then
    return 0
  fi
  if [[ "$GUI_PUBLIC_HOST" != 100.* ]]; then
    return 0
  fi
  if [[ -z "$TS_CLI" ]]; then
    if command -v tailscale >/dev/null 2>&1; then
      TS_CLI="$(command -v tailscale)"
    elif [[ -x "/Applications/Tailscale.app/Contents/MacOS/Tailscale" ]]; then
      TS_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    fi
  fi
  if [[ -z "$TS_CLI" || ! -x "$TS_CLI" ]]; then
    echo "  Tailscale CLI not found; dashboard remains local-only until Tailscale starts."
    return 0
  fi
  if "$TS_CLI" status 2>/dev/null | grep -q "^${GUI_PUBLIC_HOST}[[:space:]]"; then
    return 0
  fi
  echo "  Tailscale is not running; restoring dashboard access on ${GUI_PUBLIC_HOST}..."
  if ! "$TS_CLI" up --timeout=20s --accept-dns=false --accept-routes >/dev/null 2>&1; then
    echo "  Tailscale auto-start failed; open Tailscale manually if ${DISPLAY_DASHBOARD} is unreachable."
    return 0
  fi
}

start_detached_python() {
  local PID_FILE="$1"
  local LOG_FILE="$2"
  shift 2
  M3_DETACH_DIR="$CLUSTER" \
  M3_DETACH_PYTHON_BIN="$GUI_PYTHON" \
  M3_DETACH_PID_FILE="$PID_FILE" \
  M3_DETACH_LOG_FILE="$LOG_FILE" \
  M3_DETACH_ARGS="$*" \
  "$GUI_PYTHON" - <<'PY'
import os
import pathlib
import shlex
import subprocess

cluster = pathlib.Path(os.environ["M3_DETACH_DIR"])
pid_file = cluster / os.environ["M3_DETACH_PID_FILE"]
log_file = cluster / os.environ["M3_DETACH_LOG_FILE"]
args = shlex.split(os.environ["M3_DETACH_ARGS"])
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
log = open(log_file, "ab", buffering=0)
proc = subprocess.Popen(
    args,
    cwd=str(cluster),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True,
)
pid_file.write_text(str(proc.pid) + "\n")
PY
}

start_dashboard_ui() {
  /usr/bin/screen -S m3_gui -X quit >/dev/null 2>&1 || true
  if [[ -f "$CLUSTER/cluster_gui.pid" ]]; then
    OLD_PID=$(cat "$CLUSTER/cluster_gui.pid" 2>/dev/null)
    if [[ -n "$OLD_PID" ]]; then
      kill "$OLD_PID" >/dev/null 2>&1 || true
    fi
  fi
  pkill -TERM -f "$CLUSTER/cluster_gui.py" 2>/dev/null || true
  # Also match a GUI started with a relative path (a stale one survived every
  # restart this way, serving old routes until 2026-07-23).
  pkill -TERM -f "cluster_gui.py" 2>/dev/null || true
  sleep 1
  M3_GUI_DIR="$CLUSTER" \
  M3_GUI_PYTHON_BIN="$GUI_PYTHON" \
  M3_ENDPOINT="${M3_ENDPOINT:-$LOCAL_API_URL}" \
  M3_GUI_HOST="$GUI_HOST" \
  M3_GUI_PORT="$GUI_PORT" \
  M3_GUI_PUBLIC_HOST="$GUI_PUBLIC_HOST" \
  "$GUI_PYTHON" - <<'PY'
import os
import pathlib
import subprocess

cluster = pathlib.Path(os.environ["M3_GUI_DIR"])
python_bin = os.environ["M3_GUI_PYTHON_BIN"]
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
log = open(cluster / "cluster_gui.log", "ab", buffering=0)
proc = subprocess.Popen(
    [python_bin, "cluster_gui.py"],
    cwd=str(cluster),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True,
)
(cluster / "cluster_gui.pid").write_text(str(proc.pid) + "\n")
PY
}

start_gateway_ui() {
  if [[ "${M3_START_GATEWAY:-1}" != "1" || "${M3_GATEWAY_SKIP_START:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$CLUSTER/start_gateway.sh" ]]; then
    return 0
  fi
  /bin/zsh "$CLUSTER/start_gateway.sh" >/dev/null 2>&1 || {
    echo "  gateway start failed; check $CLUSTER/model_gateway.log"
    return 0
  }
}

wired_gb_local() {
  local pages
  pages=$(memory_pressure 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc '0-9')
  if [[ -z "$pages" ]]; then
    pages=$(vm_stat 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc '0-9')
  fi
  echo $(( ${pages:-0} * 16384 / 1024 / 1024 / 1024 ))
}

wired_gb_remote() {
  if [[ -z "${PEER:-}" ]]; then
    echo 0
    return
  fi
  local pages
  pages=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    'memory_pressure 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc "0-9"' 2>/dev/null)
  if [[ -z "$pages" ]]; then
    pages=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
      'vm_stat 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc "0-9"' 2>/dev/null)
  fi
  echo $(( ${pages:-0} * 16384 / 1024 / 1024 / 1024 ))
}

echo "Project: $CLUSTER"
echo "Endpoint: $DISPLAY_ENDPOINT"
echo "Dashboard: $DISPLAY_DASHBOARD"
echo ""

if curl -s --max-time 3 "$LOCAL_API_URL/health" >/dev/null 2>&1; then
  echo "Cluster is already running."
  if ! curl -s --max-time 1 "$LOCAL_DASHBOARD_URL/api/status" >/dev/null 2>&1; then
    echo "Dashboard is not listening; starting dashboard on ${GUI_HOST}:${GUI_PORT}..."
    start_dashboard_ui
  fi
  if [[ "${M3_START_GATEWAY:-1}" == "1" && "${M3_GATEWAY_SKIP_START:-0}" != "1" ]]; then
    echo "Ensuring model gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}..."
    start_gateway_ui
  fi
  open "$LOCAL_DASHBOARD_URL" >/dev/null 2>&1 || true
  echo "OpenWebUI base URL: $DISPLAY_ENDPOINT"
  echo "Gateway base URL: $DISPLAY_GATEWAY"
  echo "Model IDs: Minimax-M3 | Minimax-M3-No-Think | M3-Web"
  pause_and_exit 0
fi

if [[ -d "$LOCK_DIR" ]]; then
  echo "M3 start is already in progress or the guarded launcher is active."
  echo "Lock: $LOCK_DIR"
  echo "Use M3_Stop.command before starting a clean new launch."
  pause_and_exit 1
fi

echo "Applying optional host performance tune..."
if [[ "${M3_ENABLE_PERFORMANCE_TUNE:-0}" == "1" ]]; then
  if [[ "${M3_RANK0_IOGPU_WIRED_LIMIT_MB:-0}" != "0" ]] && whence priv_run >/dev/null 2>&1; then
    priv_run sysctl "iogpu.wired_limit_mb=${M3_RANK0_IOGPU_WIRED_LIMIT_MB}" >/dev/null 2>&1 || true
  elif [[ -t 0 && "${M3_RANK0_IOGPU_WIRED_LIMIT_MB:-0}" != "0" ]]; then
    sudo sysctl "iogpu.wired_limit_mb=${M3_RANK0_IOGPU_WIRED_LIMIT_MB}" >/dev/null 2>&1 || true
  elif [[ ! -t 0 ]]; then
    echo "  noninteractive start; skipping privileged local wired-limit tune"
  fi
  if [[ -n "${PEER:-}" && "${M3_RANK1_IOGPU_WIRED_LIMIT_MB:-0}" != "0" ]]; then
    if whence priv_run_rank1 >/dev/null 2>&1; then
      priv_run_rank1 sysctl "iogpu.wired_limit_mb=${M3_RANK1_IOGPU_WIRED_LIMIT_MB}" >/dev/null 2>&1 || true
    else
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
        "sudo -n sysctl iogpu.wired_limit_mb=${M3_RANK1_IOGPU_WIRED_LIMIT_MB} >/dev/null 2>&1 || true" \
        >/dev/null 2>&1 || true
    fi
  fi
  echo "  performance tune attempted"
else
  echo "  disabled (set M3_ENABLE_PERFORMANCE_TUNE=1 to opt in)"
fi
if [[ "${M3_ENABLE_MACOS_NOISE_REDUCTION:-0}" == "1" ]]; then
  if whence priv_run >/dev/null 2>&1; then
    priv_run mdutil -a -i off >/dev/null 2>&1 || true
    priv_run killall mds 2>/dev/null || true
    priv_run killall mds_stores 2>/dev/null || true
    priv_run killall corespotlightd 2>/dev/null || true
  elif [[ -t 0 ]]; then
    sudo mdutil -a -i off >/dev/null 2>&1 || true
    sudo killall mds 2>/dev/null || true
    sudo killall mds_stores 2>/dev/null || true
    sudo killall corespotlightd 2>/dev/null || true
  fi
  killall assistantd 2>/dev/null || true
  killall siriinferenced 2>/dev/null || true
  killall siriknowledged 2>/dev/null || true
  killall intelligenceplatformd 2>/dev/null || true
  killall intelligencecontextd 2>/dev/null || true
  killall knowledge-agent 2>/dev/null || true
  killall photoanalysisd 2>/dev/null || true
  killall mediaanalysisd 2>/dev/null || true
  killall photolibraryd 2>/dev/null || true
  osascript -e 'quit app "Safari"' 2>/dev/null || true
  echo "  macOS background-noise reduction attempted"
else
  echo "  macOS background-noise reduction disabled"
fi
echo ""

echo "Checking wired-memory orphan guard..."
RANK0_WIRED_GB=$(wired_gb_local)
RANK1_WIRED_GB=$(wired_gb_remote)
echo "  Rank 0 wired: ${RANK0_WIRED_GB}GB"
echo "  Rank 1 wired: ${RANK1_WIRED_GB}GB"
RANK0_LIMIT="${M3_ORPHAN_RANK0_WIRED_GB:-${M3_ORPHAN_STUDIO_WIRED_GB:-0}}"
RANK1_LIMIT="${M3_ORPHAN_RANK1_WIRED_GB:-${M3_ORPHAN_MACBOOK_WIRED_GB:-0}}"
if [[ "$RANK0_LIMIT" != "0" && "${RANK0_WIRED_GB:-0}" -gt "$RANK0_LIMIT" ]] || \
   [[ "$RANK1_LIMIT" != "0" && "${RANK1_WIRED_GB:-0}" -gt "$RANK1_LIMIT" ]]; then
  echo ""
  echo "ORPHANED METAL MEMORY DETECTED."
  echo "Reboot affected machines before starting M3. Loading now is likely to crash."
  pause_and_exit 1
fi
echo ""

echo "Unloading duplicate MiniMax-M3 from separate oMLX port ${OMLX_PORT}, if present..."
curl -s --max-time 8 -X POST \
  "http://127.0.0.1:${OMLX_PORT}/admin/api/models/MiniMax-M3-4bit/unload" \
  >/dev/null 2>&1 || true
echo ""

echo "Syncing cluster code to rank 1..."
if /bin/zsh "$CLUSTER/sync_rank1.sh"; then
  echo "  rank 1 synced"
else
  echo "  sync failed; not starting because rank 1 may run stale code."
  pause_and_exit 1
fi
echo ""

echo "Starting dashboard UI on ${GUI_HOST}:${GUI_PORT}..."
start_dashboard_ui
ensure_tailscale_access
for _ in {1..15}; do
  if curl -s --max-time 1 "$LOCAL_DASHBOARD_URL/api/status" >/dev/null 2>&1; then
    echo "  dashboard ready: $DISPLAY_DASHBOARD"
    open "$LOCAL_DASHBOARD_URL" >/dev/null 2>&1 || true
    break
  fi
  sleep 1
done
echo ""

echo "Starting guarded auto-restart launcher..."
chmod +x "$CLUSTER/auto_restart.sh" "$CLUSTER/launch_cluster.sh" "$CLUSTER/stop_cluster.sh" >/dev/null 2>&1 || true
/usr/bin/screen -S minimax_m3 -X quit >/dev/null 2>&1 || true
rm -f "$STOP_FILE"
mkdir -p "$LOG_DIR"
if [[ -f "$CLUSTER/auto_restart.pid" ]]; then
  OLD_PID=$(cat "$CLUSTER/auto_restart.pid" 2>/dev/null)
  if [[ -n "$OLD_PID" ]]; then
    kill "$OLD_PID" >/dev/null 2>&1 || true
  fi
fi
pkill -TERM -f "$CLUSTER/auto_restart.sh" 2>/dev/null || true
M3_LOG_DIR="$LOG_DIR" M3_HOSTFILE="$HOSTFILE" M3_STOP_FILE="$STOP_FILE" M3_LOCK_DIR="$LOCK_DIR" \
  start_detached_python "auto_restart.pid" "auto_restart.nohup.log" /bin/zsh "$CLUSTER/auto_restart.sh"

echo "Waiting for API health..."
for i in {1..80}; do
  if curl -s --max-time 3 "$LOCAL_API_URL/health" >/dev/null 2>&1; then
    echo ""
    echo "Cluster is UP."
    if [[ "${M3_WARMUP_ON_START:-1}" == "1" ]]; then
      echo "Starting warmup pass..."
      /usr/bin/screen -S m3_warmup -X quit >/dev/null 2>&1 || true
      if [[ -f "$CLUSTER/m3_warmup.pid" ]]; then
        OLD_PID=$(cat "$CLUSTER/m3_warmup.pid" 2>/dev/null)
        if [[ -n "$OLD_PID" ]]; then
          kill "$OLD_PID" >/dev/null 2>&1 || true
        fi
      fi
      pkill -TERM -f "$CLUSTER/m3_warmup.py" 2>/dev/null || true
      M3_WARMUP_BASE="$LOCAL_API_URL" \
        start_detached_python "m3_warmup.pid" "m3_warmup.log" "$GUI_PYTHON" "$CLUSTER/m3_warmup.py"
    fi
    echo ""
    echo "OpenWebUI base URL: $DISPLAY_ENDPOINT"
    if [[ "${M3_START_GATEWAY:-1}" == "1" && "${M3_GATEWAY_SKIP_START:-0}" != "1" ]]; then
      echo "Starting model gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}..."
      start_gateway_ui
      echo "Gateway base URL: $DISPLAY_GATEWAY"
    fi
    echo "Dashboard: $DISPLAY_DASHBOARD"
    echo "Model path: ${MLX_M3_MODEL:-mlx-community/MiniMax-M3-4bit}"
    echo "Model IDs: Minimax-M3 | Minimax-M3-No-Think | M3-Web"
    echo "Stop: double-click M3_Stop.command"
    pause_and_exit 0
  fi
  sleep 2
  printf "\r  waiting... (%ds)" $((i*2))
done

echo ""
echo "Cluster did not become healthy in 160s."
echo "Check: $LOG_DIR/startup.log and $CLUSTER/auto_restart.nohup.log"
pause_and_exit 1
