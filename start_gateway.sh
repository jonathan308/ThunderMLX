#!/bin/zsh
#
# Start the OpenAI-compatible ThunderMLX/oMLX model gateway.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/.env.local" ]]; then
  source "$SCRIPT_DIR/.env.local"
elif [[ -f "$SCRIPT_DIR/m3_cluster.env" ]]; then
  source "$SCRIPT_DIR/m3_cluster.env"
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
  source "$SCRIPT_DIR/.env"
fi

HOST="${M3_GATEWAY_HOST:-0.0.0.0}"
PORT="${M3_GATEWAY_PORT:-8010}"
PYTHON="${M3_GATEWAY_PYTHON:-${M3_GUI_PYTHON:-$(command -v python3)}}"
# Shield by default: the gateway may not auto-stop the M3 cluster (a remote
# shim probing oMLX once stopped it mid-session). Set 1 in .env.local to
# re-arm oMLX auto-switching.
export M3_GATEWAY_ALLOW_STOP_M3="${M3_GATEWAY_ALLOW_STOP_M3:-0}"
PID_FILE="$SCRIPT_DIR/model_gateway.pid"
LOG_FILE="$SCRIPT_DIR/model_gateway.log"

if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
      echo "Gateway already listening on ${HOST}:${PORT}"
      exit 0
    fi
  fi
  echo "Gateway port responds but pid file is stale; restarting gateway..."
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]]; then
    kill "$OLD_PID" >/dev/null 2>&1 || true
  fi
fi
pkill -TERM -f "$SCRIPT_DIR/model_gateway.py" 2>/dev/null || true
for _ in {1..20}; do
  if ! pgrep -f "$SCRIPT_DIR/model_gateway.py" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
if pgrep -f "$SCRIPT_DIR/model_gateway.py" >/dev/null 2>&1; then
  pkill -KILL -f "$SCRIPT_DIR/model_gateway.py" 2>/dev/null || true
fi

echo "Starting ThunderMLX model gateway on ${HOST}:${PORT}..."
M3_GATEWAY_DIR="$SCRIPT_DIR" \
M3_GATEWAY_PYTHON_BIN="$PYTHON" \
M3_GATEWAY_PID_FILE="$PID_FILE" \
M3_GATEWAY_LOG_FILE="$LOG_FILE" \
"$PYTHON" - <<'PY'
import os
import subprocess

cluster = os.environ["M3_GATEWAY_DIR"]
python_bin = os.environ["M3_GATEWAY_PYTHON_BIN"]
pid_file = os.environ["M3_GATEWAY_PID_FILE"]
log_file = os.environ["M3_GATEWAY_LOG_FILE"]
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
log = open(log_file, "ab", buffering=0)
proc = subprocess.Popen(
    [python_bin, os.path.join(cluster, "model_gateway.py")],
    cwd=cluster,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True,
)
with open(pid_file, "w") as f:
    f.write(str(proc.pid) + "\n")
PY

for _ in {1..30}; do
  if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Gateway ready: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  sleep 1
done

echo "Gateway did not become ready. Check $LOG_FILE"
exit 1
