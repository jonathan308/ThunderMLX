#!/bin/zsh
#
# Stop the OpenAI-compatible ThunderMLX/oMLX model gateway.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PID_FILE="$SCRIPT_DIR/model_gateway.pid"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$PID" ]]; then
    kill "$PID" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 "$PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
  fi
  rm -f "$PID_FILE"
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
echo "Gateway stopped."
