#!/bin/zsh
#
# M3_Gateway_Start.command - double-click launcher for the model gateway ONLY.
#
# Starts the ThunderMLX/oMLX gateway on :8010 without launching the M3 cluster
# or loading any model. Backends still come up on demand when a client asks for
# them by explicit model id. Use M3_Start.command for a full cluster start.
#
# Safe to copy to Desktop. If it is not inside the project directory, it
# resolves the real cluster folder from M3_CLUSTER_DIR or ~/minimax-m3-cluster.
set -euo pipefail

clear 2>/dev/null || true
echo "=================================================="
echo "  ThunderMLX Model Gateway - START (gateway only)"
echo "=================================================="
echo ""

SCRIPT_DIR="${0:A:h}"
if [[ -f "$SCRIPT_DIR/start_gateway.sh" && -f "$SCRIPT_DIR/model_gateway.py" ]]; then
  CLUSTER_DEFAULT="$SCRIPT_DIR"
else
  CLUSTER_DEFAULT="$HOME/minimax-m3-cluster"
fi
CLUSTER="${M3_CLUSTER_DIR:-$CLUSTER_DEFAULT}"

pause_and_exit() {
  local code="${1:-0}"
  echo ""
  if [[ -t 0 ]]; then
    read -k1 "?Press any key to close..."
  fi
  exit "$code"
}

if [[ ! -d "$CLUSTER" || ! -f "$CLUSTER/start_gateway.sh" ]]; then
  echo "Cluster folder not found: $CLUSTER"
  echo "Set M3_CLUSTER_DIR or place this script inside the minimax-m3-cluster repo."
  pause_and_exit 2
fi

cd "$CLUSTER"

if [[ -f "$CLUSTER/.env.local" ]]; then
  source "$CLUSTER/.env.local"
elif [[ -f "$CLUSTER/m3_cluster.env" ]]; then
  source "$CLUSTER/m3_cluster.env"
elif [[ -f "$CLUSTER/.env" ]]; then
  source "$CLUSTER/.env"
fi

GATEWAY_PORT="${M3_GATEWAY_PORT:-8010}"
GUI_PUBLIC_HOST="${M3_GUI_PUBLIC_HOST:-}"
if [[ -z "$GUI_PUBLIC_HOST" && -n "${M3_PUBLIC_BASE_URL:-}" ]]; then
  GUI_PUBLIC_HOST="$(python3 - <<'PY'
import os
from urllib.parse import urlparse
print(urlparse(os.environ.get("M3_PUBLIC_BASE_URL", "")).hostname or "")
PY
)"
fi
DISPLAY_GATEWAY="${M3_GATEWAY_PUBLIC_BASE_URL:-}"
if [[ -z "$DISPLAY_GATEWAY" && -n "$GUI_PUBLIC_HOST" ]]; then
  DISPLAY_GATEWAY="http://${GUI_PUBLIC_HOST}:${GATEWAY_PORT}/v1"
fi
DISPLAY_GATEWAY="${DISPLAY_GATEWAY:-http://127.0.0.1:${GATEWAY_PORT}/v1}"

# Same Tailscale bring-up as M3_Start.command: if the gateway is advertised on
# a tailnet IP but Tailscale is down, remote devices can't reach it.
ensure_tailscale_access() {
  local TS_CLI="${TAILSCALE_CLI:-}"
  if [[ -z "$GUI_PUBLIC_HOST" || "$GUI_PUBLIC_HOST" != 100.* ]]; then
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
    echo "  Tailscale CLI not found; gateway remains local-only until Tailscale starts."
    return 0
  fi
  if "$TS_CLI" status 2>/dev/null | grep -q "^${GUI_PUBLIC_HOST}[[:space:]]"; then
    return 0
  fi
  echo "  Tailscale is not running; restoring gateway access on ${GUI_PUBLIC_HOST}..."
  if ! "$TS_CLI" up --timeout=20s --accept-dns=false --accept-routes >/dev/null 2>&1; then
    echo "  Tailscale auto-start failed; open Tailscale manually if ${DISPLAY_GATEWAY} is unreachable."
    return 0
  fi
}

echo "Project: $CLUSTER"
echo "This starts ONLY the gateway. No model is loaded and the M3 cluster"
echo "stays down until a client requests an M3 model id explicitly."
echo ""

ensure_tailscale_access

if /bin/zsh "$CLUSTER/start_gateway.sh"; then
  echo ""
  echo "Gateway base URL: $DISPLAY_GATEWAY"
  echo "oMLX models route on demand; M3 model ids start the cluster on demand."
  echo "Full cluster start: M3_Start.command | Stop everything: M3_Stop.command"
  pause_and_exit 0
else
  EXIT_CODE=$?
  echo ""
  echo "Gateway start failed (exit $EXIT_CODE). Check $CLUSTER/model_gateway.log"
  pause_and_exit "$EXIT_CODE"
fi
