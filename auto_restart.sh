#!/bin/zsh
SCRIPT_DIR="${0:A:h}"
if [[ -f "$SCRIPT_DIR/.env.local" ]]; then
  source "$SCRIPT_DIR/.env.local"
elif [[ -f "$SCRIPT_DIR/m3_cluster.env" ]]; then
  source "$SCRIPT_DIR/m3_cluster.env"
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
  source "$SCRIPT_DIR/.env"
fi

CLUSTER="${M3_CLUSTER_DIR:-$SCRIPT_DIR}"
DIRECT_PEER="${M3_DIRECT_PEER:-${M3_RANK1_DIRECT_SSH:-}}"
FALLBACK_PEER="${M3_RANK1_FALLBACK_SSH:-${M3_TAILSCALE_PEER:-}}"
PEER="${M3_PEER:-}"
LOG_DIR="${M3_LOG_DIR:-/private/tmp/minimax-m3-cluster-logs}"
RESTART_LOG="$LOG_DIR/restart.log"
STARTUP_LOG="$LOG_DIR/startup.log"
STOP_FILE="${M3_STOP_FILE:-/private/tmp/minimax_m3_stop_requested}"
LOCK_DIR="${M3_LOCK_DIR:-/private/tmp/minimax_m3_start.lock}"
MAX_QUICK_FAILURES=${M3_MAX_QUICK_FAILURES:-3}
QUICK_FAILURE_WINDOW=${M3_QUICK_FAILURE_WINDOW:-120}
BACKOFF=${M3_RESTART_BACKOFF_INITIAL:-15}
MAX_BACKOFF=${M3_RESTART_BACKOFF_MAX:-300}
QUICK_FAILURES=0
GUARD_TEARDOWNS=0
LAUNCHER_RSS_GUARD_GB=${M3_LAUNCHER_RSS_GUARD_GB:-0}
LAUNCHER_RSS_GUARD_INTERVAL=${M3_LAUNCHER_RSS_GUARD_INTERVAL_SECONDS:-300}
LAUNCHER_RSS_GUARD_IDLE_GRACE=${M3_LAUNCHER_RSS_GUARD_IDLE_GRACE_SECONDS:-600}
API_GUARD_INTERVAL=${M3_API_GUARD_INTERVAL_SECONDS:-15}
API_GUARD_START_GRACE=${M3_API_GUARD_START_GRACE_SECONDS:-180}
API_DOWN_GUARD_SECONDS=${M3_API_DOWN_GUARD_SECONDS:-75}
ACTIVE_NO_START_GUARD_SECONDS=${M3_ACTIVE_NO_START_GUARD_SECONDS:-120}
ACTIVE_NO_START_CONTEXT_TPS=${M3_ACTIVE_NO_START_CONTEXT_TPS:-1000}
ACTIVE_NO_START_MARGIN_SECONDS=${M3_ACTIVE_NO_START_MARGIN_SECONDS:-120}
ACTIVE_NO_START_MAX_SECONDS=${M3_ACTIVE_NO_START_MAX_SECONDS:-900}
ACTIVE_NO_START_DRAIN_SECONDS=${M3_ACTIVE_NO_START_DRAIN_SECONDS:-600}
GUARD_TERM_GRACE_SECONDS=${M3_GUARD_TERM_GRACE_SECONDS:-120}
API_HEALTH_URL="${M3_API_HEALTH_URL:-http://127.0.0.1:${MLX_M3_PORT:-8080}/health}"
API_SHUTDOWN_URL="${M3_API_SHUTDOWN_URL:-http://127.0.0.1:${MLX_M3_PORT:-8080}/admin/shutdown}"
API_STOP_URL="${M3_API_STOP_URL:-http://127.0.0.1:${MLX_M3_PORT:-8080}/v1/stop}"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  # Self-heal a stale lock: a killed screen session can strand the dir
  # before the trap runs (blocked every start for 6 minutes on 2026-07-06).
  # Only refuse when another auto_restart is actually alive.
  others=$(pgrep -f "[a]uto_restart.sh" 2>/dev/null | grep -v "^$$\$" | grep -v "^$PPID\$" | wc -l | tr -d ' ')
  if [[ "${others:-0}" -gt 0 ]]; then
    echo "[$(date)] Start refused: another M3 auto_restart/launch is already active ($LOCK_DIR)." >> "$RESTART_LOG"
    exit 0
  fi
  echo "[$(date)] Stale start lock with no live auto_restart — taking over." >> "$RESTART_LOG"
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null || exit 0
fi
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM HUP

if [[ -z "$PEER" ]]; then
  if [[ -n "$DIRECT_PEER" ]] && ssh -o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1 \
      "$DIRECT_PEER" 'true' >/dev/null 2>&1; then
    PEER="$DIRECT_PEER"
  elif [[ -n "$FALLBACK_PEER" ]]; then
    PEER="$FALLBACK_PEER"
    echo "[$(date)] Direct SSH unavailable; using fallback SSH for control/memory checks." >> "$RESTART_LOG"
  else
    echo "[$(date)] Start refused: set M3_PEER or M3_RANK1_DIRECT_SSH in .env.local." >> "$RESTART_LOG"
    exit 2
  fi
fi

wired_gb_local() {
  local pages
  pages=$(memory_pressure 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc '0-9')
  if [[ -z "$pages" ]]; then
    pages=$(vm_stat 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc '0-9')
  fi
  echo $(( ${pages:-0} * 16384 / 1024 / 1024 / 1024 ))
}

wired_gb_remote() {
  local pages
  pages=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    'memory_pressure 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc "0-9"' 2>/dev/null)
  if [[ -z "$pages" ]]; then
    pages=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
      'vm_stat 2>/dev/null | grep "Pages wired down" | head -1 | tr -dc "0-9"' 2>/dev/null)
  fi
  echo $(( ${pages:-0} * 16384 / 1024 / 1024 / 1024 ))
}

cleanup_leftover_ranks() {
  local MODE="${1:-term}"
  local PORT="${MLX_M3_PORT:-8080}"
  echo "[$(date)] Sweeping leftover M3 rank processes after guarded exit (mode=${MODE})." >> "$RESTART_LOG"
  pkill -TERM -f "$CLUSTER/run_with_watchdog.py" 2>/dev/null
  pkill -TERM -f "$CLUSTER/sharded_server.py" 2>/dev/null
  pkill -TERM -f "mlx.launch.*$CLUSTER" 2>/dev/null
  pkill -TERM -f "$CLUSTER/launch_cluster.sh" 2>/dev/null
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "pkill -TERM -f '$CLUSTER/run_with_watchdog.py' 2>/dev/null; pkill -TERM -f '$CLUSTER/sharded_server.py' 2>/dev/null; pkill -TERM -f 'mlx.launch.*$CLUSTER' 2>/dev/null; pkill -TERM -f '$CLUSTER/launch_cluster.sh' 2>/dev/null" \
    >/dev/null 2>&1 || true

  if [[ "$MODE" == "hard" ]]; then
    local i alive_local alive_remote api_owner
    for i in $(seq 1 10); do
      sleep 1
      alive_local=$((pgrep -fl "$CLUSTER/run_with_watchdog.py|$CLUSTER/sharded_server.py|mlx.launch.*$CLUSTER|$CLUSTER/launch_cluster.sh" 2>/dev/null || true) | grep -vc "[p]grep" | tr -d ' ')
      alive_remote=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
        "(pgrep -fl '$CLUSTER/run_with_watchdog.py|$CLUSTER/sharded_server.py|mlx.launch.*$CLUSTER|$CLUSTER/launch_cluster.sh' 2>/dev/null || true) | grep -vc '[p]grep' | tr -d ' '" 2>/dev/null)
      api_owner=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ')
      if [[ "${alive_local:-0}" == "0" && "${alive_remote:-0}" == "0" && -z "$api_owner" ]]; then
        echo "[$(date)] Hard sweep TERM cleared all M3 ranks and API port ${PORT}." >> "$RESTART_LOG"
        return 0
      fi
    done
    echo "[$(date)] Hard sweep enabled for no-progress/API-down stall; escalating any lingering M3 ranks to SIGKILL." >> "$RESTART_LOG"
    pkill -KILL -f "$CLUSTER/run_with_watchdog.py" 2>/dev/null
    pkill -KILL -f "$CLUSTER/sharded_server.py" 2>/dev/null
    pkill -KILL -f "mlx.launch.*$CLUSTER" 2>/dev/null
    pkill -KILL -f "$CLUSTER/launch_cluster.sh" 2>/dev/null
    for pid in $(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null); do
      if ps -p "$pid" -o command= 2>/dev/null | grep -q "$CLUSTER"; then
        echo "[$(date)] Hard sweep killing stale API port owner pid=${pid} on ${PORT}." >> "$RESTART_LOG"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
      "pkill -KILL -f '$CLUSTER/run_with_watchdog.py' 2>/dev/null; pkill -KILL -f '$CLUSTER/sharded_server.py' 2>/dev/null; pkill -KILL -f 'mlx.launch.*$CLUSTER' 2>/dev/null; pkill -KILL -f '$CLUSTER/launch_cluster.sh' 2>/dev/null" \
      >/dev/null 2>&1 || true
    for i in $(seq 1 20); do
      sleep 1
      alive_local=$((pgrep -fl "$CLUSTER/run_with_watchdog.py|$CLUSTER/sharded_server.py|mlx.launch.*$CLUSTER|$CLUSTER/launch_cluster.sh" 2>/dev/null || true) | grep -vc "[p]grep" | tr -d ' ')
      alive_remote=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
        "(pgrep -fl '$CLUSTER/run_with_watchdog.py|$CLUSTER/sharded_server.py|mlx.launch.*$CLUSTER|$CLUSTER/launch_cluster.sh' 2>/dev/null || true) | grep -vc '[p]grep' | tr -d ' '" 2>/dev/null)
      api_owner=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ')
      if [[ "${alive_local:-0}" == "0" && "${alive_remote:-0}" == "0" && -z "$api_owner" ]]; then
        echo "[$(date)] Hard sweep KILL cleared all M3 ranks and API port ${PORT}." >> "$RESTART_LOG"
        return 0
      fi
    done
    echo "[$(date)] WARNING: hard sweep still sees local=${alive_local:-?} remote=${alive_remote:-?} api_owner='${api_owner:-}' after SIGKILL; restart may fail until stale processes exit." >> "$RESTART_LOG"
  fi
}

orphan_memory_present() {
  local studio_wired macbook_wired
  studio_wired=$(wired_gb_local)
  macbook_wired=$(wired_gb_remote)
  local rank0_limit="${M3_ORPHAN_RANK0_WIRED_GB:-${M3_ORPHAN_STUDIO_WIRED_GB:-0}}"
  local rank1_limit="${M3_ORPHAN_RANK1_WIRED_GB:-${M3_ORPHAN_MACBOOK_WIRED_GB:-0}}"
  if [[ "$rank0_limit" != "0" && "${studio_wired:-0}" -gt "$rank0_limit" ]] || \
     [[ "$rank1_limit" != "0" && "${macbook_wired:-0}" -gt "$rank1_limit" ]]; then
    local wait_tries="${M3_ORPHAN_WAIT_TRIES:-15}"
    local wait_seconds="${M3_ORPHAN_WAIT_SECONDS:-120}"
    echo "[$(date)] Orphaned Metal memory guard tripped after cluster exit: rank0 wired=${studio_wired}GB rank1 wired=${macbook_wired}GB. Sweeping leftovers, then waiting up to ${wait_tries}x${wait_seconds}s before giving up." >> "$RESTART_LOG"
    # hard: wired memory is held by LIVE spinning ranks (both machines) —
    # a TERM-only sweep left two spinners on rank1 on 2026-07-06; the hard
    # path escalates to SIGKILL and verifies pgrep+port on both ranks.
    cleanup_leftover_ranks hard
    local i
    for i in $(seq 1 "$wait_tries"); do
      sleep "$wait_seconds"
      # Honor an operator/gateway stop DURING the wait: without this check a
      # stop issued mid-orphan-recovery looked like a 30-minute hang (the
      # 2026-07-06 20:00 incident ended in an avoidable reboot).
      if [ -f "$STOP_FILE" ]; then
        echo "[$(date)] Stop requested during orphan wait; exiting restart loop (orphan sweep continues out-of-band)." >> "$RESTART_LOG"
        rm -f "$STOP_FILE"
        exit 0
      fi
      studio_wired=$(wired_gb_local)
      macbook_wired=$(wired_gb_remote)
      echo "[$(date)] Orphan wait ${i}/${wait_tries}: rank0 wired=${studio_wired}GB rank1 wired=${macbook_wired}GB." >> "$RESTART_LOG"
      if [[ ! ( "$rank0_limit" != "0" && "${studio_wired:-0}" -gt "$rank0_limit" ) ]] && \
         [[ ! ( "$rank1_limit" != "0" && "${macbook_wired:-0}" -gt "$rank1_limit" ) ]]; then
        echo "[$(date)] Orphaned Metal memory cleared during wait; restart loop may continue." >> "$RESTART_LOG"
        return 1
      fi
    done
    echo "[$(date)] Orphaned Metal memory remained after wait: rank0 wired=${studio_wired}GB rank1 wired=${macbook_wired}GB. Not restarting; reboot affected machines." >> "$RESTART_LOG"
    return 0
  fi
  return 1
}

api_idle() {
  curl -fsS --max-time 5 "$API_HEALTH_URL" 2>/dev/null | python3 -c '
import json, sys
try:
    h = json.load(sys.stdin)
except Exception:
    sys.exit(1)
active = h.get("active_request")
queue = h.get("request_queue_depth", h.get("queue_depth", 0)) or 0
ok = h.get("status") == "healthy" and not active and int(queue) == 0
sys.exit(0 if ok else 1)
' >/dev/null 2>&1
}

guarded_cluster_shutdown() {
  local launch_pid="$1"
  local reason="$2"
  local drain_limit="${3:-$ACTIVE_NO_START_DRAIN_SECONDS}"
  local waited=0
  local stop_response shutdown_response

  echo "[$(date)] Guard requesting cooperative distributed stop before shutdown: ${reason}; drain_limit=${drain_limit}s." >> "$RESTART_LOG"
  stop_response=$(curl -fsS --max-time 20 -X POST "$API_STOP_URL" 2>/dev/null || true)
  shutdown_response=$(curl -fsS --max-time 20 -X POST "$API_SHUTDOWN_URL" 2>/dev/null || true)
  echo "[$(date)] Guard stop response: ${stop_response:-unreachable}; shutdown response: ${shutdown_response:-unreachable}." >> "$RESTART_LOG"

  # /admin/shutdown marks an active request for deferred shutdown. Give the
  # distributed stop sentinel enough time to reach a prefill/decode boundary;
  # killing a rank while Metal is evaluating is what strands wired memory.
  while kill -0 "$launch_pid" 2>/dev/null && [[ "$waited" -lt "$drain_limit" ]]; do
    sleep 5
    waited=$(( waited + 5 ))
    if api_idle; then
      curl -fsS --max-time 20 -X POST "$API_SHUTDOWN_URL" >/dev/null 2>&1 || true
    elif [[ $(( waited % 30 )) -eq 0 ]]; then
      # Retry in case the event loop was temporarily occupied by a long Metal
      # evaluation when the first stop/shutdown requests arrived.
      curl -fsS --max-time 20 -X POST "$API_STOP_URL" >/dev/null 2>&1 || true
      curl -fsS --max-time 20 -X POST "$API_SHUTDOWN_URL" >/dev/null 2>&1 || true
    fi
    if [[ $(( waited % 60 )) -eq 0 ]]; then
      echo "[$(date)] Guard cooperative drain still waiting (${waited}/${drain_limit}s): ${reason}." >> "$RESTART_LOG"
    fi
  done

  if ! kill -0 "$launch_pid" 2>/dev/null; then
    echo "[$(date)] Guard cooperative drain exited launcher cleanly after ${waited}s: ${reason}." >> "$RESTART_LOG"
    return 0
  fi

  echo "[$(date)] Guard cooperative drain timed out after ${waited}s; sending SIGTERM and allowing ${GUARD_TERM_GRACE_SECONDS}s for rank-local Metal release: ${reason}." >> "$RESTART_LOG"
  kill -TERM "$launch_pid" 2>/dev/null || true
  local term_waited=0
  while kill -0 "$launch_pid" 2>/dev/null && [[ "$term_waited" -lt "$GUARD_TERM_GRACE_SECONDS" ]]; do
    sleep 5
    term_waited=$(( term_waited + 5 ))
  done
  if kill -0 "$launch_pid" 2>/dev/null; then
    echo "[$(date)] WARNING: launcher survived cooperative drain plus SIGTERM grace; SIGKILL is the final recovery action and may require reboot if Metal is still evaluating: ${reason}." >> "$RESTART_LOG"
    kill -KILL "$launch_pid" 2>/dev/null || true
    cleanup_leftover_ranks hard
  else
    cleanup_leftover_ranks term
  fi
}

launcher_rss_guard() {
  local launch_pid="$1"
  local over_since=0
  local last_log=0

  if [[ "${LAUNCHER_RSS_GUARD_GB:-0}" == "0" ]]; then
    return 0
  fi

  echo "[$(date)] Launcher RSS guard armed for pid=${launch_pid}: limit=${LAUNCHER_RSS_GUARD_GB}GB idle_grace=${LAUNCHER_RSS_GUARD_IDLE_GRACE}s interval=${LAUNCHER_RSS_GUARD_INTERVAL}s." >> "$RESTART_LOG"
  while kill -0 "$launch_pid" 2>/dev/null; do
    sleep "$LAUNCHER_RSS_GUARD_INTERVAL"
    kill -0 "$launch_pid" 2>/dev/null || break

    local rss_kb rss_gb now
    rss_kb=$(ps -o rss= -p "$launch_pid" 2>/dev/null | tr -d '[:space:]')
    [[ -n "$rss_kb" ]] || continue
    rss_gb=$(( rss_kb / 1024 / 1024 ))
    now=$(date +%s)

    if [[ "$rss_gb" -lt "$LAUNCHER_RSS_GUARD_GB" ]]; then
      over_since=0
      continue
    fi

    if api_idle; then
      if [[ "$over_since" == "0" ]]; then
        over_since="$now"
        echo "[$(date)] Launcher RSS guard saw ${rss_gb}GB >= ${LAUNCHER_RSS_GUARD_GB}GB while idle; waiting grace period before recycle." >> "$RESTART_LOG"
        continue
      fi
      if [[ $(( now - over_since )) -ge "$LAUNCHER_RSS_GUARD_IDLE_GRACE" ]]; then
        echo "[$(date)] Launcher RSS guard recycling idle cluster: launcher rss=${rss_gb}GB limit=${LAUNCHER_RSS_GUARD_GB}GB." >> "$RESTART_LOG"
        curl -fsS --max-time 10 -X POST "$API_SHUTDOWN_URL" >/dev/null 2>&1 || true
        sleep 30
        if kill -0 "$launch_pid" 2>/dev/null; then
          echo "[$(date)] Launcher RSS guard admin shutdown did not exit launcher; sending SIGTERM to pid=${launch_pid}." >> "$RESTART_LOG"
          kill -TERM "$launch_pid" 2>/dev/null || true
        fi
        return 0
      fi
    else
      over_since=0
      if [[ $(( now - last_log )) -ge 900 ]]; then
        echo "[$(date)] Launcher RSS ${rss_gb}GB is above guard limit, but cluster is busy; deferring recycle." >> "$RESTART_LOG"
        last_log="$now"
      fi
    fi
  done
}

api_liveness_guard() {
  local launch_pid="$1"
  local started_at="$2"
  local api_down_since=0
  local warmup_started=0

  echo "[$(date)] API liveness guard armed for pid=${launch_pid}: start_grace=${API_GUARD_START_GRACE}s api_down=${API_DOWN_GUARD_SECONDS}s no_start_base=${ACTIVE_NO_START_GUARD_SECONDS}s no_start_context_tps=${ACTIVE_NO_START_CONTEXT_TPS} no_start_max=${ACTIVE_NO_START_MAX_SECONDS}s drain=${ACTIVE_NO_START_DRAIN_SECONDS}s interval=${API_GUARD_INTERVAL}s." >> "$RESTART_LOG"
  while kill -0 "$launch_pid" 2>/dev/null; do
    sleep "$API_GUARD_INTERVAL"
    kill -0 "$launch_pid" 2>/dev/null || break

    local now age health_json guard_decision
    now=$(date +%s)
    age=$(( now - started_at ))

    # Generous timeout: heavy decode starves the event loop for seconds at a
    # time; a 5s probe misclassifies a busy server as dead (observed 14:11
    # 2026-07-05: guard SIGTERM'd a server that was still completing requests).
    health_json=$(curl -fsS --max-time 20 "$API_HEALTH_URL" 2>/dev/null)
    if [[ -z "$health_json" ]]; then
      if [[ "$age" -lt "$API_GUARD_START_GRACE" ]]; then
        continue
      fi
      if [[ "$api_down_since" == "0" ]]; then
        api_down_since="$now"
        echo "[$(date)] API liveness guard: API not reachable; starting down timer." >> "$RESTART_LOG"
        continue
      fi
      if [[ $(( now - api_down_since )) -ge "$API_DOWN_GUARD_SECONDS" ]]; then
        echo "[$(date)] API liveness guard recycling cluster: API down for $(( now - api_down_since ))s while launcher still alive." >> "$RESTART_LOG"
        guarded_cluster_shutdown "$launch_pid" "API down for $(( now - api_down_since ))s" "$ACTIVE_NO_START_DRAIN_SECONDS"
        return 0
      fi
      continue
    fi
    api_down_since=0

    # First healthy probe of this launch: fire the startup warmup. All boots
    # through this loop previously served COLD (warmup only ran via
    # M3_Start.command), and the first real generation kept absorbing the
    # cold-start wedge hazard (three first-generation wedges on 2026-07-06).
    # The warmup is the sacrificial first generation, inside the managed
    # loop where a guard teardown just retries the boot.
    if [[ "$warmup_started" == "0" && "${M3_WARMUP_ON_START:-1}" == "1" && -f "$SCRIPT_DIR/m3_warmup.py" ]]; then
      warmup_started=1
      echo "[$(date)] API healthy — starting warmup pass (m3_warmup.py)." >> "$RESTART_LOG"
      pkill -TERM -f "$SCRIPT_DIR/m3_warmup.py" 2>/dev/null || true
      ( cd "$SCRIPT_DIR" && M3_WARMUP_BASE="http://127.0.0.1:${MLX_M3_PORT:-8080}" \
          nohup python3 m3_warmup.py > "$LOG_DIR/m3_warmup.log" 2>&1 & ) || true
    fi

    guard_decision=$(printf '%s' "$health_json" | \
      M3_ACTIVE_NO_START_GUARD_SECONDS="$ACTIVE_NO_START_GUARD_SECONDS" \
      M3_ACTIVE_NO_START_CONTEXT_TPS="$ACTIVE_NO_START_CONTEXT_TPS" \
      M3_ACTIVE_NO_START_MARGIN_SECONDS="$ACTIVE_NO_START_MARGIN_SECONDS" \
      M3_ACTIVE_NO_START_MAX_SECONDS="$ACTIVE_NO_START_MAX_SECONDS" \
      python3 -c '
import json, os, sys
try:
    h = json.load(sys.stdin)
except Exception:
    sys.exit(1)
active = h.get("active_request") or {}
if not active:
    sys.exit(1)
try:
    since = float(active.get("seconds_since_progress") or 0)
    tokens = int(active.get("tokens_emitted") or 0)
    prefill = int(active.get("prefill_processed_tokens") or 0)
    total = int(active.get("prefill_total_tokens") or 0)
except Exception:
    sys.exit(1)
shape = active.get("request_shape") or {}
try:
    prompt = int(
        shape.get("full_prompt_tokens")
        or active.get("prompt_tokens")
        or active.get("prefill_total_tokens")
        or 0
    )
except Exception:
    prompt = 0
base = float(os.environ.get("M3_ACTIVE_NO_START_GUARD_SECONDS", "120") or "120")
context_tps = max(1.0, float(os.environ.get("M3_ACTIVE_NO_START_CONTEXT_TPS", "1000") or "1000"))
margin = max(0.0, float(os.environ.get("M3_ACTIVE_NO_START_MARGIN_SECONDS", "120") or "120"))
maximum = max(base, float(os.environ.get("M3_ACTIVE_NO_START_MAX_SECONDS", "900") or "900"))
limit = max(base, min(maximum, margin + (prompt / context_tps)))
if tokens == 0 and prefill == 0 and total == 0 and since >= limit:
    print(f"{active.get('id')}:{since:.0f}:limit={limit:.0f}:prompt={prompt}")
    sys.exit(0)
sys.exit(1)
' 2>/dev/null)
    if [[ -n "$guard_decision" ]]; then
      # 2026-07-06 cascade: during boot handover the dying previous process
      # can still answer /health with ITS stale active request — recycling on
      # that killed a 15s-old healthy boot mid-materialization and leaked
      # 156GB wired. The no-progress recycler must respect start grace too.
      if [[ "$age" -lt "$API_GUARD_START_GRACE" ]]; then
        echo "[$(date)] API liveness guard: no-progress signal (${guard_decision}) within start grace (${age}s < ${API_GUARD_START_GRACE}s) — likely the previous process's stale health; ignoring." >> "$RESTART_LOG"
        continue
      fi
      echo "[$(date)] API liveness guard recycling cluster: active request made no prefill/decode progress (${guard_decision})." >> "$RESTART_LOG"
      guarded_cluster_shutdown "$launch_pid" "active request made no progress (${guard_decision})" "$ACTIVE_NO_START_DRAIN_SECONDS"
      return 0
    fi
  done
}

while true; do
  if orphan_memory_present; then
    break
  fi

  # Rotate startup.log at 8MB so forensics don't wade through mixed boots
  if [[ -f "$STARTUP_LOG" ]] && [[ $(stat -f%z "$STARTUP_LOG" 2>/dev/null || echo 0) -gt 8388608 ]]; then
    mv "$STARTUP_LOG" "$STARTUP_LOG.1" 2>/dev/null
  fi
  START_TS=$(date +%s)
  # Relaunches must never run mixed code across ranks: a mid-session edit +
  # watchdog self-heal deployed rank0-new/rank1-stale twice on 2026-07-06
  # (the stop-desync incidents). Sync is a ~1s rsync no-op when unchanged.
  /bin/zsh "$CLUSTER/sync_rank1.sh" >> "$RESTART_LOG" 2>&1 ||     echo "[$(date)] WARN: rank1 sync failed before relaunch" >> "$RESTART_LOG"
  echo "[$(date)] Starting cluster..." >> "$RESTART_LOG"
  /bin/zsh "$CLUSTER/launch_cluster.sh" >> "$STARTUP_LOG" 2>&1 &
  LAUNCH_PID=$!
  launcher_rss_guard "$LAUNCH_PID" &
  GUARD_PID=$!
  api_liveness_guard "$LAUNCH_PID" "$START_TS" &
  API_GUARD_PID=$!
  wait "$LAUNCH_PID"
  EXIT_CODE=$?
  kill "$GUARD_PID" >/dev/null 2>&1 || true
  kill "$API_GUARD_PID" >/dev/null 2>&1 || true
  wait "$GUARD_PID" >/dev/null 2>&1 || true
  wait "$API_GUARD_PID" >/dev/null 2>&1 || true
  END_TS=$(date +%s)
  RUNTIME=$((END_TS - START_TS))
  echo "[$(date)] Cluster exited with code $EXIT_CODE" >> "$RESTART_LOG"

  if [ -f "$STOP_FILE" ]; then
    rm -f "$STOP_FILE"
    echo "[$(date)] Stop requested, exiting restart loop." >> "$RESTART_LOG"
    break
  fi

  if orphan_memory_present; then
    break
  fi

  if [ "$EXIT_CODE" -eq 0 ]; then
    if [ -f "$STOP_FILE" ]; then
      echo "[$(date)] Clean cluster exit with stop flag; not restarting." >> "$RESTART_LOG"
      break
    fi
    # A zero exit without the stop flag is NOT an operator stop: jaccl
    # progress-timeout aborts and liveness sweeps surface this way
    # (observed 2026-07-05). Relaunch so wedge conditions self-heal.
    echo "[$(date)] Cluster exited 0 WITHOUT stop flag (crash/abort or guard sweep); relaunching." >> "$RESTART_LOG"
  fi

  # Guard teardowns (jaccl ProgressGuard exit-75) are wedge SELF-HEALS, not
  # startup failures — mlx.launch swallows the rank exit code so they reach
  # here as fast zero-exits and used to stop the loop (2026-07-06). They are
  # memory-safe by construction (teardown releases pinned buffers; the
  # orphan guard re-verifies each lap), so give them their own budget.
  GUARD_TEARDOWN=0
  if tail -40 "$STARTUP_LOG" 2>/dev/null | grep -qE "exited with code 75|made no progress for [0-9]+ms"; then
    GUARD_TEARDOWN=1
  fi
  if [ "$RUNTIME" -lt "$QUICK_FAILURE_WINDOW" ]; then
    if [ "$GUARD_TEARDOWN" -eq 1 ]; then
      GUARD_TEARDOWNS=$((GUARD_TEARDOWNS + 1))
      echo "[$(date)] Guard teardown self-heal detected (${GUARD_TEARDOWNS}/${M3_MAX_GUARD_TEARDOWNS:-6} in window); relaunching without counting a startup failure." >> "$RESTART_LOG"
    else
      QUICK_FAILURES=$((QUICK_FAILURES + 1))
    fi
  else
    QUICK_FAILURES=0
    GUARD_TEARDOWNS=0
    BACKOFF=${M3_RESTART_BACKOFF_INITIAL:-15}
  fi

  if [ "$QUICK_FAILURES" -ge "$MAX_QUICK_FAILURES" ]; then
    echo "[$(date)] $QUICK_FAILURES quick startup failures; stopping restart loop to protect JACCL/Metal memory. Start again after checking logs." >> "$RESTART_LOG"
    break
  fi
  if [ "${GUARD_TEARDOWNS:-0}" -ge "${M3_MAX_GUARD_TEARDOWNS:-6}" ]; then
    echo "[$(date)] ${GUARD_TEARDOWNS} guard teardowns inside the failure window; the link is flapping — stopping so an operator can inspect." >> "$RESTART_LOG"
    break
  fi

  echo "[$(date)] Restarting in ${BACKOFF}s (quick failures: $QUICK_FAILURES/$MAX_QUICK_FAILURES)..." >> "$RESTART_LOG"
  sleep "$BACKOFF"
  BACKOFF=$((BACKOFF * 2))
  if [ "$BACKOFF" -gt "$MAX_BACKOFF" ]; then
    BACKOFF="$MAX_BACKOFF"
  fi
done
