#!/bin/zsh
# ssd_cache_toggle.sh on|off — flip SSD persistence + keepwarm and restart
# into the chosen configuration with a serialized, verified restart.
#
# "off" is the certified-stable revert point (stop certified 5/5 on
# 2026-07-06 with SSD/keepwarm disabled; RAM cache remains ON either way).
# "on" re-arms SSD persistence + restore + auto-save + keepwarm.
set -u
CLUSTER="${M3_CLUSTER_DIR:-$HOME/minimax-m3-cluster}"
ENVF="$CLUSTER/.env.local"
API=http://127.0.0.1:8080
MODE="${1:-}"

say() { echo "[ssd-toggle $(date '+%T')] $*" }

case "$MODE" in
  on)  V=1 ;;
  off) V=0 ;;
  *) echo "usage: $0 on|off"; exit 2 ;;
esac

say "setting SSD/keepwarm flags -> $MODE"
for KEY in MLX_M3_PROMPT_CACHE_SSD \
           MLX_M3_PROMPT_CACHE_SSD_RESTORE \
           MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE \
           MLX_M3_PROMPT_CACHE_KEEPWARM; do
  /usr/bin/sed -i '' "s|^${KEY}=[01]|${KEY}=${V}|" "$ENVF"
  grep -q "^${KEY}=${V}$" "$ENVF" || { say "FATAL: failed to set $KEY"; exit 1 }
done
say "flags now: $(grep -E '^MLX_M3_PROMPT_CACHE_(SSD|SSD_RESTORE|SSD_AUTO_SAVE|KEEPWARM)=' "$ENVF" | tr '\n' ' ')"

say "serialized restart..."
touch /private/tmp/minimax_m3_stop_requested
cd "$CLUSTER" && /bin/zsh stop_cluster.sh >/dev/null 2>&1
for i in $(seq 1 30); do
  sleep 1
  pgrep -f "run_with_watchdog|auto_restart.sh|mlx.launch" >/dev/null 2>&1 || break
done
pkill -9 -f "run_with_watchdog|m3_warmup" 2>/dev/null
rm -f /private/tmp/minimax_m3_stop_requested "$CLUSTER/.stop_requested"
rm -rf /private/tmp/minimax_m3_start.lock
/bin/zsh "$CLUSTER/M3_Start.command" >/dev/null 2>&1
sleep 3
pgrep -f auto_restart.sh >/dev/null || { say "FATAL: supervisor did not start"; exit 1 }

say "waiting for READY (model load ~6-9 min)..."
for i in $(seq 1 70); do
  sleep 10
  curl -s -m 3 $API/health >/dev/null 2>&1 && { say "READY"; break; }
done
curl -s -m 3 $API/health >/dev/null 2>&1 || { say "FATAL: boot timeout"; exit 1 }

sleep 3
ANSWER=$(curl -s -m 120 $API/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Minimax-M3-No-Think","messages":[{"role":"user","content":"What is 9 times 9? Just the number."}],"max_tokens":10}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>/dev/null)
say "coherence: ${ANSWER:-NO ANSWER}"
[ "$ANSWER" = "81" ] || say "WARNING: unexpected coherence answer"
say "DONE — cluster running with SSD/keepwarm $MODE"
