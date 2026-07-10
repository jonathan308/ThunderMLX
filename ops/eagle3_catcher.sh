#!/bin/zsh
# eagle3_catcher.sh — arm BEFORE firing a stall probe. Finds the REAL rank0
# worker (the biggest-RSS python; the watchdog's own capture samples the ssh
# tunnel / supervisor siblings), then every 15s: native `sample` + SIGUSR2
# (faulthandler python-stack dump -> startup.log). Run for DUR seconds.
set -u
DUR="${1:-120}"
OUT="${2:-$HOME/ThunderMLX-eagle3/ops/logs}"
mkdir -p "$OUT"
end=$(( $(date +%s) + DUR ))
i=0
while [ "$(date +%s)" -lt "$end" ]; do
  WPID=$(ps -Ao pid,rss,comm | awk '/Python/ {if ($2>maxr) {maxr=$2; p=$1}} END {print p}')
  RSS=$(ps -o rss= -p "$WPID" 2>/dev/null | awk '{printf "%.0f", $1/1048576}')
  if [ -n "${WPID:-}" ] && [ "${RSS:-0}" -gt 50 ]; then
    i=$((i+1))
    echo "[catcher] tick $i: worker pid=$WPID rss=${RSS}GB"
    kill -USR2 "$WPID" 2>/dev/null && echo "[catcher]   USR2 sent (python stacks -> startup.log)"
    sample "$WPID" 3 -file "$OUT/catch_${i}_pid${WPID}.txt" >/dev/null 2>&1 &&
      echo "[catcher]   sample -> $OUT/catch_${i}_pid${WPID}.txt"
  else
    echo "[catcher] no big worker yet (pid=${WPID:-none} rss=${RSS:-0}GB)"
  fi
  sleep 15
done
echo "[catcher] done ($i ticks)"
