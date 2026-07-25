#!/bin/zsh
# Sequential full-corpus DeepSWE driver for the mixed-4.5 thinking arm.
# Resumable: skips tasks with a harvested result in deepswe_results/.
# Pause anytime by killing this script; the in-flight task is retried on resume.
set -uo pipefail
cd ~/minimax-m3-cluster/ops/bench
export OPENAI_API_KEY=thundermlx-local
export OPENAI_BASE_URL=http://host.docker.internal:8010/v1
export OPENAI_API_BASE=http://host.docker.internal:8010/v1
PY=/usr/bin/python3
mkdir -p deepswe_results

# never overlap with an already-running pier trial (e.g. the smoke task)
while pgrep -f "pier run" >/dev/null; do sleep 60; done
$PY deepswe_harvest.py

for d in deep-swe/tasks/*/; do
  t=$(basename "$d")
  [[ -f "deepswe_results/$t.json" ]] && continue
  echo "STARTING $t at $(date '+%H:%M:%S')"
  pier run -p "$d" --agent mini-swe-agent --model openai/Minimax-M3 \
    --env docker --env-file deepswe.env > "deepswe_results/$t.log" 2>&1
  $PY deepswe_harvest.py
  if [[ -f "deepswe_results/$t.json" ]]; then
    r=$($PY -c "import json;print(json.load(open('deepswe_results/$t.json'))['reward'])")
    n=$(ls deepswe_results/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "TASK $t reward=$r ($n/113)"
  else
    echo "TASK $t FAILED-NO-RESULT (will retry on next driver start)"
  fi
  # keep docker disk in check: prune dangling images every 10 tasks
  n=$(ls deepswe_results/*.json 2>/dev/null | wc -l | tr -d ' ')
  if (( n % 10 == 0 )); then docker system prune -f >/dev/null 2>&1; fi
done
echo "DEEPSWE DRIVER COMPLETE"
