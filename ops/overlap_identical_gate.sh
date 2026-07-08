#!/bin/zsh
# Async-overlap correctness gate: with a FIXED long prompt at temp 0, the
# completion must be byte-identical with the overlap ON vs OFF. The overlap
# only reorders host-side blocking, never the math or the collective order —
# any output drift means the patch is wrong and must not ship.
#
# Usage: overlap_identical_gate.sh capture <label>   (run once per config)
#        overlap_identical_gate.sh compare <labelA> <labelB>
set -u
OPS=$HOME/minimax-m3-cluster/ops
mkdir -p $OPS/logs
PROMPT_FILE=$OPS/overlap_gate_prompt.txt

if [[ ! -f $PROMPT_FILE ]]; then
  # deterministic ~12k-token prompt, generated ONCE and reused for every capture
  python3 - <<'EOF'
from pathlib import Path
filler = ("Entry %04d: the survey vessel logged wind at %02d knots, swell %d meters, "
          "barometer %d hPa, all instruments nominal after calibration pass %d. ")
body = "".join(filler % (i, (i*7) % 40, (i*3) % 6 + 1, 980 + (i*11) % 60, i % 9) for i in range(700))
Path.home().joinpath("minimax-m3-cluster/ops/overlap_gate_prompt.txt").write_text(
    "Archive integrity review, fixed corpus A-113.\n" + body +
    "\nSummarize the overall condition of the vessel logs in exactly two sentences.")
print("gate prompt generated")
EOF
fi

cmd=${1:-}
case $cmd in
  capture)
    label=${2:?label required}
    python3 - "$label" <<'EOF'
import json, sys, requests
from pathlib import Path
label = sys.argv[1]
prompt = (Path.home() / "minimax-m3-cluster/ops/overlap_gate_prompt.txt").read_text()
r = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
    "model": "Minimax-M3-No-Think",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 96, "temperature": 0.0, "stream": False,
}, timeout=(15, 300))
r.raise_for_status()
text = r.json()["choices"][0]["message"]["content"]
out = Path.home() / f"minimax-m3-cluster/ops/logs/overlap_gate_{label}.txt"
out.write_text(text)
print(f"captured {label}: {len(text)} chars -> {out}")
EOF
    ;;
  compare)
    a=$OPS/logs/overlap_gate_${2:?}.txt
    b=$OPS/logs/overlap_gate_${3:?}.txt
    if diff -q "$a" "$b" >/dev/null 2>&1; then
      echo "IDENTICAL-OUTPUT GATE: PASS ($2 == $3)"
    else
      echo "IDENTICAL-OUTPUT GATE: FAIL — outputs differ:"
      diff "$a" "$b" | head -10
      exit 1
    fi
    ;;
  *)
    echo "usage: overlap_identical_gate.sh capture <label> | compare <a> <b>"; exit 2
    ;;
esac
