#!/usr/bin/env python3
"""Build .env.local for a trial: .env.local.pristine + the trial's env overrides.

Usage: apply_trial_env.py IDX
Prints the trial's soak/tune parameters as shell-eval-able lines:
  SOAK_ROUNDS=8 SOAK_GAP=30 SOAK_MAX_TOKENS=10000 IOGPU0=0 IOGPU1=0
Never touches .env.local.pristine. Restore after the matrix:
  cp .env.local.pristine .env.local
"""
import json
import re
import sys
from pathlib import Path

import os

REPO = Path(__file__).parent.parent
PRISTINE = REPO / ".env.local.pristine"
TARGET = REPO / ".env.local"
MATRIX = Path(os.environ.get("MATRIX_FILE", Path(__file__).parent / "trial_matrix.json"))


def main() -> int:
    idx = int(sys.argv[1])
    t = json.loads(MATRIX.read_text())["trials"][idx]
    base = PRISTINE.read_text()
    lines = base.splitlines()
    overrides = t.get("env", {})
    seen = set()
    out = []
    for line in lines:
        m = re.match(r"^([A-Z][A-Z0-9_]*)=", line)
        if m and m.group(1) in overrides:
            k = m.group(1)
            out.append(f"{k}={overrides[k]}")
            seen.add(k)
        else:
            out.append(line)
    missing = [k for k in overrides if k not in seen]
    if missing:
        out.append("")
        out.append(f"# trial {idx} ({t['name']}) additions")
        out.extend(f"{k}={overrides[k]}" for k in missing)
    TARGET.write_text("\n".join(out) + "\n")

    soak = t.get("soak", {})
    print(f"SOAK_ROUNDS={soak.get('rounds', 8)}")
    print(f"SOAK_GAP={soak.get('gap', 30)}")
    print(f"SOAK_MAX_TOKENS={soak.get('max_tokens', 10000)}")
    print(f"IOGPU0={t.get('iogpu0', 0)}")
    print(f"IOGPU1={t.get('iogpu1', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
