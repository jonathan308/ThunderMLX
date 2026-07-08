#!/usr/bin/env python3
"""Crash-safe state engine for the overnight fresh-boot trial matrix.

Commands:
  next                  claim the next pending trial (marks it running, prints its JSON);
                        exit 3 when the matrix is exhausted
  record IDX STATUS     record a trial verdict (plus optional --note / --data k=v ...)
  sweep                 mark any 'running' trial as aborted_reboot (call at orchestrator boot;
                        a trial still 'running' after a reboot means it crashed mid-flight)
  status                one line per trial

State lives in trial_matrix.json next to this script. Writes are atomic (tmp+rename).
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# MATRIX_FILE env selects an alternate matrix (e.g. soft_matrix.json for the
# no-reboot trial loop); current_trial.json handoff lives beside whichever file
PATH = Path(os.environ.get("MATRIX_FILE", Path(__file__).parent / "trial_matrix.json"))


def load():
    return json.loads(PATH.read_text())


def save(m):
    tmp = PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2) + "\n")
    tmp.rename(PATH)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    m = load()
    trials = m["trials"]

    if cmd == "sweep":
        swept = 0
        for t in trials:
            if t.get("status") == "running":
                t["status"] = "aborted_reboot"
                t["note"] = t.get("note", "") + " | still running at orchestrator boot (crashed mid-trial)"
                swept += 1
        save(m)
        print(f"swept {swept} stale running trial(s)")
        return 0

    if cmd == "next":
        for i, t in enumerate(trials):
            if t.get("status", "pending") == "pending":
                t["status"] = "running"
                t["started"] = datetime.now().isoformat()
                save(m)
                claimed = {"index": i, **t}
                # file handoff: shell-piping this JSON mangles non-ASCII under
                # the LaunchAgent's C locale (smoke-tested; it does)
                (PATH.parent / "current_trial.json").write_text(json.dumps(claimed) + "\n")
                print(json.dumps(claimed, ensure_ascii=True))
                return 0
        return 3  # exhausted

    if cmd == "field":  # field NAME [default] — read from current_trial.json
        cur = json.loads((PATH.parent / "current_trial.json").read_text())
        default = sys.argv[3] if len(sys.argv) > 3 else ""
        v = cur.get(sys.argv[2], default)
        if v is None:
            v = default
        if isinstance(v, bool):  # shell compares lowercase json-style
            v = "true" if v else "false"
        print(v)
        return 0

    if cmd == "record":
        idx, status = int(sys.argv[2]), sys.argv[3]
        t = trials[idx]
        t["status"] = status
        t["finished"] = datetime.now().isoformat()
        extra = {}
        args = sys.argv[4:]
        while args:
            if args[0] == "--note":
                t["note"] = (t.get("note", "") + " | " + args[1]).strip(" |")
                args = args[2:]
            elif args[0] == "--data":
                for kv in args[1:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        extra[k] = v
                args = []
            else:
                args = args[1:]
        if extra:
            t.setdefault("result", {}).update(extra)
        save(m)
        print(f"trial {idx} -> {status}")
        return 0

    for i, t in enumerate(trials):
        r = t.get("result", {})
        print(f"[{i}] {t.get('status','pending'):16s} {t['name']:40s} "
              f"clean={r.get('clean_rounds','-')} note={t.get('note','')[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
