#!/usr/bin/env python3
"""Harvest completed DeepSWE pier trials into ops/bench/deepswe_results/.

Idempotent: scans every jobs/*/ trial, keeps the newest result per task id,
writes deepswe_results/<task>.json (reward, error, runtime) if absent.
Prints one 'TASK <id> reward=<r> (<n>/113)' line per newly harvested task.
"""
import json
import os
import re
import sys

BENCH = os.path.expanduser("~/minimax-m3-cluster/ops/bench")
JOBS = os.path.join(BENCH, "jobs")
OUT = os.path.join(BENCH, "deepswe_results")
TASKS_DIR = os.path.join(BENCH, "deep-swe", "tasks")
os.makedirs(OUT, exist_ok=True)

total_tasks = len([d for d in os.listdir(TASKS_DIR)
                   if os.path.isdir(os.path.join(TASKS_DIR, d))])

harvested = {}
for job in sorted(os.listdir(JOBS)) if os.path.isdir(JOBS) else []:
    jdir = os.path.join(JOBS, job)
    if not os.path.isdir(jdir):
        continue
    for trial in os.listdir(jdir):
        tdir = os.path.join(jdir, trial)
        rj = os.path.join(tdir, "result.json")
        if not os.path.isdir(tdir) or not os.path.exists(rj):
            continue
        m = re.match(r"(.+)__[A-Za-z0-9]+$", trial)
        if not m:
            continue
        task = m.group(1)
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        reward = r.get("reward")
        if reward is None:
            rw = os.path.join(tdir, "verifier", "reward.json")
            if os.path.exists(rw):
                try:
                    reward = json.load(open(rw)).get("reward")
                except Exception:
                    pass
        exc = r.get("exception_info") or r.get("exception") or None
        # a trial with an exception and no reward is a FAILED RUN, not a 0 —
        # leave it unharvested so the driver retries the task
        if reward is None and exc is not None:
            continue
        harvested[task] = {"task": task, "reward": reward,
                           "exception": bool(exc), "job": job, "trial": trial,
                           "started_at": r.get("started_at"),
                           "finished_at": r.get("finished_at")}

new = 0
for task, row in harvested.items():
    p = os.path.join(OUT, f"{task}.json")
    if not os.path.exists(p):
        json.dump(row, open(p, "w"), indent=1)
        new += 1
done = len([f for f in os.listdir(OUT) if f.endswith(".json")])
for task, row in sorted(harvested.items()):
    p = os.path.join(OUT, f"{task}.json")
    if new and os.path.getsize(p) and task in harvested:
        pass
if len(sys.argv) > 1 and sys.argv[1] == "--announce-latest" and new:
    for task in sorted(harvested):
        print(f"TASK {task} reward={harvested[task]['reward']} ({done}/{total_tasks})")
print(f"harvested {new} new | {done}/{total_tasks} tasks scored", flush=True)
