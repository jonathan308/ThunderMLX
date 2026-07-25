#!/usr/bin/env python3
"""SWE-bench-derived patch benchmark (oracle-context, docker-free proxies).

Real SWE-bench Verified instances; the model sees the issue + the pre-patch
content of the files the gold fix touches, and must emit a unified diff.
Scored on objective proxies (no container execution):
  apply_ok      git apply --check succeeds on the base checkout
  files_match   patched file set == gold patch file set
  region_hit    >=1 hunk lands within ±20 lines of a gold hunk
plus thinking volume, hesitation markers, and wall time per instance.

Usage:
  swe_patch_bench.py --arm mixed45-think --model Minimax-M3
  swe_patch_bench.py --arm mixed45-nothink --model Minimax-M3-No-Think
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

BASE = "http://localhost:8010"
WORK = os.path.expanduser("~/minimax-m3-cluster/ops/bench/swe_work")
N_INSTANCES = 10
MARKER_WORDS = ("wait", "actually", "hmm", "let me reconsider", "hold on",
                "alternatively", "on second thought")


def sh(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}: {r.stderr[:300]}")
    return r


def pick_instances():
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rows = []
    for r in ds:
        patch = r["patch"]
        files = re.findall(r"^diff --git a/(\S+)", patch, re.M)
        if (len(files) == 1 and len(patch.splitlines()) <= 40
                and len(r["problem_statement"]) <= 4000):
            rows.append(r)
    rows.sort(key=lambda r: r["instance_id"])
    seen, out = set(), []
    for r in rows:                       # diversity: max 2 per repo
        if sum(1 for x in out if x["repo"] == r["repo"]) < 2:
            out.append(r)
        if len(out) == N_INSTANCES:
            break
    return out


def checkout(inst):
    repo_dir = os.path.join(WORK, inst["instance_id"])
    if not os.path.exists(repo_dir):
        url = f"https://github.com/{inst['repo']}.git"
        sh(f"git clone --filter=blob:none --quiet {url} {repo_dir}")
    sh(f"git checkout --quiet {inst['base_commit']}", cwd=repo_dir)
    sh("git clean -fdq", cwd=repo_dir)
    return repo_dir


def gold_files(inst):
    return re.findall(r"^diff --git a/(\S+)", inst["patch"], re.M)


def gold_hunk_lines(inst):
    return [int(m) for m in re.findall(r"^@@ -(\d+)", inst["patch"], re.M)]


def oracle_context(inst, repo_dir):
    parts = []
    for f in gold_files(inst):
        p = os.path.join(repo_dir, f)
        try:
            content = open(p, encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            continue
        if len(content) > 60000:
            lines = content.splitlines()
            keep = set()
            for start in gold_hunk_lines(inst):
                keep.update(range(max(0, start - 150), min(len(lines), start + 150)))
            content = "\n".join(
                f"{i+1}: {l}" for i, l in enumerate(lines) if i in keep)
            parts.append(f"### {f} (relevant excerpt, line-numbered)\n{content}")
        else:
            parts.append(f"### {f}\n{content}")
    return "\n\n".join(parts)


def ask_model(model, inst, context):
    prompt = f"""You are fixing a real GitHub issue.

REPOSITORY: {inst['repo']}

ISSUE:
{inst['problem_statement']}

CURRENT SOURCE (pre-fix):
{context}

Produce ONLY a unified diff patch that fixes the issue, in ```diff fences.
Use correct `diff --git a/... b/...` headers and accurate hunk line numbers.
No commentary outside the fences."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 8192, "temperature": 1.0, "seed": 1}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    return {"content": m.get("content") or "",
            "think": m.get("reasoning_content") or "",
            "wall_s": round(time.time() - t0, 1)}


def extract_patch(text):
    m = re.search(r"```diff\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"(^diff --git .*)", text, re.S | re.M)
    return m.group(1) if m else ""


def score(inst, repo_dir, patch):
    if not patch.strip():
        return {"apply_ok": False, "files_match": False, "region_hit": False}
    pf = os.path.join(repo_dir, "_model.patch")
    open(pf, "w").write(patch if patch.endswith("\n") else patch + "\n")
    apply_ok = sh("git apply --check --whitespace=nowarn _model.patch",
                  cwd=repo_dir, check=False).returncode == 0
    mf = set(re.findall(r"^diff --git a/(\S+)", patch, re.M))
    files_match = mf == set(gold_files(inst))
    mh = [int(x) for x in re.findall(r"^@@ -(\d+)", patch, re.M)]
    region_hit = any(abs(a - b) <= 20 for a in mh for b in gold_hunk_lines(inst))
    return {"apply_ok": apply_ok, "files_match": files_match,
            "region_hit": region_hit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    insts = pick_instances()
    print(f"[{args.arm}] {len(insts)} instances:",
          [i["instance_id"] for i in insts], flush=True)
    rows = []
    for inst in insts:
        repo_dir = checkout(inst)
        ctx = oracle_context(inst, repo_dir)
        r = ask_model(args.model, inst, ctx)
        patch = extract_patch(r["content"])
        s = score(inst, repo_dir, patch)
        tl = r["think"].lower()
        row = {"instance": inst["instance_id"], **s,
               "think_chars": len(r["think"]),
               "markers": sum(tl.count(w) for w in MARKER_WORDS),
               "wall_s": r["wall_s"], "patch": patch,
               "gold": inst["patch"]}
        rows.append(row)
        print(f"  {inst['instance_id']}: apply={s['apply_ok']} "
              f"files={s['files_match']} region={s['region_hit']} "
              f"think={len(r['think'])}c markers={row['markers']} "
              f"wall={r['wall_s']}s", flush=True)

    n = len(rows)
    print(f"=== SUMMARY [{args.arm}] ===")
    for k in ("apply_ok", "files_match", "region_hit"):
        print(f"  {k}: {sum(r[k] for r in rows)}/{n}")
    print(f"  think_avg: {sum(r['think_chars'] for r in rows)/n:.0f}c "
          f"markers_avg: {sum(r['markers'] for r in rows)/n:.1f} "
          f"wall_avg: {sum(r['wall_s'] for r in rows)/n:.1f}s")
    out = args.out or f"ops/bench/swe_{args.arm}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
