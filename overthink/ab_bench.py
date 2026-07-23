#!/usr/bin/env python3
"""A/B bench for the overthink logit penalty (arXiv 2606.00206) — lab only.

Sends identical task sets to the LAB server (:8081) across penalty arms via
the per-request `overthink_penalty` payload key (no restarts between arms).
Measures per request: thinking tokens (true tokenizer count), marker
frequencies in the reasoning trace, correctness, wall time.

Usage:
  ab_bench.py [--arms 0,0.5,1.0,1.5] [--seeds 1,2] [--base http://localhost:8081]
              [--tasks tasks.json] [--out results.json]
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request

MARKER_WORDS = [
    "wait", "but", "however", "alternatively", "actually", "maybe", "perhaps",
    "hold", "instead", "though", "reconsider", "wrong", "mistake", "error",
]

TASKS = [
    {"id": "arith-multi", "kind": "exact",
     "prompt": "A warehouse has 3 shelves with 47 boxes each. 29 boxes are shipped out, then each remaining box is split into 2 half-size boxes. How many half-size boxes are there? Give only the final number.",
     "expect": "224"},
    {"id": "primes", "kind": "exact",
     "prompt": "What are the prime factors of 1001? Answer with just the factors separated by commas, ascending.",
     "expect": "7, 11, 13", "alt": ["7,11,13", "7, 11, 13"]},
    {"id": "rate-problem", "kind": "exact",
     "prompt": "Two painters take 6 hours and 3 hours respectively to paint a room alone. Working together, how many hours to paint one room? Give only the number.",
     "expect": "2"},
    {"id": "remainder", "kind": "exact",
     "prompt": "What is the remainder when 2 to the power 20 is divided by 7? Give only the number.",
     "expect": "4"},
    {"id": "logic-order", "kind": "exact",
     "prompt": "Ana is taller than Ben. Ben is taller than Cal. Dan is shorter than Cal. Who is second-tallest? One word answer.",
     "expect": "ben"},
    {"id": "code-balanced", "kind": "exec",
     "prompt": "Write a Python function named is_balanced(s) that returns True when parentheses () [] {} in s are balanced, else False. Output only the code, no explanation.",
     "tests": [("is_balanced('([]{})')", True), ("is_balanced('([)]')", False),
               ("is_balanced('')", True), ("is_balanced('((')", False)]},
    {"id": "code-rle", "kind": "exec",
     "prompt": "Write a Python function named rle(s) that run-length-encodes a string: rle('aaabcc') == 'a3b1c2'. Output only the code.",
     "tests": [("rle('aaabcc')", "a3b1c2"), ("rle('')", ""), ("rle('z')", "z1")]},
    {"id": "code-secondmax", "kind": "exec",
     "prompt": "Write a Python function named second_max(xs) returning the second largest DISTINCT value in a list of ints, or None if it does not exist. Output only the code.",
     "tests": [("second_max([3,1,4,4,2])", 3), ("second_max([5])", None),
               ("second_max([2,2,2])", None), ("second_max([-1,-9,0])", -1)]},
    {"id": "date-reason", "kind": "exact",
     "prompt": "If the 1st of a 30-day month is a Wednesday, what weekday is the 30th? One word answer.",
     "expect": "thursday"},
    {"id": "unit-chain", "kind": "exact",
     "prompt": "A pipe fills 2.5 liters every 4 seconds. How many minutes to fill a 300 liter tank? Give only the number.",
     "expect": "8"},
]


def chat(base, prompt, lam, seed, max_tokens=6144, timeout=600):
    payload = {
        "model": "Minimax-M3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "seed": seed,
        "overthink_penalty": lam,
    }
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    wall = time.time() - t0
    msg = out["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "usage": out.get("usage") or {},
        "wall_s": round(wall, 2),
    }


def check(task, content):
    if task["kind"] == "probe":
        return None
    text = content.strip().lower()
    if task["kind"] == "exact":
        golds = [task["expect"].lower()] + [a.lower() for a in task.get("alt", [])]
        tail = text[-120:]
        return any(g in tail for g in golds)
    if task["kind"] == "exec":
        code = content
        m = re.search(r"```(?:python)?\n(.*?)```", content, re.S)
        if m:
            code = m.group(1)
        checks = "\n".join(
            f"assert (({expr}) == {expected!r}), {expr!r}"
            for expr, expected in task["tests"])
        prog = f"{code}\n{checks}\nprint('OK')\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog)
        try:
            res = subprocess.run([sys.executable, f.name],
                                 capture_output=True, timeout=10)
            return res.returncode == 0 and b"OK" in res.stdout
        except Exception:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8081")
    ap.add_argument("--arms", default="0,0.5,1.0,1.5")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--tasks", default=None, help="extra tasks json (list) to append")
    ap.add_argument("--no-default-tasks", action="store_true")
    ap.add_argument("--out", default="overthink/results.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("MLX_M3_MODEL", "mlx-community/MiniMax-M3-4bit"),
        trust_remote_code=True)

    tasks = [] if args.no_default_tasks else list(TASKS)
    if args.tasks:
        tasks += json.load(open(args.tasks))

    arms = [float(x) for x in args.arms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    total = len(arms) * len(seeds) * len(tasks)
    i = 0
    for lam in arms:
        for seed in seeds:
            for task in tasks:
                i += 1
                try:
                    r = chat(args.base, task["prompt"], lam, seed,
                             max_tokens=task.get("max_tokens", 6144))
                except Exception as e:
                    print(f"[{i}/{total}] lam={lam} seed={seed} {task['id']}: REQUEST ERROR {e}", flush=True)
                    rows.append({"task": task["id"], "lam": lam, "seed": seed, "error": str(e)})
                    continue
                think_tokens = len(tok.encode(r["reasoning"], add_special_tokens=False)) if r["reasoning"] else 0
                low = r["reasoning"].lower()
                marker_counts = {w: len(re.findall(rf"\b{w}\b", low)) for w in MARKER_WORDS}
                ok = check(task, r["content"])
                closed = bool(r["content"].strip())
                rows.append({
                    "task": task["id"], "lam": lam, "seed": seed,
                    "correct": ok, "think_closed": closed,
                    "think_tokens": think_tokens,
                    "completion_tokens": r["usage"].get("completion_tokens"),
                    "markers_total": sum(marker_counts.values()),
                    "markers": marker_counts, "wall_s": r["wall_s"],
                })
                print(f"[{i}/{total}] lam={lam} seed={seed} {task['id']}: "
                      f"{'PASS' if ok else 'FAIL'} think={think_tokens} "
                      f"markers={sum(marker_counts.values())} wall={r['wall_s']}s", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)

    print("\n=== SUMMARY by lambda ===")
    for lam in arms:
        sel = [x for x in rows if x.get("lam") == lam and "error" not in x]
        if not sel:
            continue
        graded = [x for x in sel if x["correct"] is not None]
        acc = (sum(x["correct"] for x in graded) / len(graded)) if graded else 0.0
        think = sum(x["think_tokens"] for x in sel) / len(sel)
        marks = sum(x["markers_total"] for x in sel) / len(sel)
        wall = sum(x["wall_s"] for x in sel) / len(sel)
        print(f"lam={lam:<4} n={len(sel):<3} acc={acc:.0%} "
              f"think_avg={think:7.1f} markers_avg={marks:5.1f} wall_avg={wall:6.1f}s")


if __name__ == "__main__":
    main()
