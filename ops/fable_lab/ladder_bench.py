#!/usr/bin/env python3
"""Fable speed-lab context-ladder benchmark.

Builds deterministic long-context sessions at several sizes and measures decode
tok/s + TTFT from the server's slot-release telemetry, for chat and native-tool
turns, thinking and no-thinking — with a far-needle retrieval correctness gate
baked into every rung.

Usage:
  python3 ops/fable_lab/ladder_bench.py --base http://127.0.0.1:8180 \
      --sizes 50000,80000 --modes both --kinds chat,tool --reps 3 \
      --log /private/tmp/minimax-m3-fable-lab-logs/startup.log \
      --out ops/fable_lab/results/<name>.json

Sessions are keyed fable-lab-<size>-<model>-deterministic so repeated runs reuse
the same RAM/SSD cache state (A/B/A protocol: identical prompts, seeds, session
ids, cache lineage).
"""
import argparse, json, os, re, subprocess, time, urllib.request

WORDS = ("harbor tide crane manifest ledger quartz signal beacon rail cargo "
         "vector summit meadow circuit lantern archive").split()

NEEDLES = {
    # position_fraction: (key phrase, exact expected value)
    0.35: ("the vault access code", "X-4471-KESTREL"),
    0.72: ("the auxiliary relay id", "R-9082-MARLIN"),
}

TOOLS = [
    {"type": "function", "function": {"name": "write", "description": "Write content to a file",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"},
      "content": {"type": "string"}}, "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}},
      "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
      "required": ["command"]}}},
]

SLOT_RE = re.compile(r"request (chatcmpl-[a-f0-9]+) released distributed generation slot "
                     r"\(elapsed=([\d.]+)s, first_token=([\d.]+)s, prompt_tps=([\d.]+), "
                     r"tokens=(\d+), tps=([\d.]+), decode_tps=([\d.]+)\)")


def build_doc(target_tokens: int) -> str:
    # ~1.3 tokens/word for this vocab; deterministic content, needles at fixed
    # fractions so retrieval has exact ground truth.
    n_words = int(target_tokens / 1.35)
    out, wi = [], 0
    needle_at = {int(frac * n_words): (k, v) for frac, (k, v) in NEEDLES.items()}
    for i in range(n_words):
        if i in needle_at:
            k, v = needle_at[i]
            out.append(f". Note that {k} is {v} .")
        out.append(WORDS[wi % len(WORDS)] + ("." if i % 17 == 16 else ""))
        wi += 3 if i % 5 == 0 else 1
    return " ".join(out)


def call(base, payload, timeout=3600):
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d, time.time() - t0


def last_slot_line(log_path):
    tail = subprocess.run(["tail", "-c", "400000", log_path],
                          capture_output=True, text=True).stdout
    m = SLOT_RE.findall(tail)
    if not m:
        return {}
    g = m[-1]
    return {"req": g[0], "elapsed": float(g[1]), "first_token": float(g[2]),
            "prompt_tps": float(g[3]), "tokens": int(g[4]), "tps": float(g[5]),
            "decode_tps": float(g[6])}


def run_point(base, log_path, model, size, kind, reps, doc):
    session = f"fable-lab-{size}-{model}-det"
    results = []
    base_msgs = [
        {"role": "system", "content": "You are a precise assistant. The user provides a long operations journal; answer questions about it exactly."},
        {"role": "user", "content": doc + "\n\nAcknowledge you have read the journal in one short sentence."},
        {"role": "assistant", "content": "I have read the full operations journal and I'm ready for questions."},
    ]
    for rep in range(reps):
        if kind == "chat":
            q = (f"Recall exactly (pass {rep}): what is the vault access code and the auxiliary relay id "
                 "mentioned in the journal? Then write a 250-word summary of the journal's themes.")
            p = {"model": model, "stream": False, "max_tokens": 700, "temperature": 0.0,
                 "seed": 7, "session_id": session,
                 "messages": base_msgs + [{"role": "user", "content": q}]}
        else:
            q = (f"Use the write tool to save a status file to /tmp/fable_lab_probe_{rep}.md containing: "
                 "line 1 the vault access code from the journal, line 2 the auxiliary relay id, "
                 "then a 150-word operational summary of the journal.")
            p = {"model": model, "stream": False, "max_tokens": 900, "seed": 7,
                 "session_id": session, "tools": TOOLS,
                 "messages": base_msgs + [{"role": "user", "content": q}]}
        d, wall = call(base, p)
        tel = {}
        prev_req = getattr(run_point, "_last_req", None)
        for _ in range(20):
            time.sleep(0.5)
            tel = last_slot_line(log_path)
            if tel.get("req") and tel.get("req") != prev_req:
                break
        run_point._last_req = tel.get("req")
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        args = ""
        if calls:
            args = (calls[0].get("function") or {}).get("arguments") or ""
        blob = text + " " + args
        needle_ok = all(v in blob for (_k, v) in NEEDLES.values())
        r = {"rep": rep, "wall": round(wall, 2), "tool_called": [(c.get("function") or {}).get("name") for c in calls],
             "needle_ok": needle_ok, **tel}
        results.append(r)
        print(f"    rep{rep}: decode_tps={tel.get('decode_tps','?')} ttft={tel.get('first_token','?')}s "
              f"needle_ok={needle_ok} tools={r['tool_called'] or '-'}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8180")
    ap.add_argument("--sizes", default="50000,80000")
    ap.add_argument("--modes", default="both", help="think|nothink|both")
    ap.add_argument("--kinds", default="chat,tool")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--log", default="/private/tmp/minimax-m3-fable-lab-logs/startup.log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    models = {"think": ["Minimax-M3"], "nothink": ["Minimax-M3-No-Think"],
              "both": ["Minimax-M3-No-Think", "Minimax-M3"]}[args.modes]
    kinds = args.kinds.split(",")
    out = {"started": time.strftime("%F %T"), "points": []}
    for size in sizes:
        doc = build_doc(size)
        print(f"[size {size}] doc built (~{int(len(doc.split())*1.35)} tok)", flush=True)
        for model in models:
            for kind in kinds:
                print(f"  == {model} / {kind} ==", flush=True)
                res = run_point(args.base, args.log, model, size, kind, args.reps, doc)
                out["points"].append({"size": size, "model": model, "kind": kind, "reps": res})
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                json.dump(out, open(args.out, "w"), indent=1)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
