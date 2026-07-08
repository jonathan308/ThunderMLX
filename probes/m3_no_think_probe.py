#!/usr/bin/env python3
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"


def health():
    with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
        return json.loads(r.read().decode())


def summarize_health(h):
    last = h.get("last_request") or {}
    defaults = h.get("generation_defaults") or {}
    cache = h.get("prompt_cache") or {}
    prepare = cache.get("last_prepare_event") or {}
    return {
        "status": h.get("status"),
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "active": h.get("active_request") is not None,
        "decode_eval_every": defaults.get("decode_eval_every"),
        "decode_eval_after_tokens": defaults.get("decode_eval_after_tokens"),
        "decode_eval_after_every": defaults.get("decode_eval_after_every"),
        "thinking_decode_eval_every": defaults.get("thinking_decode_eval_every"),
        "tokens": last.get("tokens"),
        "elapsed_s": last.get("elapsed_s"),
        "ttft_s": last.get("first_token_s"),
        "decode_tps": last.get("decode_tps"),
        "request_tps": last.get("tps"),
        "prompt_tokens": last.get("prompt_tokens"),
        "prompt_tps": last.get("prompt_tps"),
        "cache_action": prepare.get("action"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_prompt_tokens": prepare.get("prompt_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_missed_tokens": prepare.get("missed_tokens"),
    }


def wait_idle(before_completed, timeout=60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        if (
            last.get("active_request") is None
            and last.get("requests_completed", 0) > before_completed
        ):
            return last
        time.sleep(0.2)
    raise RuntimeError(f"request did not become idle; last={last}")


def main():
    initial = health()
    print(json.dumps({"initial": summarize_health(initial)}, sort_keys=True), flush=True)
    before_completed = initial.get("requests_completed", 0)
    payload = {
        "model": "m3-no-think",
        "messages": [{
            "role": "user",
            "content": "Write a numbered list of 120 short items about stable APIs.",
        }],
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            if raw.decode("utf-8", "replace").strip().startswith("data: "):
                chunks += 1
    final = wait_idle(before_completed)
    row = summarize_health(final)
    row["client_elapsed_s"] = round(time.time() - started, 2)
    row["chunks"] = chunks
    print(json.dumps({"result": row}, sort_keys=True), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAIL", repr(e), file=sys.stderr, flush=True)
        try:
            print("health", json.dumps(health(), sort_keys=True), file=sys.stderr, flush=True)
        except Exception:
            pass
        raise
