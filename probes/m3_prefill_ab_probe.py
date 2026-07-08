#!/usr/bin/env python3
"""Runtime prefill-step A/B probe for MiniMax-M3.

This intentionally measures cold prompt processing with unique session ids. It
restores the original runtime tuning at the end, even if a test fails.
"""
import argparse
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def get_json(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post_json(path, payload, timeout=30):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(timeout=10):
    return get_json("/health", timeout=timeout)


def runtime_tuning():
    return (health().get("generation_defaults") or {}).get("runtime_tuning") or {}


def set_tuning(values):
    return post_json("/admin/runtime-tuning", {"values": values}, timeout=30)


def reset_cache(reason):
    return post_json(
        "/admin/prompt-cache/reset",
        {"reason": reason, "clear_memory": False},
        timeout=30,
    )


def wait_idle(timeout=120):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        if last.get("active_request") is None and int(last.get("request_queue_depth") or 0) == 0:
            return last
        time.sleep(0.25)
    raise TimeoutError(f"endpoint not idle: {last}")


def stream_chat(name, prompt, *, session_id, max_tokens, timeout):
    before = health().get("requests_completed", 0)
    payload = {
        "model": "Minimax-M3-No-Think",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metadata": {"session_id": session_id, "source": "m3_prefill_ab_probe"},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_chunk_s = None
    chunks = 0
    chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            obj = json.loads(item)
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
            if piece and first_chunk_s is None:
                first_chunk_s = time.time() - started
            if piece:
                chars += len(piece)
            chunks += 1
    client_elapsed = time.time() - started
    deadline = time.time() + 60
    h = health()
    while time.time() < deadline:
        h = health()
        if h.get("active_request") is None and h.get("requests_completed", 0) > before:
            break
        time.sleep(0.2)
    last = h.get("last_request") or {}
    pc = h.get("prompt_cache") or {}
    prepare = last.get("prompt_cache_prepare") or pc.get("last_prepare_event") or {}
    ks = h.get("kernel_stats") or {}
    gd = h.get("generation_defaults") or {}
    return {
        "name": name,
        "session_id": session_id,
        "client_elapsed_s": round(client_elapsed, 3),
        "client_ttft_s": round(first_chunk_s or 0.0, 3),
        "chunks": chunks,
        "chars": chars,
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_ttft_s": last.get("first_token_s"),
        "server_tokens": last.get("tokens"),
        "server_decode_tps": last.get("decode_tps"),
        "cache_action": prepare.get("action"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "effective_prefill_step_size": gd.get("effective_prefill_step_size"),
        "last_msa_k1_impl": ks.get("last_msa_k1_impl"),
        "prefill_attention_calls": ks.get("prefill_attention_calls"),
        "prefill_standard_topk": ks.get("prefill_standard_topk"),
        "prefill_blockwise_topk": ks.get("prefill_blockwise_topk"),
        "topk_native": ks.get("topk_native"),
        "topk_native_error": ks.get("topk_native_error"),
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
    }


def build_prompt(records):
    long_prefix = "\n".join(
        f"FILE-{i:04d}: function_{i}(x) returns x + {i}; owner=team-{i % 17}; checksum={i * 97}"
        for i in range(records)
    )
    return (
        long_prefix
        + "\n\nUsing only the records above, return the exact line for FILE-1377 if it exists, then stop."
    )


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--records", type=int, default=1200)
    parser.add_argument("--steps", default="4096,5120,6144")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--session-prefix", default="")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    prefix = args.session_prefix or f"prefill-ab-{int(time.time())}"
    prompt = build_prompt(args.records)
    original = runtime_tuning()
    print(json.dumps({"initial_runtime_tuning": original}, sort_keys=True), flush=True)
    print(json.dumps({"records": args.records, "steps": steps, "max_tokens": args.max_tokens}, sort_keys=True), flush=True)
    rows = []
    try:
        for step in steps:
            wait_idle()
            print(json.dumps({"set_runtime_tuning": set_tuning({"prefill_step_size": step})}, sort_keys=True), flush=True)
            print(json.dumps({"reset": reset_cache(f"prefill A/B step {step}")}, sort_keys=True), flush=True)
            row = stream_chat(
                f"prefill_step_{step}",
                prompt,
                session_id=f"{prefix}-{step}",
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        best = max(rows, key=lambda r: float(r.get("server_prompt_tps") or 0.0))
        print(json.dumps({"best": best, "rows": rows}, sort_keys=True), flush=True)
    finally:
        restore = {"prefill_step_size": int(original.get("prefill_step_size") or 4096)}
        print(json.dumps({"restore_runtime_tuning": set_tuning(restore)}, sort_keys=True), flush=True)
        wait_idle()


if __name__ == "__main__":
    main()
