#!/usr/bin/env python3
"""A/B probe for adaptive medium-context decode cadence.

This validates the safe runtime path:
- adaptive off: medium cached prompts use the long-context cadence.
- adaptive on: medium cached prompts can use the mid-context cadence while
  high-context prompts remain clamped to the safe high-context cadence.

The probe resets cache per case, seeds a synthetic agent workspace, then
measures a cached visible-history follow-up.
"""
import argparse
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def request_json(method, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(timeout=5):
    return request_json("GET", "/health", timeout=timeout)


def runtime_tuning(values):
    return request_json("POST", "/admin/runtime-tuning", {"values": values}, timeout=20)


def reset_cache(reason):
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": reason, "clear_memory": False},
        timeout=30,
    )


def wait_idle(before_completed, timeout=300):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        pc = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and not pc.get("in_use")
            and int(last.get("requests_completed") or 0) > before_completed
        ):
            return last
        time.sleep(0.25)
    return last or health()


def stream_chat(name, messages, *, model, max_tokens, timeout):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_piece_s = None
    chunks = 0
    content_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            evt = json.loads(item)
            chunks += 1
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                )
                if piece and first_piece_s is None:
                    first_piece_s = time.time() - started
                if delta.get("content"):
                    content_parts.append(delta["content"])

    elapsed = time.time() - started
    h = wait_idle(before, timeout=max(120, timeout))
    lr = h.get("last_request") or {}
    pc = h.get("prompt_cache") or {}
    prepare = lr.get("prompt_cache_prepare") or pc.get("last_prepare_event") or {}
    row = {
        "name": name,
        "client_elapsed_s": round(elapsed, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "server_ttft_s": lr.get("first_token_s"),
        "server_elapsed_s": lr.get("elapsed_s"),
        "server_total_elapsed_s": lr.get("total_elapsed_s"),
        "server_post_generation_s": lr.get("post_generation_s"),
        "server_prompt_tokens": lr.get("prompt_tokens"),
        "server_prompt_tps": lr.get("prompt_tps"),
        "server_decode_tps": lr.get("decode_tps"),
        "server_tokens": lr.get("tokens"),
        "cache_action": prepare.get("action"),
        "cache_prompt_tokens": prepare.get("prompt_tokens"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_missed_tokens": prepare.get("missed_tokens"),
        "cache_miss_reason": prepare.get("miss_reason"),
        "cache_len": pc.get("cache_len"),
        "failed": h.get("requests_failed"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row, "".join(content_parts), h


def build_workspace(records):
    return "\n".join(
        (
            f"src/module_{i % 97:02d}/file_{i:06d}.py :: "
            f"def function_{i}(value): return value + {i}; "
            f"test_{i}(input={i % 31}) expects {i + (i % 31)}"
        )
        for i in range(records)
    )


def run_case(label, adaptive_enabled, args):
    tuning = {
        "adaptive_long_context_decode_eval": 1 if adaptive_enabled else 0,
        "mid_context_decode_eval_tokens": args.mid_tokens,
        "mid_context_decode_eval_every": args.mid_every,
        "high_context_decode_eval_tokens": args.high_tokens,
        "high_context_decode_eval_every": args.high_every,
    }
    print(json.dumps({"case": label, "runtime_tuning": runtime_tuning(tuning)}, sort_keys=True), flush=True)
    print(json.dumps({"case": label, "reset": reset_cache(f"adaptive cadence probe {label}")}, sort_keys=True), flush=True)

    workspace = build_workspace(args.records)
    user0 = {
        "role": "user",
        "content": (
            workspace
            + "\n\nYou are a coding agent. Using only the workspace facts above, "
            + "summarize the reliability implications in about 90 words and cite "
            + "the exact path for record 777 if present."
        ),
    }
    cold, answer, _ = stream_chat(
        f"{label}_cold_seed",
        [user0],
        model=args.model,
        max_tokens=args.seed_tokens,
        timeout=args.cold_timeout,
    )
    followup_messages = [
        user0,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                "Continue the analysis with a numbered list of practical "
                "agent-runtime implications. Keep it concrete."
            ),
        },
    ]
    warm, _, final = stream_chat(
        f"{label}_hot_followup",
        followup_messages,
        model=args.model,
        max_tokens=args.followup_tokens,
        timeout=args.followup_timeout,
    )
    return {"label": label, "cold": cold, "warm": warm, "failed": final.get("requests_failed")}


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--seed-tokens", type=int, default=64)
    parser.add_argument("--followup-tokens", type=int, default=448)
    parser.add_argument("--mid-tokens", type=int, default=24576)
    parser.add_argument("--mid-every", type=int, default=1)
    parser.add_argument("--high-tokens", type=int, default=98304)
    parser.add_argument("--high-every", type=int, default=1)
    parser.add_argument("--cold-timeout", type=int, default=1200)
    parser.add_argument("--followup-timeout", type=int, default=900)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    defaults = initial.get("generation_defaults") or {}
    original = defaults.get("runtime_tuning") or {}
    print(json.dumps({
        "initial": {
            "completed": initial.get("requests_completed"),
            "failed": initial.get("requests_failed"),
            "runtime_tuning": original,
        }
    }, sort_keys=True), flush=True)

    rows = []
    try:
        if not args.skip_baseline:
            rows.append(run_case("adaptive_off", False, args))
        rows.append(run_case("adaptive_on", True, args))
    finally:
        if original:
            try:
                restore = runtime_tuning(original)
            except Exception as exc:
                restore = {"ok": False, "error": repr(exc)}
            print(json.dumps({"restore": restore}, sort_keys=True), flush=True)

    final = health()
    print(json.dumps({
        "summary": {
            "rows": rows,
            "completed_delta": final.get("requests_completed", 0) - initial.get("requests_completed", 0),
            "failed_delta": final.get("requests_failed", 0) - initial.get("requests_failed", 0),
            "final_runtime_tuning": (final.get("generation_defaults") or {}).get("runtime_tuning"),
        }
    }, sort_keys=True), flush=True)
    if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
        raise SystemExit("probe saw request failure")
    for row in rows:
        warm = row.get("warm") or {}
        if (warm.get("cache_reuse_ratio") or 0) < 0.95:
            raise SystemExit(f"{row.get('label')} cache reuse below 95%")


if __name__ == "__main__":
    main()
