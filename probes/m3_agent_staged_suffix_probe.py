#!/usr/bin/env python3
"""Agent-style staged suffix cache probe.

This grows one chat session in uneven turns, matching a coding-agent workflow:
a large initial workspace context followed by progressively smaller deltas.
Healthy behavior is one cold prefill, then KV reuse with only the new suffix
processed on each later turn.
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(timeout=5):
    return request_json("GET", "/health", timeout=timeout)


def compact_cache(cache):
    if not isinstance(cache, dict):
        return cache
    return {
        "loaded": cache.get("loaded"),
        "in_use": cache.get("in_use"),
        "cache_len": cache.get("cache_len"),
        "key_tokens": cache.get("key_tokens"),
        "session_id": cache.get("session_id"),
        "last_prepare_event": cache.get("last_prepare_event"),
        "last_update_event": cache.get("last_update_event"),
    }


def reset_cache():
    payload = request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": "agent staged suffix probe", "clear_memory": False},
        timeout=30,
    )
    return {"ok": payload.get("ok"), "prompt_cache": compact_cache(payload.get("prompt_cache") or {})}


def wait_idle(before_completed, timeout=180):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health(timeout=5)
        pcache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and not pcache.get("in_use")
            and int(last.get("requests_completed") or 0) > before_completed
        ):
            return last
        time.sleep(0.25)
    return last or health(timeout=5)


def synthetic_agent_chunk(label, approx_tokens):
    target_chars = max(200, int(approx_tokens) * 4)
    lines = []
    index = 0
    chars = 0
    while chars < target_chars:
        line = (
            f"{label} file_{index:05d}.py: def unit_{index:05d}(state): "
            f"validate cache_key='{label}-{index:05d}', preserve invariant_{index % 17}, "
            f"compare expected_{index % 29}, and return state + {index % 7}. "
            "Notes: this is agent workspace evidence for incremental prompt reuse."
        )
        lines.append(line)
        chars += len(line) + 1
        index += 1
    return "\n".join(lines)


def stream_chat(label, messages, *, model, max_tokens, timeout, session_id):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metadata": {"session_id": session_id, "probe": "agent_staged_suffix"},
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
    chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            evt = json.loads(item)
            chunks += 1
            delta = (evt.get("choices") or [{}])[0].get("delta") or {}
            piece = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
            )
            if piece and first_piece_s is None:
                first_piece_s = time.time() - started
            chars += len(piece)

    final = wait_idle(before, timeout=max(180, timeout // 2))
    last = final.get("last_request") or {}
    pcache = final.get("prompt_cache") or {}
    prepare = last.get("prompt_cache_prepare") or pcache.get("last_prepare_event") or {}
    kernels = final.get("kernel_stats") or {}
    row = {
        "label": label,
        "session_id": session_id,
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "client_elapsed_s": round(time.time() - started, 3),
        "chunks": chunks,
        "output_chars": chars,
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_ttft_s": last.get("first_token_s"),
        "server_decode_tps": last.get("decode_tps"),
        "completion_tokens": last.get("completion_tokens"),
        "cache_action": prepare.get("action"),
        "cache_prompt_tokens": prepare.get("prompt_tokens"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_miss_reason": prepare.get("miss_reason"),
        "cache_len": pcache.get("cache_len"),
        "last_msa_k1_impl": kernels.get("last_msa_k1_impl"),
        "steel_mma_calls": kernels.get("msa_k1_steel_mma"),
        "requests_failed": final.get("requests_failed"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def parse_stages(raw):
    stages = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, value = item.split(":", 1)
            stages.append((label.strip(), int(value.strip())))
        else:
            value = int(item)
            stages.append((f"turn{len(stages) + 1}_plus_{value}", value))
    if not stages:
        raise ValueError("at least one stage is required")
    return stages


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--session-id", default=f"agent-staged-suffix-{int(time.time())}")
    parser.add_argument(
        "--stages",
        default="turn1_20k_base:20000,turn2_plus_8k:8000,turn3_plus_2k:2000,turn4_plus_500:500",
        help="Comma list of label:approx_tokens stages.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--min-cold-prompt-tps", type=float, default=330.0)
    parser.add_argument("--min-reuse-ratio-after-first", type=float, default=0.65)
    parser.add_argument("--max-failed-delta", type=int, default=0)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    stages = parse_stages(args.stages)

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    initial_failed = int(initial.get("requests_failed") or 0)
    defaults = initial.get("generation_defaults") or {}
    print(
        json.dumps(
            {
                "initial": {
                    "status": initial.get("status"),
                    "active_request": initial.get("active_request"),
                    "failed": initial_failed,
                    "session_id": args.session_id,
                    "stages": stages,
                    "prefill_step_size": defaults.get("prefill_step_size"),
                    "prefill_stop_check_every": defaults.get("prefill_stop_check_every"),
                    "runtime_tuning": defaults.get("runtime_tuning"),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.keep_cache:
        print(json.dumps({"reset": reset_cache()}, sort_keys=True), flush=True)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a coding agent. Preserve context and answer with concise "
                "implementation notes."
            ),
        }
    ]
    rows = []
    for idx, (label, tokens) in enumerate(stages, 1):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Agent staged suffix turn {idx}. Use prior context plus this delta. "
                    f"Include marker AGENTIC_STAGE_{idx}.\n\n"
                    + synthetic_agent_chunk(label, tokens)
                ),
            }
        )
        row = stream_chat(
            label,
            messages,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            session_id=args.session_id,
        )
        rows.append(row)
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"{label} acknowledged. Marker AGENTIC_STAGE_{idx}. "
                    "I will preserve previous workspace invariants and focus on new suffixes."
                ),
            }
        )

    failures = []
    failed_delta = int((health().get("requests_failed") or 0)) - initial_failed
    if failed_delta > args.max_failed_delta:
        failures.append(f"request failures increased by {failed_delta}")
    if (rows[0].get("server_prompt_tps") or 0.0) < args.min_cold_prompt_tps:
        failures.append(f"cold prompt_tps too low: {rows[0].get('server_prompt_tps')}")
    for row in rows[1:]:
        if (row.get("cache_reuse_ratio") or 0.0) < args.min_reuse_ratio_after_first:
            failures.append(
                f"{row['label']} reuse ratio too low: {row.get('cache_reuse_ratio')}"
            )

    print(
        json.dumps(
            {
                "summary": {
                    "session_id": args.session_id,
                    "failed_delta": failed_delta,
                    "labels": [row.get("label") for row in rows],
                    "prompt_tps": [row.get("server_prompt_tps") for row in rows],
                    "suffix_tokens": [row.get("cache_suffix_tokens") for row in rows],
                    "reuse_tokens": [row.get("cache_reuse_tokens") for row in rows],
                    "reuse_ratio": [row.get("cache_reuse_ratio") for row in rows],
                    "ttft_s": [row.get("server_ttft_s") for row in rows],
                    "decode_tps": [row.get("server_decode_tps") for row in rows],
                    "failures": failures,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit("; ".join(failures))
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
