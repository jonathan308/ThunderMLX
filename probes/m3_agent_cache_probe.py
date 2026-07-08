#!/usr/bin/env python3
"""Agent-style long-context cache reuse probe.

This sends a large synthetic "workspace" prompt, then sends a follow-up with
the previous visible assistant answer included, matching OpenWebUI/agent chat
traffic. The second turn should reuse nearly all prior tokens and prefill only
the new suffix.
"""
import argparse
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"
TIER_RECORDS = {
    "150k": 4200,
    "250k": 6900,
    "350k": 9700,
}


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


def reset_cache():
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": "agent cache probe reset", "clear_memory": False},
        timeout=30,
    )


def compact_prompt_cache(cache):
    if not isinstance(cache, dict):
        return cache
    session_map = cache.get("session_map") or {}
    return {
        "loaded": cache.get("loaded"),
        "in_use": cache.get("in_use"),
        "cache_len": cache.get("cache_len"),
        "key_tokens": cache.get("key_tokens"),
        "session_id": cache.get("session_id"),
        "last_event": cache.get("last_event"),
        "last_prepare_event": cache.get("last_prepare_event"),
        "last_update_event": cache.get("last_update_event"),
        "resident_key": session_map.get("resident_key"),
        "resident_slots": session_map.get("resident_slots"),
        "resident_total_tokens": session_map.get("resident_total_tokens"),
    }


def compact_reset_response(payload):
    if not isinstance(payload, dict):
        return payload
    return {
        "ok": payload.get("ok"),
        "prompt_cache": compact_prompt_cache(payload.get("prompt_cache") or {}),
    }


def wait_idle(before_completed, timeout=90):
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


def stream_chat(name, messages, *, model, max_tokens, timeout, session_id=None, needle=None):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if session_id:
        payload["metadata"] = {"session_id": session_id}
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
    reasoning_parts = []
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
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    reasoning_parts.append(reasoning)

    elapsed = time.time() - started
    h = wait_idle(before, timeout=120)
    lr = h.get("last_request") or {}
    prepare = lr.get("prompt_cache_prepare") or {}
    text = "".join(content_parts)
    ks = h.get("kernel_stats") or {}
    gd = h.get("generation_defaults") or {}
    row = {
        "name": name,
        "client_elapsed_s": round(elapsed, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "content_chars": len("".join(content_parts)),
        "reasoning_chars": len("".join(reasoning_parts)),
        "output_preview": text[:280],
        "needle": needle,
        "needle_found": (needle in text) if needle else None,
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
        "cache_session_id": prepare.get("session_id"),
        "cache_session_source": prepare.get("session_source"),
        "request_session_id": prepare.get("request_session_id"),
        "protected_session_id": prepare.get("protected_session_id"),
        "protected_cache_tokens": prepare.get("protected_cache_tokens"),
        "sparse_topk_override": gd.get("sparse_topk_blocks_override"),
        "compact_decode_selected_len": ks.get("last_compact_decode_selected_len"),
        "compact_decode_total_len": ks.get("last_compact_decode_total_len"),
        "compact_decode_density": ks.get("last_compact_decode_density"),
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


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--records", type=int, default=1800)
    parser.add_argument(
        "--tier",
        choices=sorted(TIER_RECORDS),
        help="Approximate long-context prompt size. Overrides --records.",
    )
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--first-max-tokens", type=int, default=256)
    parser.add_argument("--followup-max-tokens", type=int, default=96)
    parser.add_argument("--session-id", default="agent-cache-probe-a")
    parser.add_argument(
        "--interleave-short-session",
        default="agent-cache-probe-b",
        help="Send one short request from this other session before the follow-up. Use empty string to disable.",
    )
    parser.add_argument(
        "--require-needle",
        action="store_true",
        help="fail if the cold long-context answer does not include the expected synthetic record path",
    )
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    if args.tier:
        args.records = TIER_RECORDS[args.tier]

    initial = health()
    print(json.dumps({
        "initial": {
            "status": initial.get("status"),
            "completed": initial.get("requests_completed"),
            "failed": initial.get("requests_failed"),
            "active": initial.get("active_request") is not None,
            "tier": args.tier,
            "records": args.records,
            "runtime_tuning": (initial.get("generation_defaults") or {}).get("runtime_tuning"),
        }
    }, sort_keys=True), flush=True)
    if not args.keep_cache:
        print(
            json.dumps({"reset": compact_reset_response(reset_cache())}, sort_keys=True),
            flush=True,
        )

    workspace = build_workspace(args.records)
    needle = (
        "src/module_19/file_001377.py"
        if args.records > 1377
        else None
    )
    user0 = {
        "role": "user",
        "content": (
            workspace
            + "\n\nYou are a coding agent. Using only the workspace facts above, "
            + "summarize the reliability implications in about 160 words and cite "
            + "the exact path for record 1377 if present."
        ),
    }
    cold, answer, _ = stream_chat(
        "agent_context_cold",
        [user0],
        model=args.model,
        max_tokens=args.first_max_tokens,
        timeout=1800,
        session_id=args.session_id,
        needle=needle,
    )
    interleaved = None
    if args.interleave_short_session:
        interleaved, _, _ = stream_chat(
            "other_session_short",
            [{"role": "user", "content": "Say OK in one short sentence."}],
            model=args.model,
            max_tokens=32,
            timeout=180,
            session_id=args.interleave_short_session,
        )
    followup_messages = [
        user0,
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Thanks. What record path did I ask you to cite?"},
    ]
    warm, _, final = stream_chat(
        "agent_context_visible_followup",
        followup_messages,
        model=args.model,
        max_tokens=args.followup_max_tokens,
        timeout=600,
        session_id=args.session_id,
        needle=needle,
    )
    print(json.dumps({
        "final": {
            "completed": final.get("requests_completed"),
            "failed": final.get("requests_failed"),
            "prompt_cache": compact_prompt_cache(final.get("prompt_cache") or {}),
            "cold_prompt_tokens": cold.get("server_prompt_tokens"),
            "followup_prompt_tokens": warm.get("server_prompt_tokens"),
            "followup_reuse_ratio": warm.get("cache_reuse_ratio"),
            "followup_suffix_tokens": warm.get("cache_suffix_tokens"),
            "cold_needle_found": cold.get("needle_found"),
            "followup_needle_found": warm.get("needle_found"),
            "interleaved": interleaved,
        }
    }, sort_keys=True), flush=True)

    if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
        raise SystemExit("probe saw request failure")
    if (warm.get("cache_reuse_ratio") or 0) < 0.95:
        raise SystemExit("follow-up cache reuse below 95%")
    if args.require_needle and needle and not cold.get("needle_found"):
        raise SystemExit("cold answer missed requested record path")


if __name__ == "__main__":
    main()
