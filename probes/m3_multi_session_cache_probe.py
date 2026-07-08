#!/usr/bin/env python3
"""Validate in-memory multi-session prompt-cache slot restore.

This is intentionally stricter than the short-session preserve probe:

1. Build a long session A cache.
2. Build a different long session B cache, which should stash A when
   MLX_M3_PROMPT_CACHE_RESIDENT_SLOTS >= 2.
3. Send a visible-history follow-up for A and require A to restore from the
   in-memory resident slot, processing only the new suffix.
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


def reset_cache():
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": "multi-session cache probe reset", "clear_memory": False},
        timeout=30,
    )


def wait_idle(before_completed, timeout=180):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        pcache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and not pcache.get("in_use")
            and int(last.get("requests_completed") or 0) > before_completed
        ):
            return last
        time.sleep(0.25)
    return last or health()


def stream_chat(name, messages, *, model, max_tokens, session_id, timeout):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metadata": {"session_id": session_id},
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
    content = []
    reasoning = []
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
                    content.append(delta["content"])
                r = delta.get("reasoning") or delta.get("reasoning_content")
                if r:
                    reasoning.append(r)
    final = wait_idle(before)
    last = final.get("last_request") or {}
    prepare = last.get("prompt_cache_prepare") or {}
    row = {
        "name": name,
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "content_chars": len("".join(content)),
        "reasoning_chars": len("".join(reasoning)),
        "server_ttft_s": last.get("first_token_s"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_decode_tps": last.get("decode_tps"),
        "server_tokens": last.get("tokens"),
        "cache_action": prepare.get("action"),
        "cache_prompt_tokens": prepare.get("prompt_tokens"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_missed_tokens": prepare.get("missed_tokens"),
        "restored_resident_slot": prepare.get("restored_resident_slot"),
        "restored_key": prepare.get("restored_key"),
        "restored_cache_len": prepare.get("restored_cache_len"),
        "failed": final.get("requests_failed"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row, "".join(content), final


def build_workspace(prefix, records):
    return "\n".join(
        (
            f"{prefix}/module_{i % 41:02d}/file_{i:05d}.py :: "
            f"symbol_{prefix}_{i}(x) returns x + {i}; "
            f"case_{i} expects {i + (i % 17)}"
        )
        for i in range(records)
    )


def resident_slot_keys(health_doc):
    pcache = health_doc.get("prompt_cache") or {}
    smap = pcache.get("session_map") or {}
    return [row.get("key") for row in smap.get("resident_slots") or []]


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--records", type=int, default=600)
    parser.add_argument("--session-a", default="multi-session-A")
    parser.add_argument("--session-b", default="multi-session-B")
    parser.add_argument("--first-max-tokens", type=int, default=96)
    parser.add_argument("--followup-max-tokens", type=int, default=64)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    defaults = initial.get("generation_defaults") or {}
    print(json.dumps({
        "initial": {
            "status": initial.get("status"),
            "completed": initial.get("requests_completed"),
            "failed": initial.get("requests_failed"),
            "resident_slots": defaults.get("prompt_cache_resident_slots"),
            "records": args.records,
        }
    }, sort_keys=True), flush=True)
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    if int(defaults.get("prompt_cache_resident_slots") or 1) < 2:
        raise SystemExit("MLX_M3_PROMPT_CACHE_RESIDENT_SLOTS must be >= 2")
    if not args.keep_cache:
        print(json.dumps({"reset": reset_cache()}, sort_keys=True), flush=True)

    user_a = {
        "role": "user",
        "content": (
            build_workspace("alpha", args.records)
            + "\n\nSummarize alpha reliability facts in about 80 words."
        ),
    }
    a_seed, a_answer, _ = stream_chat(
        "session_a_seed",
        [user_a],
        model=args.model,
        max_tokens=args.first_max_tokens,
        session_id=args.session_a,
        timeout=900,
    )

    user_b = {
        "role": "user",
        "content": (
            build_workspace("bravo", args.records)
            + "\n\nSummarize bravo reliability facts in about 80 words."
        ),
    }
    b_seed, _, after_b = stream_chat(
        "session_b_seed",
        [user_b],
        model=args.model,
        max_tokens=args.first_max_tokens,
        session_id=args.session_b,
        timeout=900,
    )
    print(json.dumps({
        "after_b_resident_slots": resident_slot_keys(after_b),
    }, sort_keys=True), flush=True)

    a_follow, _, final = stream_chat(
        "session_a_followup_after_b",
        [
            user_a,
            {"role": "assistant", "content": a_answer},
            {"role": "user", "content": "Thanks. Give two alpha follow-up bullets."},
        ],
        model=args.model,
        max_tokens=args.followup_max_tokens,
        session_id=args.session_a,
        timeout=300,
    )
    print(json.dumps({
        "final": {
            "completed": final.get("requests_completed"),
            "failed": final.get("requests_failed"),
            "resident_slots": resident_slot_keys(final),
            "session_map": (final.get("prompt_cache") or {}).get("session_map"),
        }
    }, sort_keys=True), flush=True)

    failures = []
    if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
        failures.append("request failure count increased")
    if args.session_a not in resident_slot_keys(after_b):
        failures.append("session A was not stashed after session B")
    if not a_follow.get("restored_resident_slot"):
        failures.append(f"A follow-up did not restore a resident slot: {a_follow}")
    if (a_follow.get("cache_reuse_ratio") or 0) < 0.95:
        failures.append(f"A follow-up reuse below 95%: {a_follow.get('cache_reuse_ratio')}")
    if (a_follow.get("cache_suffix_tokens") or 999999) > 512:
        failures.append(f"A follow-up suffix too large: {a_follow.get('cache_suffix_tokens')}")
    if (a_seed.get("server_prompt_tokens") or 0) < 8192:
        failures.append("session A prompt was too small to prove long-cache behavior")
    if (b_seed.get("server_prompt_tokens") or 0) < 8192:
        failures.append("session B prompt was too small to prove long-cache behavior")
    if failures:
        raise SystemExit("; ".join(failures))
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
