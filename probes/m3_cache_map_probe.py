#!/usr/bin/env python3
"""Validate prompt-cache session-map and resident-slot switching.

The cache map keeps one distributed KV session live and can stash another in a
resident slot. This probe makes sure an interleaved OpenWebUI-style short chat
gets its own hot cache without evicting the active agent session, then proves
the agent session can be restored.
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
        {"reason": "cache map probe reset", "clear_memory": False},
        timeout=30,
    )


def wait_idle(before_completed, timeout=120):
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


def stream_chat(name, messages, *, model, max_tokens, session_id=None, timeout=900):
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
                piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
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
    shape = last.get("request_shape") or {}
    row = {
        "name": name,
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
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
        "cache_session_id": shape.get("cache_session_id"),
        "cache_session_source": shape.get("cache_session_source"),
        "protected_session_id": prepare.get("protected_session_id"),
        "protected_cache_tokens": prepare.get("protected_cache_tokens"),
        "restored_resident_slot": prepare.get("restored_resident_slot"),
        "restored_key": prepare.get("restored_key"),
        "stashed_previous_session": prepare.get("stashed_previous_session"),
        "failed": final.get("requests_failed"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row, final


def build_workspace(records):
    return "\n".join(
        f"cache_map/file_{i:05d}.py :: symbol_{i}(x) returns x + {i}"
        for i in range(records)
    )


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--records", type=int, default=600)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--session-a", default="cache-map-agent-a")
    parser.add_argument("--session-b", default="cache-map-short-b")
    parser.add_argument(
        "--implicit-sessions",
        action="store_true",
        help="omit metadata and require the server's auto conversation fingerprint to protect the resident cache",
    )
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    defaults = initial.get("generation_defaults") or {}
    resident_slots = int(defaults.get("prompt_cache_resident_slots") or 1)
    print(json.dumps({
        "initial": {
            "completed": initial.get("requests_completed"),
            "failed": initial.get("requests_failed"),
            "resident_slots": resident_slots,
        }
    }, sort_keys=True), flush=True)
    if resident_slots < 2:
        raise SystemExit("MLX_M3_PROMPT_CACHE_RESIDENT_SLOTS must be >= 2")
    print(json.dumps({"reset": reset_cache()}, sort_keys=True), flush=True)

    workspace = build_workspace(args.records)
    seed_messages = [{
        "role": "user",
        "content": (
            workspace
            + "\n\nSummarize the cache-map reliability implications in one short paragraph."
        ),
    }]
    seed, _ = stream_chat(
        "seed_agent_session",
        seed_messages,
        model=args.model,
        max_tokens=64,
        session_id=None if args.implicit_sessions else args.session_a,
        timeout=900,
    )
    if (seed.get("server_prompt_tokens") or 0) < 8192:
        raise SystemExit(f"seed prompt too small for session-protect test: {seed.get('server_prompt_tokens')}")

    short, final = stream_chat(
        "interleaved_short_session",
        [{"role": "user", "content": "Say OK."}],
        model=args.model,
        max_tokens=16,
        session_id=None if args.implicit_sessions else args.session_b,
        timeout=180,
    )
    pcache = final.get("prompt_cache") or {}
    session_map = pcache.get("session_map") or {}
    entries = session_map.get("entries") or []
    live = next((e for e in entries if e.get("resident")), None)
    slots = session_map.get("resident_slots") or []
    slot_keys = [s.get("key") for s in slots]
    row = {
        "live": live,
        "resident_slots": slots,
        "entries": entries,
        "cache_len": pcache.get("cache_len"),
        "short_action": short.get("cache_action"),
    }
    print(json.dumps({"session_map": row}, sort_keys=True), flush=True)

    prior_assistant = {"role": "assistant", "content": seed.get("content") or ""}
    if seed.get("reasoning"):
        prior_assistant["reasoning"] = seed.get("reasoning")
        prior_assistant["reasoning_content"] = seed.get("reasoning")
    follow, final = stream_chat(
        "agent_followup_after_short",
        seed_messages + [
            prior_assistant,
            {"role": "user", "content": "Give one follow-up sentence about the original cache-map prompt."},
        ],
        model=args.model,
        max_tokens=32,
        session_id=None if args.implicit_sessions else args.session_a,
        timeout=300,
    )

    failures = []
    if short.get("cache_action") != "session_switch_stash_rebuild":
        failures.append(f"short action={short.get('cache_action')}")
    expected_resident_session = seed.get("cache_session_id") if args.implicit_sessions else args.session_a
    short_session = short.get("cache_session_id")
    if not live or live.get("session_id") != short_session:
        failures.append(f"live session after short={live}; expected {short_session}")
    if expected_resident_session not in slot_keys:
        failures.append(f"agent session not stashed: expected {expected_resident_session}, slots={slot_keys}")
    if not follow.get("restored_resident_slot"):
        failures.append(f"agent follow-up did not restore resident slot: {follow}")
    if (follow.get("cache_reuse_ratio") or 0) < 0.95:
        failures.append(f"agent follow-up reuse below 95%: {follow.get('cache_reuse_ratio')}")
    if (follow.get("cache_suffix_tokens") or 999999) > 512:
        failures.append(f"agent follow-up suffix too large: {follow.get('cache_suffix_tokens')}")
    if args.implicit_sessions:
        if seed.get("cache_session_source") != "auto.conversation_fingerprint":
            failures.append(f"seed session source={seed.get('cache_session_source')}")
        if short.get("cache_session_source") != "auto.conversation_fingerprint":
            failures.append(f"short session source={short.get('cache_session_source')}")
        if not seed.get("cache_session_id") or seed.get("cache_session_id") == short.get("cache_session_id"):
            failures.append(
                "auto session ids missing or not distinct: "
                f"seed={seed.get('cache_session_id')} short={short.get('cache_session_id')}"
            )
    if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
        failures.append(f"server failures increased to {final.get('requests_failed')}")
    if failures:
        raise SystemExit("; ".join(failures))
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
