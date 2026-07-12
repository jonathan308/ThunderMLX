#!/usr/bin/env python3
"""Multi-turn MiniMax-M3 cache/speed probe for OpenWebUI-style chats.

This intentionally sends assistant history back with visible content only, which
matches clients that do not preserve reasoning_content. Healthy behavior after a
thinking response is high cache reuse and tiny suffix/missed-token counts on
short follow-ups such as "thanks" and "ok cool".
"""
import argparse
import json
import sys
import time
import urllib.error
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


def reset_cache():
    try:
        return request_json(
            "POST",
            "/admin/prompt-cache/reset",
            {"reason": "turn probe reset", "clear_memory": False},
            timeout=30,
        )
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "body": exc.read().decode("utf-8", "replace")}


def stream_chat(
    messages,
    *,
    model="Minimax-M3",
    max_tokens=384,
    temperature=0,
    session_id=None,
    timeout=1200,
):
    before_completed = health().get("requests_completed", 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if session_id:
        payload["metadata"] = {"session_id": session_id, "source": "m3_turn_probe"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=data,
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
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece and first_piece_s is None:
                    first_piece_s = time.time() - started
                if delta.get("content"):
                    content_parts.append(delta["content"])
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    reasoning_parts.append(reasoning)
    deadline = time.time() + 30
    while time.time() < deadline:
        h = health()
        pc = h.get("prompt_cache") or {}
        pe = pc.get("last_prepare_event") or {}
        ue = pc.get("last_update_event") or {}
        prepare_at = float(pe.get("at") or 0)
        update_at = float(ue.get("at") or 0)
        prewarm_published = (
            pe.get("action") != "prewarm_start"
            or (ue.get("action") and update_at >= prepare_at)
        )
        if (
            not h.get("active_request")
            and not pc.get("in_use")
            and h.get("requests_completed", 0) > before_completed
            and prewarm_published
        ):
            break
        time.sleep(0.2)
    return {
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }


def summarize(label, chat_result=None):
    h = health()
    pc = h.get("prompt_cache") or {}
    pe = pc.get("last_prepare_event") or {}
    pu = pc.get("last_update_event") or {}
    visible_pe = pe
    if pe.get("action") == "prewarm_start" and pu.get("action"):
        visible_pe = {**pe, "action": pu.get("action")}
    lr = h.get("last_request") or {}
    gd = h.get("generation_defaults") or {}
    row = {
        "label": label,
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "server_ttft_s": lr.get("first_token_s"),
        "server_prompt_tps": lr.get("prompt_tps"),
        "server_decode_tps": lr.get("decode_tps"),
        "server_request_tps": lr.get("tps"),
        "server_prompt_tokens": lr.get("prompt_tokens"),
        "server_cached_tokens": lr.get("cached_tokens"),
        "server_output_tokens": lr.get("tokens"),
        "cache_action": visible_pe.get("action"),
        "cache_reuse_tokens": visible_pe.get("reuse_tokens"),
        "cache_prompt_tokens": visible_pe.get("prompt_tokens"),
        "cache_suffix_tokens": visible_pe.get("suffix_tokens"),
        "cache_reuse_ratio": visible_pe.get("reuse_ratio"),
        "cache_missed_tokens": visible_pe.get("missed_tokens"),
        "cache_miss_reason": visible_pe.get("miss_reason"),
        "cache_previous_generated_tokens": visible_pe.get("previous_generated_tokens"),
        "cache_reused_generated_tokens": visible_pe.get("reused_generated_tokens"),
        "cache_generated_reuse_ratio": visible_pe.get("generated_reuse_ratio"),
        "cache_would_reprocess_tokens": visible_pe.get("would_reprocess_tokens"),
        "cache_last_update_action": pu.get("action"),
        "cache_last_update_reason": pu.get("reason"),
        "cache_update_generated_key_tokens": pu.get("generated_key_tokens"),
        "cache_update_generated_key_truncated": pu.get("generated_key_truncated"),
        "cache_update_exact_generated_ids": pu.get("exact_generated_ids"),
        "cache_key_tokens": pc.get("key_tokens"),
        "cache_len": pc.get("cache_len"),
        "prewarm_enabled": gd.get("visible_transcript_prewarm"),
        "prewarm_min_generated": gd.get(
            "effective_visible_transcript_prewarm_min_generated",
            gd.get("visible_transcript_prewarm_min_generated"),
        ),
    }
    if chat_result:
        row.update({
            "client_elapsed_s": chat_result["client_elapsed_s"],
            "client_ttft_s": chat_result["client_ttft_s"],
            "chunks": chat_result["chunks"],
            "content_chars": len(chat_result["content"]),
            "reasoning_chars": len(chat_result["reasoning"]),
        })
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3")
    parser.add_argument(
        "--first-max-tokens",
        type=int,
        default=768,
        help=(
            "Budget for the first thinking turn. Keep this high enough for a "
            "complete reasoning+content response; capped reasoning-only output "
            "is intentionally rejected by the cache guard."
        ),
    )
    parser.add_argument("--followup-max-tokens", type=int, default=96)
    parser.add_argument(
        "--session-id",
        default="",
        help="Stable metadata.session_id for cache-isolated probe runs",
    )
    parser.add_argument("--keep-cache", action="store_true", help="Do not reset prompt cache before the probe")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    print(json.dumps({"initial": summarize("initial")}, sort_keys=True), flush=True)
    if not args.keep_cache:
        print(json.dumps({"reset": reset_cache()}, sort_keys=True), flush=True)

    user0 = {
        "role": "user",
        "content": (
            "Cache probe: reason in one short sentence, then give a final "
            "answer with exactly two concise bullets. Each bullet must be under "
            "18 words. Explain how a stable OpenAI-compatible MiniMax-M3 "
            "gateway preserves cache reuse and tool-call compatibility."
        ),
    }
    first = stream_chat(
        [user0],
        model=args.model,
        max_tokens=args.first_max_tokens,
        session_id=args.session_id or None,
    )
    summarize("turn1_long_thinking", first)

    assistant_visible_only = {"role": "assistant", "content": first["content"]}
    user1 = {"role": "user", "content": "Thanks, give me two crisp takeaways."}
    second = stream_chat(
        [user0, assistant_visible_only, user1],
        model=args.model,
        max_tokens=args.followup_max_tokens,
        session_id=args.session_id or None,
    )
    row2 = summarize("turn2_thanks_visible_only", second)

    assistant2 = {"role": "assistant", "content": second["content"]}
    user2 = {"role": "user", "content": "ok cool, now summarize the caching risk in one sentence."}
    third = stream_chat(
        [user0, assistant_visible_only, user1, assistant2, user2],
        model=args.model,
        max_tokens=args.followup_max_tokens,
        session_id=args.session_id or None,
    )
    row3 = summarize("turn3_ok_cool_visible_only", third)

    final = health()
    if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
        raise SystemExit("probe saw a request failure")
    def healthy_followup(row):
        server_prompt_tokens = int(row.get("server_prompt_tokens") or 0)
        server_cached_tokens = int(row.get("server_cached_tokens") or 0)
        server_reuse_ratio = (
            server_cached_tokens / server_prompt_tokens
            if server_prompt_tokens > 0
            else 0.0
        )
        reuse_ok = (
            (row.get("cache_reuse_ratio") or 0) >= 0.9
            and (row.get("cache_missed_tokens") or 0) <= 512
        )
        fast_cached_followup = (
            (row.get("server_ttft_s") or 999) <= 1.0
            and server_reuse_ratio >= 0.9
            and (row.get("failed") or 0) == initial.get("requests_failed", 0)
        )
        prewarm_after_request = row.get("cache_action") == "prewarm_visible_transcript"
        return reuse_ok or (prewarm_after_request and fast_cached_followup)

    poor = [row for row in (row2, row3) if not healthy_followup(row)]
    if poor:
        raise SystemExit(f"hot-cache follow-up reuse is still poor: {poor}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {exc!r}", file=sys.stderr, flush=True)
        raise
