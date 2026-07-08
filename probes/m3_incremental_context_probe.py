#!/usr/bin/env python3
"""Grow a long agent context incrementally and validate cache reuse.

This avoids the unsafe one-shot 350k cold prompt path. Each turn appends a new
workspace chunk to the same chat session, matching how coding agents usually
grow context over time. Healthy behavior is high cache reuse after the first
chunk, bounded suffix processing per turn, and a fast final warm follow-up.
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


def reset_cache():
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": "incremental context probe reset", "clear_memory": False},
        timeout=30,
    )


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


def stream_chat(name, messages, *, model, max_tokens, timeout, session_id):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metadata": {"session_id": session_id, "source": "m3_incremental_context_probe"},
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
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            obj = json.loads(item)
            chunks += 1
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            piece = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
            )
            if piece and first_piece_s is None:
                first_piece_s = time.time() - started
            if piece:
                parts.append(piece)
    h = wait_idle(before, timeout=max(180, timeout))
    last = h.get("last_request") or {}
    pcache = h.get("prompt_cache") or {}
    prepare = last.get("prompt_cache_prepare") or pcache.get("last_prepare_event") or {}
    text = "".join(parts)
    row = {
        "name": name,
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "chars": len(text),
        "server_ttft_s": last.get("first_token_s"),
        "server_elapsed_s": last.get("elapsed_s"),
        "server_total_elapsed_s": last.get("total_elapsed_s"),
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
        "cache_miss_reason": prepare.get("miss_reason"),
        "cache_len": pcache.get("cache_len"),
        "cache_session_id": prepare.get("session_id"),
        "failed": h.get("requests_failed"),
        "preview": text[:220],
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row, text, h


def build_records(start, count):
    return "\n".join(
        (
            f"repo/service_{i % 137:03d}/feature_{i:06d}.py :: "
            f"class Feature{i} handles tenant={i % 23}, shard={i % 17}; "
            f"expected_checksum={((i * 104729) ^ (i % 97)) % 1000003}"
        )
        for i in range(start, start + count)
    )


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--session-id", default=f"incremental-context-{int(time.time())}")
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--records-per-chunk", type=int, default=2100)
    parser.add_argument("--ack-tokens", type=int, default=24)
    parser.add_argument("--final-tokens", type=int, default=256)
    parser.add_argument("--turn-timeout", type=int, default=1200)
    parser.add_argument("--final-timeout", type=int, default=900)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--min-final-tokens", type=int, default=200000)
    parser.add_argument("--min-final-reuse-ratio", type=float, default=0.95)
    parser.add_argument("--max-suffix-tokens", type=int, default=90000)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    print(
        json.dumps(
            {
                "initial": {
                    "completed": initial.get("requests_completed"),
                    "failed": initial.get("requests_failed"),
                    "session_id": args.session_id,
                    "chunks": args.chunks,
                    "records_per_chunk": args.records_per_chunk,
                    "defaults": initial.get("generation_defaults"),
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
                "You are a coding agent validating a large local context. "
                "Acknowledge each workspace chunk tersely and preserve exact paths."
            ),
        }
    ]
    rows = []
    total_records = 0
    initial_failed = int(initial.get("requests_failed") or 0)
    for idx in range(args.chunks):
        start = idx * args.records_per_chunk
        total_records += args.records_per_chunk
        chunk = build_records(start, args.records_per_chunk)
        sentinel_index = start + args.records_per_chunk - 1
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Workspace chunk {idx + 1}/{args.chunks}. Store these records "
                    f"for later code reasoning. The last record index is {sentinel_index}.\n\n"
                    + chunk
                    + "\n\nReply with one sentence: chunk acknowledged and the last record path."
                ),
            }
        )
        row, answer, _ = stream_chat(
            f"chunk_{idx + 1}",
            messages,
            model=args.model,
            max_tokens=args.ack_tokens,
            timeout=args.turn_timeout,
            session_id=args.session_id,
        )
        rows.append(row)
        messages.append({"role": "assistant", "content": answer})
        if int(row.get("failed") or 0) > initial_failed:
            raise SystemExit(f"request failure after chunk {idx + 1}")
        if idx > 0 and (row.get("cache_suffix_tokens") or 10**9) > args.max_suffix_tokens:
            raise SystemExit(f"chunk {idx + 1} suffix too large: {row.get('cache_suffix_tokens')}")

    final_index = total_records - 1
    final_path = f"repo/service_{final_index % 137:03d}/feature_{final_index:06d}.py"
    messages.append(
        {
            "role": "user",
            "content": (
                "Final validation: explain in about 180 words why this incremental "
                "cache growth is safer than one giant cold prompt. Include the exact "
                f"final record path {final_path}."
            ),
        }
    )
    final_row, final_answer, final_health = stream_chat(
        "final_cached_decode",
        messages,
        model=args.model,
        max_tokens=args.final_tokens,
        timeout=args.final_timeout,
        session_id=args.session_id,
    )
    rows.append(final_row)

    summary = {
        "total_records": total_records,
        "final_path": final_path,
        "final_path_found": final_path in final_answer,
        "final_total_prompt_tokens": final_row.get("cache_prompt_tokens"),
        "final_processed_prompt_tokens": final_row.get("server_prompt_tokens"),
        "final_reuse_ratio": final_row.get("cache_reuse_ratio"),
        "final_suffix_tokens": final_row.get("cache_suffix_tokens"),
        "final_ttft_s": final_row.get("server_ttft_s"),
        "final_decode_tps": final_row.get("server_decode_tps"),
        "failed_delta": int(final_health.get("requests_failed") or 0) - initial_failed,
        "rows": [
            {
                "name": row.get("name"),
                "total_prompt_tokens": row.get("cache_prompt_tokens"),
                "processed_prompt_tokens": row.get("server_prompt_tokens"),
                "prompt_tps": row.get("server_prompt_tps"),
                "ttft": row.get("server_ttft_s"),
                "decode_tps": row.get("server_decode_tps"),
                "reuse_ratio": row.get("cache_reuse_ratio"),
                "suffix_tokens": row.get("cache_suffix_tokens"),
            }
            for row in rows
        ],
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if summary["failed_delta"] > 0:
        raise SystemExit("probe saw request failure")
    if (final_row.get("cache_prompt_tokens") or 0) < args.min_final_tokens:
        raise SystemExit("final prompt token target was not reached")
    if (final_row.get("cache_reuse_ratio") or 0) < args.min_final_reuse_ratio:
        raise SystemExit("final follow-up cache reuse below threshold")


if __name__ == "__main__":
    main()
