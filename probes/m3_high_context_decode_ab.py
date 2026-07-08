#!/usr/bin/env python3
"""A/B high-context cached decode cadence without rebuilding the cache each time."""
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
    return request_json("POST", "/admin/runtime-tuning", {"values": values}, timeout=30)


def reset_cache(reason):
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": reason, "clear_memory": False},
        timeout=30,
    )


def compact_admin_response(payload):
    """Keep probe logs focused on tuning effects instead of full status dumps."""
    if not isinstance(payload, dict):
        return payload
    defaults = payload.get("generation_defaults") or {}
    model_tuning = payload.get("model_tuning") or {}
    prompt_cache = payload.get("prompt_cache") or {}
    runtime = payload.get("runtime_tuning") or defaults.get("runtime_tuning")
    compact = {
        "ok": payload.get("ok"),
        "changed": payload.get("changed"),
        "cache_reset_before_tuning": payload.get("cache_reset_before_tuning"),
        "runtime_tuning": runtime,
    }
    if model_tuning:
        compact["model_tuning"] = model_tuning
    if prompt_cache:
        compact["prompt_cache"] = {
            "loaded": prompt_cache.get("loaded"),
            "in_use": prompt_cache.get("in_use"),
            "cache_len": prompt_cache.get("cache_len"),
            "key_tokens": prompt_cache.get("key_tokens"),
            "session_id": prompt_cache.get("session_id"),
            "last_event": prompt_cache.get("last_event"),
        }
    return {k: v for k, v in compact.items() if v is not None}


def wait_idle(before_completed, timeout=120):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health(timeout=5)
        pcache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and int(last.get("requests_completed") or 0) > before_completed
            and not pcache.get("in_use")
        ):
            return last
        time.sleep(0.25)
    return last or health(timeout=5)


def summarize_last(name, started, first_piece_s, chunks, text, h, extra=None):
    last = h.get("last_request") or {}
    pcache = h.get("prompt_cache") or {}
    prepare = last.get("prompt_cache_prepare") or pcache.get("last_prepare_event") or {}
    kernel = h.get("kernel_stats") or {}
    defaults = h.get("generation_defaults") or {}
    row = {
        "name": name,
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "chars": len(text),
        "server_ttft_s": last.get("first_token_s"),
        "server_elapsed_s": last.get("elapsed_s"),
        "server_total_elapsed_s": last.get("total_elapsed_s"),
        "server_post_generation_s": last.get("post_generation_s"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_decode_tps": last.get("decode_tps"),
        "server_tokens": last.get("tokens"),
        "cache_action": prepare.get("action"),
        "cache_prompt_tokens": prepare.get("prompt_tokens"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_miss_reason": prepare.get("miss_reason"),
        "effective_long_context_decode_eval_every": (
            defaults.get("effective_long_context_decode_eval_every")
        ),
        "effective_sparse_topk_blocks": defaults.get("effective_sparse_topk_blocks"),
        "decode_topk_reuse_tokens": defaults.get("decode_topk_reuse_tokens"),
        "compact_decode_sort_topk": defaults.get("compact_decode_sort_topk"),
        "adaptive_long_context_decode_eval": defaults.get("adaptive_long_context_decode_eval"),
        "high_context_decode_eval_every": defaults.get("high_context_decode_eval_every"),
        "compact_decode_selected_len": kernel.get("last_compact_decode_selected_len"),
        "compact_decode_total_len": kernel.get("last_compact_decode_total_len"),
        "compact_decode_density": kernel.get("last_compact_decode_density"),
        "compact_decode_calls": kernel.get("compact_decode_calls"),
        "prefill_attention_calls": kernel.get("prefill_attention_calls"),
        "prefill_eligible": kernel.get("prefill_eligible"),
        "topk_fallback": kernel.get("topk_fallback"),
        "topk_native": kernel.get("topk_native"),
        "last_msa_k1_impl": kernel.get("last_msa_k1_impl"),
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "output_preview": text[:180],
    }
    if extra:
        row.update(extra)
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def stream_chat(name, content, *, model, session_id, max_tokens, timeout, extra=None):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metadata": {"session_id": session_id, "source": "m3_high_context_decode_ab"},
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
            chunks += 1
    h = wait_idle(before, timeout=max(120, timeout))
    return summarize_last(name, started, first_piece_s, chunks, "".join(parts), h, extra=extra)


def build_prefix(records):
    return "\n".join(
        f"FILE-{i:06d}: function_{i}(x) returns x + {i}; checksum={i * 17 % 100000}"
        for i in range(records)
    )


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--records", type=int, default=11204)
    parser.add_argument("--cadences", default="1,2,3")
    parser.add_argument(
        "--topk-values",
        default="",
        help="Optional comma-separated sparse_topk_blocks values to A/B at each cadence",
    )
    parser.add_argument(
        "--decode-reuse-values",
        default="",
        help="Optional comma-separated decode_topk_reuse_tokens values to A/B at each cadence",
    )
    parser.add_argument(
        "--compact-sort-values",
        default="",
        help="Optional comma-separated compact_decode_sort_topk values to A/B at each cadence",
    )
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--cold-timeout", type=int, default=1500)
    parser.add_argument("--decode-timeout", type=int, default=900)
    parser.add_argument("--skip-reset", action="store_true", help="Reuse existing cache/session state")
    parser.add_argument("--skip-seed", action="store_true", help="Skip cold/warm seed and run decode passes only")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    cadences = [int(item.strip()) for item in args.cadences.split(",") if item.strip()]
    topk_values = [
        int(item.strip()) for item in args.topk_values.split(",") if item.strip()
    ] or [None]
    decode_reuse_values = [
        int(item.strip()) for item in args.decode_reuse_values.split(",") if item.strip()
    ] or [None]
    compact_sort_values = [
        int(item.strip()) for item in args.compact_sort_values.split(",") if item.strip()
    ] or [None]
    session_id = args.session_id or f"decode-ab-{int(time.time())}"
    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    defaults = initial.get("generation_defaults") or {}
    original = defaults.get("runtime_tuning") or {}
    print(
        json.dumps(
            {
                "initial": {
                    "completed": initial.get("requests_completed"),
                    "failed": initial.get("requests_failed"),
                    "unsafe_runtime_tuning_allowed": defaults.get("unsafe_runtime_tuning_allowed"),
                    "runtime_tuning": original,
                    "session_id": session_id,
                    "records": args.records,
                    "cadences": cadences,
                    "topk_values": topk_values,
                    "decode_reuse_values": decode_reuse_values,
                    "compact_sort_values": compact_sort_values,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.skip_reset:
        reset = reset_cache("high context decode cadence ab")
        print(json.dumps({"reset": compact_admin_response(reset)}, sort_keys=True), flush=True)

    prefix = build_prefix(args.records)
    needle = "FILE-001377"
    seed_prompt = (
        prefix
        + "\n\nUsing only the records above, return the exact line beginning "
        + f"with {needle}, then stop."
    )
    rows = []
    try:
        if not args.skip_seed:
            rows.append(
                stream_chat(
                    "cold_seed",
                    seed_prompt,
                    model=args.model,
                    session_id=session_id,
                    max_tokens=96,
                    timeout=args.cold_timeout,
                )
            )
            rows.append(
                stream_chat(
                    "warm_seed",
                    seed_prompt,
                    model=args.model,
                    session_id=session_id,
                    max_tokens=96,
                    timeout=300,
                )
            )
        for topk in topk_values:
            for reuse in decode_reuse_values:
                for compact_sort in compact_sort_values:
                    for cadence in cadences:
                        values = {
                            "adaptive_long_context_decode_eval": 0,
                            "long_context_decode_eval_every": cadence,
                            "high_context_decode_eval_every": cadence,
                            "mid_context_decode_eval_every": cadence,
                        }
                        if topk is not None:
                            values["sparse_topk_blocks"] = topk
                        if reuse is not None:
                            values["decode_topk_reuse_tokens"] = reuse
                        if compact_sort is not None:
                            values["compact_decode_sort_topk"] = compact_sort
                        tuned = runtime_tuning(values)
                        parts = [f"cadence {cadence}"]
                        if topk is not None:
                            parts.append(f"topk {topk}")
                        if reuse is not None:
                            parts.append(f"reuse {reuse}")
                        if compact_sort is not None:
                            parts.append(f"sort {compact_sort}")
                        label = " ".join(parts)
                        print(
                            json.dumps(
                                {
                                    "cadence": cadence,
                                    "topk": topk,
                                    "decode_reuse_tokens": reuse,
                                    "compact_sort": compact_sort,
                                    "runtime_tuning": compact_admin_response(tuned),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        decode_prompt = (
                            prefix
                            + f"\n\n{label}: write about 220 words on high-context "
                            + f"decode stability, include {needle} if it exists, then stop."
                        )
                        row = stream_chat(
                            (
                                f"decode_topk_{topk}_reuse_{reuse}_sort_{compact_sort}_cadence_{cadence}"
                            ),
                            decode_prompt,
                            model=args.model,
                            session_id=session_id,
                            max_tokens=args.max_tokens,
                            timeout=args.decode_timeout,
                            extra={
                                "cadence": cadence,
                                "sparse_topk_blocks": topk,
                                "decode_reuse_tokens": reuse,
                                "compact_sort": compact_sort,
                            },
                        )
                        rows.append(row)
    finally:
        if original:
            try:
                restore = runtime_tuning(original)
            except Exception as exc:
                restore = {"ok": False, "error": repr(exc)}
            print(
                json.dumps({"restore": compact_admin_response(restore)}, sort_keys=True),
                flush=True,
            )

    final = health()
    summary = {
        "rows": [
            {
                "name": row.get("name"),
                "cadence": row.get("cadence"),
                "sparse_topk_blocks": row.get("sparse_topk_blocks"),
                "effective_sparse_topk_blocks": row.get("effective_sparse_topk_blocks"),
                "decode_reuse_tokens": row.get("decode_reuse_tokens"),
                "effective_decode_reuse_tokens": row.get("decode_topk_reuse_tokens"),
                "compact_sort": row.get("compact_sort"),
                "effective_compact_sort": row.get("compact_decode_sort_topk"),
                "prompt_tokens": row.get("server_prompt_tokens"),
                "decode_tps": row.get("server_decode_tps"),
                "ttft": row.get("server_ttft_s"),
                "cache_reuse_ratio": row.get("cache_reuse_ratio"),
                "suffix_tokens": row.get("cache_suffix_tokens"),
                "compact_decode_selected_len": row.get("compact_decode_selected_len"),
                "compact_decode_density": row.get("compact_decode_density"),
                "failed": row.get("failed"),
            }
            for row in rows
        ],
        "completed_delta": int(final.get("requests_completed") or 0)
        - int(initial.get("requests_completed") or 0),
        "failed_delta": int(final.get("requests_failed") or 0)
        - int(initial.get("requests_failed") or 0),
        "final_runtime_tuning": (final.get("generation_defaults") or {}).get("runtime_tuning"),
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if summary["failed_delta"] > 0:
        raise SystemExit("probe saw request failure")


if __name__ == "__main__":
    main()
