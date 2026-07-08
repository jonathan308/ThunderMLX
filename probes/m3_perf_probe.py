#!/usr/bin/env python3
"""Small repeatable MiniMax-M3 endpoint benchmark.

Measures client-observed streaming TTFT plus server-reported total/decode TPS.
The goal is quick A/B testing of launch knobs without changing the server.
"""
import json
import argparse
import sys
import time
import urllib.error
import urllib.request


BASE = "http://127.0.0.1:8080"


def health(timeout=5):
    with urllib.request.urlopen(BASE + "/health", timeout=timeout) as r:
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


def compact_for_log(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "prompt_cache" and isinstance(item, dict):
                out[key] = compact_prompt_cache_for_log(item)
                continue
            if key == "entries" and isinstance(item, list):
                out["entry_count"] = len(item)
                complete = sum(
                    1 for row in item
                    if isinstance(row, dict) and row.get("complete")
                )
                ready = sum(
                    1 for row in item
                    if isinstance(row, dict) and row.get("restore_ready")
                )
                if complete or ready:
                    out["complete_entries"] = complete
                    out["restore_ready_entries"] = ready
                continue
            out[key] = compact_for_log(item)
        return out
    if isinstance(value, list):
        if len(value) <= 8:
            return [compact_for_log(item) for item in value]
        return {
            "count": len(value),
            "first": compact_for_log(value[0]),
            "last": compact_for_log(value[-1]),
        }
    return value


def compact_prompt_cache_for_log(pcache):
    ssd = pcache.get("ssd") or {}
    session_map = pcache.get("session_map") or {}
    keepwarm = pcache.get("keepwarm") or {}
    return {
        "enabled": pcache.get("enabled"),
        "loaded": pcache.get("loaded"),
        "in_use": pcache.get("in_use"),
        "cache_len": pcache.get("cache_len"),
        "key_tokens": pcache.get("key_tokens"),
        "session_id": pcache.get("session_id"),
        "session_source": pcache.get("session_source"),
        "last_prepare_action": (pcache.get("last_prepare_event") or {}).get("action"),
        "last_update_action": (pcache.get("last_update_event") or {}).get("action"),
        "session_map": {
            "entry_count": session_map.get("entry_count"),
            "resident_key": session_map.get("resident_key"),
            "resident_slots_max": session_map.get("resident_slots_max"),
            "resident_total_tokens": session_map.get("resident_total_tokens"),
            "resident_total_max_tokens": session_map.get("resident_total_max_tokens"),
        },
        "keepwarm": {
            "enabled": keepwarm.get("enabled"),
            "mode": keepwarm.get("mode"),
            "count": keepwarm.get("count"),
            "last_at": keepwarm.get("last_at"),
        },
        "ssd": {
            "enabled": ssd.get("enabled"),
            "restore_enabled": ssd.get("restore_enabled"),
            "auto_save": ssd.get("auto_save"),
            "mode": ssd.get("mode"),
            "entry_count": ssd.get("entry_count"),
            "total_bytes": ssd.get("total_bytes"),
            "last_saved_tokens": ssd.get("last_saved_tokens"),
            "last_restored_tokens": ssd.get("last_restored_tokens"),
            "last_restore_miss_reason": ssd.get("last_restore_miss_reason"),
            "last_error": ssd.get("last_error"),
        },
    }


def print_event(name, payload):
    print(json.dumps({name: compact_for_log(payload)}, sort_keys=True), flush=True)


def wait_idle(timeout=90):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health(timeout=5)
        pcache = last.get("prompt_cache") or {}
        if (
            last.get("active_request") is None
            and int(last.get("queue_depth") or 0) == 0
            and not pcache.get("in_use")
        ):
            return last
        time.sleep(1)
    raise TimeoutError(f"server did not become idle before reset: {last}")


def reset_cache(reason="perf probe reset"):
    last_error = None
    for _ in range(30):
        wait_idle(timeout=30)
        try:
            return post_json(
                "/admin/prompt-cache/reset",
                {"reason": reason, "clear_memory": False},
                timeout=30,
            )
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 409:
                raise
            time.sleep(1)
    raise last_error


def summarize_health(h):
    last = h.get("last_request") or {}
    pcache = h.get("prompt_cache") or {}
    prepare = last.get("prompt_cache_prepare") or pcache.get("last_prepare_event") or {}
    update = pcache.get("last_update_event") or {}
    if prepare.get("action") == "prewarm_start" and update.get("action"):
        prepare = {**prepare, "action": update.get("action")}
    kernel = h.get("kernel_stats") or {}
    defaults = h.get("generation_defaults") or {}
    return {
        "status": h.get("status"),
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "active": h.get("active_request") is not None,
        "direct_decode": defaults.get("direct_decode_kernel"),
        "direct_decode_eval_mode": defaults.get("direct_decode_eval_mode"),
        "sparse_topk_override": defaults.get("sparse_topk_blocks_override"),
        "prefill_step_size": defaults.get("prefill_step_size"),
        "effective_prefill_step_size": defaults.get("effective_prefill_step_size"),
        "mlx_max_ops_per_buffer": defaults.get("mlx_max_ops_per_buffer"),
        "mlx_max_mb_per_buffer": defaults.get("mlx_max_mb_per_buffer"),
        "decode_eval_every": defaults.get("decode_eval_every"),
        "decode_eval_after_tokens": defaults.get("decode_eval_after_tokens"),
        "decode_eval_after_every": defaults.get("decode_eval_after_every"),
        "long_context_decode_eval_tokens": defaults.get("long_context_decode_eval_tokens"),
        "long_context_decode_eval_every": defaults.get("long_context_decode_eval_every"),
        "effective_thinking_decode_eval_every": defaults.get("effective_thinking_decode_eval_every"),
        "effective_long_context_decode_eval_tokens": defaults.get("effective_long_context_decode_eval_tokens"),
        "effective_long_context_decode_eval_every": defaults.get("effective_long_context_decode_eval_every"),
        "adaptive_long_context_decode_eval": defaults.get("adaptive_long_context_decode_eval"),
        "mid_context_decode_eval_tokens": defaults.get("mid_context_decode_eval_tokens"),
        "mid_context_decode_eval_every": defaults.get("mid_context_decode_eval_every"),
        "high_context_decode_eval_tokens": defaults.get("high_context_decode_eval_tokens"),
        "high_context_decode_eval_every": defaults.get("high_context_decode_eval_every"),
        "refresh_generation_stream": defaults.get("refresh_generation_stream"),
        "visible_transcript_prewarm": defaults.get("visible_transcript_prewarm"),
        "visible_transcript_prewarm_min_generated": defaults.get(
            "effective_visible_transcript_prewarm_min_generated",
            defaults.get("visible_transcript_prewarm_min_generated"),
        ),
        "visible_transcript_prewarm_max_tokens": defaults.get("visible_transcript_prewarm_max_tokens"),
        "visible_transcript_prewarm_max_suffix_tokens": defaults.get("visible_transcript_prewarm_max_suffix_tokens"),
        "last_tokens": last.get("tokens"),
        "last_elapsed_s": last.get("elapsed_s"),
        "last_total_elapsed_s": last.get("total_elapsed_s"),
        "last_post_generation_s": last.get("post_generation_s"),
        "last_prompt_tokens": last.get("prompt_tokens"),
        "last_cached_tokens": last.get("cached_tokens"),
        "last_prompt_tps": last.get("prompt_tps"),
        "last_ttft_s": last.get("first_token_s"),
        "last_decode_tps": last.get("decode_tps"),
        "cache_action": prepare.get("action"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_prompt_tokens": prepare.get("prompt_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_missed_tokens": prepare.get("missed_tokens"),
        "cache_miss_reason": prepare.get("miss_reason"),
        "cache_protected_cache_tokens": prepare.get("protected_cache_tokens"),
        "cache_protected_cache_reuse_ratio": prepare.get("protected_cache_reuse_ratio"),
        "cache_previous_input_tokens": prepare.get("previous_input_tokens"),
        "cache_previous_generated_tokens": prepare.get("previous_generated_tokens"),
        "cache_reused_generated_tokens": prepare.get("reused_generated_tokens"),
        "cache_generated_reuse_ratio": prepare.get("generated_reuse_ratio"),
        "cache_would_reprocess_tokens": prepare.get("would_reprocess_tokens"),
        "cache_update_generated_key_tokens": update.get("generated_key_tokens"),
        "cache_update_generated_key_truncated": update.get("generated_key_truncated"),
        "cache_update_exact_generated_ids": update.get("exact_generated_ids"),
        "cache_protect_large": pcache.get("protect_large"),
        "cache_protect_min_tokens": pcache.get("protect_min_tokens"),
        "cache_protect_bypass_max_tokens": pcache.get("protect_bypass_max_tokens"),
        "topk_native": kernel.get("topk_native"),
        "topk_fallback": kernel.get("topk_fallback"),
        "topk_native_error": kernel.get("topk_native_error"),
        "prefill_attention_calls": kernel.get("prefill_attention_calls"),
        "prefill_eligible": kernel.get("prefill_eligible"),
        "prefill_ineligible": kernel.get("prefill_ineligible"),
        "last_prefill_ineligible_reason": kernel.get("last_prefill_ineligible_reason"),
        "msa_k1_impl": defaults.get("msa_k1_impl"),
        "last_msa_k1_impl": kernel.get("last_msa_k1_impl"),
        "msa_k1_scalar": kernel.get("msa_k1_scalar"),
        "msa_k1_simd": kernel.get("msa_k1_simd"),
        "msa_k1_simd_packed": kernel.get("msa_k1_simd_packed"),
        "msa_k1_steel_mma": kernel.get("msa_k1_steel_mma"),
        "sdpa_fallback_calls": kernel.get("sdpa_fallback_calls"),
        "compact_decode_calls": kernel.get("compact_decode_calls"),
        "compact_decode_selected_len": kernel.get("last_compact_decode_selected_len"),
        "compact_decode_total_len": kernel.get("last_compact_decode_total_len"),
        "compact_decode_density": kernel.get("last_compact_decode_density"),
        "direct_decode_calls": kernel.get("decode_attention_calls"),
        "direct_decode_errors": kernel.get("direct_decode_error"),
    }


def stream_chat(
    name,
    messages,
    *,
    model="m3-no-think",
    max_tokens=256,
    session_id=None,
    needle=None,
    timeout=600,
):
    before_completed = health().get("requests_completed", 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if session_id:
        payload["metadata"] = {"session_id": session_id, "source": "m3_perf_probe"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_chunk_s = None
    chunks = 0
    text_chars = 0
    text_parts = []
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
            if piece and first_chunk_s is None:
                first_chunk_s = time.time() - started
            if piece:
                text_chars += len(piece)
                text_parts.append(piece)
            chunks += 1
    elapsed = time.time() - started
    deadline = time.time() + 60
    h = health()
    while time.time() < deadline:
        h = health()
        if (
            not h.get("active_request")
            and h.get("requests_completed", 0) > before_completed
        ):
            break
        time.sleep(0.2)
    last = h.get("last_request") or {}
    summary = summarize_health(h)
    text = "".join(text_parts)
    row = {
        "name": name,
        "client_elapsed_s": round(elapsed, 2),
        "client_ttft_s": round(first_chunk_s or 0.0, 2),
        "chunks": chunks,
        "chars": text_chars,
        "output_preview": text[:280],
        "needle": needle,
        "needle_found": (needle in text) if needle else None,
        "server_tokens": last.get("tokens"),
        "server_elapsed_s": last.get("elapsed_s"),
        "server_total_elapsed_s": last.get("total_elapsed_s"),
        "server_post_generation_s": last.get("post_generation_s"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_cached_tokens": last.get("cached_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_total_tps": last.get("tps"),
        "server_ttft_s": last.get("first_token_s"),
        "server_decode_tps": last.get("decode_tps"),
        "session_id": session_id,
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "cache_action": summary.get("cache_action"),
        "cache_reuse_tokens": summary.get("cache_reuse_tokens"),
        "cache_prompt_tokens": summary.get("cache_prompt_tokens"),
        "cache_suffix_tokens": summary.get("cache_suffix_tokens"),
        "cache_reuse_ratio": summary.get("cache_reuse_ratio"),
        "cache_missed_tokens": summary.get("cache_missed_tokens"),
        "cache_miss_reason": summary.get("cache_miss_reason"),
        "cache_protected_cache_tokens": summary.get("cache_protected_cache_tokens"),
        "cache_protected_cache_reuse_ratio": summary.get("cache_protected_cache_reuse_ratio"),
        "cache_previous_generated_tokens": summary.get("cache_previous_generated_tokens"),
        "cache_reused_generated_tokens": summary.get("cache_reused_generated_tokens"),
        "cache_generated_reuse_ratio": summary.get("cache_generated_reuse_ratio"),
        "cache_would_reprocess_tokens": summary.get("cache_would_reprocess_tokens"),
        "cache_update_generated_key_tokens": summary.get("cache_update_generated_key_tokens"),
        "cache_update_generated_key_truncated": summary.get("cache_update_generated_key_truncated"),
        "cache_update_exact_generated_ids": summary.get("cache_update_exact_generated_ids"),
        "cache_protect_large": summary.get("cache_protect_large"),
        "cache_protect_min_tokens": summary.get("cache_protect_min_tokens"),
        "cache_protect_bypass_max_tokens": summary.get("cache_protect_bypass_max_tokens"),
        "topk_native": summary.get("topk_native"),
        "topk_fallback": summary.get("topk_fallback"),
        "topk_native_error": summary.get("topk_native_error"),
        "prefill_attention_calls": summary.get("prefill_attention_calls"),
        "prefill_eligible": summary.get("prefill_eligible"),
        "prefill_ineligible": summary.get("prefill_ineligible"),
        "last_prefill_ineligible_reason": summary.get("last_prefill_ineligible_reason"),
        "msa_k1_impl": summary.get("msa_k1_impl"),
        "last_msa_k1_impl": summary.get("last_msa_k1_impl"),
        "msa_k1_scalar": summary.get("msa_k1_scalar"),
        "msa_k1_simd": summary.get("msa_k1_simd"),
        "msa_k1_simd_packed": summary.get("msa_k1_simd_packed"),
        "msa_k1_steel_mma": summary.get("msa_k1_steel_mma"),
        "sdpa_fallback_calls": summary.get("sdpa_fallback_calls"),
        "compact_decode_calls": summary.get("compact_decode_calls"),
        "direct_decode_calls": summary.get("direct_decode_calls"),
        "direct_decode_errors": summary.get("direct_decode_errors"),
        "direct_decode": summary.get("direct_decode"),
        "direct_decode_eval_mode": summary.get("direct_decode_eval_mode"),
        "sparse_topk_override": summary.get("sparse_topk_override"),
        "compact_decode_selected_len": summary.get("compact_decode_selected_len"),
        "compact_decode_total_len": summary.get("compact_decode_total_len"),
        "compact_decode_density": summary.get("compact_decode_density"),
        "effective_thinking_decode_eval_every": summary.get("effective_thinking_decode_eval_every"),
        "effective_long_context_decode_eval_tokens": summary.get("effective_long_context_decode_eval_tokens"),
        "effective_long_context_decode_eval_every": summary.get("effective_long_context_decode_eval_every"),
        "adaptive_long_context_decode_eval": summary.get("adaptive_long_context_decode_eval"),
        "mid_context_decode_eval_tokens": summary.get("mid_context_decode_eval_tokens"),
        "mid_context_decode_eval_every": summary.get("mid_context_decode_eval_every"),
        "high_context_decode_eval_tokens": summary.get("high_context_decode_eval_tokens"),
        "high_context_decode_eval_every": summary.get("high_context_decode_eval_every"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE, help="Endpoint root, default %(default)s")
    parser.add_argument("--records", type=int, default=1800, help="Synthetic records for long prompt")
    parser.add_argument("--quick", action="store_true", help="Only run the short decode case")
    parser.add_argument("--reset-cache", action="store_true", help="Reset prompt cache before measuring")
    parser.add_argument(
        "--long-cold-timeout",
        type=int,
        default=0,
        help="Timeout for cold long-prefix stream; 0 auto-scales for large contexts",
    )
    parser.add_argument(
        "--long-warm-timeout",
        type=int,
        default=300,
        help="Timeout for warm exact-repeat stream",
    )
    parser.add_argument(
        "--long-decode-timeout",
        type=int,
        default=600,
        help="Timeout for cached long-context decode stream",
    )
    parser.add_argument(
        "--session-prefix",
        default="",
        help="Stable metadata.session_id prefix for isolated benchmark cache lanes",
    )
    args = parser.parse_args()

    BASE = args.base.rstrip("/")
    session_prefix = args.session_prefix or f"perf-{int(time.time())}"
    short_session = f"{session_prefix}-short"
    long_session = f"{session_prefix}-long"
    long_cold_timeout = args.long_cold_timeout
    if long_cold_timeout <= 0:
        # Empirically MiniMax synthetic records tokenize at roughly 17-18 tokens
        # per record. Give 350k-token probes room without weakening smaller runs.
        long_cold_timeout = max(1200, int(args.records * 0.085) + 300)

    initial = health()
    print(json.dumps({"initial": summarize_health(initial)}, sort_keys=True), flush=True)
    if args.reset_cache:
        print_event("reset", reset_cache())

    stream_chat(
        "short_decode",
        [{"role": "user", "content": "Write 180 words about stable OpenAI-compatible inference endpoints."}],
        model="m3-no-think",
        max_tokens=320,
        session_id=short_session,
    )
    if args.quick:
        final = health()
        print(json.dumps({"final": summarize_health(final)}, sort_keys=True), flush=True)
        if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
            raise SystemExit("probe saw request failure")
        return

    print_event("reset_long_segment", reset_cache("perf probe long segment reset"))

    long_prefix = "\n".join(
        f"FILE-{i:04d}: function_{i}(x) returns x + {i}" for i in range(args.records)
    )
    needle = "FILE-1377: function_1377(x) returns x + 1377" if args.records > 1377 else None
    prompt = (
        long_prefix
        + "\n\nUsing only the records above, return the exact line for FILE-1377, then stop."
    )
    stream_chat(
        "long_prefix_cold",
        [{"role": "user", "content": prompt}],
        model="m3-no-think",
        max_tokens=96,
        session_id=long_session,
        needle=needle,
        timeout=long_cold_timeout,
    )
    stream_chat(
        "long_prefix_warm",
        [{"role": "user", "content": prompt}],
        model="m3-no-think",
        max_tokens=96,
        session_id=long_session,
        needle=needle,
        timeout=args.long_warm_timeout,
    )
    long_decode_prompt = (
        long_prefix
        + "\n\nUsing the records above, write about 180 words on endpoint reliability, "
        + "then include FILE-1377 if it exists."
    )
    stream_chat(
        "long_context_decode",
        [{"role": "user", "content": long_decode_prompt}],
        model="m3-no-think",
        max_tokens=320,
        session_id=long_session,
        needle=needle,
        timeout=args.long_decode_timeout,
    )

    final = health()
    print(json.dumps({"final": summarize_health(final)}, sort_keys=True), flush=True)
    if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
        raise SystemExit("probe saw request failure")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {exc!r}", file=sys.stderr, flush=True)
        raise
