#!/usr/bin/env python3
"""Probe SSD-backed persistent prompt/KV cache save and restore.

Examples:

  # Save-only artifact validation.
  python3 probes/m3_persistent_cache_probe.py --phase build --target-tokens 30000

  # Simulate a restart by dropping RAM cache, then restore from SSD.
  python3 probes/m3_persistent_cache_probe.py --phase roundtrip --target-tokens 30000

  # True restart validation:
  python3 probes/m3_persistent_cache_probe.py --phase build --session-id agent-30k
  # restart cluster
  python3 probes/m3_persistent_cache_probe.py --phase restore --session-id agent-30k
"""
import argparse
import hashlib
import json
import os
import sys
import threading
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(timeout=5):
    return request_json("GET", "/health", timeout=timeout)


def post_admin(path, payload=None, timeout=60):
    return request_json("POST", path, payload or {}, timeout=timeout)


def compact_for_log(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "prompt_cache" and isinstance(item, dict):
                out[key] = compact_prompt_cache_for_log(item)
                continue
            if key == "entries" and isinstance(item, list):
                out["entry_count"] = len(item)
                complete = sum(1 for row in item if isinstance(row, dict) and row.get("complete"))
                ready = sum(1 for row in item if isinstance(row, dict) and row.get("restore_ready"))
                if complete or ready:
                    out["complete_entries"] = complete
                    out["restore_ready_entries"] = ready
                continue
            if key in {"recent_requests", "request_history"} and isinstance(item, list):
                out[f"{key}_count"] = len(item)
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
        "cache_physical_tokens": pcache.get("cache_physical_tokens"),
        "cache_capacity_tokens": pcache.get("cache_capacity_tokens"),
        "cache_spare_tokens": pcache.get("cache_spare_tokens"),
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
            "last_restore_target_capacity": ssd.get("last_restore_target_capacity"),
            "last_restore_append_reserve_tokens": ssd.get(
                "last_restore_append_reserve_tokens"
            ),
            "last_restore_miss_reason": ssd.get("last_restore_miss_reason"),
            "last_error": ssd.get("last_error"),
        },
    }


def print_event(name, payload):
    print(json.dumps({name: compact_for_log(payload)}, sort_keys=True), flush=True)


def wait_idle(before_completed=None, timeout=900):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        pcache = last.get("prompt_cache") or {}
        completed_ok = (
            before_completed is None
            or int(last.get("requests_completed") or 0) > int(before_completed)
        )
        if not last.get("active_request") and not pcache.get("in_use") and completed_ok:
            return last
        time.sleep(0.25)
    return last or health()


def dummy_tools(count):
    names = [
        "read_file", "list_dir", "search_code", "run_tests", "apply_patch",
        "web_search", "memory_update", "task_status", "shell", "open_image",
    ]
    tools = []
    for i in range(max(0, int(count or 0))):
        name = names[i] if i < len(names) else f"tool_{i}"
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": "Compatibility probe tool; do not call unless explicitly required.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
        })
    return tools


def stream_chat(name, messages, *, model, session_id, max_tokens, timeout=1800,
                shape="plain", tools_count=0, session_mode="metadata"):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if session_mode == "metadata":
        payload["metadata"] = {
            "session_id": session_id,
            "source": f"m3_persistent_cache_probe.{shape}",
        }
    elif session_mode != "auto":
        raise ValueError(f"unsupported session_mode={session_mode}")
    if tools_count > 0:
        payload["tools"] = dummy_tools(tools_count)
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
    final = wait_idle(before, timeout=timeout)
    last = final.get("last_request") or {}
    prompt_cache = final.get("prompt_cache") or {}
    request_shape = last.get("request_shape") or {}
    prepare = last.get("prompt_cache_prepare") or {}
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    row = {
        "name": name,
        "shape": shape,
        "session_mode": session_mode,
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "content": content_text,
        "reasoning_chars": len(reasoning_text),
        "content_chars": len(content_text),
        "reasoning_text_for_probe": reasoning_text,
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
        "cache_session_id": request_shape.get("cache_session_id"),
        "cache_session_source": request_shape.get("cache_session_source"),
        "tools_count": request_shape.get("tools_count"),
        "restored_ssd_cache": bool(
            prepare.get("restored_ssd_cache") or prepare.get("restored_ssd")
        ),
        "ssd_session_hash": prepare.get("ssd_session_hash"),
        "ssd_restore_capacity": prepare.get("ssd_restore_capacity"),
        "ssd_append_reserve_tokens": prepare.get("ssd_append_reserve_tokens"),
        "cache_physical_tokens": prompt_cache.get("cache_physical_tokens"),
        "cache_capacity_tokens": prompt_cache.get("cache_capacity_tokens"),
        "cache_spare_tokens": prompt_cache.get("cache_spare_tokens"),
        "failed": final.get("requests_failed"),
    }
    log_row = dict(row)
    log_row.pop("reasoning_text_for_probe", None)
    print(json.dumps(log_row, sort_keys=True), flush=True)
    return row, final


def build_context(target_tokens):
    # MiniMax tokenization varies by runtime. These records are intentionally
    # repetitive and deterministic so separate build/restore phases render the
    # same prefix.
    target_tokens = max(1024, int(target_tokens or 30000))
    records = max(64, int(target_tokens / 18))
    return "\n".join(
        f"persistent_cache/file_{i:06d}.py :: symbol_{i}(value) returns value + {i}; owner=agent; priority={i % 17}"
        for i in range(records)
    )


def highest_file_for_target(target_tokens):
    target_tokens = max(1024, int(target_tokens or 30000))
    records = max(64, int(target_tokens / 18))
    return f"file_{records - 1:06d}", records - 1


def default_state_file(session_id):
    digest = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:24]
    return os.path.join(
        "/tmp",
        "thundermlx_persistent_cache_probe_state",
        f"{digest}.json",
    )


def write_probe_state(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
    os.replace(tmp, path)


def read_probe_state(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_messages(target_tokens, shape="plain"):
    context = build_context(target_tokens)
    if shape == "plain":
        return [{
            "role": "user",
            "content": (
                context
                + "\n\nSummarize the persistent-cache validation implications in one useful paragraph."
            ),
        }]
    if shape == "openwebui-tools":
        return [
            {
                "role": "system",
                "content": (
                    "You are a concise local assistant in OpenWebUI. "
                    "Tools may be listed for compatibility, but answer directly."
                ),
            },
            {
                "role": "user",
                "content": (
                    context
                    + "\n\nOpenWebUI cache test: summarize the durable-session implication in one paragraph."
                ),
            },
        ]
    if shape == "agent-tools":
        return [
            {
                "role": "system",
                "content": (
                    "You are a coding agent with tool schemas available. "
                    "For this validation, do not call tools; reason from the provided workspace."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Workspace snapshot follows. Treat every line as a file/symbol fact.\n"
                    + context
                    + "\n\nAgent task: identify the cache/stability implications and keep the answer brief."
                ),
            },
        ]
    raise ValueError(f"unsupported shape={shape}")


def followup_messages(target_tokens, assistant_text, shape="plain"):
    messages = seed_messages(target_tokens, shape=shape)
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})
    file_name, _ = highest_file_for_target(target_tokens)
    messages.append({
        "role": "user",
        "content": (
            "Using only the cached project context, name the highest numbered "
            f"file. The expected suffix is around {file_name}; reply briefly."
        ),
    })
    return messages


def wait_active(timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        active = health().get("active_request")
        if active:
            return active
        time.sleep(0.1)
    raise RuntimeError("no active request observed")


def stream_worker(payload, result, timeout=180):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                if raw.startswith(b"data:"):
                    result["lines"] += 1
                if b"[DONE]" in raw:
                    result["done"] = True
                    break
    except Exception as exc:
        result["error"] = repr(exc)


def cancel_after_restore(messages, *, model, session_id, shape, tools_count,
                         session_mode, timeout=120, require_enabled=False):
    h = health()
    defaults = h.get("generation_defaults") or {}
    stop_enabled = bool(defaults.get("unsafe_inflight_stop"))
    disconnect_enabled = bool(defaults.get("stop_on_client_disconnect"))
    if not (stop_enabled and disconnect_enabled):
        result = {
            "ok": True,
            "skipped": True,
            "reason": "inflight stop defaults disabled",
            "unsafe_inflight_stop": stop_enabled,
            "stop_on_client_disconnect": disconnect_enabled,
        }
        print(json.dumps({"cancel_after_restore": result}, sort_keys=True), flush=True)
        if require_enabled:
            raise SystemExit(f"cancel-after-restore skipped: {result}")
        return result

    payload = {
        "model": model,
        "messages": list(messages) + [{
            "role": "user",
            "content": "Now write a long cancellation smoke-test note. Continue until stopped.",
        }],
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0,
    }
    if session_mode == "metadata":
        payload["metadata"] = {
            "session_id": session_id,
            "source": f"m3_persistent_cache_probe.{shape}.cancel",
        }
    if tools_count > 0:
        payload["tools"] = dummy_tools(tools_count)

    baseline_failed = int(h.get("requests_failed") or 0)
    result = {"lines": 0, "done": False, "error": None}
    thread = threading.Thread(target=stream_worker, args=(payload, result, timeout), daemon=True)
    thread.start()
    before = wait_active()
    time.sleep(0.2)
    stopped = post_admin("/v1/stop", timeout=30)
    final = wait_idle(timeout=timeout)
    thread.join(timeout=10)
    if int(final.get("requests_failed") or 0) > baseline_failed:
        raise SystemExit(f"cancel-after-restore incremented failures: {final}")
    if not stopped.get("stopped"):
        raise SystemExit(f"cancel-after-restore bad stop response: {stopped}")
    row = {
        "ok": True,
        "skipped": False,
        "before": {
            key: before.get(key)
            for key in ("id", "tokens_emitted", "prefill_processed_tokens", "prefill_total_tokens")
        },
        "stop_response": stopped,
        "stream_result": result,
        "final_completed": final.get("requests_completed"),
        "final_failed": final.get("requests_failed"),
    }
    print(json.dumps({"cancel_after_restore": row}, sort_keys=True), flush=True)
    return row


def require_healthy_and_configured():
    h = health()
    if h.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {h}")
    pcache = h.get("prompt_cache") or {}
    ssd = pcache.get("ssd") or {}
    defaults = h.get("generation_defaults") or {}
    if not pcache.get("enabled"):
        raise SystemExit("prompt cache is disabled")
    if not ssd.get("enabled") and not defaults.get("prompt_cache_ssd"):
        raise SystemExit("MLX_M3_PROMPT_CACHE_SSD must be enabled for this probe")
    return h


def ssd_summary():
    h = health()
    ssd = ((h.get("prompt_cache") or {}).get("ssd") or {})
    return {
        "mode": ssd.get("mode"),
        "path": ssd.get("path"),
        "path_mode": ssd.get("path_mode"),
        "rank": ssd.get("rank"),
        "entry_count": ssd.get("entry_count"),
        "total_bytes": ssd.get("total_bytes"),
        "last_saved_tokens": ssd.get("last_saved_tokens"),
        "last_restored_tokens": ssd.get("last_restored_tokens"),
        "last_restore_target_capacity": ssd.get("last_restore_target_capacity"),
        "last_restore_append_reserve_tokens": ssd.get(
            "last_restore_append_reserve_tokens"
        ),
        "last_restore_miss_reason": ssd.get("last_restore_miss_reason"),
        "last_error": ssd.get("last_error"),
    }


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--phase", choices=("build", "restore", "roundtrip"), default="roundtrip")
    parser.add_argument("--target-tokens", type=int, default=30000)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--model", default="Minimax-M3")
    parser.add_argument(
        "--shape",
        choices=("plain", "openwebui-tools", "agent-tools"),
        default="plain",
        help="Prompt/request shape to validate.",
    )
    parser.add_argument(
        "--session-mode",
        choices=("metadata", "auto"),
        default="metadata",
        help="Use explicit metadata session ids or OpenWebUI-style auto session fingerprints.",
    )
    parser.add_argument("--tools", type=int, default=34)
    parser.add_argument("--seed-max-tokens", type=int, default=96)
    parser.add_argument("--followup-max-tokens", type=int, default=64)
    parser.add_argument("--save-timeout", type=int, default=900)
    parser.add_argument("--cancel-after-restore", action="store_true")
    parser.add_argument("--require-cancel-enabled", action="store_true")
    parser.add_argument("--skip-correctness-check", action="store_true")
    parser.add_argument(
        "--state-file",
        default=None,
        help="Local JSON file used to carry build-turn assistant text across a true restart.",
    )
    parser.add_argument(
        "--assistant-text",
        default=None,
        help="Exact build-turn assistant content to use during restore phase.",
    )
    parser.add_argument("--clear-ssd", action="store_true")
    parser.add_argument("--skip-ram-reset", action="store_true")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    session_id = args.session_id or f"persistent-cache-{args.target_tokens}"
    state_file = args.state_file or default_state_file(session_id)

    initial = require_healthy_and_configured()
    print(json.dumps({"initial": {
        "completed": initial.get("requests_completed"),
        "failed": initial.get("requests_failed"),
        "ssd": ssd_summary(),
        "shape": args.shape,
        "session_mode": args.session_mode,
    }}, sort_keys=True), flush=True)

    if args.clear_ssd:
        print_event(
            "clear_ssd",
            post_admin("/admin/prompt-cache/ssd/clear", {"reason": "persistent probe clear"}),
        )

    assistant_text = None
    if args.phase in {"build", "roundtrip"}:
        print_event(
            "reset_ram",
            post_admin(
                "/admin/prompt-cache/reset",
                {"reason": "persistent probe build reset", "clear_memory": False},
            ),
        )
        build, _ = stream_chat(
            "build_persistent_session",
            seed_messages(args.target_tokens, shape=args.shape),
            model=args.model,
            session_id=session_id,
            max_tokens=args.seed_max_tokens,
            shape=args.shape,
            tools_count=args.tools if args.shape.endswith("tools") else 0,
            session_mode=args.session_mode,
        )
        assistant_text = build.get("content") or ""
        write_probe_state(state_file, {
            "session_id": session_id,
            "model": args.model,
            "shape": args.shape,
            "session_mode": args.session_mode,
            "target_tokens": args.target_tokens,
            "assistant_text": assistant_text,
            "content_chars": build.get("content_chars"),
            "reasoning_chars": build.get("reasoning_chars"),
            "written_at": round(time.time(), 3),
        })
        print(json.dumps({"state_file": state_file}, sort_keys=True), flush=True)
        print_event(
            "manual_save",
            post_admin(
                "/admin/prompt-cache/ssd/save",
                {"reason": "persistent probe manual save"},
                timeout=args.save_timeout,
            ),
        )
        after_build = ssd_summary()
        print(json.dumps({"after_build_ssd": after_build}, sort_keys=True), flush=True)
        if not after_build.get("entry_count") and not after_build.get("last_saved_tokens"):
            raise SystemExit(f"SSD save not observed: {after_build}")
        if args.phase == "build":
            return

    if args.phase in {"restore", "roundtrip"}:
        if assistant_text is None:
            assistant_text = args.assistant_text
        if assistant_text is None:
            try:
                state = read_probe_state(state_file)
                if state.get("session_id") != session_id:
                    raise ValueError("state session_id mismatch")
                if state.get("shape") and state.get("shape") != args.shape:
                    raise ValueError(
                        f"state shape mismatch: {state.get('shape')} != {args.shape}"
                    )
                if (
                    state.get("session_mode")
                    and state.get("session_mode") != args.session_mode
                ):
                    raise ValueError(
                        "state session_mode mismatch: "
                        f"{state.get('session_mode')} != {args.session_mode}"
                    )
                assistant_text = state.get("assistant_text")
            except Exception as exc:
                raise SystemExit(
                    "restore phase needs the exact build-turn assistant text; "
                    f"run build first or pass --assistant-text/--state-file ({exc})"
                ) from exc
        if not args.skip_ram_reset:
            print_event(
                "reset_ram_before_restore",
                post_admin(
                    "/admin/prompt-cache/reset",
                    {"reason": "persistent probe restore reset", "clear_memory": False},
                ),
            )
        restore, restore_health = stream_chat(
            "restore_persistent_session",
            followup_messages(args.target_tokens, assistant_text, shape=args.shape),
            model=args.model,
            session_id=session_id,
            max_tokens=args.followup_max_tokens,
            shape=args.shape,
            tools_count=args.tools if args.shape.endswith("tools") else 0,
            session_mode=args.session_mode,
        )
        after_restore = ssd_summary()
        print(json.dumps({"after_restore_ssd": after_restore}, sort_keys=True), flush=True)
        if not restore.get("restored_ssd_cache"):
            raise SystemExit(f"SSD restore not observed: {restore}")
        if float(restore.get("cache_reuse_ratio") or 0.0) < 0.90:
            raise SystemExit(f"SSD restore reuse too low: {restore}")
        restore_capacity = int(restore.get("ssd_restore_capacity") or 0)
        append_reserve = int(restore.get("ssd_append_reserve_tokens") or 0)
        required_capacity = int(restore.get("cache_prompt_tokens") or 0) + max(
            0, append_reserve
        )
        if restore_capacity < required_capacity:
            raise SystemExit(
                "SSD restore did not reserve its bounded append capacity: "
                f"capacity={restore_capacity}, required={required_capacity}, "
                f"row={restore}"
            )
        # Capacity is step-rounded, but must not silently expand back to the
        # server's full output ceiling after generation. Tool-shaped requests
        # may legitimately run one bounded internal recovery generation after
        # a short first attempt emits no usable call. Account for that explicit
        # recovery ceiling while still rejecting the old 32K over-reservation.
        live_capacity = int(restore.get("cache_capacity_tokens") or 0)
        defaults = restore_health.get("generation_defaults") or {}
        bounded_generation_reserve = max(
            int(args.followup_max_tokens or 0),
            append_reserve,
        )
        if args.shape.endswith("tools"):
            bounded_generation_reserve = max(
                bounded_generation_reserve,
                int(defaults.get("tool_action_no_call_token_budget") or 0),
                int(defaults.get("tool_unusable_retry_max_tokens") or 0),
            )
        max_bounded_capacity = (
            int(restore.get("cache_prompt_tokens") or 0)
            + bounded_generation_reserve
            + 4095
        )
        if live_capacity > max_bounded_capacity:
            raise SystemExit(
                "warm batch conversion re-expanded to the output ceiling: "
                f"live_capacity={live_capacity}, bounded_max={max_bounded_capacity}, "
                f"request_ceiling={args.followup_max_tokens}, "
                f"bounded_recovery_ceiling={bounded_generation_reserve}, row={restore}"
            )
        expected_file, _ = highest_file_for_target(args.target_tokens)
        restore_text = (
            (restore.get("content") or "")
            + "\n"
            + (restore.get("reasoning_text_for_probe") or "")
        )
        if not args.skip_correctness_check and expected_file not in restore_text:
            clean_restore = dict(restore)
            clean_restore.pop("reasoning_text_for_probe", None)
            raise SystemExit(
                f"restore answer did not mention expected {expected_file}: {clean_restore}"
            )
        if args.cancel_after_restore:
            cancel_after_restore(
                followup_messages(args.target_tokens, assistant_text, shape=args.shape),
                model=args.model,
                session_id=session_id,
                shape=args.shape,
                tools_count=args.tools if args.shape.endswith("tools") else 0,
                session_mode=args.session_mode,
                require_enabled=args.require_cancel_enabled,
            )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
