#!/usr/bin/env python3
"""Validate coalesced SSD autosaves without disturbing the RAM hot path."""

import argparse
import json
import time
import urllib.request


def request_json(base, method, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health(base):
    return request_json(base, "GET", "/health", timeout=10)


def wait_idle(base, completed_before, timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health(base)
        cache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and not cache.get("in_use")
            and int(last.get("requests_completed") or 0) > completed_before
        ):
            return last
        time.sleep(0.25)
    raise RuntimeError(f"request did not become idle: {last}")


def stream_chat(base, payload, timeout):
    completed_before = int(health(base).get("requests_completed") or 0)
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content = []
    reasoning = []
    started = time.time()
    first_piece_s = None
    last_piece_s = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            event = json.loads(item)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if piece or thought:
                    elapsed = time.time() - started
                    first_piece_s = first_piece_s or elapsed
                    last_piece_s = elapsed
                if piece:
                    content.append(piece)
                if thought:
                    reasoning.append(thought)
    final = wait_idle(base, completed_before, timeout)
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "last_piece_s": round(last_piece_s or 0.0, 3),
        "health": final,
    }


def context_for_target(target_tokens):
    records = max(64, int(target_tokens / 18))
    return "\n".join(
        f"autosave/file_{i:06d}.py :: symbol_{i}(value) returns value + {i}"
        for i in range(records)
    )


def compact_result(label, result):
    last = result["health"].get("last_request") or {}
    cache = result["health"].get("prompt_cache") or {}
    ssd = cache.get("ssd") or {}
    prepare = last.get("prompt_cache_prepare") or {}
    return {
        "label": label,
        "client_elapsed_s": result["client_elapsed_s"],
        "client_ttft_s": result["client_ttft_s"],
        "last_piece_s": result["last_piece_s"],
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_decode_tps": last.get("decode_tps"),
        "server_tokens": last.get("tokens"),
        "server_post_generation_s": last.get("post_generation_s"),
        "cache_action": prepare.get("action"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "ssd_last_saved_tokens": ssd.get("last_saved_tokens"),
        "ssd_deferred_count": ssd.get("auto_save_deferred_count"),
        "ssd_deferred_reason": ssd.get("last_auto_save_deferred_reason"),
        "failures": result["health"].get("requests_failed"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--target-tokens", type=int, default=10000)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    session_id = f"ssd-autosave-delta-{int(time.time())}"

    initial = health(base)
    ssd_initial = (initial.get("prompt_cache") or {}).get("ssd") or {}
    if not ssd_initial.get("enabled") or not ssd_initial.get("auto_save"):
        raise SystemExit("SSD autosave must be enabled")
    failures_before = int(initial.get("requests_failed") or 0)
    deferred_before = int(ssd_initial.get("auto_save_deferred_count") or 0)
    request_json(
        base,
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": "ssd autosave delta probe", "clear_memory": False},
        timeout=30,
    )

    user = {
        "role": "user",
        "content": context_for_target(args.target_tokens)
        + "\n\nSummarize this cache checkpoint in one short paragraph.",
    }
    metadata = {"session_id": session_id, "source": "m3_ssd_autosave_delta_probe"}
    first = stream_chat(
        base,
        {
            "model": args.model,
            "messages": [user],
            "metadata": metadata,
            "stream": True,
            "max_tokens": 96,
            "temperature": 0,
        },
        args.timeout,
    )
    first_row = compact_result("initial_checkpoint", first)
    if int(first_row.get("ssd_last_saved_tokens") or 0) < 8192:
        raise SystemExit(f"initial autosave was not observed: {first_row}")

    assistant = {"role": "assistant", "content": first["content"]}
    if first["reasoning"]:
        assistant["reasoning_content"] = first["reasoning"]
    second = stream_chat(
        base,
        {
            "model": args.model,
            "messages": [
                user,
                assistant,
                {"role": "user", "content": "Reply with one cache takeaway."},
            ],
            "metadata": metadata,
            "stream": True,
            "max_tokens": 48,
            "temperature": 0,
        },
        args.timeout,
    )
    second_row = compact_result("small_hot_followup", second)
    if int(second_row.get("ssd_deferred_count") or 0) <= deferred_before:
        raise SystemExit(f"follow-up autosave was not deferred: {second_row}")
    if not str(second_row.get("ssd_deferred_reason") or "").startswith(
        "delta_below_threshold:"
    ):
        raise SystemExit(f"unexpected autosave decision: {second_row}")
    if float(second_row.get("cache_reuse_ratio") or 0.0) < 0.90:
        raise SystemExit(f"hot cache reuse too low: {second_row}")
    if int(second["health"].get("requests_failed") or 0) != failures_before:
        raise SystemExit(f"request failures increased: {second_row}")

    print(json.dumps(first_row, sort_keys=True))
    print(json.dumps(second_row, sort_keys=True))
    print("PASS: SSD autosave delta probe")


if __name__ == "__main__":
    main()
