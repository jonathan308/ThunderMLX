#!/usr/bin/env python3
"""Warm the MiniMax-M3 endpoint after startup without keeping user cache state."""
import json
import os
import sys
import time
import urllib.error
import urllib.request


BASE = os.environ.get("M3_WARMUP_BASE", "http://127.0.0.1:8080").rstrip("/")
TIMEOUT_S = int(os.environ.get("M3_WARMUP_TIMEOUT_SECONDS", "300") or "300")
MAX_TOKENS = int(os.environ.get("M3_WARMUP_MAX_TOKENS", "96") or "96")
MODEL = os.environ.get("M3_WARMUP_MODEL", "Minimax-M3-No-Think")
MODELS = [
    m.strip()
    for m in os.environ.get(
        "M3_WARMUP_MODELS", "Minimax-M3-No-Think,Minimax-M3"
    ).split(",")
    if m.strip()
]
PROMPT = os.environ.get(
    "M3_WARMUP_PROMPT",
    "Warm up the endpoint with exactly 70 words about stable local inference, live streaming, and low latency.",
)
INTERACTIVE_MODEL = os.environ.get("M3_WARMUP_INTERACTIVE_MODEL", "Minimax-M3")
INTERACTIVE_PROMPTS = [
    p.strip()
    for p in os.environ.get(
        "M3_WARMUP_INTERACTIVE_PROMPTS",
        (
            "Hello. Reply in one short sentence."
            "||Say one short friendly sentence about cache reuse."
            "||Give one concise sentence about distributed MiniMax inference now."
        ),
    ).split("||")
    if p.strip()
]
METAL_WARMUP = os.environ.get(
    "M3_WARMUP_METAL", "1"
).strip().lower() in {"1", "true", "yes", "on"}
METAL_WARMUP_SIZE = int(os.environ.get("M3_WARMUP_METAL_SIZE", "64") or "64")
METAL_WARMUP_REPEATS = int(os.environ.get("M3_WARMUP_METAL_REPEATS", "1") or "1")
RESET_PROMPT_CACHE = os.environ.get(
    "M3_WARMUP_RESET_PROMPT_CACHE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
OPENWEBUI_SIM_WARMUP = os.environ.get(
    "M3_WARMUP_OPENWEBUI_SIM", "1"
).strip().lower() in {"1", "true", "yes", "on"}
OPENWEBUI_SIM_TURNS = int(os.environ.get("M3_WARMUP_OPENWEBUI_SIM_TURNS", "4") or "4")
OPENWEBUI_SIM_MAX_TOKENS = int(
    os.environ.get(
        "M3_WARMUP_OPENWEBUI_SIM_MAX_TOKENS",
        str(min(MAX_TOKENS, 48)),
    )
    or str(min(MAX_TOKENS, 48))
)
OPENWEBUI_SIM_SYSTEM = os.environ.get(
    "M3_WARMUP_OPENWEBUI_SIM_SYSTEM",
    (
        "Use this date/time context if relevant. Full current datetime: "
        "2026-06-30 00:00:00. Current timezone: America/Los_Angeles. "
        "You are running locally as MiniMax-M3 through an OpenAI-compatible "
        "endpoint. Be concise, helpful, and preserve context."
    ),
)
OPENWEBUI_SIM_PROMPTS = [
    p.strip()
    for p in os.environ.get(
        "M3_WARMUP_OPENWEBUI_SIM_PROMPTS",
        "remember apple||what word?||thanks !||ok",
    ).split("||")
    if p.strip()
]


def request_json(method, path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health(timeout=5):
    return request_json("GET", "/health", timeout=timeout)


def wait_ready():
    deadline = time.time() + TIMEOUT_S
    last = None
    while time.time() < deadline:
        try:
            last = health()
            if (
                last.get("status") == "healthy"
                and last.get("active_request") is None
                and int(last.get("request_queue_depth") or 0) == 0
            ):
                return last
        except Exception as exc:
            last = {"error": repr(exc)}
        time.sleep(2)
    raise TimeoutError(f"endpoint did not become idle/healthy; last={last}")


def stream_warmup(model_id, *, prompt=PROMPT, metadata=True, max_tokens=MAX_TOKENS):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if metadata:
        payload["metadata"] = {"session_id": f"startup-warmup:{model_id}"}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = 0
    token_chunks = 0
    started = time.time()
    first = None
    with urllib.request.urlopen(req, timeout=max(60, TIMEOUT_S)) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            chunks += 1
            try:
                evt = json.loads(item)
            except json.JSONDecodeError:
                continue
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
                if piece:
                    token_chunks += 1
                    if first is None:
                        first = time.time() - started
    return {
        "model": model_id,
        "chunks": chunks,
        "token_chunks": token_chunks,
        "client_ttft_s": round(first or 0.0, 3),
        "prompt_chars": len(prompt),
        "metadata": bool(metadata),
    }


def stream_payload(payload, *, label):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = 0
    token_chunks = 0
    started = time.time()
    first = None
    visible = ""
    reasoning_chars = 0
    with urllib.request.urlopen(req, timeout=max(60, TIMEOUT_S)) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            chunks += 1
            try:
                evt = json.loads(item)
            except json.JSONDecodeError:
                continue
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reasoning = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                piece = content or reasoning
                if piece:
                    token_chunks += 1
                    if first is None:
                        first = time.time() - started
                if content:
                    visible += content
                if reasoning:
                    reasoning_chars += len(reasoning)
    final = wait_idle_and_cache_ready()
    last = final.get("last_request") or {}
    return {
        "label": label,
        "model": payload.get("model"),
        "chunks": chunks,
        "token_chunks": token_chunks,
        "client_ttft_s": round(first or 0.0, 3),
        "server_ttft_s": last.get("first_token_s"),
        "decode_tps": last.get("decode_tps"),
        "prompt_tps": last.get("prompt_tps"),
        "processed_prompt_tokens": last.get("processed_prompt_tokens"),
        "full_prompt_tokens": last.get("full_prompt_tokens"),
        "cache_efficiency": last.get("cache_efficiency"),
        "visible_chars": len(visible),
        "reasoning_chars": reasoning_chars,
        "visible": visible,
    }


def openwebui_sim_warmup():
    if not OPENWEBUI_SIM_WARMUP:
        return []
    messages = [{"role": "system", "content": OPENWEBUI_SIM_SYSTEM}]
    results = []
    prompts = OPENWEBUI_SIM_PROMPTS[: max(1, OPENWEBUI_SIM_TURNS)]
    for idx, prompt in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": INTERACTIVE_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": max(1, min(MAX_TOKENS, OPENWEBUI_SIM_MAX_TOKENS)),
            "temperature": 0,
        }
        result = stream_payload(payload, label=f"openwebui_sim_turn_{idx}")
        results.append({k: v for k, v in result.items() if k != "visible"})
        messages.append({"role": "assistant", "content": result.get("visible") or ""})
    return results


def metal_warmup():
    if not METAL_WARMUP:
        return {"skipped": True}
    return request_json(
        "POST",
        "/admin/metal-warmup",
        {
            "matrix_size": METAL_WARMUP_SIZE,
            "repeats": METAL_WARMUP_REPEATS,
            "reason": "startup warmup",
        },
        timeout=max(60, TIMEOUT_S),
    )


def reset_prompt_cache():
    try:
        return request_json(
            "POST",
            "/admin/prompt-cache/reset",
            {"reason": "warmup reset", "clear_memory": False},
            timeout=30,
        )
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code}


def wait_idle_and_cache_ready(timeout=45):
    deadline = time.time() + timeout
    last = health()
    while time.time() < deadline:
        cache = last.get("prompt_cache") or {}
        prepare = cache.get("last_prepare_event") or {}
        update = cache.get("last_update_event") or {}
        prepare_at = float(prepare.get("at") or 0.0)
        update_at = float(update.get("at") or 0.0)
        prewarm_done = (
            prepare.get("action") != "prewarm_start"
            or (update.get("action") and update_at >= prepare_at)
        )
        if (
            last.get("active_request") is None
            and int(last.get("request_queue_depth") or 0) == 0
            and not cache.get("in_use")
            and prewarm_done
        ):
            return last
        time.sleep(0.2)
        last = health()
    return last


def main():
    initial = wait_ready()
    skip_after_completed = int(
        os.environ.get("M3_WARMUP_SKIP_AFTER_COMPLETED", "0") or "0"
    )
    if skip_after_completed >= 0 and int(initial.get("requests_completed") or 0) > skip_after_completed:
        print(json.dumps({"skipped": "endpoint already served requests"}), flush=True)
        return 0
    metal = metal_warmup()
    results = []
    for model_id in MODELS:
        result = stream_warmup(model_id)
        final = wait_idle_and_cache_ready()
        last = final.get("last_request") or {}
        result.update({
            "server_ttft_s": last.get("first_token_s"),
            "decode_tps": last.get("decode_tps"),
            "prompt_tps": last.get("prompt_tps"),
        })
        results.append(result)
    for prompt in INTERACTIVE_PROMPTS:
        result = stream_warmup(
            INTERACTIVE_MODEL,
            prompt=prompt,
            metadata=False,
            max_tokens=min(MAX_TOKENS, 48),
        )
        final = wait_idle_and_cache_ready()
        last = final.get("last_request") or {}
        result.update({
            "server_ttft_s": last.get("first_token_s"),
            "decode_tps": last.get("decode_tps"),
            "prompt_tps": last.get("prompt_tps"),
            "interactive": True,
        })
        results.append(result)
    results.extend(openwebui_sim_warmup())
    reset = reset_prompt_cache() if RESET_PROMPT_CACHE else {"ok": None, "skipped": True}
    final = health()
    print(json.dumps({
        "warmup": results[-1] if results else None,
        "warmups": results,
        "metal_warmup": metal,
        "reset_ok": reset.get("ok"),
        "reset_skipped": reset.get("skipped", False),
        "cache_loaded": (final.get("prompt_cache") or {}).get("loaded"),
        "cache_session_id": (final.get("prompt_cache") or {}).get("session_id"),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": repr(exc)}), file=sys.stderr, flush=True)
        raise
