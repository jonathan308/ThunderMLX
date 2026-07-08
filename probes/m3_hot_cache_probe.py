#!/usr/bin/env python3
"""Probe multi-turn prompt-cache reuse with and without reasoning metadata."""
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def post_json(path, payload, timeout=900):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health():
    with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def reset_cache():
    payload = json.dumps({
        "reason": "hot cache probe reset",
        "clear_memory": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/admin/prompt-cache/reset",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(messages, max_tokens=384, model="Minimax-M3"):
    before_completed = health().get("requests_completed", 0)
    started = time.time()
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content_parts = []
    reasoning_parts = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            evt = json.loads(data)
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    reasoning_parts.append(reasoning)
    elapsed = time.time() - started
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
        "elapsed_s": round(elapsed, 3),
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }


def cache_summary(label):
    h = health()
    pc = h.get("prompt_cache") or {}
    pe = pc.get("last_prepare_event") or {}
    ue = pc.get("last_update_event") or {}
    visible_pe = pe
    if pe.get("action") == "prewarm_start" and ue.get("action"):
        visible_pe = {**pe, "action": ue.get("action")}
    lr = h.get("last_request") or {}
    return {
        "label": label,
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "cache_action": visible_pe.get("action"),
        "reuse_tokens": pe.get("reuse_tokens"),
        "prompt_tokens": pe.get("prompt_tokens"),
        "suffix_tokens": pe.get("suffix_tokens"),
        "reuse_ratio": pe.get("reuse_ratio"),
        "missed_tokens": pe.get("missed_tokens"),
        "miss_reason": pe.get("miss_reason"),
        "previous_key_tokens": pe.get("previous_key_tokens"),
        "previous_input_tokens": pe.get("previous_input_tokens"),
        "previous_generated_tokens": pe.get("previous_generated_tokens"),
        "reused_generated_tokens": pe.get("reused_generated_tokens"),
        "generated_reuse_ratio": pe.get("generated_reuse_ratio"),
        "would_reprocess_tokens": pe.get("would_reprocess_tokens"),
        "cache_update_action": ue.get("action"),
        "cache_update_key_tokens": ue.get("key_tokens"),
        "cache_update_cache_len": ue.get("cache_len"),
        "last_ttft_s": lr.get("first_token_s"),
        "last_prompt_tps": lr.get("prompt_tps"),
        "last_decode_tps": lr.get("decode_tps"),
        "last_prompt_tokens": lr.get("prompt_tokens"),
    }


def run_case(include_reasoning):
    reset_cache()
    case_name = "with_reasoning" if include_reasoning else "content_only"
    user = {
        "role": "user",
        "content": (
            "Think carefully, then explain in about 220 words how a stable "
            "OpenAI-compatible inference gateway should preserve cache reuse "
            "across multi-turn agent chats."
        ),
    }
    first = chat([user], max_tokens=360)
    first_summary = cache_summary(f"{case_name}:after_first")
    print(json.dumps(first_summary))
    assistant = {"role": "assistant", "content": first["content"]}
    if include_reasoning and first["reasoning"]:
        assistant["reasoning_content"] = first["reasoning"]
    follow = {"role": "user", "content": "Thanks, give me two crisp takeaways."}
    second = chat([user, assistant, follow], max_tokens=96)
    summary = cache_summary(case_name)
    summary["first_content_chars"] = len(first["content"])
    summary["first_reasoning_chars"] = len(first["reasoning"])
    summary["second_content_chars"] = len(second["content"])
    summary["second_elapsed_s"] = second["elapsed_s"]
    print(json.dumps(summary))


def main():
    run_case(False)
    run_case(True)


if __name__ == "__main__":
    main()
