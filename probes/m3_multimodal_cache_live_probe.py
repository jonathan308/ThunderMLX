#!/usr/bin/env python3
"""Live two-rank acceptance probe for exact multimodal prompt/KV reuse."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
import uuid

from PIL import Image


BASE = "http://127.0.0.1:8080"


def _request_json(method, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _health():
    return _request_json("GET", "/health", timeout=5)


def _wait_idle(previous_completed, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = _health()
        if (
            not status.get("active_request")
            and int(status.get("requests_completed") or 0) > previous_completed
            and not (status.get("prompt_cache") or {}).get("in_use")
        ):
            return status
        time.sleep(0.2)
    raise TimeoutError("server did not return to idle")


def _image_uri(left, right):
    image = Image.new("RGB", (32, 16))
    pixels = image.load()
    for y in range(16):
        for x in range(32):
            pixels[x, y] = left if x < 16 else right
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _image_user(image_uris, text="Describe the colors and their positions."):
    content = [{"type": "text", "text": text}]
    for image_uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": image_uri}})
    return {"role": "user", "content": content}


def _chat(
    *,
    model,
    messages,
    session_id,
    stream,
    max_tokens,
    tools=None,
    tool_choice=None,
    timeout=300,
):
    before = _health()
    completed = int(before.get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
        "metadata": {"session_id": session_id},
        "stream": bool(stream),
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    started = time.time()
    first_piece = None
    content_parts = []
    reasoning_parts = []
    streamed_tool_calls = {}
    response_message = None
    if stream:
        request = urllib.request.Request(
            BASE + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get(
                        "reasoning"
                    )
                    if (content or reasoning) and first_piece is None:
                        first_piece = time.time() - started
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    for call in delta.get("tool_calls") or []:
                        index = int(call.get("index") or 0)
                        target = streamed_tool_calls.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if call.get("id"):
                            target["id"] += str(call["id"])
                        function = call.get("function") or {}
                        if function.get("name"):
                            target["function"]["name"] += str(function["name"])
                        if function.get("arguments"):
                            target["function"]["arguments"] += str(
                                function["arguments"]
                            )
        response_message = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if streamed_tool_calls:
            response_message["tool_calls"] = [
                streamed_tool_calls[index]
                for index in sorted(streamed_tool_calls)
            ]
    else:
        response = _request_json(
            "POST", "/v1/chat/completions", payload, timeout=timeout
        )
        response_message = (response.get("choices") or [{}])[0].get("message") or {}
        content_parts.append(response_message.get("content") or "")
        reasoning_parts.append(
            response_message.get("reasoning_content")
            or response_message.get("reasoning")
            or ""
        )
        first_piece = time.time() - started
    final = _wait_idle(completed, timeout=timeout)
    last = final.get("last_request") or {}
    prepare = last.get("prompt_cache_prepare") or (
        (final.get("prompt_cache") or {}).get("last_prepare_event")
    ) or {}
    return {
        "message": response_message,
        "content": "".join(content_parts),
        "reasoning_chars": len("".join(reasoning_parts)),
        "tokens": int(last.get("tokens") or 0),
        "client_first_piece_s": round(first_piece or 0.0, 3),
        "client_elapsed_s": round(time.time() - started, 3),
        "server_ttft_s": last.get("first_token_s"),
        "decode_tps": last.get("decode_tps"),
        "prompt_tps": last.get("prompt_tps"),
        "prompt_tokens": last.get("prompt_tokens"),
        "cached_tokens": last.get("cached_tokens"),
        "cache_action": prepare.get("action"),
        "physical_cache_hit": bool(prepare.get("physical_cache_hit")),
        "physical_reuse_tokens": int(prepare.get("physical_reuse_tokens") or 0),
        "media_safe_prefix_min": int(prepare.get("media_safe_prefix_min") or 0),
        "image_fingerprint": prepare.get("image_fingerprint"),
        "requests_failed": int(final.get("requests_failed") or 0),
        "prepare": prepare,
    }


def _reset():
    return _request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": "multimodal live probe", "clear_memory": False},
    )


def _assert_cold(row, label):
    if row["physical_cache_hit"] or row["physical_reuse_tokens"]:
        raise AssertionError(f"{label} unexpectedly reused visual KV: {row}")


def _assert_hot(row, label):
    if not row["physical_cache_hit"]:
        raise AssertionError(f"{label} did not report a physical hit: {row}")
    if row["physical_reuse_tokens"] < row["media_safe_prefix_min"]:
        raise AssertionError(f"{label} reused before media boundary: {row}")
    if int(row["cached_tokens"] or 0) <= 0:
        raise AssertionError(f"{label} reported no cached tokens: {row}")


def _summary(label, row):
    compact = {
        key: row.get(key)
        for key in (
            "client_first_piece_s",
            "server_ttft_s",
            "decode_tps",
            "prompt_tps",
            "prompt_tokens",
            "cached_tokens",
            "cache_action",
            "physical_cache_hit",
            "physical_reuse_tokens",
            "media_safe_prefix_min",
            "image_fingerprint",
            "reasoning_chars",
            "requests_failed",
        )
    }
    print(json.dumps({"label": label, **compact}, sort_keys=True), flush=True)


def _image_tool_case(image_a, timeout):
    _reset()
    session_id = f"mm-tool-{uuid.uuid4().hex[:10]}"
    user = _image_user(
        [image_a],
        "Use the report_colors tool to record the left and right colors.",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "report_colors",
            "description": "Record the colors visible on each side of an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "left": {"type": "string"},
                    "right": {"type": "string"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
        },
    }]
    called = _chat(
        model="Minimax-M3-No-Think",
        messages=[user],
        session_id=session_id,
        stream=False,
        max_tokens=1024,
        tools=tools,
        tool_choice="required",
        timeout=timeout,
    )
    _assert_cold(called, "image tool cold")
    tool_calls = (called["message"] or {}).get("tool_calls") or []
    if not tool_calls:
        raise AssertionError(f"image request emitted no native tool call: {called}")
    call = tool_calls[0]
    if (call.get("function") or {}).get("name") != "report_colors":
        raise AssertionError(f"unexpected image tool call: {call}")
    _summary("image_native_tool:cold", called)

    summarized = _chat(
        model="Minimax-M3-No-Think",
        messages=[
            user,
            called["message"],
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": "report_colors",
                "content": json.dumps({"ok": True}),
            },
            {
                "role": "user",
                "content": "Use report_colors once more to confirm the same result.",
            },
        ],
        session_id=session_id,
        stream=False,
        max_tokens=1024,
        tools=tools,
        tool_choice="required",
        timeout=timeout,
    )
    _assert_hot(summarized, "image tool-result follow-up")
    repeated_calls = (summarized["message"] or {}).get("tool_calls") or []
    if not repeated_calls:
        raise AssertionError("image tool-result follow-up emitted no tool call")
    _summary("image_tool_result:safe_prefix_hit", summarized)


def _thinking_image_tool_case(image_a, timeout):
    """Gate the exact ZCode-sensitive image + thinking + native-tool path."""
    _reset()
    session_id = f"mm-thinking-tool-{uuid.uuid4().hex[:10]}"
    tools = [{
        "type": "function",
        "function": {
            "name": "report_colors",
            "description": "Record the colors visible on each side of an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "left": {"type": "string"},
                    "right": {"type": "string"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
        },
    }]
    row = _chat(
        model="Minimax-M3",
        messages=[_image_user(
            [image_a],
            "Inspect the image, then use report_colors exactly once.",
        )],
        session_id=session_id,
        stream=True,
        max_tokens=2048,
        tools=tools,
        tool_choice="auto",
        timeout=timeout,
    )
    _assert_cold(row, "thinking image tool cold")
    calls = (row["message"] or {}).get("tool_calls") or []
    if not calls:
        raise AssertionError(
            f"thinking image request emitted no native tool call: {row}"
        )
    if (calls[0].get("function") or {}).get("name") != "report_colors":
        raise AssertionError(f"unexpected thinking image tool call: {calls[0]}")
    if int(row.get("tokens") or 0) >= 2048:
        raise AssertionError(f"thinking image tool exhausted its token budget: {row}")
    if not row.get("reasoning_chars"):
        raise AssertionError("thinking image tool emitted no streamed reasoning")
    _summary("thinking_image_native_tool:cold", row)


def _same_image_case(model, stream, image_a, timeout):
    _reset()
    session_id = f"mm-live-{uuid.uuid4().hex[:10]}"
    user = _image_user([image_a])
    first = _chat(
        model=model,
        messages=[user],
        session_id=session_id,
        stream=stream,
        max_tokens=768 if model == "Minimax-M3" else 192,
        timeout=timeout,
    )
    _assert_cold(first, f"{model} cold")
    _summary(f"{model}:cold", first)
    followup = _chat(
        model=model,
        messages=[
            user,
            {"role": "assistant", "content": first["content"]},
            {"role": "user", "content": "Which color is on the left?"},
        ],
        session_id=session_id,
        stream=stream,
        max_tokens=768 if model == "Minimax-M3" else 128,
        timeout=timeout,
    )
    _assert_hot(followup, f"{model} same-image follow-up")
    _summary(f"{model}:same_image", followup)
    return first, followup, user, session_id


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = _health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    failed_before = int(initial.get("requests_failed") or 0)
    image_a = _image_uri((255, 0, 0), (0, 0, 255))
    image_b = _image_uri((0, 255, 0), (255, 255, 0))

    first, _followup, user, session_id = _same_image_case(
        "Minimax-M3-No-Think", True, image_a, args.timeout
    )
    changed = _chat(
        model="Minimax-M3-No-Think",
        messages=[_image_user([image_b])],
        session_id=session_id,
        stream=True,
        max_tokens=192,
        timeout=args.timeout,
    )
    _assert_cold(changed, "changed image")
    if changed["image_fingerprint"] == first["image_fingerprint"]:
        raise AssertionError("changed image retained the old fingerprint")
    _summary("changed_image:cold", changed)

    removed = _chat(
        model="Minimax-M3-No-Think",
        messages=[{"role": "user", "content": "Answer only: no image."}],
        session_id=session_id,
        stream=True,
        max_tokens=64,
        timeout=args.timeout,
    )
    _assert_cold(removed, "image removal")
    _summary("image_removed:cold", removed)

    first_think, _hot_think, _think_user, _think_sid = _same_image_case(
        "Minimax-M3", False, image_a, args.timeout
    )
    if not first_think["reasoning_chars"]:
        raise AssertionError("thinking image request emitted no reasoning")

    _reset()
    compact_sid = f"mm-compact-{uuid.uuid4().hex[:10]}"
    compact_first = _chat(
        model="Minimax-M3-No-Think",
        messages=[user],
        session_id=compact_sid,
        stream=True,
        max_tokens=192,
        timeout=args.timeout,
    )
    compacted = _chat(
        model="Minimax-M3-No-Think",
        messages=[
            user,
            {
                "role": "assistant",
                "content": "Compacted summary: the image contains red and blue.",
            },
            {"role": "user", "content": "Name the right-side color."},
        ],
        session_id=compact_sid,
        stream=True,
        max_tokens=128,
        timeout=args.timeout,
    )
    _assert_hot(compacted, "compacted same-image history")
    _summary("compacted_history:safe_prefix_hit", compacted)

    _reset()
    order_sid = f"mm-order-{uuid.uuid4().hex[:10]}"
    ordered_user = _image_user([image_a, image_b], "Describe image one, then image two.")
    ordered = _chat(
        model="Minimax-M3-No-Think",
        messages=[ordered_user],
        session_id=order_sid,
        stream=True,
        max_tokens=192,
        timeout=args.timeout,
    )
    _assert_cold(ordered, "ordered multi-image cold")
    reversed_row = _chat(
        model="Minimax-M3-No-Think",
        messages=[_image_user([image_b, image_a], "Describe image one, then image two.")],
        session_id=order_sid,
        stream=True,
        max_tokens=192,
        timeout=args.timeout,
    )
    _assert_cold(reversed_row, "reversed multi-image order")
    if reversed_row["image_fingerprint"] == ordered["image_fingerprint"]:
        raise AssertionError("image order did not change the fingerprint")
    _summary("multi_image_reordered:cold", reversed_row)

    _image_tool_case(image_a, args.timeout)
    _thinking_image_tool_case(image_a, args.timeout)

    final = _health()
    if int(final.get("requests_failed") or 0) != failed_before:
        raise AssertionError(
            f"failure count changed: {failed_before} -> {final.get('requests_failed')}"
        )
    if final.get("active_request"):
        raise AssertionError("server still has an active request")
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
