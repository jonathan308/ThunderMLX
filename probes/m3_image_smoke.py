#!/usr/bin/env python3
"""MiniMax-M3 OpenAI-compatible image-to-text smoke test.

This sends a tiny generated red/blue PNG as a data URI through the same
`image_url` content shape used by OpenWebUI and dashboard chat. It validates
that the endpoint streams a VLM answer, identifies both colors, and returns to
idle without increasing the server failure count.
"""
import argparse
import base64
import io
import json
import time
import urllib.request

from PIL import Image


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


def wait_idle(before_completed, timeout=120):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        pcache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and int(last.get("requests_completed") or 0) > before_completed
            and not pcache.get("in_use")
        ):
            return last
        time.sleep(0.2)
    return last or health()


def red_blue_png_data_uri():
    image = Image.new("RGB", (32, 16))
    pixels = image.load()
    for y in range(16):
        for x in range(32):
            pixels[x, y] = (255, 0, 0) if x < 16 else (0, 0, 255)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def stream_image_chat(model, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What colors are visible? Answer in one concise sentence.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": red_blue_png_data_uri()},
                },
            ],
        }],
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    chunks = 0
    first_piece_s = None
    content_parts = []
    reasoning_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            chunks += 1
            evt = json.loads(item)
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                content = delta.get("content")
                if (reasoning or content) and first_piece_s is None:
                    first_piece_s = time.time() - started
                if reasoning:
                    reasoning_parts.append(reasoning)
                if content:
                    content_parts.append(content)
    return {
        "client_elapsed_s": round(time.time() - started, 3),
        "client_first_piece_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "content": "".join(content_parts),
        "reasoning_chars": len("".join(reasoning_parts)),
    }


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    before_completed = int(initial.get("requests_completed") or 0)
    before_failed = int(initial.get("requests_failed") or 0)

    row = stream_image_chat(args.model, args.max_tokens, args.timeout)
    final = wait_idle(before_completed)
    last = final.get("last_request") or {}
    text = row["content"].lower()
    result = {
        **row,
        "server_failed_before": before_failed,
        "server_failed_after": final.get("requests_failed"),
        "server_ttft_s": last.get("first_token_s"),
        "server_decode_tps": last.get("decode_tps"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "image_count": (last.get("request_shape") or {}).get("image_count"),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if "red" not in text or "blue" not in text:
        raise SystemExit("image answer did not mention red and blue")
    if int(final.get("requests_failed") or 0) > before_failed:
        raise SystemExit("server failure count increased")
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
