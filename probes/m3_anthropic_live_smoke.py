#!/usr/bin/env python3
"""Live smoke test for the Anthropic Messages gateway route."""

from __future__ import annotations

import argparse
import json
import urllib.request


BASE = "http://127.0.0.1:8010"


def post(path, payload, timeout=240):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": "local",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def payload(stream=False):
    return {
        "model": "Minimax-M3",
        "max_tokens": 160,
        "stream": bool(stream),
        "system": "You are a tool-calling assistant. When asked for weather, call get_weather. Do not answer from memory.",
        "messages": [{
            "role": "user",
            "content": "Use the tool to get the weather in Paris.",
        }],
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        }],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }


def run_nonstream(timeout):
    with post("/v1/messages", payload(stream=False), timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tools = [block for block in data.get("content", []) if block.get("type") == "tool_use"]
    print(json.dumps({"stream": False, "stop_reason": data.get("stop_reason"), "content": data.get("content")}, sort_keys=True))
    if data.get("stop_reason") != "tool_use" or not tools:
        raise SystemExit("Anthropic non-stream tool_use missing")
    args = tools[0].get("input") or {}
    if args.get("location") != "Paris":
        raise SystemExit(f"Anthropic non-stream args malformed: {args}")


def run_stream(timeout):
    events = []
    with post("/v1/messages", payload(stream=True), timeout=timeout) as resp:
        current_event = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                current_event = None
                continue
            if line.startswith("event: "):
                current_event = line[7:]
                continue
            if line.startswith("data: "):
                events.append((current_event, json.loads(line[6:])))
    tool_start = [
        data for event, data in events
        if event == "content_block_start"
        and (data.get("content_block") or {}).get("type") == "tool_use"
    ]
    json_delta = [
        data for event, data in events
        if event == "content_block_delta"
        and (data.get("delta") or {}).get("type") == "input_json_delta"
    ]
    stop = [data for event, data in events if event == "message_stop"]
    print(json.dumps({
        "stream": True,
        "events": len(events),
        "tool_start": bool(tool_start),
        "json_delta": bool(json_delta),
        "message_stop": bool(stop),
    }, sort_keys=True))
    if not tool_start or not json_delta or not stop:
        raise SystemExit("Anthropic stream tool_use events missing")


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    run_nonstream(args.timeout)
    run_stream(args.timeout)
    print("PASS")


if __name__ == "__main__":
    main()
