#!/usr/bin/env python3
"""Gate a long-context thinking + image + native-tool request."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid


PROBES = os.path.dirname(os.path.abspath(__file__))
if PROBES not in sys.path:
    sys.path.insert(0, PROBES)

import m3_multimodal_cache_live_probe as live


def _long_instruction(records):
    lines = [
        f"record_{index:05d}: alpha beta gamma delta epsilon value_{index:05d}"
        for index in range(records)
    ]
    return (
        "Inspect the image and retain the records below. After inspection, "
        "call report_colors exactly once with the color on the left and the "
        "color on the right.\n"
        + "\n".join(lines)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=live.BASE)
    parser.add_argument("--records", type=int, default=6500)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--followup-only", action="store_true")
    args = parser.parse_args()
    live.BASE = args.base.rstrip("/")

    health = live._health()
    if health.get("status") != "healthy" or health.get("active_request"):
        raise SystemExit(f"endpoint is not idle and healthy: {health}")
    failed_before = int(health.get("requests_failed") or 0)
    if not args.followup_only:
        live._reset()

    image_uri = live._image_uri((255, 0, 0), (0, 0, 255))
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
    session_id = (
        args.session_id
        or f"mm-long-thinking-tool-{uuid.uuid4().hex[:10]}"
    )
    initial_user = live._image_user(
        [image_uri],
        _long_instruction(args.records),
    )
    if args.followup_only:
        row = live._chat(
            model="Minimax-M3",
            messages=[
                initial_user,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_report_colors",
                        "type": "function",
                        "function": {
                            "name": "report_colors",
                            "arguments": json.dumps({
                                "left": "red",
                                "right": "blue",
                            }),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_report_colors",
                    "content": "colors recorded",
                },
                {
                    "role": "user",
                    "content": (
                        "The color report succeeded. Reply with only DONE and "
                        "do not call another tool."
                    ),
                },
            ],
            session_id=session_id,
            stream=True,
            max_tokens=128,
            tools=tools,
            tool_choice="auto",
            timeout=args.timeout,
        )
        live._assert_hot(row, "long multimodal tool-result follow-up")
        if "DONE" not in (row.get("content") or "").upper():
            raise AssertionError(f"unexpected follow-up answer: {row}")
        print(json.dumps({
            "ok": True,
            "followup": True,
            "prompt_tokens": row.get("prompt_tokens"),
            "physical_reuse_tokens": row.get("physical_reuse_tokens"),
            "cache_action": row.get("cache_action"),
            "server_ttft_s": row.get("server_ttft_s"),
            "decode_tps": row.get("decode_tps"),
            "content": row.get("content"),
            "failed": live._health().get("requests_failed"),
        }, sort_keys=True), flush=True)
        print("PASS", flush=True)
        return

    row = live._chat(
        model="Minimax-M3",
        messages=[initial_user],
        session_id=session_id,
        stream=True,
        max_tokens=args.max_tokens,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "report_colors"},
        },
        timeout=args.timeout,
    )
    calls = (row.get("message") or {}).get("tool_calls") or []
    if not calls:
        raise AssertionError(f"long multimodal turn emitted no tool call: {row}")
    function = calls[0].get("function") or {}
    if function.get("name") != "report_colors":
        raise AssertionError(f"unexpected tool call: {calls[0]}")
    arguments = json.loads(function.get("arguments") or "{}")
    if not arguments.get("left") or not arguments.get("right"):
        raise AssertionError(f"incomplete color arguments: {arguments}")
    if int(row.get("tokens") or 0) >= args.max_tokens:
        raise AssertionError(f"generation exhausted its token budget: {row}")
    if not row.get("reasoning_chars"):
        raise AssertionError("thinking stream emitted no reasoning")

    final = live._health()
    if final.get("active_request"):
        raise AssertionError(f"request slot did not return to idle: {final}")
    if int(final.get("requests_failed") or 0) != failed_before:
        raise AssertionError("server failure count changed")
    print(json.dumps({
        "ok": True,
        "records": args.records,
        "prompt_tokens": row.get("prompt_tokens"),
        "prompt_tps": row.get("prompt_tps"),
        "server_ttft_s": row.get("server_ttft_s"),
        "decode_tps": row.get("decode_tps"),
        "generation_tokens": row.get("tokens"),
        "reasoning_chars": row.get("reasoning_chars"),
        "tool": function.get("name"),
        "arguments": arguments,
        "failed": final.get("requests_failed"),
    }, sort_keys=True), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
