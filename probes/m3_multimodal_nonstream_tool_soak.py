#!/usr/bin/env python3
"""Repeat the non-stream image/tool shape that previously wedged rank sync."""

from __future__ import annotations

import argparse
import json

import m3_multimodal_cache_live_probe as live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=live.BASE)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    live.BASE = args.base.rstrip("/")
    initial = live._health()
    failed_before = int(initial.get("requests_failed") or 0)
    completed_before = int(initial.get("requests_completed") or 0)
    image = live._image_uri((255, 0, 0), (0, 0, 255))

    for cycle in range(1, max(1, args.repeats) + 1):
        live._image_tool_case(image, args.timeout)
        status = live._health()
        if status.get("active_request"):
            raise AssertionError(f"cycle {cycle} left an active request")
        if int(status.get("requests_failed") or 0) != failed_before:
            raise AssertionError(
                f"cycle {cycle} changed failure count: "
                f"{failed_before} -> {status.get('requests_failed')}"
            )
        print(
            json.dumps(
                {
                    "cycle": cycle,
                    "completed": status.get("requests_completed"),
                    "failed": status.get("requests_failed"),
                    "status": status.get("status"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    final = live._health()
    completed_delta = int(final.get("requests_completed") or 0) - completed_before
    expected = max(1, args.repeats) * 2
    if completed_delta != expected:
        raise AssertionError(
            f"expected {expected} completed requests, got {completed_delta}"
        )
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
