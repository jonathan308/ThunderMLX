#!/usr/bin/env python3
"""Build and restore an exact image-bearing SSD prompt/KV cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


PROBES = os.path.dirname(os.path.abspath(__file__))
if PROBES not in sys.path:
    sys.path.insert(0, PROBES)

import m3_multimodal_cache_live_probe as live
import m3_persistent_cache_probe as persistent


def _seed_user(records, image_uri):
    lines = [
        f"record_{index:05d}: alpha beta gamma delta epsilon value_{index:05d}"
        for index in range(records)
    ]
    text = (
        "Memorize these records and inspect the image. "
        "When ready, answer exactly CACHE_READY.\n"
        + "\n".join(lines)
    )
    return live._image_user([image_uri], text)


def _state_path(session_id):
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in session_id)
    return f"/private/tmp/{safe}-multimodal-ssd.json"


def _print(label, value):
    print(json.dumps({label: value}, sort_keys=True), flush=True)


def _build(args, image_uri, state_file):
    persistent.post_admin(
        "/admin/prompt-cache/reset",
        {"reason": "multimodal SSD build reset", "clear_memory": False},
    )
    user = _seed_user(args.records, image_uri)
    built = live._chat(
        model="Minimax-M3-No-Think",
        messages=[user],
        session_id=args.session_id,
        stream=True,
        max_tokens=64,
        timeout=args.timeout,
    )
    if built["physical_cache_hit"]:
        raise AssertionError("SSD build unexpectedly started from a hot image cache")
    if int(built["prompt_tokens"] or 0) < 8192:
        raise AssertionError(f"build prompt too small for SSD gate: {built}")
    state = {
        "session_id": args.session_id,
        "records": args.records,
        "assistant_content": built["content"],
        "prompt_tokens": built["prompt_tokens"],
        "image_fingerprint": built["image_fingerprint"],
        "written_at": round(time.time(), 3),
    }
    persistent.write_probe_state(state_file, state)
    _print("build", {
        "prompt_tokens": built["prompt_tokens"],
        "prompt_tps": built["prompt_tps"],
        "decode_tps": built["decode_tps"],
        "server_ttft_s": built["server_ttft_s"],
        "image_fingerprint": built["image_fingerprint"],
        "state_file": state_file,
    })
    saved = persistent.post_admin(
        "/admin/prompt-cache/ssd/save",
        {"reason": "multimodal SSD acceptance save"},
        timeout=args.timeout,
    )
    _print("save", persistent.compact_for_log(saved))
    time.sleep(2)
    summary = persistent.ssd_summary()
    if int(summary.get("last_saved_tokens") or 0) < 8192:
        raise AssertionError(f"multimodal SSD save was not observed: {summary}")
    return state


def _restore(args, image_uri, state_file, state=None):
    state = state or persistent.read_probe_state(state_file)
    if state.get("session_id") != args.session_id:
        raise AssertionError("state session id does not match")
    if int(state.get("records") or 0) != args.records:
        raise AssertionError("state record count does not match")
    persistent.post_admin(
        "/admin/prompt-cache/reset",
        {"reason": "multimodal SSD restore RAM reset", "clear_memory": False},
    )
    user = _seed_user(args.records, image_uri)
    restored = live._chat(
        model="Minimax-M3-No-Think",
        messages=[
            user,
            {"role": "assistant", "content": state["assistant_content"]},
            {
                "role": "user",
                "content": (
                    f"Reply with only the value for record_{args.records - 1:05d}."
                ),
            },
        ],
        session_id=args.session_id,
        stream=True,
        max_tokens=64,
        timeout=args.timeout,
    )
    live._assert_hot(restored, "multimodal SSD restore")
    prepare = restored["prepare"]
    if not prepare.get("restored_ssd_cache"):
        raise AssertionError(f"request did not report SSD restoration: {restored}")
    summary = persistent.ssd_summary()
    if int(summary.get("last_restored_tokens") or 0) < 8192:
        raise AssertionError(f"multimodal SSD restore was not observed: {summary}")
    _print("restore", {
        "server_ttft_s": restored["server_ttft_s"],
        "prompt_tokens": restored["prompt_tokens"],
        "cached_tokens": restored["cached_tokens"],
        "physical_reuse_tokens": restored["physical_reuse_tokens"],
        "media_safe_prefix_min": restored["media_safe_prefix_min"],
        "restored_ssd_cache": prepare.get("restored_ssd_cache"),
        "ssd_last_restored_tokens": summary.get("last_restored_tokens"),
        "decode_tps": restored["decode_tps"],
        "content": restored["content"][:120],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=live.BASE)
    parser.add_argument(
        "--phase", choices=("build", "restore", "roundtrip"), default="roundtrip"
    )
    parser.add_argument("--records", type=int, default=700)
    parser.add_argument("--session-id", default="multimodal-ssd-10k")
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--clear-ssd", action="store_true")
    args = parser.parse_args()
    live.BASE = args.base.rstrip("/")
    persistent.BASE = live.BASE
    state_file = args.state_file or _state_path(args.session_id)
    image_uri = live._image_uri((255, 0, 0), (0, 0, 255))

    health = live._health()
    defaults = health.get("generation_defaults") or {}
    if health.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {health}")
    if not defaults.get("image_prompt_cache_enabled"):
        raise SystemExit("MLX_M3_IMAGE_PROMPT_CACHE is disabled")
    if args.clear_ssd:
        persistent.post_admin(
            "/admin/prompt-cache/ssd/clear",
            {"reason": "multimodal SSD probe clear"},
            timeout=args.timeout,
        )

    state = None
    if args.phase in {"build", "roundtrip"}:
        state = _build(args, image_uri, state_file)
    if args.phase in {"restore", "roundtrip"}:
        _restore(args, image_uri, state_file, state=state)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
