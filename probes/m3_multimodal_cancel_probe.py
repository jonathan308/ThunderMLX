#!/usr/bin/env python3
"""Validate coordinated cancellation for image prefill and image decode."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import uuid


PROBES = os.path.dirname(os.path.abspath(__file__))
if PROBES not in sys.path:
    sys.path.insert(0, PROBES)

import m3_multimodal_cache_live_probe as live


def request_json(base, method, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health(base):
    return request_json(base, "GET", "/health", timeout=5)


def wait_idle(base, timeout=180):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health(base)
        prompt_cache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and int(last.get("request_queue_depth") or 0) == 0
            and not prompt_cache.get("in_use")
        ):
            return last
        time.sleep(0.2)
    raise TimeoutError(f"server did not return to clean idle: {last}")


def stream_worker(base, payload, result):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                item = line[6:]
                if item == "[DONE]":
                    result["done"] = True
                    break
                event = json.loads(item)
                result["events"] += 1
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        result["visible_events"] += 1
    except Exception as exc:  # Probe reports the exact transport outcome.
        result["error"] = repr(exc)


def start_stream(base, payload):
    result = {"events": 0, "visible_events": 0, "done": False, "error": None}
    thread = threading.Thread(
        target=stream_worker,
        args=(base, payload, result),
        daemon=True,
    )
    thread.start()
    return thread, result


def wait_active(base, predicate, timeout=90):
    deadline = time.time() + timeout
    samples = []
    while time.time() < deadline:
        active = health(base).get("active_request")
        if active:
            samples.append(active)
            if predicate(active):
                return active, samples
        time.sleep(0.15)
    raise TimeoutError(f"requested active phase was not observed: {samples[-4:]}")


def stop_active(base, active):
    request_id = active.get("id")
    if not request_id:
        raise AssertionError(f"active request has no id: {active}")
    stopped = request_json(
        base,
        "POST",
        "/v1/stop",
        {"request_id": request_id},
        timeout=30,
    )
    if not stopped.get("stopped"):
        raise AssertionError(f"server rejected coordinated stop: {stopped}")
    return stopped


def assert_clean(base, failed_before, thread, result, label):
    final = wait_idle(base, timeout=180)
    thread.join(timeout=20)
    if thread.is_alive():
        raise AssertionError(f"{label} stream worker did not exit")
    if int(final.get("requests_failed") or 0) != failed_before:
        raise AssertionError(
            f"{label} changed failure count: {failed_before} -> "
            f"{final.get('requests_failed')}"
        )
    if (final.get("prompt_cache") or {}).get("in_use"):
        raise AssertionError(f"{label} left prompt cache in use")
    return final


def image_payload(image_uri, text, session_id, max_tokens=4096):
    return {
        "model": "Minimax-M3-No-Think",
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "metadata": {"session_id": session_id},
        "messages": [live._image_user([image_uri], text)],
    }


def cancel_during_image_prefill(base, image_uri, failed_before):
    nonce = uuid.uuid4().hex
    records = "\n".join(
        f"cancel_record_{index:05d}_{nonce}: alpha beta gamma delta epsilon"
        for index in range(3500)
    )
    payload = image_payload(
        image_uri,
        "Inspect the image and read every record before replying.\n" + records,
        f"mm-cancel-prefill-{nonce}",
    )
    thread, result = start_stream(base, payload)

    def in_prefill(active):
        total = int(active.get("prefill_total_tokens") or 0)
        processed = int(active.get("prefill_processed_tokens") or 0)
        phase = str(active.get("phase") or active.get("stage") or "").lower()
        return (total > 0 and processed < total) or "prefill" in phase

    active, samples = wait_active(base, in_prefill, timeout=90)
    stopped = stop_active(base, active)
    final = assert_clean(base, failed_before, thread, result, "image prefill cancel")
    max_processed = max(
        (int(sample.get("prefill_processed_tokens") or 0) for sample in samples),
        default=0,
    )
    max_total = max(
        (int(sample.get("prefill_total_tokens") or 0) for sample in samples),
        default=0,
    )
    if max_total and max_processed >= max_total:
        raise AssertionError(
            "image prefill completed before cancellation "
            f"({max_processed}/{max_total})"
        )
    return {
        "request_id": active.get("id"),
        "prefill_processed_tokens": max_processed,
        "prefill_total_tokens": max_total,
        "stop_mode": stopped.get("mode"),
        "stream": result,
        "completed": final.get("requests_completed"),
    }


def cancel_during_image_decode(base, image_uri, failed_before):
    nonce = uuid.uuid4().hex
    payload = image_payload(
        image_uri,
        (
            "Describe the image, then write a numbered list of 2,000 distinct "
            "facts about distributed inference. Continue until explicitly stopped."
        ),
        f"mm-cancel-decode-{nonce}",
        max_tokens=8192,
    )
    thread, result = start_stream(base, payload)

    def in_decode(active):
        tokens = int(active.get("tokens_emitted") or active.get("tokens") or 0)
        phase = str(active.get("phase") or active.get("stage") or "").lower()
        return tokens >= 12 or ("decode" in phase and result["visible_events"] >= 2)

    active, _samples = wait_active(base, in_decode, timeout=120)
    stopped = stop_active(base, active)
    final = assert_clean(base, failed_before, thread, result, "image decode cancel")
    if result["visible_events"] < 1:
        raise AssertionError(f"decode cancel observed no generated output: {result}")
    return {
        "request_id": active.get("id"),
        "tokens_emitted": active.get("tokens_emitted"),
        "stop_mode": stopped.get("mode"),
        "stream": result,
        "completed": final.get("requests_completed"),
    }


def post_cancel_smoke(base, image_uri, failed_before):
    live.BASE = base
    row = live._chat(
        model="Minimax-M3-No-Think",
        messages=[live._image_user([image_uri], "Name the left-side color only.")],
        session_id=f"mm-post-cancel-{uuid.uuid4().hex}",
        stream=True,
        max_tokens=64,
        timeout=180,
    )
    if not row["content"].strip():
        raise AssertionError(f"post-cancel image smoke returned empty output: {row}")
    if row["requests_failed"] != failed_before:
        raise AssertionError(f"post-cancel image smoke changed failures: {row}")
    return {
        "content": row["content"][:80],
        "ttft_s": row["server_ttft_s"],
        "decode_tps": row["decode_tps"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    initial = wait_idle(base)
    defaults = initial.get("generation_defaults") or {}
    if not defaults.get("image_prompt_cache_enabled"):
        raise SystemExit("MLX_M3_IMAGE_PROMPT_CACHE is disabled")
    failed_before = int(initial.get("requests_failed") or 0)
    image_uri = live._image_uri((255, 0, 0), (0, 0, 255))
    results = {
        "prefill": cancel_during_image_prefill(base, image_uri, failed_before),
        "decode": cancel_during_image_decode(base, image_uri, failed_before),
        "followup": post_cancel_smoke(base, image_uri, failed_before),
    }
    final = wait_idle(base)
    print(json.dumps({
        "ok": True,
        "failed_before": failed_before,
        "failed_after": final.get("requests_failed"),
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
