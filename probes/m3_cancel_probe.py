#!/usr/bin/env python3
"""Validate ThunderMLX in-flight cancellation behavior.

Checks three OpenAI-compatible stop paths:
  1. POST /v1/stop during a live stream.
  2. Client-side SSE disconnect by closing the HTTP response.
  3. Dashboard proxy POST /api/generation/stop.
  4. Client-side SSE disconnect while a long prompt is still prefilling.

The probe intentionally uses small prompts and low token thresholds so it can be
run on the production cluster without burning a long generation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request


def post_json(url: str, payload: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_idle(base_url: str, *, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = get_json(f"{base_url}/health")
        if not last.get("active_request") and int(last.get("request_queue_depth") or 0) == 0:
            return last
        time.sleep(0.25)
    raise RuntimeError(f"server did not become idle; last health={last}")


def assert_stop_defaults(health: dict, *, require_unsafe_stop: bool = False) -> dict:
    defaults = health.get("generation_defaults") or {}
    actual = {
        "unsafe_inflight_stop": defaults.get("unsafe_inflight_stop"),
        "stop_on_client_disconnect": defaults.get("stop_on_client_disconnect"),
        "stop_check_every": defaults.get("stop_check_every"),
    }
    if require_unsafe_stop:
        expected = {
            "unsafe_inflight_stop": True,
            "stop_on_client_disconnect": True,
            "stop_check_every": 4,
        }
        if actual != expected:
            raise RuntimeError(f"unexpected stop defaults: expected={expected} actual={actual}")
    elif actual.get("stop_check_every") != 4:
        raise RuntimeError(f"unexpected stop_check_every: {actual}")
    if "lifetime_tokens" not in health:
        raise RuntimeError("/health missing lifetime_tokens")
    return actual


def idle_stop_safe_mode(base_url: str) -> dict:
    stopped = post_json(f"{base_url}/v1/stop")
    if stopped.get("stopped"):
        raise RuntimeError(f"safe-mode idle stop unexpectedly stopped a request: {stopped}")
    if stopped.get("mode") != "drain_only":
        raise RuntimeError(f"unexpected safe-mode stop response: {stopped}")
    return stopped


def stream_worker(base_url: str, payload: dict, result: dict) -> None:
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for raw in response:
                if raw.startswith(b"data:"):
                    result["lines"] += 1
                if b"[DONE]" in raw:
                    result["done"] = True
                    break
    except Exception as exc:  # noqa: BLE001 - probe reports exact transport failure.
        result["error"] = repr(exc)


def active_request(base_url: str) -> dict | None:
    return get_json(f"{base_url}/health").get("active_request")


def wait_active(base_url: str, *, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        active = active_request(base_url)
        if active:
            return active
        time.sleep(0.1)
    raise RuntimeError("no active request observed")


def explicit_stop(base_url: str, baseline_failed: int) -> dict:
    payload = {
        "model": "Minimax-M3",
        "stream": True,
        "max_tokens": 2048,
        "metadata": {"session_id": "probe-explicit-stop"},
        "messages": [{
            "role": "user",
            "content": "Write a long checklist about cancelling distributed inference. Continue until stopped.",
        }],
    }
    result = {"lines": 0, "done": False, "error": None}
    thread = threading.Thread(target=stream_worker, args=(base_url, payload, result), daemon=True)
    thread.start()
    before = wait_active(base_url)
    time.sleep(0.2)
    stopped = post_json(f"{base_url}/v1/stop")
    if not stopped.get("stopped") or stopped.get("mode") != "distributed_token_boundary":
        raise RuntimeError(f"bad /v1/stop response: {stopped}")
    final = wait_idle(base_url)
    thread.join(timeout=10)
    if int(final.get("requests_failed") or 0) > baseline_failed:
        raise RuntimeError(f"explicit stop incremented failures: {final}")
    return {
        "before": before,
        "stop_response": stopped,
        "stream_result": result,
        "final_completed": final.get("requests_completed"),
    }


def client_disconnect(base_url: str, baseline_failed: int) -> dict:
    payload = {
        "model": "Minimax-M3",
        "stream": True,
        "max_tokens": 2048,
        "metadata": {"session_id": "probe-client-disconnect"},
        "messages": [{
            "role": "user",
            "content": "Write a detailed operations note about server cancellation and streaming clients.",
        }],
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    lines = 0
    with urllib.request.urlopen(req, timeout=60) as response:
        for raw in response:
            if raw.startswith(b"data:"):
                lines += 1
            if lines >= 4:
                response.close()
                break
    samples = []
    for _ in range(80):
        health = get_json(f"{base_url}/health")
        active = health.get("active_request")
        samples.append(active)
        if not active:
            final = health
            break
        time.sleep(0.25)
    else:
        raise RuntimeError(f"client disconnect did not release slot; samples={samples[-4:]}")
    first_active = next((item for item in samples if item), None)
    if not first_active or not first_active.get("cancel_requested"):
        raise RuntimeError(f"disconnect did not expose cancel_requested; samples={samples[:4]}")
    if first_active.get("cancel_reason") != "client_disconnect":
        raise RuntimeError(f"unexpected cancel reason: {first_active}")
    if int(final.get("requests_failed") or 0) > baseline_failed:
        raise RuntimeError(f"client disconnect incremented failures: {final}")
    return {
        "lines_before_close": lines,
        "first_active_after_close": {
            key: first_active.get(key)
            for key in ("id", "tokens_emitted", "client_connected", "cancel_requested", "cancel_reason")
        },
        "final_completed": final.get("requests_completed"),
    }


def client_disconnect_during_prefill(base_url: str, baseline_failed: int) -> dict:
    unique = f"{time.time():.6f}"
    unit = (
        f" cancellation-prefill-boundary-regression-{unique} "
        + ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 80)
    )
    content = (unit * 70)[:240_000]
    payload = {
        "model": "Minimax-M3-No-Think",
        "stream": True,
        "max_tokens": 1024,
        "metadata": {"session_id": f"probe-prefill-disconnect-{unique}"},
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": (
                    "Read this context, then answer briefly. Context:\n"
                    + content
                    + "\nQuestion: summarize one cancellation detail."
                ),
            },
        ],
    }
    fd, payload_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(payload_path, "w", encoding="utf-8") as payload_file:
        json.dump(payload, payload_file)
    output = tempfile.NamedTemporaryFile(delete=False)
    output.close()
    out_handle = open(output.name, "wb")
    proc = subprocess.Popen(
        [
            "curl", "-N", "-sS", "--no-buffer",
            "-H", "Content-Type: application/json",
            "-X", "POST", f"{base_url}/v1/chat/completions",
            "--data-binary", "@" + payload_path,
        ],
        stdout=out_handle,
        stderr=subprocess.STDOUT,
    )

    prefill_started_sample = None
    saw_active = False
    close_deadline = time.time() + 30
    active_started_at = None
    while time.time() < close_deadline:
        active = active_request(base_url)
        if active:
            saw_active = True
            if active_started_at is None:
                active_started_at = time.time()
        if active and int(active.get("prefill_total_tokens") or 0) > 0:
            processed = int(active.get("prefill_processed_tokens") or 0)
            total = int(active.get("prefill_total_tokens") or 0)
            if processed > 0 and processed < total:
                prefill_started_sample = active
                break
        # Some runtime paths do not surface prefill telemetry before the client
        # is dropped, but keeping the socket open for this long still exercises
        # cancellation while a cold long prompt is in flight.
        if active_started_at is not None and time.time() - active_started_at >= 14:
            break
        time.sleep(0.5)
    if not saw_active:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        out_handle.close()
        raise RuntimeError("long prefill request did not become active before timeout")

    proc.terminate()

    samples = []
    final = None
    for _ in range(120):
        health = get_json(f"{base_url}/health")
        active = health.get("active_request")
        samples.append(active)
        if not active:
            final = health
            break
        time.sleep(0.5)
    if final is None:
        raise RuntimeError(f"prefill disconnect did not release slot; samples={samples[-4:]}")

    active_samples = [item for item in samples if item]
    cancel_sample = next((item for item in active_samples if item.get("cancel_requested")), None)
    if not cancel_sample and active_samples:
        raise RuntimeError(f"prefill disconnect did not expose cancel_requested; samples={active_samples[:4]}")
    if cancel_sample and cancel_sample.get("cancel_reason") != "client_disconnect":
        raise RuntimeError(f"unexpected prefill cancel reason: {cancel_sample}")
    if int(final.get("requests_failed") or 0) > baseline_failed:
        raise RuntimeError(f"prefill disconnect incremented failures: {final}")

    prefill_samples = [
        item for item in active_samples
        if int(item.get("prefill_total_tokens") or 0) > 0
    ]
    max_processed = max(
        (int(item.get("prefill_processed_tokens") or 0) for item in prefill_samples),
        default=0,
    )
    max_total = max(
        (int(item.get("prefill_total_tokens") or 0) for item in prefill_samples),
        default=0,
    )
    if max_total and max_processed >= max_total:
        raise RuntimeError(
            "prefill disconnect drained the entire prompt instead of cancelling: "
            f"processed={max_processed} total={max_total}"
        )
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    out_handle.close()
    try:
        with open(output.name, "rb") as output_file:
            lines_before_close = sum(
                1 for line in output_file if line.startswith(b"data:")
            )
    except OSError:
        lines_before_close = 0
    for temp_path in (payload_path, output.name):
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return {
        "lines_before_close": lines_before_close,
        "reader_result": {
            "process_returncode": proc.returncode,
            "output_bytes": os.path.getsize(output.name) if os.path.exists(output.name) else 0,
        },
        "prefill_started_sample": (
            {
                key: prefill_started_sample.get(key)
                for key in (
                    "id", "prefill_processed_tokens", "prefill_total_tokens",
                    "tokens_emitted",
                )
            }
            if prefill_started_sample else None
        ),
        "first_cancel_sample": {
            key: cancel_sample.get(key)
            for key in (
                "id", "client_connected", "cancel_requested", "cancel_reason",
                "prefill_processed_tokens", "prefill_total_tokens",
                "tokens_emitted",
            )
        } if cancel_sample else None,
        "max_prefill_processed": max_processed,
        "max_prefill_total": max_total,
        "final_completed": final.get("requests_completed"),
    }


def dashboard_proxy_stop(base_url: str, dashboard_url: str, baseline_failed: int) -> dict:
    payload = {
        "model": "Minimax-M3",
        "stream": True,
        "max_tokens": 2048,
        "metadata": {"session_id": "probe-dashboard-stop"},
        "messages": [{
            "role": "user",
            "content": "Think through a cancellation smoke test. Continue until stopped.",
        }],
    }
    result = {"lines": 0, "done": False, "error": None}
    thread = threading.Thread(target=stream_worker, args=(base_url, payload, result), daemon=True)
    thread.start()
    before = wait_active(base_url)
    stopped = post_json(f"{dashboard_url}/api/generation/stop")
    if not stopped.get("stopped") or stopped.get("mode") != "distributed_token_boundary":
        raise RuntimeError(f"bad dashboard stop response: {stopped}")
    final = wait_idle(base_url)
    thread.join(timeout=10)
    if int(final.get("requests_failed") or 0) > baseline_failed:
        raise RuntimeError(f"dashboard stop incremented failures: {final}")
    return {
        "before": before,
        "stop_response": stopped,
        "stream_result": result,
        "final_completed": final.get("requests_completed"),
    }


def post_cancel_followup(base_url: str, baseline_failed: int) -> dict:
    payload = {
        "model": "Minimax-M3-No-Think",
        "stream": True,
        "max_tokens": 64,
        "metadata": {"session_id": "probe-post-cancel-followup"},
        "messages": [{
            "role": "user",
            "content": "Reply with exactly: OK.",
        }],
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = 0
    text = ""
    with urllib.request.urlopen(req, timeout=60) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            part = delta.get("content") or delta.get("reasoning") or ""
            text += part
            chunks += 1
    final = wait_idle(base_url)
    if int(final.get("requests_failed") or 0) > baseline_failed:
        raise RuntimeError(f"post-cancel followup incremented failures: {final}")
    if not text.strip():
        raise RuntimeError("post-cancel followup returned empty text")
    return {
        "chunks": chunks,
        "text": text.strip(),
        "final_completed": final.get("requests_completed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--require-unsafe-stop",
        action="store_true",
        help=(
            "Require MLX_M3_ENABLE_UNSAFE_INFLIGHT_STOP=1 and "
            "MLX_M3_STOP_ON_CLIENT_DISCONNECT=1, then run live abort tests."
        ),
    )
    args = parser.parse_args()

    initial = wait_idle(args.base_url)
    stop_defaults = assert_stop_defaults(
        initial, require_unsafe_stop=args.require_unsafe_stop
    )
    baseline_failed = int(initial.get("requests_failed") or 0)
    before_lifetime = dict(initial.get("lifetime_tokens") or {})

    if args.require_unsafe_stop:
        results = {
            "explicit_stop": explicit_stop(args.base_url, baseline_failed),
            "client_disconnect": client_disconnect(args.base_url, baseline_failed),
            "client_disconnect_during_prefill": client_disconnect_during_prefill(
                args.base_url, baseline_failed
            ),
            "dashboard_proxy_stop": dashboard_proxy_stop(
                args.base_url, args.dashboard_url, baseline_failed
            ),
            "post_cancel_followup": post_cancel_followup(args.base_url, baseline_failed),
        }
    else:
        results = {
            "safe_mode": {
                "stop_defaults": stop_defaults,
                "idle_stop_response": idle_stop_safe_mode(args.base_url),
                "skipped_live_abort_tests": True,
                "reason": (
                    "unsafe distributed token-boundary stop and client-disconnect "
                    "abort are disabled in production defaults"
                ),
            },
            "post_cancel_followup": post_cancel_followup(args.base_url, baseline_failed),
        }
    final = wait_idle(args.base_url)
    if int(final.get("requests_failed") or 0) > baseline_failed:
        raise RuntimeError(f"failure count increased: before={baseline_failed} final={final}")
    after_lifetime = dict(final.get("lifetime_tokens") or {})
    if int(after_lifetime.get("processed_total_live") or 0) <= int(before_lifetime.get("processed_total_live") or 0):
        raise RuntimeError(
            f"lifetime token counter did not increase: before={before_lifetime} after={after_lifetime}"
        )
    print(json.dumps({
        "ok": True,
        "baseline_failed": baseline_failed,
        "final_failed": final.get("requests_failed"),
        "final_completed": final.get("requests_completed"),
        "lifetime_before": before_lifetime,
        "lifetime_after": after_lifetime,
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
