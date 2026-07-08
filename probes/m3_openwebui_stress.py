#!/usr/bin/env python3
import base64
import io
import json
import sys
import time
import urllib.request

from PIL import Image


BASE = "http://127.0.0.1:8080"
BASELINE_FAILED = 0


def post(path, payload, timeout=180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(timeout=5):
    with urllib.request.urlopen(BASE + "/health", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_idle(label, timeout=120):
    start = time.time()
    last = None
    while time.time() - start < timeout:
        h = health()
        last = h
        if h.get("active_request") is None and h.get("request_queue_depth") == 0:
            return h
        time.sleep(1)
    raise RuntimeError(f"{label}: server did not go idle; last health={last}")


def stream_chat(payload, *, read_chunks=None, timeout=240):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            if line.startswith("data: "):
                chunks.append(line[6:])
            if read_chunks is not None and len(chunks) >= read_chunks:
                break
            if time.time() - started > timeout:
                raise TimeoutError("stream timed out")
    return chunks


def extract_text(chunks):
    text = []
    reasoning = []
    keepalives = 0
    for chunk in chunks:
        if chunk == "[DONE]":
            continue
        obj = json.loads(chunk)
        delta = obj["choices"][0].get("delta") or {}
        if not delta:
            keepalives += 1
        text.append(delta.get("content") or "")
        reasoning.append(delta.get("reasoning") or "")
        reasoning.append(delta.get("reasoning_content") or "")
    return "".join(text), "".join(reasoning), keepalives


def png_data_uri():
    im = Image.new("RGB", (32, 16))
    px = im.load()
    for y in range(16):
        for x in range(32):
            px[x, y] = (255, 0, 0) if x < 16 else (0, 0, 255)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def assert_idle_clean(label):
    h = wait_idle(label)
    if h.get("requests_failed", 0) > BASELINE_FAILED:
        raise RuntimeError(f"{label}: server has failed requests: {h}")
    return h


def main():
    global BASELINE_FAILED
    initial = health()
    BASELINE_FAILED = int(initial.get("requests_failed", 0) or 0)
    print("initial", json.dumps(initial, sort_keys=True), flush=True)
    messages = [
        {"role": "system", "content": "You are concise and accurate."},
        {"role": "user", "content": "Think briefly, then answer exactly: alpha-ok"},
    ]

    print("turn1 thinking stream", flush=True)
    chunks = stream_chat({
        "model": "m3",
        "messages": messages,
        "stream": True,
        "max_tokens": 512,
        "temperature": 0,
    })
    text, reasoning, keepalives = extract_text(chunks)
    print("turn1", text[:120], "reasoning_chars", len(reasoning), "keepalives", keepalives, flush=True)
    if "alpha-ok" not in text.lower():
        raise RuntimeError("turn1 missing expected answer")
    assert_idle_clean("after turn1")

    messages.append({"role": "assistant", "content": text, "reasoning": reasoning})
    messages.append({"role": "user", "content": "Second turn: answer exactly beta-ok and nothing else."})
    print("turn2 OpenWebUI-style with prior reasoning", flush=True)
    chunks = stream_chat({
        "model": "m3",
        "messages": messages,
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0,
    }, timeout=300)
    text2, reasoning2, keepalives2 = extract_text(chunks)
    print("turn2", text2[:160], "reasoning_chars", len(reasoning2), "keepalives", keepalives2, flush=True)
    if "beta-ok" not in text2.lower():
        raise RuntimeError("turn2 missing expected answer")
    assert_idle_clean("after turn2")

    print("deliberate disconnect after initial chunks", flush=True)
    _ = stream_chat({
        "model": "m3",
        "messages": [{"role": "user", "content": "Write a numbered list of 120 short items about stable APIs."}],
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.2,
    }, read_chunks=1, timeout=120)
    post_disconnect = wait_idle("after disconnect", timeout=180)
    print("post_disconnect", json.dumps(post_disconnect, sort_keys=True), flush=True)

    print("queued follow-up after disconnect", flush=True)
    rsp = post("/v1/chat/completions", {
        "model": "m3-no-think",
        "messages": [{"role": "user", "content": "Return exactly gamma-ok"}],
        "stream": False,
        "max_tokens": 128,
        "temperature": 0,
    })
    content = rsp["choices"][0]["message"]["content"]
    print("queued", content[:120], flush=True)
    if "gamma-ok" not in content.lower():
        raise RuntimeError("queued follow-up missing expected answer")
    assert_idle_clean("after queued follow-up")

    print("image recognition stream", flush=True)
    chunks = stream_chat({
        "model": "m3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What colors are visible? Answer in one sentence."},
                {"type": "image_url", "image_url": {"url": png_data_uri()}},
            ],
        }],
        "stream": True,
        "temperature": 0,
    }, timeout=180)
    image_text, image_reasoning, image_keepalives = extract_text(chunks)
    print("image", image_text[:200], "reasoning_chars", len(image_reasoning), "keepalives", image_keepalives, flush=True)
    if not ("red" in image_text.lower() and "blue" in image_text.lower()):
        raise RuntimeError("image answer did not mention red and blue")
    assert_idle_clean("after image")

    records = "\n".join(f"REC-{i:03d}: value HC-{i:03d}" for i in range(180))
    print("high-context lookup", flush=True)
    rsp = post("/v1/chat/completions", {
        "model": "m3-no-think",
        "messages": [{
            "role": "user",
            "content": records + "\n\nReturn only the value for REC-137.",
        }],
        "stream": False,
        "max_tokens": 96,
        "temperature": 0,
    }, timeout=240)
    hc = rsp["choices"][0]["message"]["content"]
    print("high_context", hc[:120], flush=True)
    if "HC-137" not in hc:
        raise RuntimeError("high-context lookup failed")
    final = assert_idle_clean("final")
    print("final", json.dumps(final, sort_keys=True), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAIL", repr(e), file=sys.stderr, flush=True)
        try:
            print("health", json.dumps(health(), sort_keys=True), file=sys.stderr, flush=True)
        except Exception:
            pass
        raise
