#!/usr/bin/env python3
"""Offline unit tests for m3_capture (feature/capture-only).

Exercises the dump writer + accumulate/finalize path with SYNTHETIC arrays
shaped like the real captures (concat width 3*6144 = 18432), with no model,
server, or cluster. Verifies the written corpus is consumable by the SAME
dataset builder the trainer uses (ops/eagle3_finetune.build_request_sequence)
and that the (hidden_t -> token_{t+1}) pairing survives the round trip.

Run:  mlx-vlm064-env/bin/python3.14 ops/test_capture_only.py
"""
import os
import sys
import glob
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# m3_capture reads config at import, so arm it BEFORE importing.
_TMP = tempfile.mkdtemp(prefix="capture_test_")
os.environ["MLX_M3_EAGLE3_CAPTURE_ONLY"] = "1"
os.environ["MLX_M3_EAGLE3_DUMP_DIR"] = _TMP
os.environ["MLX_M3_EAGLE3_CAPTURE_LAYERS"] = "1,29,56"

import numpy as np
import mlx.core as mx
import m3_capture

D = 3 * 6144  # concat capture width, matches the real dumps
_FAILS = []


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        _FAILS.append(msg)


def _reset_caps():
    # Restore module caps between tests (some tests shrink them) through the
    # live-tuning surface the server uses.
    m3_capture.set_limits(
        max_request_bytes=int(200 * (1 << 20)),
        max_total_bytes=int(100 * (1 << 30)),
    )
    m3_capture._DISABLED["value"] = False
    m3_capture._DIR_BYTES["value"] = None
    for k in m3_capture._WARNED:
        m3_capture._WARNED[k] = False


# --------------------------------------------------------------------------
def test_pure_writers():
    print("test_pure_writers")
    d = tempfile.mkdtemp(prefix="pw_", dir=_TMP)
    n, T = 5, 4
    ph = np.random.randn(1, n, D).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    m3_capture.write_prompt_file(os.path.join(d, "prompt_1.npz"), ph, ids, 77)
    hid = np.random.randn(T, D).astype(np.float32)
    tks = np.arange(T + 1, dtype=np.int64)
    m3_capture.write_decode_file(os.path.join(d, "decode_1.npz"), hid, tks)

    zp = np.load(os.path.join(d, "prompt_1.npz"))
    check(zp["prompt_hidden"].shape == (1, n, D), "prompt_hidden shape (1,n,D)")
    check(zp["prompt_hidden"].dtype == np.float16, "prompt_hidden fp16")
    check(zp["prompt_ids"].shape == (n,), "prompt_ids shape (n,)")
    check(zp["prompt_ids"].dtype == np.int32, "prompt_ids int32")
    check(int(zp["first_token"]) == 77, "first_token value")

    zd = np.load(os.path.join(d, "decode_1.npz"))
    check(zd["hidden"].shape == (T, D), "decode hidden shape (T,D)")
    check(zd["hidden"].dtype == np.float16, "decode hidden fp16")
    check(zd["tokens"].shape == (T + 1,), "decode tokens shape (T+1,)")
    check(zd["tokens"].dtype == np.int32, "decode tokens int32")


# --------------------------------------------------------------------------
def test_accumulate_finalize():
    print("test_accumulate_finalize (chunked prefill + decode singles)")
    _reset_caps()
    a, b, K = 6, 4, 5           # prompt in 2 chunks (a,b); K decode steps
    n_prompt = a + b
    m3_capture.begin_request()
    check(m3_capture.is_capturing(), "request armed after begin_request")
    # prefill chunks (multi-token) then per-step decode singles (value j marks h_j)
    m3_capture.push(mx.ones((1, a, D), dtype=mx.float32) * 0.5)
    m3_capture.push(mx.ones((1, b, D), dtype=mx.float32) * 0.5)
    for j in range(K):
        m3_capture.push(mx.ones((1, 1, D), dtype=mx.float32) * float(j))

    prompt_ids = list(range(100, 100 + n_prompt))
    tokens = [500 + j for j in range(K)]         # y0..y_{K-1}
    req_dir = m3_capture.finalize_request(prompt_ids, tokens)
    check(req_dir is not None and os.path.isdir(req_dir), "req dir written")

    zp = np.load(glob.glob(os.path.join(req_dir, "prompt_*.npz"))[0])
    check(zp["prompt_hidden"].shape == (1, n_prompt, D),
          f"prompt split n_prompt={n_prompt} (= N - K)")
    check(list(zp["prompt_ids"]) == prompt_ids, "prompt_ids preserved")
    check(int(zp["first_token"]) == tokens[0], "first_token == tokens[0]")

    zd = np.load(glob.glob(os.path.join(req_dir, "decode_*.npz"))[0])
    check(zd["hidden"].shape == (K - 1, D), "decode hidden T = K-1 (tail dropped)")
    check(list(zd["tokens"]) == tokens, "decode tokens == full stream (T+1=K)")
    # pairing: written hidden[j] must be h_j (marked value j) -> tokens[j+1]
    ok = all(abs(float(zd["hidden"][j, 0]) - j) < 1e-2 for j in range(K - 1))
    check(ok, "hidden[j] is h_j (hidden_t -> token_{t+1} preserved)")


# --------------------------------------------------------------------------
def test_consumable_by_trainer():
    print("test_consumable_by_trainer (eagle3_finetune.build_request_sequence)")
    _reset_caps()
    try:
        import eagle3_finetune as ft
    except Exception as e:
        check(False, f"import eagle3_finetune failed: {e}")
        return
    a, K = 8, 6
    m3_capture.begin_request()
    m3_capture.push(mx.ones((1, a, D), dtype=mx.float32) * 0.3)
    for j in range(K):
        m3_capture.push(mx.ones((1, 1, D), dtype=mx.float32) * float(j))
    prompt_ids = list(range(200, 200 + a))
    tokens = [900 + j for j in range(K)]
    req_dir = m3_capture.finalize_request(prompt_ids, tokens)

    seq = ft.build_request_sequence(req_dir)
    check(seq is not None, "build_request_sequence returns a sequence")
    if seq is None:
        return
    inp_tokens, inp_hidden, labels = seq
    check(inp_hidden.shape[1] == D, "consumed hidden width == D_concat")
    check(inp_tokens.shape[0] == inp_hidden.shape[0] == labels.shape[0],
          "tokens/hidden/labels lengths aligned")
    # prompt(a) + decode(K-1) positions, teacher-forcing drops the final one.
    check(inp_tokens.shape[0] == a + (K - 1) - 1,
          f"pair count == n_prompt + T - 1 ({a}+{K-1}-1)")


# --------------------------------------------------------------------------
def test_safety_caps_and_empty():
    print("test_safety (per-request cap, no-tokens, guards)")
    _reset_caps()
    # No tokens -> no dump.
    m3_capture.begin_request()
    m3_capture.push(mx.ones((1, 4, D), dtype=mx.float32))
    check(m3_capture.finalize_request([1, 2, 3, 4], []) is None,
          "K=0 writes nothing")

    # Per-request cap -> abort with no files.
    m3_capture.set_limits(max_request_bytes=1 << 10)   # 1 KB, far below one chunk
    m3_capture.begin_request()
    m3_capture.push(mx.ones((1, 64, D), dtype=mx.float32))
    for j in range(4):
        m3_capture.push(mx.ones((1, 1, D), dtype=mx.float32))
    check(not m3_capture.is_capturing(), "request marked capped")
    check(m3_capture.finalize_request(list(range(64)), [1, 2, 3, 4]) is None,
          "capped request writes nothing")
    _reset_caps()

    # OFF by default: push before begin_request is a no-op (no active request).
    m3_capture.abort_request()
    m3_capture.push(mx.ones((1, 1, D), dtype=mx.float32))  # must not raise
    check(m3_capture.finalize_request([1], [1]) is None,
          "no active request -> finalize is a no-op")

    # prompt_ids length disagreeing with n_prompt (tokenizer drift on the
    # uncached path): keep the self-consistent decode file, skip prompt.
    _reset_caps()
    a, K = 6, 4
    m3_capture.begin_request()
    m3_capture.push(mx.ones((1, a, D), dtype=mx.float32))
    for j in range(K):
        m3_capture.push(mx.ones((1, 1, D), dtype=mx.float32))
    rd = m3_capture.finalize_request(list(range(a - 2)), [1, 2, 3, 4])  # wrong len
    check(rd is not None and not glob.glob(os.path.join(rd, "prompt_*.npz")),
          "prompt_ids mismatch -> prompt file skipped")
    check(bool(glob.glob(os.path.join(rd, "decode_*.npz"))),
          "prompt_ids mismatch -> decode file still written")


# --------------------------------------------------------------------------
def main():
    try:
        test_pure_writers()
        test_accumulate_finalize()
        test_consumable_by_trainer()
        test_safety_caps_and_empty()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): " + "; ".join(_FAILS))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
