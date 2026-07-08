#!/usr/bin/env python3
"""Offline structural test: depth-2 chunked-prefill overlap (m3_batch_cancel).

Exercises docs/DESIGN-chunk-overlap-native.md's pipelined prefill loop against
the REAL mlx-vlm 0.6.4 PromptProcessingBatch/GenerationBatch machinery with a
tiny mock LM (CPU device, hidden=16, 3 layers with the production
MiniMaxM3KVCache/KVCache mixed cache layout, 84-token prompt) so it is safe to
run on a box hosting a live cluster. No distributed group: the cancel
ctrl-word all_sum degenerates to a size-1 group.

Sections:
  0. Drift guard: _require_lazy_prefill_support rejects a shape-less object
     and accepts a real 0.6.4 PromptProcessingBatch (proven implicitly by
     section 1 taking the lazy path for every chunk).
  1. Identical output: _run over a 10-chunk prompt (prefill_step_size=8,
     84 tokens) with the overlap gate OFF (serial upstream prompt_step) and
     ON (overlap loop). Greedy sampling; asserts byte-identical generated
     token sequences, identical final cache offsets/contents across all
     layers, identical prefill progress callbacks, exactly 10 lazy chunk
     steps in the ON run and 0 in the OFF run, and that the
     m3_pipeline_patch overlap flag was set then cleared.
  2. Fallback wiring: with the gate ON but support-check forced to raise
     Unsupported, _run silently uses the serial loop (0 lazy steps, flag
     never touched) and still produces the baseline tokens.

Run:
  ~/mlx-vlm064-env/bin/python3.14 ops/test_overlap_shapes.py
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Keep every allocation off the GPU: a live inference cluster shares this box.
mx.set_default_device(mx.cpu)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import m3_batch_cancel as mbc  # noqa: E402
import m3_pipeline_patch as mpp  # noqa: E402
from mlx_vlm.models.cache import KVCache  # noqa: E402
from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache  # noqa: E402

VOCAB, HIDDEN = 96, 16
PREFILL_STEP = 8
PROMPT_LEN = 10 * PREFILL_STEP + 4  # 10 chunked steps, 4 tokens for generate()
MAX_TOKENS = 6
PROMPT_IDS = [(i * 2654435761) % VOCAB for i in range(PROMPT_LEN)]


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _bytes(a: mx.array) -> bytes:
    return np.asarray(a).tobytes()


# ---------------------------------------------------------------------------
# Tiny mock model: real cache API, deterministic causal attention per layer so
# every chunk's logits depend on all previous chunks' cache contents (any
# overlap mis-ordering changes the tokens).
# ---------------------------------------------------------------------------

class _MockInner:
    """The .model of the LM: owns embed_tokens like MiniMaxM3Model."""

    def __init__(self, n_layers: int):
        self.embed = mx.random.normal((VOCAB, HIDDEN)) * 0.5
        self.layers = [
            tuple(mx.random.normal((HIDDEN, HIDDEN)) * 0.3 for _ in range(3))
            for _ in range(n_layers)
        ]
        self.w_out = mx.random.normal((HIDDEN, VOCAB)) * 0.5

    def embed_tokens(self, ids: mx.array) -> mx.array:
        return self.embed[ids]

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(inputs)
        for c, (wq, wk, wv) in zip(cache, self.layers):
            B, L, D = h.shape
            q = (h @ wq)[:, None]  # [B, 1(head), L, D]
            k = (h @ wk)[:, None]
            v = (h @ wv)[:, None]
            keys, values = c.update_and_fetch(k, v)
            if hasattr(c, "update_index_and_fetch"):  # M3 sparse-index layers
                c.update_index_and_fetch(k)
            S = keys.shape[2]
            scores = (q @ keys.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(D))
            q_pos = mx.arange(S - L, S)[:, None]
            k_pos = mx.arange(S)[None, :]
            scores = mx.where(k_pos <= q_pos, scores, mx.array(-1e9, scores.dtype))
            h = h + (mx.softmax(scores, axis=-1) @ values)[:, 0]
        return h @ self.w_out


class _MockLanguageModel:
    def __init__(self):
        self.model = _MockInner(n_layers=3)

    def make_cache(self):
        # Production MiniMax-M3 layout: sparse-index and full-attention layers.
        return [MiniMaxM3KVCache(), KVCache(), MiniMaxM3KVCache()]

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        return self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)


class _MockModel:
    def __init__(self):
        self.language_model = _MockLanguageModel()

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(self, inputs, cache=None, inputs_embeds=None, n_to_process=None,
                 **kwargs):
        return self.language_model(inputs, cache=cache, inputs_embeds=inputs_embeds)


class _MockDetokenizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokens = []
        self.last_segment = ""

    def add_token(self, tok):
        self.tokens.append(int(tok))
        self.last_segment = f"<{int(tok)}>"

    def finalize(self):
        self.last_segment = ""


class _MockProcessor:
    def __init__(self):
        self.detokenizer = _MockDetokenizer()


class _MockTokenizer:
    @staticmethod
    def stopping_criteria(tok) -> bool:
        return False  # generation always ends via max_tokens ("length")


def _cache_refs(prompt_cache):
    out = []
    for c in prompt_cache:
        if isinstance(c, MiniMaxM3KVCache):
            (k, v), idx = c.state
            out.append((int(c.offset), int(c.index_offset),
                        _bytes(k), _bytes(v), _bytes(idx)))
        else:
            k, v = c.state
            out.append((int(c.offset), _bytes(k), _bytes(v)))
    return out


def _run_once(model, overlap: bool):
    """One full _run pass on fresh caches. Returns a dict of observables."""
    greedy = lambda logprobs: mx.argmax(logprobs, axis=-1)  # noqa: E731
    prompt_cache = model.make_cache()
    progress = []

    lazy_calls = {"n": 0}
    orig_lazy = mbc._prompt_step_lazy

    def counting_lazy(pb):
        lazy_calls["n"] += 1
        return orig_lazy(pb)

    flag_calls = []
    orig_setter = mpp.set_prefill_overlap_active

    def recording_setter(active):
        flag_calls.append(bool(active))
        orig_setter(active)

    orig_gate = mbc._OVERLAP_NATIVE_ENABLED
    mbc._OVERLAP_NATIVE_ENABLED = overlap
    mbc._prompt_step_lazy = counting_lazy
    mpp.set_prefill_overlap_active = recording_setter
    try:
        results = list(mbc._run(
            rank=0,
            model=model,
            processor=_MockProcessor(),
            tokenizer=_MockTokenizer(),
            input_ids=mx.array([PROMPT_IDS], dtype=mx.int32),
            max_tokens=MAX_TOKENS,
            sampler=greedy,
            processors=None,
            prompt_cache=prompt_cache,
            prefill_step_size=PREFILL_STEP,
            progress_cb=lambda p, t: progress.append((p, t)),
        ))
    finally:
        mbc._OVERLAP_NATIVE_ENABLED = orig_gate
        mbc._prompt_step_lazy = orig_lazy
        mpp.set_prefill_overlap_active = orig_setter

    return {
        "tokens": [r.token for r in results],
        "finish": [r.finish_reason for r in results],
        "prompt_tokens": {r.prompt_tokens for r in results},
        "progress": progress,
        "lazy_calls": lazy_calls["n"],
        "flag_calls": flag_calls,
        "cache": _cache_refs(prompt_cache),
        "offsets": [int(c.offset) for c in prompt_cache],
    }


def main() -> int:
    # Section 2's expected fallback warning should interleave with the [PASS]
    # lines instead of landing on unbuffered stderr out of order.
    logging.basicConfig(stream=sys.stdout, format="[log] %(message)s")

    # Seed chosen so the greedy decode does NOT collapse to a fixed point:
    # a varied token sequence is a much stronger equivalence check (asserted
    # below) than a repeated one.
    mx.random.seed(1)
    model = _MockModel()

    # ---- Section 0: drift guard ----------------------------------------
    print("== 0. upstream-drift guard ==")
    try:
        mbc._require_lazy_prefill_support(object())
    except mbc.Unsupported as e:
        assert "PromptProcessingBatch drift" in str(e), str(e)
        _ok(f"shape-less object rejected with Unsupported: {e}")
    else:
        raise AssertionError("expected Unsupported for a shape-less batch")

    # ---- Section 1: overlap off vs on, identical output ------------------
    print("== 1. overlap off vs on ==")
    off = _run_once(model, overlap=False)
    on = _run_once(model, overlap=True)

    n_chunks = (PROMPT_LEN - 4) // PREFILL_STEP
    assert off["lazy_calls"] == 0, f"serial run used lazy path: {off['lazy_calls']}"
    assert off["flag_calls"] == [], f"serial run touched flag: {off['flag_calls']}"
    assert on["lazy_calls"] == n_chunks, (
        f"overlap run took {on['lazy_calls']} lazy steps, expected {n_chunks} "
        "(support check may have silently fallen back to serial)"
    )
    assert on["flag_calls"] == [True, False], on["flag_calls"]
    assert mpp._M3_PREFILL_OVERLAP_ACTIVE is False, "flag left set after run"
    _ok(f"overlap loop engaged: {n_chunks} lazy chunk steps, flag set then cleared")

    assert len(off["tokens"]) == MAX_TOKENS, off["tokens"]
    assert off["finish"][-1] == "length" and all(
        f is None for f in off["finish"][:-1]
    ), off["finish"]
    assert off["prompt_tokens"] == {PROMPT_LEN}
    assert len(set(off["tokens"])) >= 4, (
        f"degenerate greedy sequence weakens the check: {off['tokens']}"
    )
    assert off["tokens"] == on["tokens"], (
        f"token divergence:\n  off={off['tokens']}\n  on ={on['tokens']}"
    )
    assert on["finish"] == off["finish"]
    _ok(f"generated tokens identical (greedy, {MAX_TOKENS} tokens): {on['tokens']}")

    expect_off = PROMPT_LEN + MAX_TOKENS
    assert off["offsets"] == [expect_off] * 3, off["offsets"]
    assert on["offsets"] == off["offsets"], (
        f"final cache offsets differ: off={off['offsets']} on={on['offsets']}"
    )
    assert on["cache"] == off["cache"], "final cache contents differ between runs"
    _ok(f"final cache offsets identical across all layers: {on['offsets']} "
        "(KV + index bytes identical too)")

    expected_progress = [
        ((i + 1) * PREFILL_STEP, PROMPT_LEN) for i in range(n_chunks)
    ]
    assert off["progress"] == expected_progress, off["progress"]
    assert on["progress"] == expected_progress, (
        f"overlap progress bookkeeping differs: {on['progress']}"
    )
    _ok("prefill progress callbacks identical: "
        f"{n_chunks} x {PREFILL_STEP} tokens up to {n_chunks * PREFILL_STEP}")

    # ---- Section 2: Unsupported -> serial fallback ----------------------
    print("== 2. drift fallback ==")
    orig_check = mbc._require_lazy_prefill_support

    def forced_unsupported(pb):
        raise mbc.Unsupported("forced by test")

    mbc._require_lazy_prefill_support = forced_unsupported
    try:
        fb = _run_once(model, overlap=True)
    finally:
        mbc._require_lazy_prefill_support = orig_check
    assert fb["lazy_calls"] == 0, "fallback still took the lazy path"
    assert fb["flag_calls"] == [], "fallback touched the overlap flag"
    assert fb["tokens"] == off["tokens"] and fb["offsets"] == off["offsets"]
    _ok("gate on + Unsupported support-check falls back to the serial loop "
        "with baseline-identical output")

    print("== VERDICT ==")
    print("overlap on/off: byte-identical tokens, offsets, cache contents; "
          "pipelined loop engaged for every chunk; drift guard + fallback OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
