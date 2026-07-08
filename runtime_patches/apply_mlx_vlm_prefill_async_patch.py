#!/usr/bin/env python3
"""Install the MiniMax-M3 async prefill chunk-overlap patch into MLX-VLM.

The chunked prefill loop in ``mlx_vlm.generate.ar.generate_step`` runs a
synchronous ``mx.eval`` per chunk: the host thread drains the full pipeline
(rank1 compute -> send -> rank0 compute) before building the next chunk's
graph, so rank1 idles 25-40% of prefill. This patch adds an env-gated overlap:
``mx.async_eval`` the current chunk, immediately build the next, and block on
the PREVIOUS chunk — one-chunk pipelining depth, bounded memory.

Stream discipline: everything stays on the default stream per rank, so the
per-rank ordering of jaccl collectives is unchanged (the proven-fatal pattern
is concurrent collectives on DIFFERENT streams — this patch introduces none).

Gate: MLX_M3_PREFILL_ASYNC_OVERLAP=1 (default 0 = stock behavior, byte-exact).
Run on BOTH ranks (rank1: ~/mlx-env python). launch_cluster.sh must forward
the env var or ranks silently stay on the sync path.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

MARKER = "# MiniMax-M3 cluster patch: prefill async chunk overlap"

FLAG_ANCHOR = "# MiniMax-M3 cluster patch: prefill clear_cache cadence"

FLAG_BLOCK = f"""{FLAG_ANCHOR}
{MARKER}
import os as _m3_os
_M3_PREFILL_ASYNC_OVERLAP = _m3_os.environ.get("MLX_M3_PREFILL_ASYNC_OVERLAP", "0") == "1"
"""

INIT_ANCHOR = """            total_tokens = inputs_embeds.shape[1]
            processed_tokens = 0"""

INIT_REPLACEMENT = """            total_tokens = inputs_embeds.shape[1]
            processed_tokens = 0
            _m3_prev_states = None  # async overlap: previous chunk's cache states"""

EVAL_ANCHOR = """                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])"""

EVAL_REPLACEMENT = """                    quantize_cache_fn(prompt_cache)
                    if _M3_PREFILL_ASYNC_OVERLAP:
                        _m3_states = [c.state for c in prompt_cache]
                        mx.async_eval(_m3_states)
                        if _m3_prev_states is not None:
                            mx.eval(_m3_prev_states)
                        _m3_prev_states = _m3_states
                    else:
                        mx.eval([c.state for c in prompt_cache])"""

DRAIN_ANCHOR = """            input_ids = input_ids[:, -1:]"""

DRAIN_REPLACEMENT = """            if _M3_PREFILL_ASYNC_OVERLAP and _m3_prev_states is not None:
                mx.eval(_m3_prev_states)  # drain the last in-flight chunk
                _m3_prev_states = None

            input_ids = input_ids[:, -1:]"""


def main() -> int:
    check_only = "--check" in sys.argv
    spec = importlib.util.find_spec("mlx_vlm.generate.ar")
    if spec is None or spec.origin is None:
        print("mlx_vlm.generate.ar not found", file=sys.stderr)
        return 2
    path = pathlib.Path(spec.origin)
    text = path.read_text()
    if MARKER in text:
        print(f"async overlap patch already installed: {path}")
        return 0
    for anchor, name in ((FLAG_ANCHOR, "flag"), (INIT_ANCHOR, "init"),
                         (EVAL_ANCHOR, "eval"), (DRAIN_ANCHOR, "drain")):
        if text.count(anchor) != 1:
            print(f"anchor '{name}' matched {text.count(anchor)} times (need exactly 1) — "
                  f"mlx_vlm layout changed; NOT patching", file=sys.stderr)
            return 3
    if check_only:
        print(f"--check OK: all anchors match once, patch is applicable to {path}")
        return 0
    text = text.replace(FLAG_ANCHOR, FLAG_BLOCK, 1)
    text = text.replace(INIT_ANCHOR, INIT_REPLACEMENT, 1)
    text = text.replace(EVAL_ANCHOR, EVAL_REPLACEMENT, 1)
    text = text.replace(DRAIN_ANCHOR, DRAIN_REPLACEMENT, 1)
    path.write_text(text)
    print(f"async overlap patch installed: {path}")
    print("REMINDER: apply on BOTH ranks; forward MLX_M3_PREFILL_ASYNC_OVERLAP in launch_cluster.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
