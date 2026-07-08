#!/usr/bin/env python3
"""Make the per-chunk mx.clear_cache() in chunked prefill cadence-gated.

The MLX-VLM AR generator calls ``mx.clear_cache()`` after every prefill chunk,
returning the whole Metal buffer pool to the OS each time. On the ThunderMLX
2-node pipeline that contributes to rank 1 unwiring/rewiring its ~84GB working
set once per chunk (measured 3GB<->89GB wired sawtooth during every long
prefill). ``MLX_M3_PREFILL_CLEAR_CACHE_EVERY`` gates the call:

- ``1``: clear after every chunk — identical to stock behavior.
- ``N``: clear after every Nth chunk.
- ``0`` (default): never clear during chunked prefill (the MLX cache limit still
  bounds the pool; the cluster launches with MLX_M3_CACHE_LIMIT_GB=32).
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys


MARKER = "# MiniMax-M3 cluster patch: prefill clear_cache cadence"


def main() -> int:
    spec = importlib.util.find_spec("mlx_vlm.generate.ar")
    if spec is None or spec.origin is None:
        print("mlx_vlm.generate.ar not found", file=sys.stderr)
        return 2
    path = pathlib.Path(spec.origin)
    text = path.read_text()
    if MARKER in text:
        updated = re.sub(
            r'os\.environ\.get\("MLX_M3_PREFILL_CLEAR_CACHE_EVERY",\s*"[0-9]+"\) or "[0-9]+"',
            'os.environ.get("MLX_M3_PREFILL_CLEAR_CACHE_EVERY", "0") or "0"',
            text,
            count=1,
        )
        if updated != text:
            path.write_text(updated)
            print(f"updated prefill clear_cache cadence default: {path}")
            return 0
        print(f"prefill clear_cache cadence patch already installed: {path}")
        return 0

    anchor = "import os\n"
    if anchor not in text:
        print("import anchor not found", file=sys.stderr)
        return 3
    header = (
        "import os\n"
        f"{MARKER}\n"
        "_M3_PREFILL_CLEAR_CACHE_EVERY = max(\n"
        "    0, int(os.environ.get(\"MLX_M3_PREFILL_CLEAR_CACHE_EVERY\", \"0\") or \"0\")\n"
        ")\n"
        "_M3_PREFILL_CHUNK_COUNTER = {\"n\": 0}\n"
    )
    text = text.replace(anchor, header, 1)

    old_clear = (
        "                    mx.clear_cache()\n"
        "                    pbar.update(n_to_process)\n"
    )
    new_clear = (
        "                    _M3_PREFILL_CHUNK_COUNTER[\"n\"] += 1\n"
        "                    if _M3_PREFILL_CLEAR_CACHE_EVERY > 0 and (\n"
        "                        _M3_PREFILL_CHUNK_COUNTER[\"n\"]\n"
        "                        % _M3_PREFILL_CLEAR_CACHE_EVERY == 0\n"
        "                    ):\n"
        "                        mx.clear_cache()\n"
        "                    pbar.update(n_to_process)\n"
    )
    if old_clear not in text:
        print("clear_cache patch anchor not found", file=sys.stderr)
        return 4
    text = text.replace(old_clear, new_clear, 1)

    path.write_text(text)
    print(f"installed prefill clear_cache cadence patch: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
