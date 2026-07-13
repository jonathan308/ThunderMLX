#!/usr/bin/env python3
"""Exact-output and latency A/B for the optional native MiniMax MSA top-k."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MLX_M3_KERNEL_STATS", "1")

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[3]
MSA_PATH = ROOT / "MSA Support/mlx_vlm/models/minimax_m3_vl/msa.py"
spec = importlib.util.spec_from_file_location("m3_msa_native_ab", MSA_PATH)
MSA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MSA)

from omlx.custom_kernels.minimax_m3 import fast as native_fast


def bench(fn, *, warmup: int, iters: int):
    out = None
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    return out, (time.perf_counter() - started) * 1000.0 / iters


def run_case(
    length: int,
    context: int,
    block_chunk: int,
    iters: int,
    native_input_dtype: str,
) -> dict:
    if context < length:
        raise ValueError("context must be at least length")
    mx.random.seed(length * 1_000_003 + context)
    queries = mx.random.normal((1, 4, length, 128)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 1, context, 128)).astype(mx.bfloat16)
    mx.eval(queries, keys)
    args = (queries, keys, context - length, 128**-0.5, 128, 16, 0, 1)

    MSA._NATIVE_MSA_TOPK_MODE = "off"
    fallback, fallback_ms = bench(
        lambda: MSA.build_grouped_msa_topk_blockwise(
            *args, block_chunk_size=block_chunk
        ),
        warmup=1,
        iters=iters,
    )

    native_queries = queries.astype(mx.float32) if native_input_dtype == "fp32" else queries
    native_keys = keys.astype(mx.float32) if native_input_dtype == "fp32" else keys
    native_args = (
        native_queries,
        native_keys,
        context - length,
        128**-0.5,
        128,
        16,
        0,
        1,
    )
    native, native_ms = bench(
        lambda: native_fast.minimax_msa_topk(
            native_args[0],
            native_args[1],
            q_start=native_args[2],
            scale=native_args[3],
            block_size=native_args[4],
            topk=native_args[5],
            init_blocks=native_args[6],
            local_blocks=native_args[7],
        ),
        warmup=1,
        iters=iters,
    )
    exact = bool(mx.array_equal(fallback, native).item())
    mismatch = int(mx.sum(fallback != native).item())
    row = {
        "length": length,
        "context": context,
        "block_chunk": block_chunk,
        "native_input_dtype": native_input_dtype,
        "exact": exact,
        "mismatch": mismatch,
        "fallback_ms": round(fallback_ms, 3),
        "native_ms": round(native_ms, 3),
        "speedup": round(fallback_ms / native_ms, 3),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
    }
    del queries, keys, native_queries, native_keys, fallback, native
    mx.clear_cache()
    return row


def parse_shape(value: str) -> tuple[int, int]:
    length, context = value.lower().split("x", 1)
    return int(length), int(context)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        default="512x4096,2048x16384,4096x32768,4096x80000",
    )
    parser.add_argument("--block-chunk", type=int, default=32)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--native-input-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()

    if not native_fast.is_native_available():
        raise SystemExit(f"native extension unavailable: {native_fast.import_error()}")
    rows = []
    for value in args.shapes.split(","):
        length, context = parse_shape(value.strip())
        row = run_case(
            length,
            context,
            args.block_chunk,
            args.iters,
            args.native_input_dtype,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not row["exact"] and not args.allow_mismatch:
            raise SystemExit(f"native output mismatch: {row}")
    print(json.dumps({"pass": True, "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
