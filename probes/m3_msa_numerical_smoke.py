#!/usr/bin/env python3
"""Numerically check ThunderMLX's production MiniMax-M3 MSA kernels."""

from __future__ import annotations

import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLX_M3_OMLX_MINIMAX_OVERLAY", "1")
os.environ.setdefault("MLX_M3_KERNEL_STATS", "1")

import mlx.core as mx
import sharded_server as server


assert server._install_omlx_minimax_overlay()

from mlx_vlm.models.minimax_m3_vl import msa


BLOCK_SIZE = 128
HEAD_DIM = 128
H_Q = 64
H_KV = 4
SCALE = HEAD_DIM ** -0.5


def _selected_tokens(blocks):
    return mx.concatenate([
        mx.arange(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE)
        for block in blocks
    ])


def _dense_selected_decode(q, k, v, blocks):
    token_ids = _selected_tokens(blocks)
    keys = mx.take(k, token_ids, axis=2)
    values = mx.take(v, token_ids, axis=2)
    repeats = H_Q // H_KV
    keys = mx.repeat(keys, repeats, axis=1)
    values = mx.repeat(values, repeats, axis=1)
    scores = mx.matmul(
        q.astype(mx.float32),
        keys.astype(mx.float32).swapaxes(-1, -2),
    ) * SCALE
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, values.astype(mx.float32)).astype(q.dtype)


def _dense_selected_prefill(q, k, v, blocks):
    token_ids = _selected_tokens(blocks)
    keys = mx.take(k, token_ids, axis=0).transpose(1, 0, 2)
    values = mx.take(v, token_ids, axis=0).transpose(1, 0, 2)
    repeats = H_Q // H_KV
    keys = mx.repeat(keys, repeats, axis=0)
    values = mx.repeat(values, repeats, axis=0)
    queries = q.transpose(1, 0, 2)
    scores = mx.matmul(
        queries.astype(mx.float32),
        keys.astype(mx.float32).swapaxes(-1, -2),
    ) * SCALE
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, values.astype(mx.float32)).transpose(1, 0, 2).astype(q.dtype)


def _assert_close(label, actual, expected, *, max_abs_limit=0.003, rel_limit=0.02):
    mx.eval(actual, expected)
    max_abs = float(mx.max(mx.abs(
        actual.astype(mx.float32) - expected.astype(mx.float32)
    )).item())
    scale = float(mx.max(mx.abs(expected.astype(mx.float32))).item())
    relative = max_abs / max(scale, 1e-8)
    assert max_abs <= max_abs_limit, (label, max_abs, relative)
    assert relative <= rel_limit, (label, max_abs, relative)
    print(f"{label}: max_abs={max_abs:.7f} relative={relative:.7f}")


def check_topk_builders():
    length = 16
    total = 1024
    topk = 4
    idx_q = mx.random.normal((1, H_KV, length, HEAD_DIM)).astype(mx.bfloat16)
    idx_k = mx.random.normal((1, 1, total, HEAD_DIM)).astype(mx.bfloat16)
    standard = msa.build_grouped_msa_topk(
        idx_q, idx_k, total - length, SCALE, BLOCK_SIZE, topk, 0, 1
    )
    blockwise = msa.build_grouped_msa_topk_blockwise(
        idx_q,
        idx_k,
        total - length,
        SCALE,
        BLOCK_SIZE,
        topk,
        0,
        1,
        block_chunk_size=2,
    )
    mx.eval(standard, blockwise)
    mismatches = int(mx.sum(standard != blockwise).item())
    assert mismatches == 0, mismatches
    print("topk builders: exact match")


def check_prefill_kernel():
    length = 8
    total = 1024
    blocks = [0, 2, 4, 6]
    q = mx.random.normal((length, H_Q, HEAD_DIM)).astype(mx.bfloat16)
    k = mx.random.normal((total, H_KV, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal((total, H_KV, HEAD_DIM)).astype(mx.bfloat16)
    q2k = mx.broadcast_to(
        mx.array(blocks, dtype=mx.int32)[None, None, :],
        (H_KV, length, len(blocks)),
    )
    actual = msa.msa_sparse_attention_b1(
        q,
        k,
        v,
        q2k,
        q_start=total - length,
        scale=SCALE,
        block_size=BLOCK_SIZE,
        k1_impl="steel_mma",
        full_splits=True,
    )
    expected = _dense_selected_prefill(q, k, v, blocks)
    _assert_close("prefill steel_mma", actual, expected)


def check_decode_kernel():
    total = 1024
    blocks = [0, 2, 4, 6]
    q = mx.random.normal((1, H_Q, 1, HEAD_DIM)).astype(mx.bfloat16)
    k = mx.random.normal((1, H_KV, total, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal((1, H_KV, total, HEAD_DIM)).astype(mx.bfloat16)
    topk_idx = mx.array(blocks, dtype=mx.int32)[None, None, None, :]
    topk_valid = mx.ones(topk_idx.shape, dtype=mx.bool_)
    actual = msa.msa_sparse_decode_b1_mma(
        q,
        k,
        v,
        topk_idx,
        topk_valid,
        q_pos=total - 1,
        scale=SCALE,
        block_size=BLOCK_SIZE,
    )
    expected = _dense_selected_decode(q, k, v, blocks)
    _assert_close("decode steel_mma", actual, expected)


def main():
    assert mx.metal.is_available()
    mx.random.seed(20260714)
    check_topk_builders()
    check_prefill_kernel()
    check_decode_kernel()
    print("PASS")


if __name__ == "__main__":
    main()
