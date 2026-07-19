#!/usr/bin/env python3
"""Bit-parity proof for the v2 block-scores kernel.

For many randomized cases: run v1 and v2 on identical inputs; require
(1) block_scores arrays BITWISE identical, (2) full msa_decode_select_topk
outputs identical. Covers ragged tails, small/large ctx, NaN keys, both
bf16 and fp16 dtypes.
"""
import importlib.util, random, sys
import mlx.core as mx

import os
_MSA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "MSA Support", "mlx_vlm", "models", "minimax_m3_vl", "msa.py")
_spec = importlib.util.spec_from_file_location("fable_msa", _MSA_PATH)
msa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msa)

H_IDX, D_IDX = 4, 128
TOPK, INIT_B, LOCAL_B = 16, 1, 2
THREADS = 256


def run_scores(kernel, q, keys, total_k, num_blocks, block, per_head_grid):
    blocks_per_tg = max(THREADS // block, 1)
    tgs_per_head = (num_blocks + blocks_per_tg - 1) // blocks_per_tg
    grid_x = (H_IDX if per_head_grid else 1) * tgs_per_head * THREADS
    params = mx.array([total_k, num_blocks], dtype=mx.int32)
    out = kernel(
        inputs=[q, keys, params],
        template=[("T", q.dtype), ("H_IDX", H_IDX), ("D_IDX", D_IDX),
                  ("BLOCK_SIZE", block), ("THREADS", THREADS)],
        grid=(grid_x, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(H_IDX, num_blocks)], output_dtypes=[mx.float32])[0]
    mx.eval(out)
    return out


def main():
    random.seed(11)
    mx.random.seed(11)
    fails = 0
    cases = 0
    for trial in range(60):
        block = random.choice([32, 64, 128])
        ctx = random.choice([1500, 4096, 9999, 20000, 50001, 120000])
        if ctx // block < TOPK + 4:
            ctx = block * (TOPK + 8)
        dtype = random.choice([mx.bfloat16, mx.float16])
        num_blocks = (ctx + block - 1) // block
        q = (mx.random.normal((1, H_IDX, 1, D_IDX)) * 0.7).astype(dtype)
        keys = (mx.random.normal((1, 1, ctx, D_IDX)) * 0.7).astype(dtype)
        if trial % 7 == 3:  # NaN injection
            k2 = mx.array(keys)
            k2[0, 0, ctx // 3, :] = mx.array(float("nan")).astype(dtype)
            keys = k2
        mx.eval(q, keys)
        s1 = run_scores(msa._MSA_DECODE_BLOCK_SCORES, q, keys, ctx, num_blocks, block, True)
        s2 = run_scores(msa._MSA_DECODE_BLOCK_SCORES_V2, q, keys, ctx, num_blocks, block, False)
        bitwise = bool(mx.all(
            s1.view(mx.uint32) == s2.view(mx.uint32)).item())
        # full-API selection equality (v1 vs v2 via the env switch)
        msa._MSA_SELECT_V2 = False
        t1 = msa.msa_decode_select_topk(q, keys, q_pos=ctx - 1, block_size=block,
                                        topk=TOPK, init_blocks=INIT_B,
                                        local_blocks=LOCAL_B)
        msa._MSA_SELECT_V2 = True
        t2 = msa.msa_decode_select_topk(q, keys, q_pos=ctx - 1, block_size=block,
                                        topk=TOPK, init_blocks=INIT_B,
                                        local_blocks=LOCAL_B)
        msa._MSA_SELECT_V2 = False
        sel_eq = (t1 is not None and t2 is not None
                  and bool(mx.all(t1 == t2).item()))
        cases += 1
        if not (bitwise and sel_eq):
            fails += 1
            print(f"FAIL trial={trial} block={block} ctx={ctx} dtype={dtype} "
                  f"bitwise={bitwise} sel_eq={sel_eq}")
    print(f"{cases - fails}/{cases} parity cases passed "
          f"({'BIT-EXACT' if fails == 0 else 'FAILURES PRESENT'})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
