#!/usr/bin/env python3
"""Decompose the per-token cost of exact MSA decode selection (reuse=0).

Replicates the decode-step selection sequence across N_LAYERS sparse layers at
several context sizes, isolating:
  A. block-scores kernel only
  B. + top-k select kernel (the full fused pair, params PREBUILT once)
  C. + fresh mx.array params per layer (current production behavior)
  D. index-cache append+fetch per layer, donation-friendly (view dropped)
  E. index-cache append+fetch per layer, donation-BLOCKED (view retained)

Run ONLY when the cluster is idle/stopped on this machine (GPU contention
invalidates results):
  ~/mlx-vlm064-env/bin/python3.14 ops/fable_lab/selection_microbench.py
"""
import importlib.util, sys, time
import mlx.core as mx
import os
_MSA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "MSA Support", "mlx_vlm", "models", "minimax_m3_vl", "msa.py")
_spec = importlib.util.spec_from_file_location("fable_msa", _MSA_PATH)
msa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msa)

N_LAYERS = 57          # sparse layers per token (60 minus dense)
H_IDX, D_IDX = 4, 128  # index heads (queries), index dim
BLOCK, TOPK = 64, 16
INIT_B, LOCAL_B = 1, 2
REPS = 30


def bench(fn, warmup=5, reps=REPS):
    for _ in range(warmup):
        fn()
        mx.eval(mx.zeros(1))
    mx.synchronize() if hasattr(mx, "synchronize") else mx.eval(mx.zeros(1))
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    mx.synchronize() if hasattr(mx, "synchronize") else mx.eval(mx.zeros(1))
    return (time.perf_counter() - t0) / reps * 1000  # ms per "token"


def main():
    print(f"mlx {mx.__version__ if hasattr(mx,'__version__') else '?'}; "
          f"layers={N_LAYERS} h_idx={H_IDX} d={D_IDX} block={BLOCK} topk={TOPK}")
    for ctx in (25_000, 50_000, 100_000, 200_000):
        num_blocks = (ctx + BLOCK - 1) // BLOCK
        q = mx.random.normal((1, H_IDX, 1, D_IDX)).astype(mx.bfloat16)
        keys = mx.random.normal((1, 1, ctx, D_IDX)).astype(mx.bfloat16)
        mx.eval(q, keys)
        results = {}

        # --- A: block-scores kernel only, prebuilt params, eval at end
        params_a = mx.array([ctx, num_blocks], dtype=mx.int32)
        threads = 256
        blocks_per_tg = max(threads // BLOCK, 1)
        tgs_per_head = (num_blocks + blocks_per_tg - 1) // blocks_per_tg
        def run_a():
            outs = []
            for _ in range(N_LAYERS):
                s = msa._MSA_DECODE_BLOCK_SCORES(
                    inputs=[q, keys, params_a],
                    template=[("T", q.dtype), ("H_IDX", H_IDX), ("D_IDX", D_IDX),
                              ("BLOCK_SIZE", BLOCK), ("THREADS", threads)],
                    grid=(H_IDX * tgs_per_head * threads, 1, 1),
                    threadgroup=(threads, 1, 1),
                    output_shapes=[(H_IDX, num_blocks)],
                    output_dtypes=[mx.float32])[0]
                outs.append(s)
            mx.eval(*outs)
        results["A_scores_only"] = bench(run_a)

        # --- B: full pair via the public API but with our own prebuilt params
        #        (msa_decode_select_topk builds params internally = variant C;
        #         here we call the two kernels directly with shared params)
        params_b = mx.array([num_blocks, num_blocks - 1], dtype=mx.int32)
        def run_b():
            outs = []
            for _ in range(N_LAYERS):
                s = msa._MSA_DECODE_BLOCK_SCORES(
                    inputs=[q, keys, params_a],
                    template=[("T", q.dtype), ("H_IDX", H_IDX), ("D_IDX", D_IDX),
                              ("BLOCK_SIZE", BLOCK), ("THREADS", threads)],
                    grid=(H_IDX * tgs_per_head * threads, 1, 1),
                    threadgroup=(threads, 1, 1),
                    output_shapes=[(H_IDX, num_blocks)],
                    output_dtypes=[mx.float32])[0]
                t = msa._MSA_DECODE_TOPK_SELECT_SMEM(
                    inputs=[s, params_b],
                    template=[("H_IDX", H_IDX), ("TOPK", TOPK),
                              ("INIT_BLOCKS", INIT_B), ("LOCAL_BLOCKS", LOCAL_B),
                              ("THREADS", threads), ("SMEM_CAP", 4096)],
                    grid=(H_IDX * threads, 1, 1),
                    threadgroup=(threads, 1, 1),
                    output_shapes=[(1, H_IDX, 1, TOPK)],
                    output_dtypes=[mx.int32])[0]
                outs.append(t)
            mx.eval(*outs)
        results["B_pair_shared_params"] = bench(run_b)

        # --- C: production path — msa_decode_select_topk (fresh params inside)
        def run_c():
            outs = []
            for _ in range(N_LAYERS):
                t = msa.msa_decode_select_topk(
                    q, keys, q_pos=ctx - 1, block_size=BLOCK, topk=TOPK,
                    init_blocks=INIT_B, local_blocks=LOCAL_B)
                outs.append(t)
            mx.eval(*outs)
        results["C_production_api"] = bench(run_c)

        # --- F: production API with the v2 kernel
        msa._MSA_SELECT_V2 = True
        results["F_v2_api"] = bench(run_c)
        msa._MSA_SELECT_V2 = False

        # --- D/E: index append + fetch (slice_update donation behavior)
        step = 256
        cap = ((ctx + step) // step + 1) * step
        def make_buf():
            b = mx.zeros((1, 1, cap, D_IDX), dtype=mx.bfloat16)
            mx.eval(b)
            return b
        newk = mx.random.normal((1, 1, 1, D_IDX)).astype(mx.bfloat16)
        mx.eval(newk)

        bufs_d = [make_buf() for _ in range(N_LAYERS)]
        off_d = [ctx] * N_LAYERS
        def run_d():  # donation-friendly: returned view dropped each step
            touched = []
            for i in range(N_LAYERS):
                b = bufs_d[i]
                o = off_d[i]
                b[..., o:o + 1, :] = newk
                bufs_d[i] = b
                off_d[i] = o + 1
                touched.append(b[..., :o + 1, :].sum())  # consume view, drop
            mx.eval(*touched)
            for i in range(N_LAYERS):
                off_d[i] = ctx  # reset so buffer never overflows
        results["D_append_donation_ok"] = bench(run_d)

        bufs_e = [make_buf() for _ in range(N_LAYERS)]
        off_e = [ctx] * N_LAYERS
        retained = [None] * N_LAYERS
        def run_e():  # donation-BLOCKED: previous view retained across steps
            touched = []
            for i in range(N_LAYERS):
                b = bufs_e[i]
                o = off_e[i]
                b[..., o:o + 1, :] = newk
                bufs_e[i] = b
                retained[i] = b[..., :o + 1, :]  # keep a live reference
                touched.append(retained[i][0, 0, 0, 0])
                off_e[i] = ctx
            mx.eval(*touched)
        results["E_append_donation_blocked"] = bench(run_e)

        line = " ".join(f"{k}={v:6.2f}ms" for k, v in results.items())
        print(f"ctx={ctx:>7,}: {line}", flush=True)


if __name__ == "__main__":
    main()
