#!/usr/bin/env python3.14
"""Numerics + microbench for the EAGLE3 split-verify attention optimization.

Split-verify replaces the dense-over-full-context masked attention that an
L=(K+1)-token EAGLE3 verify block currently takes with a two-region split that
is algebraically identical to a single softmax over the union of the regions:

  * HISTORY  [0, base)         -- fused sparse-decode MMA over top-k blocks,
                                  one call per query (~L x the L=1 decode path),
                                  keeping per-head LSE for the merge.
  * TAIL     [base, base+L)    -- dense causal L x L attention over the
                                  speculative tokens themselves.
  * MERGE    log-sum-exp weighted combine of the two region outputs.

This standalone harness imports the edited ``msa.py`` DIRECTLY by file path
(it is self-contained: stdlib + mlx.core only), so no serving package / overlay
path wiring is needed and no server is touched.

It validates two things and reports both honestly:

  1. NUMERICS -- split-verify vs
       (a) an EXACT-TARGET reference: dense f32 attention over exactly the
           blocks split-verify selected (history) + causal tail. The only
           difference should be numerical (bf16 kernel vs f32 reference).
       (b) a DENSE-FULL reference: dense f32 attention over the entire context.
           This delta is the block-sparsity approximation (inherent to the
           model's trained sparse attention), NOT a bug.

  2. MICROBENCH -- dense-full (mx.fast.scaled_dot_product_attention, the naive
     fallback) vs split-verify, single-layer MiniMax-M3 shapes, at several
     context sizes x L. Also times a "gather+dense-selected" proxy for the
     current sparse-prefill verify path (reads only the selected KV).

Shapes are MiniMax-M3's sparse-index attention layer: H_q=64, H_kv=4, D=128,
block_size=128, topk=16 blocks, index_heads=4, index_dim=128.
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import mlx.core as mx

# ---------------------------------------------------------------------------
# Import the edited msa.py directly by path (self-contained module).
# ---------------------------------------------------------------------------
_MSA_PATH = (
    "~/ThunderMLX-eagle3/MSA Support/"
    "mlx_vlm/models/minimax_m3_vl/msa.py"
)
_spec = importlib.util.spec_from_file_location("m3_msa_split_verify", _MSA_PATH)
msa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msa)

_LANG_PATH = (
    "~/ThunderMLX-eagle3/MSA Support/"
    "mlx_vlm/models/minimax_m3_vl/language.py"
)


def _load_current_verify_ops():
    """Extract + exec the CURRENT production verify ops from language.py: the
    per-query block SELECTION (``_select_sparse_block_indices_compiled``) and
    the one-pass sparse-prefill ATTENTION kernel. Both are self-contained
    (``mx`` + ``lru_cache``), so they can be lifted out without importing the
    whole serving package. Returns (select_fn, attn_fn) or (None, None).
    """
    import functools
    from typing import Optional as _Optional
    try:
        lines = open(_LANG_PATH).read().splitlines()
        # Locate the self-contained block by name markers (robust to edits
        # elsewhere in language.py): from the select fn to just before the
        # _swiglu_oai definition that follows the prefill helpers.
        start = next(
            i for i, ln in enumerate(lines)
            if ln.startswith("def _select_sparse_block_indices_compiled(")
        )
        end = next(
            i for i, ln in enumerate(lines)
            if i > start and ln.startswith("def _swiglu_oai(")
        )
        # back up over the @partial(mx.compile...) decorator line(s)
        while end > start and lines[end - 1].lstrip().startswith("@"):
            end -= 1
        segment = "\n".join(lines[start:end])
        ns = {
            "mx": mx,
            "lru_cache": functools.lru_cache,
            "partial": functools.partial,
            "Optional": _Optional,
        }
        exec(compile(segment, "<lang_verify_extract>", "exec"), ns)
        return (
            ns.get("_select_sparse_block_indices_compiled"),
            ns.get("_minimax_m3_sparse_prefill_attention"),
        )
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load current verify ops: {e}")
        return None, None


_CUR_SELECT, _CUR_SPARSE_PREFILL = _load_current_verify_ops()

# MiniMax-M3 sparse-index attention layer geometry.
H_Q = 64
H_KV = 4
D = 128
GQA = H_Q // H_KV
BLOCK = 128
TOPK = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
INDEX_HEADS = 4
INDEX_DIM = 128
SCALE = D ** -0.5

# Attention layers in MiniMax-M3 that carry a sparse index (dense-attn layers
# and any linear/lightning layers do not use this path). Used only for the
# honest verify-pass extrapolation at the end.
N_SPARSE_LAYERS = 57  # 60 layers, first 3 are dense; adjust if config differs.
N_TOTAL_LAYERS = 60


def _clear_cache():
    for fn in ("clear_cache",):
        f = getattr(mx, fn, None)
        if callable(f):
            f()
            return
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "clear_cache"):
        metal.clear_cache()


def make_inputs(base: int, L: int, seed: int = 0):
    """Synthetic RoPE'd Q/K/V + index projections for one sparse-attn layer.

    Returns q [1,H_q,L,D], k/v [1,H_kv,base+L,D], idx_q [1,H_idx,L,idx_dim],
    idx_k [1,1,base+L,idx_dim]. Values are treated as already-RoPE'd and
    already-normalized -- the attention math is position-agnostic apart from
    the causal structure fixed by ``base`` and ``L``.
    """
    mx.random.seed(seed)
    total = base + L
    q = mx.random.normal((1, H_Q, L, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, H_KV, total, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, H_KV, total, D)).astype(mx.bfloat16)
    idx_q = mx.random.normal((1, INDEX_HEADS, L, INDEX_DIM)).astype(mx.bfloat16)
    idx_k = mx.random.normal((1, 1, total, INDEX_DIM)).astype(mx.bfloat16)
    mx.eval(q, k, v, idx_q, idx_k)
    return q, k, v, idx_q, idx_k


def select_history_blocks(idx_q, idx_k, base: int):
    """Shared top-k block selection over the history [0, base).

    Mirrors the L=1 decode selection: score the history index-keys against a
    single representative query (the last speculative query), force the local
    block near ``base``, and take the top-k blocks per index head. Returns
    (topk_idx [1,H_idx,1,topk] int32, topk_valid same-shape bool).
    """
    idx_q_rep = mx.contiguous(idx_q[:, :, -1:, :])          # [1,H_idx,1,idx_dim]
    idx_k_hist = mx.contiguous(idx_k[:, :, :base, :])       # [1,1,base,idx_dim]
    topk_idx = msa.msa_decode_select_topk(
        idx_q_rep,
        idx_k_hist,
        q_pos=base - 1,
        block_size=BLOCK,
        topk=TOPK,
        init_blocks=INIT_BLOCKS,
        local_blocks=LOCAL_BLOCKS,
    )
    topk_idx = topk_idx.astype(mx.int32)
    topk_valid = topk_idx >= 0
    return topk_idx, topk_valid


def split_verify(q, k, v, topk_idx, topk_valid, base: int, impl: str = "fused"):
    return msa.msa_split_verify_attention(
        q,
        k,
        v,
        mx.maximum(topk_idx, 0),
        topk_valid,
        base=base,
        scale=SCALE,
        block_size=BLOCK,
        impl=impl,
    )


# ---------------------------------------------------------------------------
# f32 references (per KV head to bound peak memory).
# ---------------------------------------------------------------------------

def _ref_attention(q, k, v, base: int, L: int, allowed_builder):
    """Generic per-KV-head f32 masked attention.

    ``allowed_builder(hkv, total)`` returns a bool [L, total] mask of which key
    positions query i may attend. Returns out [1,H_q,L,D] f32.
    """
    total = base + L
    outs = []
    for hkv in range(H_KV):
        q_g = q[0, hkv * GQA : (hkv + 1) * GQA].astype(mx.float32)  # [gqa,L,D]
        k_h = k[0, hkv].astype(mx.float32)                          # [total,D]
        v_h = v[0, hkv].astype(mx.float32)                          # [total,D]
        s = mx.matmul(q_g, k_h.swapaxes(-1, -2)) * SCALE           # [gqa,L,total]
        allowed = allowed_builder(hkv, total)                      # [L,total] bool
        s = mx.where(allowed[None], s, mx.array(-float("inf"), mx.float32))
        p = mx.softmax(s, axis=-1)
        o = mx.matmul(p, v_h)                                      # [gqa,L,D]
        outs.append(o)
        mx.eval(o)
    out = mx.concatenate(outs, axis=0)[None]                       # [1,H_q,L,D]
    return out


def dense_full_ref(q, k, v, base: int, L: int):
    kpos = mx.arange(base + L)

    def builder(hkv, total):
        # query i has absolute position base+i; attend keys [0, base+i].
        qi = mx.arange(L)[:, None] + base
        return kpos[None, :] <= qi

    return _ref_attention(q, k, v, base, L, builder)


def exact_target_ref(q, k, v, topk_idx, base: int, L: int):
    """Dense f32 over exactly the blocks split-verify selected + causal tail."""
    total = base + L
    kpos = mx.arange(total)
    num_blocks = (base + BLOCK - 1) // BLOCK
    block_ids = mx.arange(num_blocks)
    is_hist = kpos < base
    j = kpos - base
    tail_causal = is_hist == False  # noqa: E712  (tail region)
    ti = mx.arange(L)[:, None]

    def builder(hkv, total_):
        sel = topk_idx[0, hkv, 0]                       # [topk] int32, -1 pad
        valid = sel >= 0
        keep = mx.any((sel[:, None] == block_ids[None, :]) & valid[:, None], axis=0)
        hist_ok = keep[mx.clip(kpos // BLOCK, 0, num_blocks - 1)] & is_hist  # [total]
        tail_ok = tail_causal[None, :] & (j[None, :] <= ti)                  # [L,total]
        return hist_ok[None, :] | tail_ok

    return _ref_attention(q, k, v, base, L, builder)


def dense_full_fast(q, k, v):
    """Realistic bf16 dense fallback used for the microbench baseline."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal")


def native_select(idx_q, idx_k, base: int, L: int):
    """Current per-query block selection: score all L queries over the FULL
    base+L index history (the O(ctx*L) cost the overlay comments flag as the
    verify bottleneck). Returns head-reduced block_indices [1, L, topk]."""
    if _CUR_SELECT is None:
        return None
    qpos = mx.arange(base, base + L, dtype=mx.int32)[None]     # [1, L]
    return _CUR_SELECT(
        idx_q, idx_k, qpos, SCALE, BLOCK, TOPK, INIT_BLOCKS, LOCAL_BLOCKS
    )


def fused_select(idx_q, idx_k, base: int):
    """Split-verify's selection: fused L=1 top-k select shared across the L
    queries, scoring a single representative query over the history [0, base).
    O(ctx) once instead of O(ctx*L)."""
    return select_history_blocks(idx_q, idx_k, base)[0]


def current_sparse_prefill(q, k, v, topk_idx, base: int, L: int):
    """The current production verify path at ctx >= ~14k: the one-pass
    sparse-prefill Metal kernel. Returns its output, or None when ineligible
    (ctx < selected_length*7 = 14336, where the live path is instead dense).

    Feeds head-reduced per-query block indices [1,L,topk] + q_positions [1,L]
    of the shape/dtype the kernel demands (values only affect which blocks are
    read, not the timing).
    """
    if _CUR_SPARSE_PREFILL is None:
        return None
    head_red = mx.maximum(topk_idx[0, 0, 0], 0).astype(mx.int32)   # [topk]
    block_indices = mx.broadcast_to(head_red[None, None, :], (1, L, TOPK))
    q_positions = mx.arange(base, base + L, dtype=mx.int32)[None]   # [1, L]
    return _CUR_SPARSE_PREFILL(
        q, k, v, block_indices, q_positions, SCALE, BLOCK
    )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def max_abs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def mean_abs(a, b):
    return float(mx.mean(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def run_numerics(ctx_list, L_list):
    print("=" * 78)
    print("NUMERICS  (max|.| abs diff of attention output, values ~O(1))")
    print("=" * 78)
    print(
        f"{'ctx':>7} {'L':>3} | {'fused-vs-exact':>15} {'gather-vs-exact':>16} | "
        f"{'split-vs-dense':>15}"
    )
    print("-" * 78)
    worst_exact = 0.0
    for base in ctx_list:
        for L in L_list:
            q, k, v, idx_q, idx_k = make_inputs(base, L, seed=100 + L)
            topk_idx, topk_valid = select_history_blocks(idx_q, idx_k, base)
            mx.eval(topk_idx, topk_valid)
            out_f = split_verify(q, k, v, topk_idx, topk_valid, base, "fused")
            out_g = split_verify(q, k, v, topk_idx, topk_valid, base, "gather")
            ref_exact = exact_target_ref(q, k, v, topk_idx, base, L)
            ref_dense = dense_full_ref(q, k, v, base, L)
            mx.eval(out_f, out_g, ref_exact, ref_dense)
            df = max_abs(out_f, ref_exact)
            dg = max_abs(out_g, ref_exact)
            dd = max_abs(out_f, ref_dense)
            worst_exact = max(worst_exact, df, dg)
            print(
                f"{base:>7} {L:>3} | {df:>15.5f} {dg:>16.5f} | {dd:>15.5f}"
            )
            del q, k, v, idx_q, idx_k, out_f, out_g, ref_exact, ref_dense
            _clear_cache()
    print("-" * 78)
    verdict = "PASS" if worst_exact < 1e-2 else "CHECK"
    print(
        f"[{verdict}] worst split-vs-exact-target = {worst_exact:.5f} "
        f"(gate: < 1e-2 bf16-scale), both impls. split-vs-dense is the "
        f"block-sparsity approximation, larger for random synthetic selection."
    )
    print()
    return worst_exact


# ---------------------------------------------------------------------------
# Microbench
# ---------------------------------------------------------------------------

def bench(fn, iters=20, warmup=6, repeats=5):
    """Min over ``repeats`` timed windows. The box shares its GPU with live
    serving; contention only ADDS time, so the min window is the least-polluted
    estimate of the true dispatch cost."""
    for _ in range(warmup):
        mx.eval(fn())
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        best = min(best, (time.perf_counter() - t0) / iters * 1e3)
    return best


def run_microbench(ctx_list, L_list):
    print("=" * 78)
    print("MICROBENCH  (single sparse-attn layer, ms/verify-forward, min-of-5)")
    print("=" * 78)
    print("ATTENTION only (block indices supplied):")
    print(
        f"{'ctx':>7} {'L':>3} | {'dense':>8} {'cur-sparse':>10} "
        f"{'spl-fused':>10} {'spl-gather':>11}"
    )
    print("-" * 78)
    rows = []
    for base in ctx_list:
        for L in L_list:
            q, k, v, idx_q, idx_k = make_inputs(base, L, seed=7 + L)
            topk_idx, topk_valid = select_history_blocks(idx_q, idx_k, base)
            mx.eval(topk_idx, topk_valid)

            t_dense = bench(lambda: dense_full_fast(q, k, v))
            t_fused = bench(
                lambda: split_verify(q, k, v, topk_idx, topk_valid, base, "fused")
            )
            t_gather = bench(
                lambda: split_verify(q, k, v, topk_idx, topk_valid, base, "gather")
            )
            elig = current_sparse_prefill(q, k, v, topk_idx, base, L)
            if elig is not None:
                mx.eval(elig)
                t_curattn = bench(
                    lambda: current_sparse_prefill(q, k, v, topk_idx, base, L)
                )
                cur_lbl = f"{t_curattn:>10.3f}"
                live_attn = t_curattn
                sparse_live = True
            else:
                cur_lbl = f"{'(dense)':>10}"
                live_attn = t_dense
                sparse_live = False

            # Selection cost (the flagged verify bottleneck).
            t_nsel = bench(lambda: native_select(idx_q, idx_k, base, L))
            t_fsel = bench(lambda: fused_select(idx_q, idx_k, base))

            t_split = min(t_fused, t_gather)
            rows.append((base, L, t_dense, live_attn, sparse_live,
                         t_fused, t_gather, t_split, t_nsel, t_fsel))
            print(
                f"{base:>7} {L:>3} | {t_dense:>8.3f} {cur_lbl} "
                f"{t_fused:>10.3f} {t_gather:>11.3f}"
            )
            del q, k, v, idx_q, idx_k, topk_idx, topk_valid
            _clear_cache()
    print("-" * 78)
    print(
        "dense       = scaled_dot_product_attention (causal) over FULL context\n"
        "              (live verify attention at ctx < ~14k).\n"
        "cur-sparse  = live one-pass sparse-prefill kernel (ctx >= 14336; shown\n"
        "              '(dense)' where ineligible, i.e. live path is dense).\n"
        "spl-fused   = fused sparse-decode history (L dispatches) + LSE merge.\n"
        "spl-gather  = gather selected KV once + single batched masked SDPA."
    )
    print()

    # Full per-layer verify cost: SELECT + ATTENTION.
    print("SELECT + ATTENTION (full per-layer verify attention cost):")
    print(
        f"{'ctx':>7} {'L':>3} | {'nat-sel':>8} {'fus-sel':>8} | "
        f"{'live(sel+at)':>12} {'split(sel+at)':>13} | {'speedup':>8}"
    )
    print("-" * 78)
    for (base, L, t_dense, live_attn, sparse_live,
         t_fused, t_gather, t_split, t_nsel, t_fsel) in rows:
        live_total = t_nsel + live_attn
        split_total = t_fsel + t_split
        print(
            f"{base:>7} {L:>3} | {t_nsel:>8.3f} {t_fsel:>8.3f} | "
            f"{live_total:>12.3f} {split_total:>13.3f} | "
            f"{live_total / split_total:>7.2f}x"
        )
    print("-" * 78)
    print(
        "nat-sel = per-query selection over full base+L history (O(ctx*L)).\n"
        "fus-sel = split-verify's shared fused L=1 select (O(ctx), 1 query).\n"
        "live(sel+at)  = nat-sel + live attention (dense or cur-sparse).\n"
        "split(sel+at) = fus-sel + best split attention."
    )
    print()
    return rows


def extrapolate(rows):
    print("=" * 78)
    print("VERIFY-PASS EXTRAPOLATION (honest; attention-only)")
    print("=" * 78)
    print(
        f"Per-layer (SELECT+ATTN) delta (live - split) x {N_SPARSE_LAYERS} "
        f"sparse-attn layers\n(of {N_TOTAL_LAYERS} total; dense/linear layers "
        "unaffected). live = nat-select + live\nattention (dense at ctx<~14k, "
        "one-pass sparse kernel at ctx>=14336).\n"
        "Does NOT capture: MoE/FFN (unchanged; small at L=K+1 tokens), other\n"
        "proj/RoPE (unchanged), pipeline & collective overheads. Attention +\n"
        "selection only. Timings share the GPU with live serving (min-of-5)."
    )
    print("-" * 78)
    for (base, L, t_dense, live_attn, sparse_live,
         t_fused, t_gather, t_split, t_nsel, t_fsel) in rows:
        best_name = "gather" if t_gather <= t_fused else "fused"
        which = "sparse" if sparse_live else "dense"
        live_total = t_nsel + live_attn
        split_total = t_fsel + t_split
        per_layer = live_total - split_total
        pass_save = per_layer * N_SPARSE_LAYERS
        print(
            f"ctx={base:>6} L={L}: live={which:>6} best={best_name:>6} "
            f"per-layer {per_layer:+.3f} ms -> verify-pass {pass_save:+.1f} ms "
            f"over {N_SPARSE_LAYERS} layers"
        )
    print("-" * 78)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--num-L", type=int, nargs="+", default=[2, 4, 6, 8],
                    dest="num_L", help="L values for the numerics test")
    ap.add_argument("--bench-L", type=int, nargs="+", default=[4, 6],
                    dest="bench_L", help="L values for the microbench")
    ap.add_argument("--skip-numerics", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    print(f"mlx {mx.__version__}  metal={mx.metal.is_available()}  "
          f"device={mx.default_device()}")
    print(f"shapes: H_q={H_Q} H_kv={H_KV} D={D} block={BLOCK} topk={TOPK} "
          f"index_heads={INDEX_HEADS}  scale={SCALE:.6f}")
    print()

    if not args.skip_numerics:
        run_numerics(args.ctx, args.num_L)
    if not args.skip_bench:
        rows = run_microbench(args.ctx, args.bench_L)
        extrapolate(rows)


if __name__ == "__main__":
    main()
