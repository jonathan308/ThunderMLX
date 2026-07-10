"""Offline microbench: where does MiniMax-M3 MSA prefill time go, per layer, per chunk?

Compares CUSTOM MSA kernels (score-matrix build, topk/mask, sparse attention)
vs STOCK qmm GEMMs (attn projections + MoE gather_qmm) at model shapes.
Synthetic tensors only. rank0 (M3 Ultra). Keeps peak mem well under 20GB.
"""
import os, sys, time, importlib.util, gc, json

os.environ.setdefault("MLX_M3_MSA_PREFILL", "1")
os.environ.setdefault("MLX_M3_KERNEL_STATS", "1")
# match real prefill env defaults for blockwise builder
os.environ.setdefault("MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_MIN_KV_LEN", "32768")
os.environ.setdefault("MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_BLOCK_CHUNK", "2")

import mlx.core as mx

MSA_PATH = "~/ThunderMLX-eagle3/MSA Support/mlx_vlm/models/minimax_m3_vl/msa.py"
spec = importlib.util.spec_from_file_location("m3_msa", MSA_PATH)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

# installed mlx_vlm switch layers for the MoE qmm proxy
sys.path.append("~/mlx-vlm064-env/lib/python3.14/site-packages")
from mlx_vlm.models.switch_layers import SwitchLinear

DT = mx.bfloat16
GS, BITS = 64, 4

# ---- model dims (MiniMax-M3-4bit) ----
HID = 6144
HQ, HKV, D = 64, 4, 128           # attention heads
IDX_H, IDX_D = 4, 128             # indexer
BS, TOPK, INITB, LOCALB = 128, 16, 0, 1
MOE_INT = 3072
N_EXPERTS = 128
EXPERTS_PER_TOK = 4               # +1 shared packed = 5
DENSE_INT = 12288
SCALE = D ** -0.5


def bench(fn, iters=20, warmup=5):
    o = None
    for _ in range(warmup):
        o = fn()
        mx.eval(o)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn()
        mx.eval(o)
    mx.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000.0
    del o
    return dt


def qlin(in_dim, out_dim):
    w = mx.random.normal((out_dim, in_dim)).astype(DT)
    wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
    mx.eval(wq, s, b)
    return wq, s, b


def make_switch(in_dim, out_dim, n_exp):
    sl = SwitchLinear(in_dim, out_dim, n_exp, bias=False)
    sl = sl.to_quantized(group_size=GS, bits=BITS)
    mx.eval(sl.parameters())
    return sl


print("STEEL_MMA present:", M._MSA_CSR_K1_STEEL_MMA is not None,
      "| COMPACT:", M._MSA_CSR_K1_STEEL_MMA_COMPACT is not None)

# ---- build stock-qmm proxies once (weights reused across shapes) ----
wq_q, s_q, b_q = qlin(HID, HQ * D)      # q_proj 6144->8192
wq_k, s_k, b_k = qlin(HID, HKV * D)     # k_proj 6144->512
wq_v, s_v, b_v = qlin(HID, HKV * D)     # v_proj 6144->512
wq_o, s_o, b_o = qlin(HQ * D, HID)      # o_proj 8192->6144
wq_iq, s_iq, b_iq = qlin(HID, IDX_H * IDX_D)  # index_q 6144->512
wq_ik, s_ik, b_ik = qlin(HID, IDX_D)          # index_k 6144->128

# MoE packed: gate_up [E, 2*int, hid], down [E, hid, int]; E = 128 routed + 1 shared
E = N_EXPERTS + 1
gate_up = make_switch(HID, 2 * MOE_INT, E)   # 6144 -> 6144
down = make_switch(MOE_INT, HID, E)          # 3072 -> 6144
# dense mlp (first 3 layers) as quantized_matmul proxy
wq_dg, s_dg, b_dg = qlin(HID, DENSE_INT)     # gate 6144->12288
wq_du, s_du, b_du = qlin(HID, DENSE_INT)     # up 6144->12288
wq_dd, s_dd, b_dd = qlin(DENSE_INT, HID)     # down 12288->6144


def qmm(x, wsb):
    w, s, b = wsb
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BITS)


def attn_proj_call(x):
    q = qmm(x, (wq_q, s_q, b_q))
    k = qmm(x, (wq_k, s_k, b_k))
    v = qmm(x, (wq_v, s_v, b_v))
    iq = qmm(x, (wq_iq, s_iq, b_iq))
    ik = qmm(x, (wq_ik, s_ik, b_ik))
    o = qmm(q, (wq_o, s_o, b_o))  # o_proj on attn output (same width 8192)
    return o + k.sum() + v.sum() + iq.sum() + ik.sum()


def _gather_sort(x, indices):
    *_, m = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv = mx.argsort(order)
    return x.flatten(0, -3)[order // m], indices[order], inv


def _scatter_unsort(x, inv, shape):
    return mx.unflatten(x[inv], 0, shape)


def moe_call(x, inds):
    # mirror MiniMaxPackedSwitchGLU.__call__
    xx = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(xx, inds)
    gu = gate_up(xs, idx, sorted_indices=True)
    gate, up = mx.split(gu, 2, axis=-1)
    act = mx.sigmoid(gate) * up  # swiglu-ish (proxy; timing not numerics)
    y = down(act, idx, sorted_indices=True)
    y = _scatter_unsort(y, inv, inds.shape)
    return y.squeeze(-2)


def dense_mlp_call(x):
    g = qmm(x, (wq_dg, s_dg, b_dg))
    u = qmm(x, (wq_du, s_du, b_du))
    return qmm(mx.sigmoid(g) * u, (wq_dd, s_dd, b_dd))


def run_config(L, ctx, iters=20):
    q_start = ctx - L
    total_k = ctx
    num_blocks = (total_k + BS - 1) // BS

    # --- inputs (pre-evaluated) ---
    x = mx.random.normal((1, L, HID)).astype(DT); mx.eval(x)
    idx_q = mx.random.normal((1, IDX_H, L, IDX_D)).astype(DT)
    idx_k = mx.random.normal((1, 1, total_k, IDX_D)).astype(DT)
    mx.eval(idx_q, idx_k)
    q = mx.random.normal((L, HQ, D)).astype(DT)
    k = mx.random.normal((total_k, HKV, D)).astype(DT)
    v = mx.random.normal((total_k, HKV, D)).astype(DT)
    mx.eval(q, k, v)
    # routing indices: 4 routed (0..127) + shared expert index 128
    routed = mx.random.randint(0, N_EXPERTS, (1, L, EXPERTS_PER_TOK)).astype(mx.int32)
    shared = mx.full((1, L, 1), N_EXPERTS, dtype=mx.int32)
    inds = mx.concatenate([routed, shared], axis=-1); mx.eval(inds)

    use_blockwise = (L * total_k * 4 >= 64 * 1024 * 1024) and (total_k >= 32768)
    builder = M.build_grouped_msa_topk_blockwise if use_blockwise else M.build_grouped_msa_topk
    bkw = {"block_chunk_size": 2} if use_blockwise else {}

    # real q2k for CSR / sparse-attn bench
    q2k = builder(idx_q, idx_k, q_start, SCALE, BS, TOPK, INITB, LOCALB, **bkw)[0]
    mx.eval(q2k)

    # synthetic block_scores for the topk-select microbench (shape-driven cost)
    block_scores = mx.random.normal((1, IDX_H, L, num_blocks)).astype(mx.float32)
    mx.eval(block_scores)

    # prebuilt CSR for the K1/K2-only microbench
    row_ptr, _, qsplit, split_counts = M.build_k2q_csr_b1(
        q2k, total_k=total_k, block_size=BS,
        return_qsplits=True, return_q_indices=False, return_split_counts=True)
    mx.eval(row_ptr, qsplit, split_counts)

    res = {}
    # ---- CUSTOM MSA kernels ----
    res["msa_topk_total"] = bench(
        lambda: builder(idx_q, idx_k, q_start, SCALE, BS, TOPK, INITB, LOCALB, **bkw),
        iters=iters)
    res["topk_select"] = bench(
        lambda: M._select_msa_topk_from_block_scores(
            block_scores, q_start=q_start, block_size=BS, topk=TOPK,
            init_blocks=INITB, local_blocks=LOCALB),
        iters=iters)
    res["score_build"] = max(res["msa_topk_total"] - res["topk_select"], 0.0)
    res["sparse_attn_csr"] = bench(
        lambda: M.build_k2q_csr_b1(
            q2k, total_k=total_k, block_size=BS, return_qsplits=True,
            return_q_indices=False, return_split_counts=True)[0],
        iters=iters)
    res["sparse_attn_k1k2"] = bench(
        lambda: M.msa_sparse_attention_b1_from_csr(
            q, k, v, row_ptr, qsplit, split_counts, q_start=q_start, scale=SCALE,
            block_size=BS, topk=TOPK, k1_impl="auto", full_splits=False),
        iters=iters)
    res["sparse_attn_total"] = bench(
        lambda: M.msa_sparse_attention_b1(
            q, k, v, q2k, q_start=q_start, scale=SCALE, block_size=BS,
            k1_impl="auto", full_splits=False),
        iters=iters)
    res["k1_impl"] = M.get_kernel_stats().get("last_msa_k1_impl")

    # ---- STOCK qmm GEMMs ----
    res["attn_proj"] = bench(lambda: attn_proj_call(x), iters=iters)
    res["moe_qmm"] = bench(lambda: moe_call(x, inds), iters=iters)
    res["dense_mlp"] = bench(lambda: dense_mlp_call(x), iters=iters)

    res["peak_gb"] = mx.get_peak_memory() / 1e9
    del x, idx_q, idx_k, q, k, v, q2k, block_scores, row_ptr, qsplit, split_counts, inds
    gc.collect(); mx.clear_cache()
    return res, use_blockwise


CONFIGS = [(2048, 16384), (4096, 16384), (2048, 32768), (4096, 32768)]
ALL = {}
for L, ctx in CONFIGS:
    iters = 12 if (L * ctx >= 4096 * 16384) else 20
    r, bw = run_config(L, ctx, iters=iters)
    ALL[f"L{L}_ctx{ctx}"] = r
    print(f"\n=== L={L} ctx={ctx}  (builder={'blockwise' if bw else 'standard'}, "
          f"K1={r['k1_impl']}, peak={r['peak_gb']:.1f}GB) ===")
    # per-MoE-layer chunk composition
    msa = r["score_build"] + r["topk_select"] + r["sparse_attn_total"]
    stock = r["attn_proj"] + r["moe_qmm"]
    layer = msa + stock
    print(f"  {'component':<22}{'ms':>9}{'%layer':>9}")
    rows = [
        ("MSA score_build", r["score_build"]),
        ("MSA topk_select", r["topk_select"]),
        ("MSA sparse_attn", r["sparse_attn_total"]),
        ("  (csr)", r["sparse_attn_csr"]),
        ("  (k1+k2)", r["sparse_attn_k1k2"]),
        ("stock attn_proj qmm", r["attn_proj"]),
        ("stock MoE qmm", r["moe_qmm"]),
    ]
    for name, ms in rows:
        pct = 100 * ms / layer if not name.startswith("  ") else float("nan")
        print(f"  {name:<22}{ms:>9.3f}{pct:>9.1f}" if pct == pct
              else f"  {name:<22}{ms:>9.3f}{'':>9}")
    print(f"  {'-'*40}")
    print(f"  {'CUSTOM MSA total':<22}{msa:>9.3f}{100*msa/layer:>9.1f}")
    print(f"  {'STOCK qmm total':<22}{stock:>9.3f}{100*stock/layer:>9.1f}")
    print(f"  {'MoE-LAYER total':<22}{layer:>9.3f}{100:>9.1f}")
    print(f"  (dense-layer mlp qmm, x3 layers only: {r['dense_mlp']:.3f} ms)")

outp = "/tmp/msa_prefill_bench_out.json"
with open(outp, "w") as f:
    json.dump(ALL, f, indent=2)
print("\nwrote", outp)
