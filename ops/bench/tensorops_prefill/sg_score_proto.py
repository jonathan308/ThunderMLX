"""Deliverable-3 prototype: a simdgroup_matrix-tiled score-build kernel (S = scale * Q.Kt),
the matmul-shaped core of build_grouped_msa_topk. Self-contained (uses installed mlx
steel header on either rank). Numerics-checks vs mx.matmul, then benches:
  (1) this custom simdgroup_matrix kernel   vs
  (2) stock mx.matmul (nax on M5, steel-simdgroup on M3)
on whatever rank it runs. Cross-rank ratios reveal whether a HAND-WRITTEN
simdgroup_matrix custom kernel benefits from the M5 Neural Accelerators."""
import time, sys
from pathlib import Path
import mlx.core as mx

def steel_header():
    root = Path(mx.__file__).parent / "include"
    seen = set()
    def expand(p):
        fp = root / p
        if fp in seen: return ""
        seen.add(fp)
        out = []
        for ln in fp.read_text().splitlines():
            s = ln.strip()
            if s.startswith('#include "mlx/') and s.endswith('"'):
                out.append(expand(s[len('#include "'):-1]))
            elif s != "#pragma once":
                out.append(ln)
        return "\n".join(out)
    return expand("mlx/backend/metal/kernels/steel/attn/attn.h")

HDR = steel_header() + "\nusing namespace metal;\nusing namespace mlx::steel;\n"

_SG_SCORE = mx.fast.metal_kernel(
    name="sg_score_qkt",
    input_names=["q", "k", "scale"],
    output_names=["s"],
    header=HDR,
    source=r"""
        constexpr short BQ = 32;      // M rows per threadgroup
        constexpr short BN = 64;      // N cols per threadgroup
        constexpr short BD = D;       // contraction
        constexpr short WM = BQ / 8;  // 4 warps
        uint tgn = threadgroup_position_in_grid.x;
        uint tgm = threadgroup_position_in_grid.y;
        int row0 = int(tgm) * BQ;
        int col0 = int(tgn) * BN;
        uint simd_group_id = simdgroup_index_in_threadgroup;
        uint simd_lane_id = thread_index_in_simdgroup;

        constexpr short kFrag = 8;
        using Frag = BaseMMAFrag<float, kFrag, kFrag>;
        constexpr int TQ = BQ / (WM * kFrag);   // 1
        constexpr int TN = BN / kFrag;          // 8
        constexpr int TD = BD / kFrag;          // 16
        MMATile<float, TQ, 1, Frag> Qtile;
        MMATile<float, 1, TN, Frag> Ktile;
        MMATile<float, TQ, TN, Frag> Stile;
        Stile.clear();
        const short2 sc = Frag::get_coord(simd_lane_id);
        const short sm = sc.y, sn = sc.x;
        const short tm = kFrag * TQ * simd_group_id;

        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short iq = 0; iq < TQ; iq++) {
                int m = row0 + tm + sm + iq * kFrag;
                for (short jj = 0; jj < Frag::kElemCols; jj++) {
                    int d = dd * kFrag + sn + jj;
                    Qtile.frag_at(iq, 0)[jj] = m < M ? float(q[m * D + d]) : 0.0f;
                }
            }
            for (short ik = 0; ik < TN; ik++) {
                int d = dd * kFrag + sm;
                for (short jj = 0; jj < Frag::kElemCols; jj++) {
                    int n = col0 + ik * kFrag + sn + jj;
                    Ktile.frag_at(0, ik)[jj] = n < N ? float(k[n * D + d]) : 0.0f;
                }
            }
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(Stile, Qtile, Ktile, Stile);
        }

        using st = decltype(Stile);
        for (short i = 0; i < st::kTileRows; i++) {
            int m = row0 + tm + sm + i * st::kFragRows;
            for (short j = 0; j < st::kTileCols; j++) {
                for (short jj = 0; jj < st::MMAFrag_t::kElemCols; jj++) {
                    int n = col0 + sn + j * st::kFragCols + jj;
                    if (m < M && n < N) {
                        s[m * N + n] = T(Stile.frag_at(i, j)[jj] * scale);
                    }
                }
            }
        }
    """,
)

def sg_score(q, k, scale):
    M, D = q.shape
    N = k.shape[0]
    tgx = 128  # 4 warps
    return _SG_SCORE(
        inputs=[q, k, float(scale)],
        template=[("T", mx.float32), ("M", M), ("N", N), ("D", D)],
        grid=(((N + 63) // 64) * tgx, (M + 31) // 32, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
    )[0]

def bench(fn, it=30, w=8):
    for _ in range(w): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(it): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter() - t0) / it

try: arch = mx.device_info().get("architecture", "?")
except Exception: arch = mx.metal.device_info().get("architecture", "?")
print(f"arch={arch}  mlx={mx.__version__}")

# score-build-ish shapes: M=chunk rows (L*heads collapsed), N=ctx, D=128
for (M, N, D) in [(4096, 16384, 128), (4096, 32768, 128), (2048, 16384, 128)]:
    q = mx.random.normal((M, D)).astype(mx.bfloat16)
    k = mx.random.normal((N, D)).astype(mx.bfloat16)
    mx.eval(q, k)
    scale = D ** -0.5
    # numerics vs reference
    ref = (q.astype(mx.float32) @ k.astype(mx.float32).T) * scale
    got = sg_score(q, k, scale)
    mx.eval(ref, got)
    rel = float(mx.max(mx.abs(got - ref)) / (mx.max(mx.abs(ref)) + 1e-6))
    t_sg = bench(lambda: sg_score(q, k, scale))
    t_mm = bench(lambda: (q @ k.T) * scale)   # stock (nax on M5)
    fl = 2 * M * N * D
    print(f"  M={M} N={N}: relerr={rel:.2e} | "
          f"sg_kernel {t_sg*1e3:6.2f}ms {fl/t_sg/1e12:5.1f}TF | "
          f"mx.matmul {t_mm*1e3:6.2f}ms {fl/t_mm/1e12:5.1f}TF | "
          f"sg/mm={t_sg/t_mm:.2f}x")
