# TensorOps Prefill — Phase 1: Capability Audit, Time-Breakdown, Prototype

_Branch `feature/tensorops-prefill` off `feature/split-verify`. Offline research + rank0/rank1 microbenches only; no servers touched. Goal was: port the custom MSA sparse-prefill path toward simdgroup/TensorOps matmul to push prefill past 500 tok/s (from ~384/377 at 16k/32k)._

## TL;DR verdict — the premise is already satisfied where it matters, and cannot be satisfied where it was aimed

1. **The dominant prefill cost is stock quantized-matmul GEMMs, not our custom MSA kernels.** Measured per MoE layer (heaviest chunk): **stock qmm = 80–87%** of the layer (MoE gather_qmm alone 62–68%, attention projections 17–20%); **custom MSA kernels = only 13–20%** (sparse attention ~9%, score-build 3–10% growing with ctx, topk <1%). The research's "GEMMs dominate" assumption is **confirmed**.
2. **Our custom MSA sparse-prefill attention is ALREADY simdgroup_matrix-tiled** (`_MSA_CSR_K1_STEEL_MMA`, `mlx::steel` `MMATile`/`tile_matmad`, auto-selected for the M3 config). There is no "port to simdgroup" work left to do on it.
3. **The pinned mlx (`c110f69e`) ALREADY contains and runs the Metal-4 / M5 Neural-Accelerator ("nax") tensor-op path** for stock `matmul`/`quantized_matmul`/`sdpa`. On **rank1 (M5 Max) it is active and correct right now**: measured **3.3–3.5× matmul/qmm throughput vs rank0**, zero code change. This is the "free 1.3× from rank1 accelerators" the research anticipated — **it is already banked.**
4. **A hand-written `simdgroup_matrix` custom kernel gets ZERO nax benefit on M5** (measured: 5.8 TF vs stock nax matmul 14–17 TF on the *same* M5). `simdgroup_matrix` does not reach the Neural Accelerators; only MLX's built-in ops (via `mpp::tensor_ops::matmul2d`) do. So porting our custom kernel to simdgroup tiles buys nothing on M5, and porting it to the raw tensor-op API has no MLX wrapper and no working precedent (high risk).
5. **Upgrading mlx buys ~0 for M5:** the nax feature landed upstream in **v0.30.0 (Nov 2025)**, our pin is later, and v0.32.0 is still the newest release. An upgrade only risks the jaccl ProgressGuard regression and the nax miscompile landmine.

**Net recommended sequence:** lock in / assert the rank1 nax path (done, zero-cost) → do **not** upgrade mlx for M5 → do **not** port custom kernels to simdgroup_matrix → attack **rank0 (M3 Ultra, no nax)** which is now the prefill floor, via layer-split rebalance toward the faster M5 and pipeline overlap. If the MSA slice is still worth chasing on rank1, the low-risk route is gather-selected-KV + stock `mx.fast.scaled_dot_product_attention` (which has a nax path), not a hand-written tensor-op kernel.

---

## 1. Capability audit (facts, cited)

Source of truth: MLX checkout `~/mlx-src` (HEAD `99d3e3ec` = guard-032c; pinned production build `c110f69e` = guard-032b is one commit back; both = upstream base `de7b4ed9`, just past tag `v0.32.0`, + jaccl ProgressGuard patch). Installed metallib cross-checked at `mlx-vlm064-env/.../mlx/lib/mlx.metallib`.

### (a) `mx.matmul` dense GEMM
- Steel GEMM kernels emit hardware simdgroup-matrix tiles: `mlx/backend/metal/kernels/steel/gemm/mma.h` — L6 `#include <metal_simdgroup_matrix>`, L46 `simdgroup_matrix<T,8,8>` fragment, L205 `simdgroup_multiply_accumulate(D,A,B,C)`; tiled by `BlockMMA` (L213+). Dtypes fp16/bf16/fp32(+complex) (`steel_gemm_fused.metal:29-33`).
- **nax path present**: `matmul.cpp:915` `use_nax = is_nax_available() && !complex && (enable_tf32() || dtype != float32)`; `enable_tf32()` defaults **1** → fp16/bf16/fp32 all route to nax on capable HW.

### (b) `mx.quantized_matmul` (the MoE / projection path)
- Dispatch `quantized.cpp:QuantizedMatmul::eval_gpu` (L1486): `M >= vector_limit` → matrix-matrix `qmm`; else vector `qmv/qvm`. `get_qmv_batch_limit` returns ~6–32 (typically ~18 on gen-15+). **Prefill is M≫8 → qmm path.**
- qmm non-nax fallback DOES use simdgroup-matrix tiles: `qmm_t_impl` (`quantized.h:1193`) dequantizes into threadgroup memory then `mlx::steel::BlockMMA` → `simdgroup_multiply_accumulate`. (qmv/qvm decode path is scalar/`simd_`-reduction `qdot`, `quantized.h:192` — irrelevant to prefill.)
- **qmm nax path present**: `quantized.cpp:787` `if (metal::is_nax_available() && transpose && K%64==0 …)` → `qmm_nax` → `affine_qmm_t_nax` → `tile_matmad_nax` (`quantized_nax.h:1045`) → tensor-op MMA. **Our MoE gather_qmm is `transpose=True`, K=6144 (÷64) → nax-eligible.**

### (c) Does the pin target M5 TensorOps? YES — `simdgroup_matrix` is NOT the newest primitive present
- Tensor-op ("nax") source: `steel/gemm/nax.h` L12 `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, L401 `mpp::tensor_ops::matmul2d_descriptor(...)`, L411 `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>`, L414-425 cooperative tensors, L448 `gemm_op.run(...)`. Also `steel/attn/nax.h` (sdpa).
- **Compiled into the shipped runtime** (not dormant): build gate `kernels/CMakeLists.txt:158` `MLX_METAL_VERSION >= 400`; `build_0320.log` shows *"macOS 26.5 SDK, Metal version 400"* and compiles `*_nax.air` into `mlx.metallib`; installed metallib contains demangled `mpp::tensor_ops::matmul2d_descriptor::mode`.
- Runtime gate `device.cpp:943 is_nax_available()`: needs `MLX_METAL_NO_NAX` undefined + `__builtin_available(macOS 26.2)` + `gen >= (arch=='p'?18:17)`. **Gen 17 = Apple g17 = M5.** So nax runs only on M5-class + macOS 26.2+; M1–M4 fall back to the simdgroup Steel path.
- Keyword sweep: `simdgroup_matrix`/`simdgroup_multiply_accumulate` present (mma.h, attn/mma.h). `mpp::tensor_ops`/`MetalPerformancePrimitives` present **only** in the two `nax.h`. `<metal_tensor>`, literal `Metal 4`, `simdgroup_async` **ABSENT** (MLX uses MPP, not `<metal_tensor>`).

### (d) Metal-4 TensorOps from custom `mx.fast.metal_kernel`?
- Custom kernels JIT-compile at the OS-max Metal language version (`device.cpp:get_metal_version()` → `LanguageVersion4_0` on macOS 26; applied `build_library_` L661). So the compile *target* is Metal 4.0 on this box.
- **But there is no exposed switch and no wrapper.** `parse_compile_options` (`python/src/fast.cpp:100`) accepts only `"math_mode"` (PR #3728, present: Safe/Relaxed/Fast, default Safe) and throws on any other key. No `<metal_tensor>` support anywhere. A user *could* `#include <MetalPerformancePrimitives/…>` via the `header=` arg, but there is **no confirmed working example** of `mpp::tensor_ops` inside `metal_kernel`, and int4/int8 tensor types need macOS-26 updates (WWDC26 s330). **Practical ceiling for custom kernels today = `simdgroup_matrix`, which does not reach the M5 accelerators (empirically confirmed, §3).**

### Upstream M5 timeline (web, cited)
- **PR #2772 "Add Neural Accelerator Support"** merged **2025-11-19** (`54f1cc6e…`), shipped in **v0.30.0** ("Support for Neural Accelerators on M5 (macOS ≥ 26.2)"). One PR added `_nax` for GEMM **and** qmm **and** sdpa.
- Follow-ups all pre-v0.32.0: #2916 (JIT nax, v0.30.1), #2982 (runtime guard, v0.30.3), **#3622 (build gate: needs `MACOSX_DEPLOYMENT_TARGET=26.2`, 2026-06-13)**.
- After v0.32.0: only #3824 (open, cosmetic build-warning). **No new op acceleration. v0.32.0 is newest as of 2026-07-09.**
- ⚠️ **Miscompile landmine — PR #3593 (UNMERGED report):** some macOS-26 Metal toolchains miscompile nax so **matmul with M>8 returns garbage** (corrupts prefill, decode M=1 unaffected). Opt-out `MLX_DISABLE_NAX` (not present in our pin — verified no-op). **We tested for it and our rank1 build is clean (§3).**
- Apple's ~3.5–4.06× TTFT figures are **M5-vs-M4** (M4 has no accelerators at all), *not* a same-chip software delta. Same-M5 path-on/off proxies (llama.cpp) ≈ **2.0–2.5×** real prefill, ~4× pure-matmul ceiling. Our measured cross-rank qmm ratio (3.3×) folds M5-arch + nax + core-count together.

---

## 2. Where prefill time actually goes (measured, rank0 M3 Ultra, synthetic)

Per **MoE layer** (57 of 60 layers), **heaviest chunk** (`total_k = ctx`, `q_start = ctx-L`). Builder auto-selects standard (<32k kv) vs blockwise (≥32k). K1 auto-selects `steel_mma`. Bench: `ops/bench/tensorops_prefill/msa_prefill_bench.py`.

| component | L2048/16k | L4096/16k | L2048/32k | L4096/32k | nature |
|---|--:|--:|--:|--:|---|
| MSA score_build (`mx.matmul`) | 4.57 (3.1%) | 8.74 (3.2%) | 16.85 (10.5%) | 29.79 (10.0%) | stock matmul, fp32 |
| MSA topk_select (custom) | 0.98 (0.7%) | 1.65 (0.6%) | 1.19 (0.7%) | 2.00 (0.7%) | custom reduce kernel |
| MSA sparse_attn (custom **steel_mma**) | 13.73 (9.2%) | 25.95 (9.4%) | 14.06 (8.7%) | 26.29 (8.8%) | **already simdgroup_matrix** |
| — of which CSR build | 0.50 | 0.58 | 0.58 | 0.83 | metadata |
| — of which K1+K2 kernels | 13.08 | 25.21 | 13.63 | 25.37 | steel_mma + combine |
| **stock attn_proj qmm** | 27.82 (18.7%) | 54.89 (19.9%) | 27.83 (17.3%) | 54.85 (18.4%) | q/k/v/o + idx, 4-bit qmm |
| **stock MoE gather_qmm** | 101.77 (68.4%) | 184.95 (67.0%) | 100.82 (62.7%) | 184.44 (62.0%) | 129-expert top-4+shared |
| **CUSTOM MSA total** | **19.28 (13.0%)** | **36.34 (13.2%)** | **32.11 (20.0%)** | **58.08 (19.5%)** | |
| **STOCK qmm total** | **129.59 (87.0%)** | **239.85 (86.8%)** | **128.65 (80.0%)** | **239.29 (80.5%)** | |
| MoE-layer total (ms) | 148.9 | 276.2 | 160.8 | 297.4 | |

(Dense-layer MLP, only 3/60 layers: ~56 ms @L2048, ~112 ms @L4096 — pure stock qmm.)

**Reading:** the qmm GEMMs are 80–87% of the layer and the MoE alone is the single biggest block. The custom MSA attention (already MMA) is ~9%; the only growing custom-ish cost is the score-build (`mx.matmul`, fp32) which is 3%→10% as ctx doubles and would keep growing at the 200k context the cluster serves. Topk is negligible. **Verdict: the win is in stock qmm, and stock qmm already has an M5 path.**

---

## 3. Prototype + cross-rank nax test

Two artifacts, both self-contained (installed mlx steel header), benched rank0 + rank1.

### 3a. Cross-rank stock throughput (`ops/bench/tensorops_prefill/nax_probe.py`)
| | arch | macOS | bf16 matmul | 4-bit qmm | M>8 correctness |
|---|---|---|--:|--:|---|
| rank0 (M3 Ultra) | g15d | 26.5.1 | 17.6 / 17.4 TF | 16.5 / 16.4 TF | OK (relerr 4e-3) |
| rank1 (M5 Max) | **g17s** | 26.5.1 | **61.1 / 59.4 TF** | **55.2 / 54.4 TF** | **OK (relerr 2e-2, NOT the #3593 garbage)** |
| **rank1 / rank0** | | | **~3.4×** | **~3.3×** | |

`MLX_DISABLE_NAX=1` on rank1 = no-op (opt-out not in this build), so nax cannot be turned off here; the on/off delta is inferred from §3b instead. **nax is active + correct on rank1 today.**

### 3b. Custom `simdgroup_matrix` score kernel vs stock (`ops/bench/tensorops_prefill/sg_score_proto.py`)
A hand-written `mlx::steel` `MMATile`/`tile_matmad` kernel computing `S = scale·Q·Kᵀ` (the matmul core of `build_grouped_msa_topk`). Numerics vs `mx.matmul`: **relerr ≤ 3e-7** (exact).

| shape (M×N×D) | rank0 sg_kernel | rank0 mx.matmul | rank1 sg_kernel | rank1 mx.matmul (nax) | rank1 sg/mm |
|---|--:|--:|--:|--:|--:|
| 4096×16384×128 | 9.8 TF | 9.7 TF | **5.8 TF** | 13.9 TF | **2.42× slower** |
| 4096×32768×128 | 10.7 TF | 10.7 TF | **5.9 TF** | 17.2 TF | **2.90× slower** |
| 2048×16384×128 | 8.5 TF | 9.0 TF | **5.3 TF** | 14.1 TF | **2.65× slower** |

**Conclusion:** on rank0 the custom simdgroup kernel *ties* `mx.matmul` (both are simdgroup Steel — the M3 ceiling). On rank1 the custom simdgroup kernel is **2.4–2.9× slower than stock `mx.matmul` on the identical chip**, and even slower than the same kernel on rank0 (M5 Max has ~half M3 Ultra's shader cores). **`simdgroup_matrix` does not touch the M5 Neural Accelerators — only MLX's built-in nax ops do.** A "port our custom MSA kernel to simdgroup tiles" delivers no M5 gain; the tensor-op API route (`mpp::tensor_ops`) has no MLX custom-kernel wrapper and no working precedent.

_(Score-build shapes are tall-skinny/D=128 → memory-bound, so nax uplift here is 2.4–2.9×; compute-bound square GEMMs like MoE see the full ~3.3×.)_

---

## 4. The mlx-upgrade question — quantified

- **"Upgrade mlx on rank1 to get M5 nax" → expected gain ≈ 0.** nax is already in the pin (v0.30.0 feature; our base is later) and already active on rank1 (§3). No newer release exists. Upgrading would re-open the jaccl ProgressGuard patch (two-stage rebuild) and re-expose the #3593 miscompile risk for **no throughput upside**.
- **"Port custom MSA kernels toward tensorops" → expected end-to-end gain small and high-risk.** The custom MSA slice is 13–20% of a rank0 layer. On rank1, because qmm is nax-accelerated, the same MSA work is a *larger* fraction of rank1's (smaller) per-layer time — but rank1 owns only 22/60 layers and is already the fast node. Even optimistically routing the sparse attention to nax (~2–2.5× on that slice) nets a few % end-to-end **only if** rank1 isn't already idle-waiting on rank0. And the port itself needs raw `mpp::tensor_ops` MSL with no wrapper/precedent.
- **The actual floor is rank0 (M3 Ultra), which has no nax path at all.** It runs the dominant qmm at ~17 TF vs rank1's ~55 TF. Per-layer prefill compute on rank1 is ~2× faster than rank0; the 38/22 (rank0/rank1) split therefore likely under-utilizes the M5. **The highest-ROI lever is rebalancing layers toward rank1 + pipeline-chunk overlap (research levers #3–4), not kernel porting.**

**Recommended sequence**
1. **(zero-cost, do now)** Add a boot assertion on rank1: arch `g17*` + macOS ≥ 26.2 + run the M>8 correctness probe; fail loud if nax silently drops to the simdgroup fallback or the #3593 miscompile appears after any OS/toolchain change.
2. **Do not upgrade mlx** for M5 reasons; keep `c110f69e`.
3. **Do not port custom MSA kernels to `simdgroup_matrix`** (already done / no M5 benefit).
4. **Rebalance the layer split toward rank1 and overlap chunks** — measure rank1 idle-in-recv; A/B a 34/26 or 32/28 split (memory permitting: 22→28 layers ≈ +12 GB on rank1's 128 GB). This directly attacks rank0-as-floor.
5. **If still chasing the MSA slice on rank1:** prototype gather-selected-KV → `mx.fast.scaled_dot_product_attention` (built-in nax sdpa) instead of the custom simdgroup kernel; validate numerics/perf offline first.

---

## 5. Runtime / build recipe status (documented, NOT rebuilt)

- **Pinned runtime = split two-wheel artifact** in `minimax-m3-cluster/runtime_patches/variants/guard-032b/`: `mlx-0.32.0.dev20260706+c110f69e` (core, 560 KB) + `mlx_metal-…+c110f69e` (libmlx + `mlx.metallib`, 56 MB). Newer variant `guard-032c` (`0a8142d6`, "release Metal memory before ProgressGuard teardown-exit") also staged.
- **Build recipe** = `minimax-m3-cluster/ops/build_mlx_variant.sh` (HANDOFF-2026-07-05). **Two-stage, mandatory** (`setup.py:263-290`): Stage 1 `MLX_BUILD_STAGE=1 python -m build -w` → `mlx`; clean; Stage 2 `MLX_BUILD_STAGE=2 …` → `mlx-metal`. **Never single-stage** — a bare `pip wheel .` yields corrupt Metal kernels ("token salad at normal t/s").
- **jaccl ProgressGuard** patch (progress guard + QP retry + teardown-exit) sits on upstream `de7b4ed9`; **any mlx upgrade MUST re-apply it** (see `mlx-src` guard-032/032b/032c commits and `runtime_patches/jaccl-progress-timeout.patch`).
- **NAX build requirements (must preserve on any rebuild):** Metal-4 SDK (macOS 26.5 SDK / `__METAL_VERSION__==400`) + macOS deployment target ≥ 26.2 (PR #3622), else nax kernels silently don't compile and the M5 falls back to the slow simdgroup path with no error. Our current metallib satisfies this (verified: `mpp::tensor_ops` symbols present + nax active on rank1).

### Validation-window plan (pass/fail)
Baseline: 384/377 tok/s @ 16k/32k. Respect the ThunderMLX sync whitelist / wedge-orphan rules.
1. **nax health assert (no serving needed):** on rank1 run `nax_probe.py` → PASS if bf16 matmul ≥ 45 TF and M>8 relerr < 0.05; FAIL (investigate OS/build) otherwise. Bake into launch preflight.
2. **Split-rebalance A/B (needs boot):** boot with 34/26 then 32/28 vs baseline 38/22. Instrument rank1 idle-in-`recv` (`sharded_server.py:5274`) and per-chunk prefill ms. PASS = ≥1.1× end-to-end prefill @ 32k with reduced rank1 idle and no wedge/orphan regression; else revert.
3. **(optional) gather+nax-sdpa sparse attention:** offline numerics (<1e-2 bf16) + per-layer bench on rank1 vs current steel_mma; only if it beats steel_mma on rank1 do an armed-by-env serving A/B. PASS = ≥1.15× on rank1's isolated attention with bit-comparable output; else shelve.

**Bench artifacts:** `ops/bench/tensorops_prefill/{msa_prefill_bench.py, nax_probe.py, sg_score_proto.py}`.
