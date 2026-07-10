# CPU/AMX Offload for MSA Prefill — Research & Engineering Plan

_Target: raise MiniMax-M3 (4-bit) prefill from ~384/378/366 tok/s (30k/80k/200k) toward 500+ tok/s on the 2-node pipeline-parallel cluster (rank0 = M3 Ultra 80-GPU/32-CPU/256GB, rank1 = M5 Max 128GB, Thunderbolt-5 jaccl RDMA, 38/22 layer split)._
_Scope: prefill only. Decode (fused sparse-decode + EAGLE3) is explicitly out of scope. Read-only research; no code changed._

---

## TL;DR verdict

**CPU/AMX offload is NOT the lever that gets you to 500+ tok/s prefill — it is at best a 0–5% side-dish, and offloading the heavy GEMMs to AMX is a dead end.** Prefill is *compute*-bound (big batched GEMMs), and on rank0 the AMX coprocessor delivers only ~2–5% of the GPU's matmul FLOPS (~1.2–2.0 TFLOPS FP32 aggregate AMX vs 32.8 TFLOPS FP32 / 65.5 TFLOPS FP16 GPU), so moving matmul to the CPU makes prefill *slower*, not faster. The unified-memory bandwidth (819 GB/s) is shared, but during compute-bound prefill the GPU is ALU-bound, not bandwidth-bound, so there is spare bandwidth — just no spare *matmul* worth using on the CPU. **The #1 real lever is GPU-side matmul throughput on rank1: the M5 Max has GPU Neural Accelerators (Metal-4 TensorOps) that give ~3.5–4× prefill/TTFT, and your hand-written MSA Metal kernel almost certainly does not use them.** Ranked below; CPU offload sits near the bottom, and the only CPU idea worth a probe is overlapping the small MSA mask/top-k *glue* on `stream=mx.cpu` — plumbing you already use for the `all_sum` barrier.

**Ranked levers (expected prefill tok/s gain, high→low):**
1. **Make rank1's prefill GEMMs use the M5 Neural Accelerators (Metal-4 TensorOps).** Biggest lever; up to ~3.5–4× on rank1's 22-layer share if it's currently on general ALUs. Requires an MLX/Metal-4 build that exposes TensorOps and a MSA kernel that uses `simdgroup_matrix`/TensorOps.
2. **Fuse / enlarge the GPU MSA prefill kernel** (fewer launches, `simdgroup_matrix` tiles, bigger block-chunk) — attacks the actual compute path on both ranks.
3. **Pipeline-chunk overlap + layer-split rebalance** — double-buffer chunks so rank0 computes chunk N+1 while rank1 + jaccl handle chunk N; shift a few layers toward the faster-prefill M5. Kills the rank0-idle-in-`recv` bubble.
4. **Prefill step-size / batching tuning within the Metal command-buffer timeout; strip any per-chunk `clear_cache`/redundant `eval`.** Mostly already spent (you run step 4096–16384), but verify no per-chunk sync barrier remains.
5. **Algorithmic sparsity (lower top-k / denser blocks).** 1.1–1.5× but trades answer quality; already being A/B'd (topk2/4/8 logs).
6. **CPU-stream overlap of MSA mask/top-k/RoPE glue** (`stream=mx.cpu`) — ≤5%, low risk, only if it's on the critical path. One cheap probe.
7. **CPU/AMX GEMM offload** — dead end (2–5% of GPU FLOPS, high integration risk). Do not pursue.

---

## Bottleneck analysis (compute vs bandwidth, with numbers)

### Prefill is compute-bound; decode is bandwidth-bound — confirmed by two independent sources
- Apple's own MLX/M5 writeup states it plainly: _"Generating the first token is compute-bound, and takes full advantage of the Neural Accelerators. Generating subsequent tokens is bounded by memory bandwidth, rather than by compute ability."_ [Apple ML Research, M5 MLX]
- Independent confirmation: prefill _"is compute-bound, meaning it saturates GPU compute resources by performing matrix-matrix multiplication operations,"_ while decode _"is memory-bound … generating output tokens one at a time."_ [dasroot, Prefill Bottleneck]
- Mechanism: prefill runs `[seq_len × d] · [d × d]` **matrix–matrix** GEMMs with high arithmetic intensity (FLOPs ≫ bytes), so it lives on the compute-bound side of the roofline. Decode runs `[1 × d] · [d × d]` matrix–vector products that re-read the whole weight/KV set per token — bandwidth-bound. This is exactly why decode on this cluster is already ~24–32 tok/s (bandwidth) and prefill is the compute story.

**Implication for CPU offload:** because prefill is compute-bound and there is spare *bandwidth*, the question is purely "is there spare *matmul compute* on the CPU worth stealing?" The FLOPS numbers say no.

### The FLOPS gap that kills CPU GEMM offload (rank0, M3 Ultra 80-core)
| Unit | Throughput | Source |
|---|---|---|
| GPU FP32 (peak) | **32.77 TFLOPS** | WareDB M3 Ultra 80-core |
| GPU FP16 (peak) | **65.54 TFLOPS** | WareDB M3 Ultra 80-core |
| GPU FP32 (measured, base-M3 scaled) | ~20 TFLOPS sustained | arXiv 2502.05317 (base M3 2.47 TFLOPS × core-count scaling) |
| AMX FP32, per P-core cluster | ~0.35 TFLOPS (cblas peak) … 0.61–0.68 TFLOPS (hand-tuned, loaded) | community / arXiv 2606.25426 |
| AMX FP32, **aggregate** (2 dies → 2 clusters) | **~1.2–2.0 TFLOPS** | derived |
| CPU vector (vDSP) | ~1.5 TFLOPS (M4 base ref) | arXiv 2502.05317 |
| Unified memory bandwidth | **819 GB/s** (shared CPU+GPU) | Apple / WareDB |

- **AMX matmul is ~2–5% of the GPU's** (1.5 TFLOPS FP32 AMX ÷ 32.8 TFLOPS GPU ≈ 4.6%; ÷ 65.5 TFLOPS FP16 ≈ 2.3%). Even with a maximally generous BF16 AMX of ~3–4 TFLOPS against a sustained-45-TFLOPS GPU, the ceiling is ~7–9% *if* perfectly overlapped with zero contention — which is unphysical because AMX GEMM would compete for the same 819 GB/s.
- The strongest published pro-AMX result — a hand-written AMX prefill-GEMM kernel that _"exceeds Accelerate … ~2.0× over cblas_sgemm"_ and lifted llama.cpp full-forward from _"291 to 420 tokens/s (1.44×) at 128-token prefill"_ [arXiv 2606.25426] — was on an **M1 with no GPU in the loop** (M1 GPU is only 1.36 TFLOPS). That is a CPU-only inference win. On a box where the GPU is 20–30× the AMX, the same offload is strictly negative.

**Verdict on the bottleneck question:** during GPU-bound prefill the CPU is largely idle (mlx-vlm measured CPU at ~1.3% during a stalling prefill [mlx-vlm #945]) **and** there is spare memory bandwidth — but the idle resource that matters (matmul FLOPS) does not exist on the CPU in useful quantity. CPU offload of the compute buys ~0%. The real prefill bottleneck is (a) GPU matmul throughput and (b) scheduling/pipeline stalls, not a busy CPU or saturated bandwidth.

### The asymmetry that IS the opportunity: rank1 is an M5 Max
- M5 introduced **GPU Neural Accelerators** (matmul units per GPU core) reached via **Metal-4 TensorOps / Metal Performance Primitives**; MLX uses them. Because _"prefill is compute-bound … the Neural Accelerators directly attack the bottleneck,"_ Apple/third-party measured **~3.5–4.06× TTFT (prefill) speedup M5 vs M4** (e.g., Qwen3-14B 4-bit 4.06×; Qwen3-8B 20k-prompt 158→579 tok/s = 3.65×), while decode gains only 19–27% (bandwidth). [Apple ML Research; MacGPU; Skorppio]
- **This is where the free 1.3× lives.** rank1 owns 22 of 60 layers. If those layers' prefill GEMMs run on M5's general ALUs instead of the Neural Accelerators, you're leaving multiples on the table for rank1's share. A hand-written `mx.fast.metal_kernel` (which is what `_minimax_m3_sparse_prefill_one_pass_kernel` in `language.py` is) does **not** use the accelerators unless it explicitly emits `simdgroup_matrix`/TensorOps — so today the MSA path likely misses them on rank1 entirely.

---

## MLX CPU/GPU capabilities (what the API actually allows, with citations)

- **Device streams exist and are per-op.** Every op takes an optional `stream=` kwarg; it can be a `Stream` or a `Device` (`mx.cpu` / `mx.gpu`), defaulting to `mx.default_stream(mx.default_device())`. API: `mx.default_stream`, `mx.new_stream`, `mx.set_default_stream`, `mx.synchronize`, `mx.stream(dev)` context manager. [MLX docs: Using Streams / Devices and Streams]
- **CPU and GPU ops can run concurrently.** MLX's canonical unified-memory example runs a compute-dense `matmul` on `mx.gpu` while small overhead-bound elementwise ops run on `mx.cpu`, cutting an M1 Max micro-benchmark from **2.8 ms → 1.4 ms**. The doc's own framing: put compute-dense work on GPU, put _"very small … overhead bound"_ work on CPU. [MLX docs: Unified Memory]
- **Cross-stream dependencies are automatic.** If a GPU op consumes a CPU op's output, _"MLX will automatically insert a dependency between the two streams."_ No manual sync needed (but that dependency also means offload only helps when the two sides are genuinely independent). [MLX docs: Unified Memory / Streams]
- **The MLX CPU backend uses Accelerate → AMX for matmul.** A CPU-stream `matmul` dispatches through Apple's Accelerate BLAS onto the AMX block (~350 GFLOPS FP32 single-core via `cblas_sgemm`). So `stream=mx.cpu` GEMM *does* hit AMX — it's just tiny (see bottleneck table). Small single GEMMs at batch=1 are ~0.03 ms on Accelerate vs ~1 ms Metal-dispatch, i.e. CPU wins only on *small/latency-bound* ops, never on prefill-scale GEMMs. [community/DEV: MLX + Accelerate/AMX]
- **You already use this pattern.** `sharded_server.py:9676` runs `mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu)` _"concurrent with the model's own"_ work (comment at :5152). So the CPU-stream plumbing and precedent are in-tree — integration risk for a glue-offload probe is low.
- **You already rely on compute/transfer overlap.** `sharded_server.py:9699–9709`: you deliberately leave `mx.async_eval` native because _"forcing async_eval synchronous destroys prefill performance (no compute/transfer overlap)."_ So rank0-compute ∥ jaccl-transfer overlap already exists; the remaining bubble is rank0 idle **inside `jaccl recv`** while rank1 computes (`:5274`, `:8976`, `:9254`).
- **Chunked prefill is mandatory and bounded.** MLX's `prefill_step_size` chunks the prompt; raising 512→8192 gives up to ~1.5× [lmstudio-js #507; thornad patch], but per-chunk `mx.eval()` + `mx.clear_cache()` create sync barriers that stall the GPU (_"GPU utilization low and intermittent … CPU 1.3%"_) [mlx-vlm #945], and the **Metal command-buffer timeout** caps chunk size — unchunked PP prefill times out at ~1,500 tokens [mlx #2990]. You already run adaptive `prefill_step_size` in the 4096–16384 range (`sharded_server.py:351,467,5168`), so the easy chunk-size win is largely banked.
- **Interconnect is not the bottleneck.** Over TB5 RDMA, pipeline-parallel is within **2.3%** of tensor-parallel (PP4 14.49 vs TP4 14.82 tok/s) because _"RDMA makes the all-reduce … nearly free."_ So jaccl/Thunderbolt is not where the prefill time goes — don't chase the wire. [mlx #2990]

---

## Ranked offload candidates (the CPU-offload question, answered honestly)

| # | Candidate | Mechanism | Est. prefill payoff | Integration risk | How to prototype |
|---|---|---|---|---|---|
| A | **Overlap MSA glue (block-mask build, top-k index math, RoPE, norms) on `stream=mx.cpu`** while GPU runs the sparse-attention/GEMM kernel | Tag the small pre-attention index ops (`mx.argpartition`/`take_along_axis` at `language.py:241,320`; block-mask construction) with `stream=mx.cpu`; scheduler auto-syncs the dependency into the GPU kernel | **Low: 0–5%.** Only helps if that glue currently serializes *before* the GPU kernel on the critical path. Most of it is already tiny and overlaps. | **Low** — pattern already in-tree (`:9676`) | Wrap the mask/top-k build in `with mx.stream(mx.cpu):`, keep the one-pass kernel on GPU; A/B tok/s at 30k/80k. Pass = ≥3% and no correctness drift. |
| B | **CPU-side producer/prefetch of next chunk's token embeddings / KV index math** during the rank0 `jaccl recv` bubble | Do embedding gather + position/RoPE tables for chunk N+1 on `mx.cpu` while rank0 blocks in `recv` for chunk N | **Low: 0–4%.** Bubble is short vs GEMM time; embeddings are gather (bandwidth), not matmul | **Low–Med** | Time the `recv` stall window (`:5274`); move only cheap index/gather ops into it on `mx.cpu`. Pass = measurable reduction in idle-in-recv with net tok/s gain. |
| C | **Fractional attention/MLP split CPU+GPU for the same layer** (e.g., a few heads or the gate/up on AMX) | Run part of the GEMM on `stream=mx.cpu` (Accelerate/AMX), rest on GPU | **~0% / negative.** AMX is 2–5% of GPU FLOPS and contends for the same 819 GB/s | **High** — dtype/quant mismatch (4-bit weights need CPU dequant), correctness, contention | Not recommended. If tested at all, one head on `mx.cpu` as a falsification probe; expect it to be slower. |
| D | **CPU dequant of the 4-bit path feeding a CPU GEMM** | Dequant + `cblas` GEMM on AMX | **Negative.** Adds dequant cost to a unit that's already 20–30× too slow | **High** | Skip. Documented here only to close the loop: the compute deficit is decisive. |
| E | **Full CPU/AMX GEMM offload of whole layers** | Route layers to `mx.cpu` | **Strongly negative** (see FLOPS table) | **High** | Do not pursue. |

**Net:** only **A** (and marginally **B**) are worth a single low-cost probe — and their upside is ≤5%, i.e. they cannot by themselves close the 384→500 (~1.3×) gap. Everything that actually reaches 1.3× is GPU-side or pipeline-side (next section).

---

## Recommended experiment sequence (maintenance-window, measurable pass/fail)

Order = highest expected tok/s per unit risk first. Baseline for all: current 384/378/366 tok/s at 30k/80k/200k; measure with the existing bench harness (`bench_*_*.jsonl`, `perf_probe_*`).

1. **Confirm the M5 TensorOps path on rank1 (diagnostic, zero code).** Verify the pinned MLX build (c110f69e / mlx-vlm064-env, per memory) is a Metal-4 build that exposes GPU Neural Accelerators, and check whether the dense prefill GEMMs on rank1 dispatch to TensorOps (Metal capture / `MTL_CAPTURE`, or a 22-layer-only micro-bench of a dense matmul on rank1 vs rank0 normalized per-core). **Pass:** rank1 dense-GEMM TFLOPS ≳2× its general-ALU expectation ⇒ accelerators active. **Fail (accelerators idle):** this is your biggest lever — pursue #2.

2. **Emit `simdgroup_matrix`/TensorOps in the MSA one-pass prefill kernel (rank1 first).** The current `_minimax_m3_sparse_prefill_one_pass_kernel` is a scalar/threadgroup Metal kernel (`language.py:331–572`); rewrite the inner Q·Kᵀ and P·V tiles with `simdgroup_matrix` (and Metal-4 tensor ops where available). **Pass:** ≥1.5× on rank1's isolated 22-layer prefill with bit-comparable outputs; ≥1.15× end-to-end at 80k. **Prototype safely** behind an env flag (mirrors `MLX_M3_MSA_PREFILL`), one rank at a time.

3. **Pipeline-chunk double-buffering + split rebalance.** With chunked prefill already in place, make rank0 launch chunk N+1's forward while it would otherwise block in `jaccl recv` for chunk N (async_eval is already native — extend the overlap window), and A/B a 36/24 or 34/26 split to move load onto the faster-prefill M5. **Pass:** reduced rank0 idle-in-`recv` (instrument `:5274`) **and** ≥1.1× end-to-end at 200k with no wedge/orphan regressions (respect the ThunderMLX sync whitelist).

4. **Sync-barrier audit + step-size sweep within the Metal timeout.** Trace the prefill loop (`:1051`, `:5168`) for any per-chunk `mx.eval`/`mx.clear_cache` (killer per mlx-vlm #945); confirm `clear_cache` fires only at request/error boundaries (`:5288–5290`), not per chunk. Sweep `prefill_step_size` up to the largest value that stays under the command-buffer timeout at 200k. **Pass:** ≥1.05× and no timeouts; **or** documented "already optimal" if no per-chunk barrier exists.

5. **CPU glue-overlap probe (candidate A) — the honest CPU-offload test.** Move MSA block-mask build + top-k index math to `with mx.stream(mx.cpu):` while the sparse kernel runs on GPU. **Pass:** ≥3% at 30k/80k with identical top-k selection. **Expected outcome:** ≤5% or noise — which empirically closes the "should we offload to CPU?" question and lets you stop investing there.

---

## Sources (URLs actually consulted)

- Apple ML Research — Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- MLX docs — Using Streams: https://ml-explore.github.io/mlx/build/html/usage/using_streams.html
- MLX docs — Unified Memory (concurrent CPU/GPU example): https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- MLX docs — Devices and Streams: https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html
- MLX GitHub Discussion #2990 — TB5 RDMA: Pipeline vs Tensor Parallelism (Kimi-K2): https://github.com/ml-explore/mlx/discussions/2990
- mlx-vlm Issue #945 — server prefill slow due to per-chunk mx.eval()/mx.clear_cache() sync barriers: https://github.com/Blaizzy/mlx-vlm/issues/945
- lmstudio-js Issue #507 — prompt-processing chunk size 8192 → up to 1.5× prefill: https://github.com/lmstudio-ai/lmstudio-js/issues/507
- thornad/lmstudio-mlx-patch — 2× prompt processing on Apple Silicon: https://github.com/thornad/lmstudio-mlx-patch
- arXiv 2606.25426 — "Above the Inner Loop: Exceeding Accelerate at LLM Prefill GEMM on the M1 AMX": https://arxiv.org/abs/2606.25426
- arXiv 2502.05317 — "Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC" (GPU/CPU/AMX FLOPS, bandwidth): https://arxiv.org/html/2502.05317v1
- arXiv 2606.12765 — "Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU": https://arxiv.org/pdf/2606.12765
- arXiv 2308.16369 — "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills": https://arxiv.org/pdf/2308.16369
- WareDB — Apple M3 Ultra (80-core GPU) specs (32.77 TFLOPS FP32 / 65.54 TFLOPS FP16 / 819 GB/s): https://www.waredb.com/processor/apple-m3-ultra-gpu-80-cores
- Apple Newsroom — M3 Ultra (>800 GB/s, up to 80-core GPU): https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
- dasroot — "The Prefill Bottleneck Problem" (prefill compute-bound vs decode memory-bound): https://dasroot.net/posts/2026/05/prefill-bottleneck-token-generation-latency-prompt-processing/
- MacGPU — 2026 M5 MLX Neural Accelerators TTFT/decode benchmarks: https://macgpu.com/en/blog/2026-0425-mac-m5-neural-accelerators-mlx-ttft-decode-benchmark-remote.html
- Skorppio — Apple M5 Max vs NVIDIA DGX Spark LLM benchmark (M5 GPU matmul TFLOPS): https://skorppio.com/blog/apple-m5-max-vs-nvidia-ai-deep-dive
- meekolab — "The Elusive Apple Matrix Coprocessor (AMX)": https://research.meekolab.com/the-elusive-apple-matrix-coprocessor-amx
