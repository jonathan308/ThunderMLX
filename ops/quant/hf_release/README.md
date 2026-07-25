---
base_model: MiniMaxAI/MiniMax-M3
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- minimax
- mixture-of-experts
- mixed-precision
- quantization
- thundermlx
license: other
license_name: minimax
---

# MiniMax-M3 Mixed-4.5bit MLX — the anti-overthinking quant

A **mixed-precision MLX quantization of MiniMax-M3** (428B parameters, 23B
active) that puts precision where decisions are made instead of spreading it
evenly. Built for and served by
[ThunderMLX](https://github.com/jonathan308/ThunderMLX), a 2-Mac pipeline
serving stack for Apple Silicon.

**TL;DR:** at +45 GB over the standard flat 4-bit (270 vs 225 GB), this quant
closes ~28% of the entire fidelity gap to the bf16 model, cuts reasoning-loop
"doom spirals" by 42–60%, eliminates 92% of hesitation markers, ships complete
agentic artifacts instead of drafting them inside thinking — and finishes real
tasks **15% faster in wall time** despite ~12% slower raw decode, because it
stops second-guessing itself.

## Why: flat 4-bit quantization causes overthinking

Running MiniMax-M3 4-bit in agentic use, we kept hitting a failure family:
thinking spirals that re-analyze the same paragraph with mutating wording,
hesitation cascades ("wait… actually… let me reconsider"), and a stubborn
habit of drafting entire code artifacts inside the thinking block while
ignoring steering. Following arXiv 2606.00206 (quantization inflates
hesitation-marker probabilities at high-entropy positions), we first shipped a
runtime logit-penalty guard — it helped, but treated the symptom.

The cause turned out to be *where* flat quantization spends its error budget.
Rounding noise in a handful of small, decision-critical modules flips discrete
choices: which experts fire, which KV blocks sparse attention reads, and which
token wins the final logit race. This quant fixes those modules directly.

## The recipe

| Tier | Modules | Precision | Rationale |
|---|---|---|---|
| Decision | lm_head, all 57 MoE router gates, sparse-attention indexer projections | **8-bit / g64** | rounding noise here flips discrete choices — the literal overthinking mechanism |
| Every-token | embeddings, all attention projections, dense-MLP layers | **6-bit / g64** | error compounds across all 60 layers with no routing dilution |
| Bulk | all 129-expert fused MoE tensors | **4-bit / g32** | halved group size halves in-group rounding error; the cheapest quality lever on 96% of the weights |
| Native | vision tower, norms (bf16), e_score_correction_bias (f32) | untouched | matches upstream |

Effective average: ~4.8 bits/weight. Identical tensor names and MLX affine
format to the standard 4-bit conversion — **loads anywhere the flat 4-bit
loads**, no code changes.

## Benchmarks

### Distribution fidelity (teacher-forced EAR vs a bf16-grade reference, ~10k positions)

EAR = per-position overlap between the quant's and the reference model's
next-token distributions (metric from arXiv 2605.02404), normalized, higher
is better. Reference = the bf16 checkpoint itself (experts at lossless 8-bit),
evaluated with a layer-streaming pass.

| Quant | Size | EAR mean | Worst-5% positions |
|---|---|---|---|
| flat 4-bit / g64 | 225 GB | 0.8747 | 0.5236 |
| same-budget control (extra bits spread across bulk experts) | 268 GB | 0.8806 | 0.5493 |
| **this quant** | 270 GB | **0.9103** | **0.6656** |

The control experiment is the point: an equal-size quant that spends its extra
bits on bulk experts recovers ~5% of the gap to bf16. Spending the same bits
on the decision path recovers **~28%** — and **~30% at the hard-position tail**
where reasoning behavior lives. Where the bits go matters far more than how
many.

### Behavior (identical prompts and seeds vs flat 4-bit, guard disabled)

| Suite | flat 4-bit | this quant |
|---|---|---|
| Graded tasks — accuracy | 100% | **100%** |
| Graded — avg thinking tokens | 176 | **121 (−31%)** |
| Graded — hesitation markers/run | 0.60 | **0.05 (−92%)** |
| Graded — avg wall time | 8.0 s | **6.8 s (−15%)** |
| Loop probes (3 seeds) — avg thinking tokens | 1992 | **1159 (−42%)** |
| Loop probes — hesitation markers | 28.9 | **7.7 (−73%)** |

Ungoverned, this quant out-behaves the flat 4-bit running its most aggressive
anti-overthinking logit penalty. On the flagship two-turn agentic test (build
a complete single-file game, then steer), it plans in ~1k characters of
thinking and ships a complete 46.8k-character working artifact in the answer —
the flat 4-bit drafted the entire artifact inside its thinking block and
resisted steering. Long thinking is preserved where it's warranted: hard
constraint-solving still gets ~4k tokens of *forward-moving* reasoning
(2.3% repeated-phrase churn vs >10% in true spirals).

### Speed (2-Mac ThunderMLX pipeline, Thunderbolt RDMA, 38/22 layer split)

| Metric | flat 4-bit | this quant |
|---|---|---|
| Decode, short context | ~28 tok/s | 23–26 tok/s |
| Decode @ 70k context | ~27–29 tok/s | 23.8 tok/s (no depth collapse) |
| Prefill @ 70k | — | 342 tok/s |
| TTFT (warm) | ~1.4 s | ~1.4 s (unchanged) |

The ~12% decode tax is repaid with interest on real tasks by shorter,
non-redundant thinking (see wall times above).

## Serving

Built for [ThunderMLX](https://github.com/jonathan308/ThunderMLX) across two
Apple Silicon Macs (tested: Mac Studio + MacBook Pro, 38/22 pipeline split,
~187 GB + ~96 GB wired). Any MLX stack that serves the standard 4-bit
conversion can load this model unchanged — same tensor names, same config
schema, per-path quantization overrides declared in `config.json`.

## Reproduce / adapt

The converter, verification suite, and EAR evaluator are open source in the
ThunderMLX repo (`ops/quant/`):

- `m3_mixed_quant.py` — streaming mixed-precision converter: plan pass with a
  name-set parity gate, per-expert rebuild of fused MoE tensors, incremental
  5 GB shards, ~15 GB peak memory while converting an 854 GB checkpoint.
- `ear_eval.py` / `ear_compare.py` — layer-streaming EAR evaluator: exact
  next-token distributions from models far larger than RAM, including the
  bf16 reference itself.

Two upstream findings the tooling works around, relevant to anyone quantizing
very large MoE models with MLX: (1) kernels evaluated on tensors above ~2³¹
elements can silently corrupt output — fused MoE expert tensors are exactly
that size, so the converter rebuilds them per-expert; (2) GPU kernels fed
directly from memory-mapped files on slow external drives stall past the
Metal watchdog — the converter materializes on the CPU stream first.

## Acknowledgements

- MiniMax for MiniMax-M3.
- arXiv 2606.00206 (quantization-induced overthinking) for the mechanism, and
  arXiv 2605.02404 (statistically-lossless quantization) for the EAR metric.
- The MLX team — this entire pipeline runs on MLX.
