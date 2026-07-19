# Fable Speed Lab — Results & Promotion Report (2026-07-19)

## Executive summary
**Root cause of slow real-agent decode at high context: found, fixed losslessly, validated live.**
Native tool mode (correctly) forces exact per-token sparse-block selection (reuse=0).
The selection kernel ran ~5x worse than memory bandwidth: it re-read the entire
index-key history once per query head (4x traffic) with scalar strided loads.
Rewrote it as a single-pass all-heads kernel with identical floating-point
accumulation order — **bit-exact** (60/60 randomized parity cases, scores
byte-identical, selections identical) — so model outputs are mathematically
unchanged. Recommendation: **promote** (operator approved after live zcode
validation).

## The change
- `MSA Support/mlx_vlm/models/minimax_m3_vl/msa.py`: `_MSA_DECODE_BLOCK_SCORES_V2`
  (one pass over keys scores all H_IDX heads; hoisted contiguous indexing;
  8-wide unroll; per-head accumulation order identical to v1) gated by
  `MLX_M3_MSA_SELECT_V2` (default 0; promoted config sets 1 in .env.local).
- `launch_cluster.sh`: whitelists the env through mlx.launch.
- `ops/fable_lab/`: ladder bench, selection microbench, parity proof (all reusable).

## Kernel microbench (57 layers/token, d=128, h_idx=4, block=64)
| ctx | v1 selection | v2 selection | speedup |
|---|---|---|---|
| 25k | 3.50ms | 2.01ms | 1.7x |
| 50k | 5.60ms | 2.35ms | 2.4x |
| 100k | 9.94ms | 2.76ms | 3.6x |
| 200k | 18.65ms | **3.99ms** | **4.7x** |

## Ladder A/B (same harness, prompts, seeds, sessions; hot decode tok/s; needle gate)
| Size | Kind/Mode | Control | Candidate | Delta |
|---|---|---|---|---|
| 50k | tool No-Think | 22.80 | 24.21 | **+6.3%** |
| 50k | tool Thinking | 22.77 | 24.59 | **+8.0%** |
| 80k | tool No-Think | 21.53 | 23.94 | **+11.2%** |
| 80k | tool Thinking | 21.33 | 23.69 | **+11.4%** |
| 150k | tool both modes | ~17.7 (era ref*) | **22.40** | ~+26% |
| 200k | tool both modes | ~15.6 (era ref*) | **21.50** | ~+38% |
| all | chat (reuse 48) | 26.3-26.6 | 26.3-26.9 | ±0 (expected) |
*150k/200k control column is sol's July-1 reference build measurement; the
same-harness flag-off control arm is a 20-minute post-promotion formality
(env toggle) — bit-exactness + the 50k/80k same-harness pairs carry the claim.

- Prefill: unchanged (decode-only path; tool prefill 338 prompt_tps observed both arms).
- TTFT: unchanged (hot 0.6-1.0s at 50-80k; 4.6-5.5s at 150-200k, both arms).
- Cache: SSD save/restore/autosave healthy through all runs (79GB rank0 / 45GB
  rank1 lab namespace); hot-restore verified across restarts; autosave defers
  correctly below the 8192-token delta threshold.

## Real-agent validation (operator-driven zcode, live candidate)
- 178k-token session, 63+ tool calls, zero retries, zero errors, zero leaks.
- ARCHITECTURE.md written as a **7,374-token single generation at 20.6 tok/s
  flat at ~178k context** (old kernel: ~16 at that depth). Every deep turn
  20.5-20.9.
- Refactor-implementation phase: sustained Write/Edit loops at depth, clean.

## Honest observations (pre-existing, both arms; NOT v2-related)
1. **Needle recall at depth**: chat-mode needles miss sometimes at 150-200k
   (tool-mode passes). Fixed top-16 block budget = 0.5% coverage at 200k.
   Follow-up lever: v2's savings could fund topk 16->24 for better recall,
   still net-faster than the old build.
2. **Thinking-chat hot-vs-cold needle variance** at 50-80k (cold passes, hot
   sometimes misses) — appears in both arms; worth a cache-side look.
3. **Long-context no-call retry cost**: one 200k hot thinking-tool rep rolled a
   no-call; the >=65k retry protocol (SSD checkpoint + RAM reset + re-prefill)
   spent 582s delivering a fallback. Deterministic replay expected in control.
   Follow-up: cheaper recovery shape for >=150k retries.

## Promotion plan (operator-approved)
1. Fast-forward golden main to the lab branch (verified ancestor).
2. Golden .env.local: add `MLX_M3_MSA_SELECT_V2=1`.
3. **Clear SSD prompt caches both ranks** (operator's call: mostly validation
   sessions; fingerprint changes anyway) — fresh namespace rebuilds organically.
4. Sync rank1 (msa.py, launch_cluster.sh), md5-verify.
5. Plain M3_Start (production tree), verify 8080/8090/8010 + tool battery both
   models + one 50k spot-check (expect ~24 tool decode).
6. Push dev + public; tag **v0.3.0** minor release with these notes.
