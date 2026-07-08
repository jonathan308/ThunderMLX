# Design: Native in-flight cancel via mirrored BatchGenerator (lockstep)

Status: SHIPPED + CERTIFIED 5/5 2026-07-06 20:28 (commit 14444cd).
Gate: MLX_M3_BATCH_CANCEL=1 (live on main, both ranks).
As built: no per-step ctrl word in decode — the EOS swap in the synced
sampler IS the cancel (the batch loop's step-synchrony makes it safe; the
stream path's pre-building was the deadlock all along). Prefill cancel =
one int32 all_sum per chunk boundary. See m3_batch_cancel.py. CRITICAL:
the swap must be mx.depends(eos, sampled) — a bare constant starves the
peer's h-send (three-bug chain in memory + commit 14444cd).

## Why
Three mid-stream break designs failed identically (photographed mutual
MeshImpl::send deadlock in the lookahead step: rank0 token-sync send vs
rank1 pipeline-h send, each awaiting a recv the broken peer never posts).
Conclusion: on pipeline jaccl, generation may only end at a STEP BOUNDARY
agreed by both ranks. mlx-vlm 0.6.4 ships exactly that primitive:
`BatchGenerator.remove(uid)` (generate/ar.py:2570) — removes a sequence
between forwards, handling queued/prefilling/decoding phases.

## Architecture
Per request (both ranks, after the existing rank_request _bcast):
1. Both construct/reuse `BatchGenerator(model.language_model, processor,
   prefill_batch_size=1, completion_batch_size=1, **kwargs)`.
2. Both call `gen.insert([token_ids], [max_tokens], prompt_kwargs=...)`
   with byte-identical args -> identical uid on both ranks.
3. Step loop (BOTH ranks, identical):
       ctrl = _bcast(None-or-cmd, rank)   # rank0 sends {"op":"step"} or
                                          # {"op":"cancel"} — tiny,
                                          # SEQUENTIAL (before the forward,
                                          # never concurrent with model
                                          # collectives; EXO-style consensus)
       if ctrl.op == "cancel":
           gen.remove(uid)                 # step-boundary removal, both ranks
           break                           # nothing in flight by construction
       responses = gen.next()              # ONE forward step, identical
                                           # collectives on both ranks
       rank0: feed response.token -> streaming detokenizer -> SSE delta
       both: finish_reason set -> break (natural end)
4. rank0 finalize: existing barrier/prewarm/cache path unchanged.

## Key integration points
- Token sync: existing _synced_sample_with_positions patch already wraps
  the sampler the batch path calls (ar.py:1028 passes positions) — rank1
  keeps consuming rank0's tokens. EOS-swap scaffolding can be DELETED.
- Prompt cache: our prepared MiniMaxM3KVCache converts via `c.to_batch()`
  (ar.py:754-757, added upstream FOR this cache class). Plumb the prepared
  cache into insert's prompt_kwargs; verify trim/finalize/filter behavior
  (language.py:569/634/737/759).
- Patched __call__ must tolerate batch kwargs (return_hidden/skip_logits) —
  LanguageModel pops them natively; verify with temp=0 greedy probe (the
  fast path triggers on greedy + speculative_argmax_from_hidden).
- Detokenizer: rank0 builds text via make_streaming_detokenizer (upstream
  server pattern) since Response carries token ids, not text.
- Watchdog: tick per gen.next(); prefill progress from gen.stats().
- The control-word bcast replaces run_mirror's blocking idle _bcast during
  generation only; idle wait unchanged (guard timeout covers it).

## Why the per-step control bcast is safe (vs the historical wedge)
The 2026-07-05 wedge root cause was a per-token all_sum on stream=mx.cpu
CONCURRENT with model collectives on the same QP/CQ. The control word here
is strictly SEQUENTIAL: it completes before gen.next() dispatches any model
collective, at the same position in both ranks' op order — no cross-stream
race, no reordering.

## Acceptance
ops/stop_acceptance_rig.sh all-5 PASS + 10k decode + agent cycles + idle
survival, then soak. Also retest the residual-wedge hypothesis: with
step-boundary endings everywhere, natural-stop wedges should disappear —
if they do, the "driver CQE loss" residual was this race all along.

## Cleanup once landed
- Delete _FORCE_EOS scaffolding + stop-file decode branches.
- /v1/stop + disconnect -> enqueue "cancel" ctrl (instant, phase-agnostic).
- Consider completion_batch_size>1 next: continuous batching (multi-client).
