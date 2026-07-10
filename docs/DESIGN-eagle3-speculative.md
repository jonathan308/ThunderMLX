# EAGLE3 speculative decode on the 2-rank pipeline — buildout design

Workspace: `~/ThunderMLX-eagle3` (branch `feature/eagle3-speculative`, clone of
golden @ 17e4d4a). Golden (`~/minimax-m3-cluster`) stays untouched until the
maintenance window validates this build.

## Goal
Decode 17 t/s @35k / 20 @19k / 23 short → ~2× via EAGLE3 speculative decoding,
using the released drafter `Inferact/MiniMax-M3-EAGLE3` (staged at
`~/.exo/models/Inferact--MiniMax-M3-EAGLE3`) and mlx-vlm 0.6.4's speculative
module (`draft_kind="eagle3"`), with zero regression to prefill, stops,
cache reuse, or wedge-freedom.

## Upstream facts (source-verified)
- `mlx_vlm.speculative.drafters.load_drafter(path)` → `(model, "eagle3")`
  (auto-detects from HF `model_type`).
- `stream_generate(draft_model=…, draft_kind="eagle3")` replaces the decode
  loop with `_eagle3_rounds`; a batch variant `_eagle3_rounds_batch` exists
  behind `get_speculative_rounds_batch("eagle3")` — our m3_batch_cancel
  mirror drives the batch generator, so the batch variant is the target seam.
- EAGLE3 drafts from CAPTURED target hidden states:
  `draft_model.config.target_layer_ids` (or `capture_layer_ids`) names the
  layers. The verify walk is greedy prefix-accept
  (`_eagle3_walk`: accepted = longest draft/target token match, +1 bonus).
- The target model must expose `rollback_speculative_cache` (KV rollback of
  rejected draft tokens) — check native minimax_m3_vl for it; if absent,
  implement via `_trim_prompt_cache_in_place`-style trim (MiniMaxM3KVCache
  supports trim incl. index cache).

## Pipeline (38,22 split: rank1 owns layers 0-21, rank0 owns 22-59) — the
## four distributed problems and their designs

1. **Capture shipping (drafter inputs).** The pipeline patch captures
   rank-locally only (`hidden_sink.append` inside the layer loop). If
   `target_layer_ids` include any layer <22, rank1 must ship that hidden to
   rank0 every round. DESIGN: piggyback — rank1 concatenates
   `[h_boundary, h_capture_l2, …]` along the feature axis into the ONE send
   it already performs per forward; rank0 splits. Zero new collectives ⇒
   the collective count/order stays identical to today's proven decode.
   (Payload grows by n_captures × K tokens × hidden ≈ hundreds of KB —
   negligible on TB5.)

2. **Accept-count synchronization.** rank0 (sole decode owner, has logits)
   computes the accepted prefix length per round; rank1 must trim the SAME
   number of rejected draft tokens from its KV. DESIGN: reuse the existing
   sampled-token sync channel — the per-round "sampled token" broadcast
   becomes (accept_count, bonus_token); rank1 derives its trim locally.
   Again zero new collectives. The 2026-07-08 consensus/physical-truth
   fixes are the safety net: any trim slip = one loud identical rebuild,
   not a wedge.

3. **Verify forward = K-token step.** The pipeline patch already handles
   L>1 forwards (prefill chunks); a K≈16 verify step is a mini-prefill.
   Index caches update+trim symmetrically via the same cache objects.
   CAUTION: the kv_step=4096 boundary send during decode is the historical
   freeze surface — verify steps crossing a boundary must be exercised in
   the rig (generate across 4096·n totals deliberately).

4. **Stops/cancel.** The EOS-swap stop rides the sampled-token sync; with
   (2) the swap applies to the bonus token. Batch-cancel drain semantics
   unchanged (rounds end at the same lockstep boundary). Keep-cache-on-
   cancel already trims to input prefix — works unchanged.

## Integration shape
New module `m3_eagle3.py` (mirrors m3_batch_cancel's pattern):
- env `MLX_M3_EAGLE3_DRAFT` (path; empty = feature off — golden default),
  `MLX_M3_EAGLE3_BLOCK` (override draft block, default drafter-configured).
- rank0: `load_drafter()` at model-load time (bf16, ~1-2 GB, Studio-resident;
  rank1 never loads it).
- `_generation_iter` gains an eagle3 branch when the drafter is armed and
  the request shape is supported (text-only, B=1, no images) — falls back
  to the existing batch path otherwise, same Unsupported pattern.
- Pipeline patch: extend `capture_layer_ids` handling with the piggyback
  send/split (gated on the same env so golden's path is byte-identical
  when off).

## Test plan (maintenance window)
1. Offline: drafter load smoke (no cluster) — DONE-able pre-window.
2. Boot eagle3 build (ports as golden, golden stopped): health, overlay
   install, drafter-armed log on rank0 only.
3. Correctness: pinned-seed A/B — eagle3 ON vs OFF must produce identical
   greedy outputs (speculative is lossless); tool-call turn; thinking turn.
4. Decode speed: short-ctx, 19k, 35k (targets: 23→~40, 20→~35, 17→~28+,
   acceptance-dependent).
5. Stability: 3×10k decode incl. kv_step boundary crossings; stop/cancel
   mid-round; keep-on-cancel retry; 16-turn tool loop.
6. Fallback drill: MLX_M3_EAGLE3_DRAFT="" → byte-identical golden behavior.
7. If green: merge branch → golden, publish, restore via M3_Start.

## Open items (resolve when drafter config lands)
- `target_layer_ids` values → is the piggyback needed at all? (If all ≥22,
  rank0-local capture suffices and step 1 collapses.)
- Drafter dtype/quantization (config.json) + actual RAM.
- Whether native minimax_m3_vl implements `rollback_speculative_cache`.
