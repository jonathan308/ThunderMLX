"""Native in-flight cancel for the 2-rank MiniMax-M3 pipeline.

Gate: MLX_M3_BATCH_CANCEL=1 (read identically on both ranks via .env.local).

Why the upstream stream path cannot stop early on pipeline jaccl
----------------------------------------------------------------
generate_step() pre-builds step N+1 before yielding token N (double-buffered
decode). On the split model that pre-built step contains live cross-rank
collectives (rank1's blocking h-send eval, rank0's token-sync send). Any exit
decision made by the CONSUMER therefore lands while asymmetric work is queued
— photographed three times as the mutual MeshImpl::send deadlock. The stream
geometry is unfixable from outside the generator.

Why this path can
-----------------
mlx-vlm 0.6.4's PromptProcessingBatch/GenerationBatch are step-synchronous:
work for step K is built only inside next() call K, prompt_step() fully
evals each prefill chunk, and GenerationBatch.filter() materializes pending
state before teardown. Both ranks run the identical loop on the identical
synced token stream, so every exit decision is made from identical data at
an identical step boundary — nothing asymmetric is ever in flight.

Cancel mechanics (no new decode-phase collectives):
- decode: rank 0 arms sharded_server._FORCE_EOS (the /v1/stop and
  tool-complete paths already do). The token-sync sampler patch swaps the
  next sampled token for EOS before the send; both ranks' stop_criteria fire
  on the same token at the same next() boundary. Latency ~1-2 tokens.
- prefill: one int32 all_sum per chunk boundary (prompt_step is synchronous,
  so nothing else is on the wire when it runs). Rank 0 contributes 1 when a
  stop is pending; both ranks then abandon the prompt batch identically.
  Under MLX_M3_PREFILL_OVERLAP_NATIVE=1 the chunks are software-pipelined
  two deep (docs/DESIGN-chunk-overlap-native.md); the ctrl-word still fires
  every chunk, and the one in-flight chunk is drained before any abandon so
  teardown stays at a fully-evaluated boundary on both ranks.
- consumer break/disconnect: surfaces here as GeneratorExit at a yield.
  We arm EOS and DRAIN the loop (no further yields) until it ends naturally,
  so the ranks stay lockstep even while the client is already gone.

Prompt-cache reuse is preserved: the server's prepared per-layer caches are
converted to warm batch caches (contents intact, B=1) for the loop, then the
grown state is written back into the original cache objects so the server's
persistence/trim machinery is unchanged.
"""

import logging
import os
import sys
import time

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("MLX_M3_BATCH_CANCEL", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# Depth-2 chunked-prefill overlap (docs/DESIGN-chunk-overlap-native.md).
# Read identically on both ranks via .env.local, like _ENABLED above.
_OVERLAP_NATIVE_ENABLED = os.environ.get(
    "MLX_M3_PREFILL_OVERLAP_NATIVE", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Prefill cancel-check cadence. _ctrl_prefill() is a BLOCKING all_sum + host
# sync at every chunk boundary; on the 2-node jaccl pipeline that extra
# collective (and its cross-stream contention with the pipeline send/recv)
# costs ~20-25% of prefill wall time versus stream_generate, which has no such
# check. Running it every Nth chunk instead keeps /v1/stop responsive
# (worst-case cancel latency = N prefill chunks) while recovering the speed.
# MUST be read identically on both ranks (forwarded env) so the collective
# count stays matched -- a per-rank mismatch would desync the ring and wedge.
# 1 = every chunk (original), N = every Nth chunk, 0 = never (no prefill cancel).
_PREFILL_CANCEL_EVERY = max(
    0, int(os.environ.get("MLX_M3_PREFILL_CANCEL_EVERY", "1") or "1")
)


def _rank_aware_int(name, default, *, fallback_name=None):
    rank = os.environ.get("MLX_RANK", "").strip()
    keys = []
    if rank:
        keys.append(f"{name}_RANK{rank}")
    keys.append(name)
    if fallback_name:
        if rank:
            keys.append(f"{fallback_name}_RANK{rank}")
        keys.append(fallback_name)
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip():
            return int(value)
    return int(default)


# A request's max_tokens is a safety ceiling, not its expected output size.
# Reserving the whole ceiling on every warm-cache conversion pushed a 45k
# session to 78k physical KV slots and exhausted the 128GB rank. Keep a small
# rank-aware window and let the native cache grow if a response truly needs it.
_BATCH_APPEND_RESERVE_TOKENS = max(
    0,
    _rank_aware_int(
        "MLX_M3_BATCH_APPEND_RESERVE_TOKENS",
        4096,
        fallback_name="MLX_M3_PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS",
    ),
)


class Unsupported(Exception):
    """Request shape the batch-cancel path does not handle; use stream_generate."""


def enabled():
    return _ENABLED


def _server():
    return sys.modules.get("sharded_server")


def _cancel_pending_rank0():
    """True when a client stop is pending (rank 0 only; others always False).

    Covers both arming styles: _FORCE_EOS already active (decode poll armed
    it) and a fresh, nonce-matched stop file that decode's n%8 poll has not
    consumed yet — so prefill-phase stops don't wait for decode to begin.
    """
    srv = _server()
    if srv is None:
        return False
    fe = getattr(srv, "_FORCE_EOS", None)
    if fe and fe.get("active"):
        return True
    try:
        sp = srv._read_prefill_stop_file()
        if sp and sp.get("reason"):
            file_nonce = sp.get("nonce")
            cur_nonce = srv._STOP_NONCE.get("value")
            # The prefill stop file is cleared at every request start on BOTH
            # ranks, so a stop file present during prefill was written for THIS
            # request. Accept it unless it carries a DIFFERENT nonce (a
            # distinguishably-stale file). A missing file nonce is treated as
            # current: /v1/stop writes the file with nonce=_STOP_NONCE.value,
            # but that is None for request paths that never populate the nonce
            # (e.g. non-stream), and the strict `nonce and ==` test then silently
            # failed to abort prefill. Existence + reason + non-conflicting nonce
            # is the robust signal.
            if file_nonce is None or file_nonce == cur_nonce:
                return True
    except Exception:
        pass
    return False


def _arm_force_eos():
    srv = _server()
    fe = getattr(srv, "_FORCE_EOS", None) if srv else None
    if fe is None or fe.get("eos_id") is None:
        logger.warning("batch-cancel: _FORCE_EOS unavailable; drain continues")
        return
    fe["active"] = True


def _ctrl_prefill(rank, cancel):
    """One tiny all_sum at a fully-evaluated chunk boundary. Returns True to
    abandon prefill on BOTH ranks. Sequential with all model collectives —
    prompt_step() has already mx.eval'd the chunk when this runs."""
    word = mx.distributed.all_sum(
        mx.array(1 if (rank == 0 and cancel) else 0, dtype=mx.int32)
    )
    mx.eval(word)
    return int(word.item()) > 0


# ---------------------------------------------------------------------------
# Chunked-prefill overlap (docs/DESIGN-chunk-overlap-native.md)
# ---------------------------------------------------------------------------
# prompt_step() fully evals each chunk, so on the pipeline split each rank
# idles while the other computes its half (~63% of chunk wall-time on rank1
# at 38/22). The depth-2 pipeline below issues chunk N+1's lazy forward while
# chunk N executes: async_eval launches N, eval completes N-1. Both ranks
# issue identical graphs in identical order, so every send/recv pair stays
# matched — the same async_eval discipline decode already uses.

# Everything _prompt_step_lazy touches on PromptProcessingBatch. Checked up
# front (not mid-loop) so an upstream-drift fallback to the serial path is
# decided identically on both ranks before any chunk is in flight.
_LAZY_STEP_ATTRS = (
    "needs_processing",
    "model",
    "prompt_cache",
    "prefill_step_size",
    "_inputs_embeds",
    "_input_ids",
    "_processed_prompt_columns",
    "_prompt_kwargs",
    "_prompt_length_aware_keys",
    "_prompt_kwargs_for_step",
    "_next_apc_checkpoint_column",
    "_store_apc_exact_checkpoints",
    "_apc_manager",
)


def _require_lazy_prefill_support(pb):
    """_prompt_step_lazy vendors mlx-vlm 0.6.4 internals; verify the installed
    build still has them before entering the overlap loop. The verdict is a
    pure function of the installed package (identical on both ranks)."""
    from mlx_vlm.generate import ar as ar_mod

    missing = [a for a in _LAZY_STEP_ATTRS if not hasattr(pb, a)]
    if missing:
        raise Unsupported(
            "PromptProcessingBatch drift: missing " + ", ".join(missing)
        )
    if pb._apc_manager is not None:
        raise Unsupported("overlap prefill does not handle APC-managed batches")
    if pb._prompt_length_aware_keys and not hasattr(
        ar_mod, "_slice_sequence_aligned_prompt_kwarg"
    ):
        raise Unsupported("ar._slice_sequence_aligned_prompt_kwarg missing")


_WARNED_NO_OVERLAP_FLAG = False


def _set_pipeline_overlap_flag(active):
    """Toggle m3_pipeline_patch._M3_PREFILL_OVERLAP_ACTIVE (imported lazily:
    sharded_server plain-imports the patch module, so this resolves to the
    same instance whose _patched_call closure reads the flag). If the setter
    is unavailable, overlap stays CORRECT but degraded: rank1's per-chunk
    eager h-eval re-serializes the chunks it was meant to overlap."""
    global _WARNED_NO_OVERLAP_FLAG
    try:
        import m3_pipeline_patch

        m3_pipeline_patch.set_prefill_overlap_active(active)
    except Exception as e:
        if active and not _WARNED_NO_OVERLAP_FLAG:
            _WARNED_NO_OVERLAP_FLAG = True
            logger.warning(
                "batch-cancel: overlap flag unavailable (%s); prefill overlap "
                "degrades to serial on rank1", e
            )


def _prompt_step_lazy(pb):
    """PromptProcessingBatch.prompt_step vendored from mlx-vlm 0.6.4
    (generate/ar.py:1791) with the per-chunk mx.eval([c.state ...]) and
    mx.clear_cache() HOISTED OUT so the caller can pipeline chunk N+1 under
    chunk N. Returns (n, states): tokens queued and the chunk's un-evaluated
    cache-state list. All bookkeeping past the model call matches upstream
    line for line, so pb.generate() sees the exact serial-path state."""
    if not pb.needs_processing():
        return 0, None

    step = pb.prefill_step_size or pb._inputs_embeds.shape[1]
    n = min(step, pb._inputs_embeds.shape[1] - 1)
    checkpoint_col = pb._next_apc_checkpoint_column()
    if checkpoint_col is not None:
        n = min(n, checkpoint_col - pb._processed_prompt_columns)
    if n <= 0:
        return 0, None
    prompt_kwargs = pb._prompt_kwargs_for_step(n)
    pb.model(
        pb._input_ids[:, :n],
        cache=pb.prompt_cache,
        inputs_embeds=pb._inputs_embeds[:, :n],
        n_to_process=n,
        **prompt_kwargs,
    )
    states = [c.state for c in pb.prompt_cache]
    pb._processed_prompt_columns += n
    pb._store_apc_exact_checkpoints()
    pb._inputs_embeds = pb._inputs_embeds[:, n:]
    pb._input_ids = pb._input_ids[:, n:]
    if pb._prompt_length_aware_keys:
        from mlx_vlm.generate import ar as ar_mod

        for k in pb._prompt_length_aware_keys:
            pb._prompt_kwargs[k] = ar_mod._slice_sequence_aligned_prompt_kwarg(
                k, pb._prompt_kwargs[k], start=n
            )
    return n, states


def _prefill_overlap_loop(pb, rank, progress_cb, total):
    """Depth-2 software-pipelined prefill: launch chunk N (async_eval), then
    complete chunk N-1 (eval). The cancel ctrl-word still fires every chunk;
    on cancel the in-flight chunk is drained first so both ranks tear down
    from the same fully-evaluated boundary (cancel latency worsens by <= 1
    chunk — see the design doc's cancel note). Returns (processed, cancelled).
    """
    processed = 0
    pending_states = None
    pending_n = 0

    def _complete_pending():
        nonlocal pending_states, pending_n, processed
        if pending_states is None:
            return
        mx.eval(pending_states)
        mx.clear_cache()  # upstream's per-chunk hygiene, hoisted with the eval
        processed += pending_n
        pending_states, pending_n = None, 0
        if progress_cb is not None:
            try:
                progress_cb(processed, total)
            except Exception:
                pass

    while pb.needs_processing():
        if _ctrl_prefill(rank, _cancel_pending_rank0()):
            _complete_pending()
            return processed, True
        n, states = _prompt_step_lazy(pb)
        if n == 0:
            break
        mx.async_eval(states)  # launch chunk N
        _complete_pending()  # complete chunk N-1
        pending_states, pending_n = states, n
    _complete_pending()
    return processed, False


# ---------------------------------------------------------------------------
# Warm cache <-> batch cache conversion (B=1, left_padding=[0])
# ---------------------------------------------------------------------------

def _pad_sequence_capacity(value, target_capacity):
    if value is None or len(value.shape) < 3:
        return value
    current = int(value.shape[2])
    target = max(current, int(target_capacity or 0))
    if target <= current:
        return value
    padding = [(0, 0)] * len(value.shape)
    padding[2] = (0, target - current)
    return mx.pad(value, padding)


def _rounded_capacity(target_capacity, step):
    target = max(0, int(target_capacity or 0))
    step = max(1, int(step or 1))
    return ((target + step - 1) // step) * step


def _warm_batch_caches(prompt_cache, target_capacity=0):
    from mlx_vlm.models import cache as vcache

    out = []
    for c in prompt_cache:
        if hasattr(c, "to_batch") and not isinstance(c, vcache.KVCache):
            # MiniMaxM3KVCache.to_batch() goes through KVCache.state, which
            # slices backing tensors to the logical offset. A restored 350k
            # cache then becomes exactly full and its first appended token
            # reallocates/copies the entire KV on every layer. B=1 with zero
            # padding can safely share the full backing arrays while carrying
            # the logical offsets separately.
            bc = c.to_batch([0])
            source = getattr(c, "kv_cache", None)
            target = getattr(bc, "kv_cache", None)
            # Keep MLX's native BatchKVCache growth cadence. Propagating the
            # single-cache 4096-token step into the batch cache triggered an
            # M5 IOGPUFamily "completeMemory prepare count underflow" panic at
            # 315k prefill tokens. Capacity reservation below avoids a whole-
            # KV append copy without changing the proven batch allocator.
            batch_step = max(
                1,
                int(getattr(target, "step", 256) or 256),
            )
            if (
                source is not None
                and target is not None
                and getattr(source, "keys", None) is not None
            ):
                desired = _rounded_capacity(
                    max(int(c.offset), int(target_capacity or 0)),
                    batch_step,
                )
                keys = _pad_sequence_capacity(source.keys, desired)
                values = _pad_sequence_capacity(source.values, desired)
                source.keys = keys
                source.values = values
                target.keys = keys
                target.values = values
                target.left_padding = mx.array([0], dtype=mx.int32)
                target.offset = mx.array([int(c.offset)], dtype=mx.int32)
                target._idx = int(c.offset)
                bc._can_skip_decode_mask = True
                index_keys = _pad_sequence_capacity(
                    getattr(c, "index_keys", None), desired
                )
                c.index_keys = index_keys
                bc.index_keys = index_keys
                bc.index_offset = int(getattr(c, "index_offset", 0) or 0)
                arrays = [keys, values]
                if index_keys is not None:
                    arrays.append(index_keys)
                mx.eval(*arrays)
            out.append(bc)
        elif isinstance(c, vcache.KVCache):
            bc = vcache.BatchKVCache([0])
            batch_step = max(1, int(getattr(bc, "step", 256) or 256))
            if c.offset > 0:
                desired = _rounded_capacity(
                    max(int(c.offset), int(target_capacity or 0)),
                    batch_step,
                )
                keys = _pad_sequence_capacity(c.keys, desired)
                values = _pad_sequence_capacity(c.values, desired)
                c.keys = keys
                c.values = values
                bc.keys = keys
                bc.values = values
                bc.left_padding = mx.array([0], dtype=mx.int32)
                bc.offset = mx.array([int(c.offset)], dtype=mx.int32)
                bc._idx = int(c.offset)
                mx.eval(keys, values)
            out.append(bc)
        else:
            raise Unsupported(f"no warm batch conversion for {type(c).__name__}")
    return out


def _restore_single_caches(batch_caches, single_caches):
    """Write grown B=1 batch state back into the server's original cache
    objects, in place, so persistence/trim/reuse machinery is untouched.
    Evaluating the states here is also the symmetric drain point: it forces
    the final step's lazy graph to complete on both ranks before teardown."""
    from mlx_vlm.models import cache as vcache

    states = []
    for bc in batch_caches:
        inner = getattr(bc, "kv_cache", bc)
        if getattr(inner, "keys", None) is not None:
            states.extend([inner.keys, inner.values])
        ik = getattr(bc, "index_keys", None)
        if ik is not None:
            states.append(ik)
    if states:
        mx.eval(states)

    for bc, sc in zip(batch_caches, single_caches):
        if hasattr(sc, "kv_cache"):  # MiniMaxM3KVCache
            inner = bc.kv_cache
            off = int(getattr(inner, "_idx", 0) or 0)
            if off > 0 and inner.keys is not None:
                sc.kv_cache.keys = inner.keys
                sc.kv_cache.values = inner.values
                sc.kv_cache.offset = off
            else:
                sc.kv_cache.keys = None
                sc.kv_cache.values = None
                sc.kv_cache.offset = 0
            sc.index_keys = bc.index_keys
            sc.index_offset = int(getattr(bc, "index_offset", 0) or 0)
        elif isinstance(sc, vcache.KVCache):
            off = int(getattr(bc, "_idx", 0) or 0)
            if off > 0 and bc.keys is not None:
                sc.keys = bc.keys
                sc.values = bc.values
                sc.offset = off
            else:
                sc.keys = None
                sc.values = None
                sc.offset = 0


# ---------------------------------------------------------------------------
# The drop-in generator
# ---------------------------------------------------------------------------

def batch_cancel_stream_generate(rank, model, processor, prompt, **kwargs):
    """Validated construction, then returns the generator. Raises Unsupported
    BEFORE any state is touched so the caller can fall back to
    stream_generate for shapes we don't handle (multimodal, kv-quant)."""
    from mlx_vlm.generate import ar as ar_mod

    if kwargs.get("kv_bits") is not None:
        raise Unsupported("kv-quantized cache")
    for k in ("image", "pixel_values", "draft_model"):
        if kwargs.get(k) is not None:
            raise Unsupported(k)

    tokenizer = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )

    max_tokens = int(kwargs.get("max_tokens") or 256)
    input_ids = kwargs.get("input_ids")
    if input_ids is None:
        from mlx_vlm.utils import prepare_inputs

        inputs = prepare_inputs(processor, prompts=prompt, add_special_tokens=True)
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise Unsupported("tokenization produced no input_ids")

    # Sampler + logits processors: replicate generate_step's construction so
    # sampling semantics (incl. seeded _PositionedTargetSampler) are identical
    # to the stream path.
    temperature = float(kwargs.get("temperature", 0.0) or 0.0)
    top_p = float(kwargs.get("top_p", ar_mod.DEFAULT_TOP_P))
    min_p = float(kwargs.get("min_p", ar_mod.DEFAULT_MIN_P))
    top_k = int(kwargs.get("top_k", ar_mod.DEFAULT_TOP_K))
    seed = kwargs.get("seed")
    sampler = kwargs.get("sampler")
    if sampler is None:
        if (
            seed is not None
            and temperature > 0
            and min_p == ar_mod.DEFAULT_MIN_P
            and top_k == ar_mod.DEFAULT_TOP_K
        ):
            sampler = ar_mod._PositionedTargetSampler(
                temperature=temperature, top_p=top_p, seed=seed
            )
        else:
            sampler = ar_mod.make_sampler(
                temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
            )
    processors = ar_mod.make_logits_processors(
        kwargs.get("logit_bias"),
        kwargs.get("repetition_penalty"),
        kwargs.get("repetition_context_size", ar_mod.DEFAULT_REPETITION_CONTEXT_SIZE),
        kwargs.get("presence_penalty"),
        kwargs.get("presence_context_size", ar_mod.DEFAULT_REPETITION_CONTEXT_SIZE),
        kwargs.get("frequency_penalty"),
        kwargs.get("frequency_context_size", ar_mod.DEFAULT_REPETITION_CONTEXT_SIZE),
    )
    if kwargs.get("logits_processors"):
        processors = list(processors) + list(kwargs["logits_processors"])

    prompt_cache = kwargs.get("prompt_cache")
    prefill_step_size = kwargs.get("prefill_step_size")
    progress_cb = kwargs.get("prefill_progress_callback")

    return _run(
        rank=rank,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        processors=processors or None,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
        progress_cb=progress_cb,
        max_kv_size=kwargs.get("max_kv_size"),
    )


def _run(rank, model, processor, tokenizer, input_ids, max_tokens, sampler,
         processors, prompt_cache, prefill_step_size, progress_cb,
         max_kv_size=None):
    from mlx_vlm.generate import ar as ar_mod
    from mlx_vlm.generate.dispatch import GenerationResult
    from mlx_vlm.tokenizer_utils import make_streaming_detokenizer

    if sampler is None:
        sampler = lambda x: mx.argmax(x, axis=-1)

    suffix_list = input_ids.flatten().tolist()
    reused = 0
    if prompt_cache:
        try:
            reused = int(prompt_cache[0].offset)
        except Exception:
            reused = 0
    total_prompt_tokens = reused + len(suffix_list)

    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    ids_arr = mx.array([suffix_list], dtype=mx.int32)
    inputs_embeds = inner.embed_tokens(ids_arr)

    append_reserve_tokens = min(
        max(0, int(max_tokens or 0)),
        _BATCH_APPEND_RESERVE_TOKENS,
    )
    target_capacity = total_prompt_tokens + append_reserve_tokens
    if max_kv_size is not None and int(max_kv_size or 0) > 0:
        target_capacity = min(target_capacity, int(max_kv_size))
    logger.debug(
        "rank %s: warm batch KV target prompt=%d output_ceiling=%d "
        "append_reserve=%d target=%d",
        rank,
        total_prompt_tokens,
        max_tokens,
        append_reserve_tokens,
        target_capacity,
    )

    if prompt_cache:
        batch_caches = _warm_batch_caches(
            prompt_cache,
            target_capacity=target_capacity,
        )
    else:
        batch_caches = _warm_batch_caches(
            lm.make_cache(),
            target_capacity=target_capacity,
        )

    pb = ar_mod.PromptProcessingBatch(
        model=model,
        uids=[0],
        input_ids=[suffix_list],
        max_tokens=[max_tokens],
        inputs_embeds=inputs_embeds,
        prompt_kwargs={},
        logits_processors=[processors] if processors else None,
        prefill_step_size=prefill_step_size,
        # A COPY: GenerationBatch.filter([]) clears its cache list in place at
        # sequence end; our reference must survive for the write-back below.
        warm_cache=list(batch_caches),
    )

    tic = time.perf_counter()
    processed = 0
    total = len(suffix_list)
    cancelled_in_prefill = False

    overlap = _OVERLAP_NATIVE_ENABLED
    if overlap:
        try:
            _require_lazy_prefill_support(pb)
        except Unsupported as e:
            # Both ranks reach the same verdict (it depends only on the
            # installed mlx-vlm), so this fallback keeps them lockstep.
            logger.warning("batch-cancel: prefill overlap off (%s)", e)
            overlap = False

    if overlap:
        _set_pipeline_overlap_flag(True)
        try:
            processed, cancelled_in_prefill = _prefill_overlap_loop(
                pb, rank, progress_cb, total
            )
        finally:
            _set_pipeline_overlap_flag(False)
    else:
        _chunk_i = 0
        pending_cancel = None  # all_sum launched last chunk, read this chunk
        while pb.needs_processing():
            _chunk_i += 1
            # Act on the PREVIOUS chunk's cancel all_sum. It was launched last
            # iteration and its ring round-trip overlapped last iteration's
            # prompt_step, so reading it now does not stall the pipeline. Both
            # ranks read the same collective result and break together. Cancel
            # latency is one prefill chunk.
            if pending_cancel is not None:
                hit = int(pending_cancel.item()) > 0
                pending_cancel = None
                if hit:
                    cancelled_in_prefill = True
                    break
            # Launch THIS chunk's cancel all_sum WITHOUT blocking, then compute
            # the chunk. Deferring the read (above) is what removes the ~23%
            # per-chunk blocking-collective cost that the batch path used to pay
            # versus stream_generate. The predicate is identical on both ranks
            # (deterministic _chunk_i + forwarded _PREFILL_CANCEL_EVERY), so the
            # all_sum count/order stays matched -- a mismatch would desync/wedge.
            if _PREFILL_CANCEL_EVERY > 0 and (_chunk_i % _PREFILL_CANCEL_EVERY == 0):
                pending_cancel = mx.distributed.all_sum(
                    mx.array(
                        1 if (rank == 0 and _cancel_pending_rank0()) else 0,
                        dtype=mx.int32,
                    )
                )
                mx.async_eval(pending_cancel)
            processed += pb.prompt_step()
            if progress_cb is not None:
                try:
                    progress_cb(processed, total)
                except Exception:
                    pass
        # Catch a cancel that landed on the final launched chunk before we fall
        # through to decode (where the EOS-swap decode-stop takes over).
        if not cancelled_in_prefill and pending_cancel is not None:
            if int(pending_cancel.item()) > 0:
                cancelled_in_prefill = True

    if cancelled_in_prefill:
        del pb
        mx.clear_cache()
        logger.info("rank %s: batch-cancel: prefill abandoned at %d/%d tokens",
                    rank, processed, total)
        # Carry the EOS id so the consumer's stopped-bookkeeping fires and the
        # cache registry is reset — the originals only hold the old prefix,
        # not the abandoned prefill (generation_tokens=0 keeps it out of the
        # generated-ids list).
        srv = _server()
        fe = getattr(srv, "_FORCE_EOS", None) if srv else None
        eos_id = fe.get("eos_id") if fe else None
        yield GenerationResult(
            text="",
            token=int(eos_id) if eos_id is not None else None,
            logprobs=None,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=0,
            total_tokens=total_prompt_tokens,
            prompt_tps=0.0,
            generation_tps=0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            cached_tokens=reused,
            finish_reason="stop",
        )
        return

    gb = pb.generate(
        sampler, tokenizer.stopping_criteria,
        compute_logprobs=True, top_logprobs_k=0,
    )
    prompt_time = time.perf_counter() - tic
    prompt_tps = total_prompt_tokens / prompt_time if prompt_time > 0 else 0.0

    detok = make_streaming_detokenizer(processor)
    tic = time.perf_counter()
    n = 0
    closing = False
    finish_reason = None
    last_token = None

    while len(gb):
        r = gb.next()[0]
        n += 1
        last_token = r.token
        if r.finish_reason == "stop":
            # EOS (natural or swap-injected): never enters the text, but the
            # final flush must carry the detokenizer tail (tail-loss rule).
            finish_reason = "stop"
            break
        detok.add_token(r.token)
        if r.finish_reason == "length":
            finish_reason = "length"
            break
        if closing:
            continue
        try:
            yield GenerationResult(
                text=detok.last_segment,
                token=r.token,
                logprobs=None,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=n,
                total_tokens=total_prompt_tokens + n,
                prompt_tps=prompt_tps,
                generation_tps=n / max(time.perf_counter() - tic, 1e-9),
                peak_memory=mx.get_peak_memory() / 1e9,
                cached_tokens=reused,
            )
        except GeneratorExit:
            # Consumer closed us mid-stream (client break / disconnect).
            # Do NOT abandon the lockstep loop: arm the EOS swap and drain
            # silently to the shared step boundary, then return.
            closing = True
            if rank == 0:
                _arm_force_eos()
            logger.info("rank %s: batch-cancel: consumer closed at token %d; "
                        "draining to lockstep boundary", rank, n)

    if prompt_cache:
        _restore_single_caches(batch_caches, prompt_cache)
    else:
        states = []
        for bc in batch_caches:
            inner_c = getattr(bc, "kv_cache", bc)
            if getattr(inner_c, "keys", None) is not None:
                states.extend([inner_c.keys, inner_c.values])
        if states:
            mx.eval(states)

    if closing:
        logger.info("rank %s: batch-cancel: drained clean after %d tokens "
                    "(finish=%s)", rank, n, finish_reason)
        return

    detok.finalize()
    yield GenerationResult(
        text=detok.last_segment,
        token=last_token,
        logprobs=None,
        prompt_tokens=total_prompt_tokens,
        generation_tokens=n,
        total_tokens=total_prompt_tokens + n,
        prompt_tps=prompt_tps,
        generation_tps=n / max(time.perf_counter() - tic, 1e-9),
        peak_memory=mx.get_peak_memory() / 1e9,
        cached_tokens=reused,
        finish_reason=finish_reason or "length",
    )
