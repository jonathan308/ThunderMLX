"""EAGLE3 speculative decode for the 2-rank MiniMax-M3 pipeline (ThunderMLX).

Lockstep port of mlx_vlm.speculative.eagle3._eagle3_rounds for the 38,22
pipeline split. The upstream loop assumes a single process that both drafts
and verifies; here rank 0 owns the drafter (and logits) while rank 1 must run
the identical verify forwards in lockstep. Design (docs/DESIGN-eagle3-
speculative.md):

- rank 0 drafts a block locally, then BROADCASTS (draft_tokens) to rank 1
  through one tiny all_sum — rank 1 needs the token ids to embed them for
  its share of the verify forward.
- both ranks run the SAME verify forward (a K+1-token mini-prefill through
  the pipeline patch, which already supports L>1 and rank-local capture).
- rank 0 walks acceptance against its logits and BROADCASTS
  (accepted, bonus) in a second tiny all_sum; both ranks roll their caches
  back identically (rollback_speculative_cache is rank-local math).
- layer-2 capture lives on rank 1 (owns layers 0-21): the pipeline patch
  piggybacks captured hiddens onto the boundary activation it already sends
  (see m3_pipeline_patch NOTE), so the collective count per forward is
  UNCHANGED from the proven decode path.

Sampling is greedy on the speculative path (temp 0 semantics) — matching
the tool-mode sampling the agent endpoints already pin (temperature 0.2 is
near-greedy; the verify walk keeps outputs exactly equal to the target's
greedy stream). Non-greedy requests fall back to the normal batch path.

Env:
  MLX_M3_EAGLE3_DRAFT       path to the (adapted) drafter dir; empty = off
  MLX_M3_EAGLE3_BLOCK       draft block size override (default: drafter's,
                            clamped to 8; README reference uses 3-4)
  MLX_M3_EAGLE3_NORM_RESIDUAL  norm_before_residual flag (default 0; only
                            affects acceptance rate, never correctness)
"""

import os
import sys
import logging
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("m3_eagle3")

_DRAFT_PATH = os.environ.get("MLX_M3_EAGLE3_DRAFT", "").strip()
_BLOCK_OVERRIDE = int(os.environ.get("MLX_M3_EAGLE3_BLOCK", "0") or "0")
_NORM_RESIDUAL = os.environ.get(
    "MLX_M3_EAGLE3_NORM_RESIDUAL", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Runtime toggle (dashboard button): rank 0 reads this per request and puts
# the decision INTO the broadcast request op, so both ranks always pick the
# same generator. Never read directly on rank 1.
RUNTIME_ENABLED = {"value": True}

# Per-request flag, set on BOTH ranks from the broadcast request op before
# run_generation (rank 0 at op build, rank 1 in the mirror loop). Single
# generation slot => race-free module state, same pattern as _STOP_NONCE.
REQUEST_ACTIVE = {"value": False}

# v1 samples greedily on the speculative path; only greedy-resolved requests
# ride it unless forced (testing / near-greedy tool traffic).
_FORCE = os.environ.get(
    "MLX_M3_EAGLE3_FORCE", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# DIAGNOSTIC A/B (deadlock hunt): override the capture layers. Setting all
# rank0-owned layers (e.g. "30,57,58" on the 38,22 split) makes the pipeline
# piggyback INERT — drafter quality becomes garbage (fc gets wrong layers;
# harmless, verification rejects) but the pipeline verdict is binary:
# hang gone => piggyback send/recv is the bug; hang stays => rank0-side.
_CAPTURE_OVERRIDE = [
    int(x) for x in os.environ.get(
        "MLX_M3_EAGLE3_CAPTURE_LAYERS", ""
    ).replace(",", " ").split()
]

# Offline-acceptance capture dir (rank 0): dumps per-round verify hiddens +
# draft/target tokens so drafter experiments run offline. Empty = off.
_DUMP_DIR = os.environ.get("MLX_M3_EAGLE3_DUMP_DIR", "").strip()
if _DUMP_DIR:
    try:
        os.makedirs(_DUMP_DIR, exist_ok=True)
    except Exception:
        _DUMP_DIR = ""

_DRAFTER = None          # rank-0 only
_TARGET_LAYER_IDS = [2, 30, 57]   # Inferact/MiniMax-M3-EAGLE3 (README)


class Unsupported(Exception):
    """Request shape the eagle3 path does not handle; use the batch path."""


def _server():
    return sys.modules.get("sharded_server")


def enabled():
    return bool(_DRAFT_PATH)


def armed(rank: int) -> bool:
    return bool(_DRAFT_PATH) and (_DRAFTER is not None or rank != 0)


# --------------------------------------------------------------------------
# TorchSpec checkpoint adapter
# --------------------------------------------------------------------------
# Inferact/MiniMax-M3-EAGLE3 is a TorchSpec/HF export: model_type "llama",
# THREE per-capture fc_norms (fc_norm.{0,1,2}, each RMSNorm(6144)) instead of
# mlx's single concat-RMSNorm(18432), and no speculators wrapper. The tensor
# names otherwise match Eagle3DraftModel's module tree exactly, so the
# "conversion to mlx" is: build the Eagle3Config in code, subclass the model
# with the per-segment norms, and load the safetensors verbatim (bf16).

def _build_drafter_class():
    from mlx_vlm.speculative.drafters.eagle3.eagle3 import Eagle3DraftModel
    from mlx_vlm.speculative.drafters.eagle3.config import (
        Eagle3Config,
        TextConfig,
    )

    class TorchSpecEagle3(Eagle3DraftModel):
        """Eagle3DraftModel with TorchSpec's per-capture fc norms."""

        def __init__(self, config: Eagle3Config):
            super().__init__(config)
            text = config.transformer_layer_config
            # Three RMSNorms named fc_norm.{0,1,2} so load_weights maps the
            # checkpoint names directly.
            self.fc_norm = [
                nn.RMSNorm(self.target_hidden_size, eps=text.rms_norm_eps)
                for _ in range(3)
            ]

        def _prepare_target_hidden(self, hidden: mx.array) -> mx.array:
            if hidden.shape[-1] == self.hidden_size:
                return hidden
            th = self.target_hidden_size
            parts = [
                self.fc_norm[i](hidden[..., i * th : (i + 1) * th])
                for i in range(3)
            ]
            return self.fc(mx.concatenate(parts, axis=-1))

        def bind(self, target_model):
            # Upstream bind() swaps in the target's embedding when shapes
            # match. On the pipeline, rank 0 may not materialize the target
            # embedding (rank 1 owns the early layers), so ALWAYS keep the
            # drafter's own shipped embedding.
            return self

    return TorchSpecEagle3, Eagle3Config, TextConfig


def load_drafter_rank0():
    """Load + adapt the TorchSpec checkpoint on rank 0. Idempotent."""
    global _DRAFTER, _TARGET_LAYER_IDS
    if _DRAFTER is not None or not _DRAFT_PATH:
        return _DRAFTER
    import json
    import glob

    TorchSpecEagle3, Eagle3Config, TextConfig = _build_drafter_class()
    with open(os.path.join(_DRAFT_PATH, "config.json")) as f:
        hf = json.load(f)

    text = TextConfig(
        model_type="llama",
        hidden_size=int(hf["hidden_size"]),
        intermediate_size=int(hf["intermediate_size"]),
        num_hidden_layers=int(hf.get("num_hidden_layers", 1)),
        num_attention_heads=int(hf["num_attention_heads"]),
        num_key_value_heads=int(hf.get("num_key_value_heads",
                                       hf["num_attention_heads"])),
        head_dim=int(hf.get("head_dim") or 0) or None,
        rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
        vocab_size=int(hf["vocab_size"]),
        max_position_embeddings=int(hf.get("max_position_embeddings", 1048576)),
        rope_theta=float(hf.get("rope_theta", 5000000)),
        attention_bias=bool(hf.get("attention_bias", False)),
        hidden_act=str(hf.get("hidden_act", "silu")),
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
    )
    cfg = Eagle3Config(
        model_type="eagle3",
        transformer_layer_config=text,
        draft_vocab_size=int(hf.get("draft_vocab_size", hf["vocab_size"])),
        target_hidden_size=int(hf["hidden_size"]),
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
        norm_before_residual=_NORM_RESIDUAL,
        norm_before_fc=False,  # TorchSpec per-capture norms live in the subclass
        eagle_aux_hidden_state_layer_ids=list(_TARGET_LAYER_IDS),
        block_size=_BLOCK_OVERRIDE or 4,
    )
    model = TorchSpecEagle3(cfg)
    files = sorted(glob.glob(os.path.join(_DRAFT_PATH, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {_DRAFT_PATH}")
    weights = {}
    for fpath in files:
        weights.update(mx.load(fpath))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    _TARGET_LAYER_IDS = list(cfg.target_layer_ids or _TARGET_LAYER_IDS)
    _DRAFTER = model
    n_params = sum(v.size for _, v in weights.items())
    logger.info(
        "eagle3 drafter loaded: %s (%.2fB params, block=%d, targets=%s, "
        "norm_residual=%s)",
        _DRAFT_PATH, n_params / 1e9, cfg.block_size, _TARGET_LAYER_IDS,
        _NORM_RESIDUAL,
    )
    return _DRAFTER


# --------------------------------------------------------------------------
# Lockstep collectives — tiny fixed-shape all_sums, mirroring the sampled-
# token sync discipline: rank 0 contributes the payload, rank 1 zeros, both
# read the same sum. Fixed shapes keep the collective schedule identical on
# both ranks no matter the round outcome.
# --------------------------------------------------------------------------

# Sentinel: any rank injects this into ANY loop collective; token ids and
# counts are always >= 0, so a summed slot going strongly negative is an
# unambiguous "peer exited" signal regardless of what the peer contributed.
_SENTINEL = -(10**6)


def _bcast_ints(rank: int, values: Optional[List[int]], width: int) -> List[int]:
    """One fixed-width all_sum. Contribution model: pass a list to contribute,
    None to contribute zeros — EITHER rank may contribute (all_sum is
    symmetric), which is what lets the exit sentinel work from both sides."""
    if values is not None:
        assert len(values) == width
        payload = mx.array(values, dtype=mx.int32)
    else:
        payload = mx.zeros((width,), dtype=mx.int32)
    out = mx.distributed.all_sum(payload)
    mx.eval(out)
    return [int(v) for v in out.tolist()]


# --------------------------------------------------------------------------
# The lockstep rounds generator
# --------------------------------------------------------------------------

def eagle3_stream_generate(
    rank: int,
    model,
    processor,
    prompt,
    *,
    prompt_cache=None,
    input_ids=None,
    max_tokens: int,
    prefill_step_size: int = 4096,
    prefill_progress_callback=None,
    **kwargs,
):
    """Speculative lockstep generation. Yields objects shaped like upstream
    GenerationResult (token, text, generation_tokens, prompt_tokens...) so
    the existing consume loops work unchanged.

    Request-shape gate (checked identically on both ranks from the broadcast
    request): text-only, B=1. Anything else raises Unsupported BEFORE any
    collective, so both ranks fall back together.
    """
    # EAGER GATES — this is a plain function that validates and returns the
    # inner generator. A generator body does not run until first next(), which
    # happens outside the caller's try/except; raising Unsupported lazily
    # turns a clean fallback into a request error (window take-1 finding).
    if kwargs.get("image") is not None or kwargs.get("pixel_values") is not None:
        raise Unsupported("eagle3 path is text-only")
    if kwargs.get("kv_bits"):
        raise Unsupported("kv-quant not exercised on the eagle3 path")
    if prompt_cache is None:
        raise Unsupported("no prompt cache for this request")
    temp = kwargs.get("temperature", 0.0) or 0.0
    if temp > 0 and not _FORCE:
        # Gate is a pure function of the broadcast request + shared env, so
        # both ranks fall through together.
        raise Unsupported(f"non-greedy sampling (temperature={temp})")

    from mlx_vlm.generate.common import GenerationResult  # upstream result shape
    from mlx_vlm.speculative.eagle3 import _eagle3_walk

    return _eagle3_rounds_lockstep(
        rank, model, processor, prompt,
        prompt_cache=prompt_cache,
        input_ids=input_ids,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        prefill_progress_callback=prefill_progress_callback,
        GenerationResult=GenerationResult,
        _eagle3_walk=_eagle3_walk,
        **kwargs,
    )


def _eagle3_rounds_lockstep(
    rank: int,
    model,
    processor,
    prompt,
    *,
    prompt_cache,
    input_ids,
    max_tokens: int,
    prefill_step_size: int,
    prefill_progress_callback,
    GenerationResult,
    _eagle3_walk,
    **kwargs,
):
    srv = _server()

    lm = model.language_model if hasattr(model, "language_model") else model
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos_ids = set()
    for attr in ("eos_token_id", "eos_token_ids"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            eos_ids.add(v)
        elif isinstance(v, (list, tuple, set)):
            eos_ids.update(int(x) for x in v)

    drafter = load_drafter_rank0() if rank == 0 else None
    block = _BLOCK_OVERRIDE or (int(drafter.config.block_size) if drafter else 4)
    block = max(2, min(8, block))

    # ---- tokenize / prefill (both ranks, standard pipeline prefill with
    # capture enabled so the drafter can be seeded from the prompt hiddens).
    if input_ids is not None:
        ids = input_ids
    else:
        ids = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)
    n_prompt = int(ids.shape[1])

    capture_ids = list(_CAPTURE_OVERRIDE or _TARGET_LAYER_IDS)
    hidden_chunks = []
    logits = None
    processed = 0
    while processed < n_prompt:
        chunk = ids[:, processed : processed + prefill_step_size]
        out = lm(
            chunk,
            cache=prompt_cache,
            capture_layer_ids=capture_ids,
        )
        logits = out.logits
        if getattr(out, "hidden_states", None):
            hidden_chunks.append(mx.concatenate(out.hidden_states, axis=-1))
        mx.eval(logits)
        processed += int(chunk.shape[1])
        if prefill_progress_callback is not None:
            try:
                prefill_progress_callback(min(processed, n_prompt), n_prompt)
            except Exception:
                pass

    # first target token: rank 0 samples greedily, broadcasts.
    if rank == 0:
        first = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        [first] = _bcast_ints(rank, [first], 1)
    else:
        [first] = _bcast_ints(rank, None, 1)


    sampler = (lambda lg: mx.argmax(lg, axis=-1))
    greedy_kwargs = {"greedy": True} if rank == 0 else {}

    def _result(tok: int, n: int, text_piece: str) -> Any:
        return GenerationResult(
            text=text_piece,
            token=tok,
            logprobs=None,
            prompt_tokens=n_prompt,
            generation_tokens=n,
            prompt_tps=0.0,
            generation_tps=0.0,
            peak_memory=0.0,
        )

    # Incremental detokenization (window take-2 finding: mlx-vlm's HF
    # tokenizer has no .detokenizer, so streaming pieces were empty and the
    # no-visible guard killed the turn at 32 silent tokens). Naive full
    # re-decode + string diff is correct for any tokenizer; O(n^2) decode
    # cost is acceptable at agent turn sizes (streaming detok = follow-up).
    _out_tokens: List[int] = []
    _prev_text = [""]

    def _detok_add(tok: int) -> str:
        _out_tokens.append(int(tok))
        try:
            text = tokenizer.decode(_out_tokens, skip_special_tokens=False)
        except TypeError:
            text = tokenizer.decode(_out_tokens)
        piece = text[len(_prev_text[0]):]
        _prev_text[0] = text
        return piece

    # Exit protocol (v3, after the 2026-07-09 double-machine wedge):
    #   - EVERY loop collective uses ONE constant width W, so any interleave
    #     of round-start / accept / sentinel pairs at the wire level.
    #   - EITHER rank may inject the sentinel (strongly negative slot 0 —
    #     token ids and counts are >= 0, so the SUM stays unambiguous no
    #     matter what the peer contributed to the same collective).
    #   - EVERY exit path — normal return, GeneratorExit (consume-loop
    #     break/stop/disconnect), or exception — flows through one finally
    #     that sends the sentinel exactly once, UNLESS this rank already
    #     consumed the peer's sentinel (that consumption balanced the
    #     books). The v2 bug: a rank returning normally while its peer
    #     waited at round-start left an unpaired collective armed for the
    #     NEXT request => both machines wedged.
    _n_draft_width = max(1, min(block, max_tokens) - 1) if max_tokens > 1 else 1
    _W = _n_draft_width + 4
    _exit = {"balanced": False}

    emitted = 0
    b = first

    try:
        # Drafter prefill lives INSIDE the try so a rank-0-only exception
        # here still flows through the sentinel finally (rank 1 would
        # otherwise march to its round-start and wait forever).
        prompt_hidden = (
            mx.concatenate(hidden_chunks, axis=1) if hidden_chunks else None
        )
        if rank == 0 and drafter is not None:
            drafter.reset(model)
            prefill_draft = getattr(drafter, "prefill_from_target_hidden", None)
            if callable(prefill_draft) and prompt_hidden is not None:
                prefill_draft(ids, prompt_hidden, first,
                              sampler, mx.int32, greedy=True)
        # Offline-harness + fine-tune capture: the PROMPT-phase hiddens are
        # what the 2026-07-09 offline sweep was missing (replayed drafter had
        # no KV/context => 0.07 accept vs 1.07 live; flags unjudged). One
        # fp16 file per request; doubles as calibration-training input.
        if rank == 0 and _DUMP_DIR and prompt_hidden is not None:
            try:
                import numpy as _np
                import time as _tt
                _np.savez(
                    os.path.join(_DUMP_DIR, f"prompt_{int(_tt.time())}.npz"),
                    prompt_hidden=_np.array(
                        prompt_hidden.astype(mx.float16), copy=False
                    ),
                    prompt_ids=_np.array(
                        ids.reshape(-1).astype(mx.int32), copy=False
                    ),
                    first_token=_np.int32(first),
                )
                logger.info(
                    "eagle3 capture: prompt hiddens dumped (%d tokens)",
                    int(ids.shape[1]),
                )
            except Exception as e:
                logger.warning("eagle3 prompt dump failed: %s", e)

        hidden = prompt_hidden[:, -1:, :] if prompt_hidden is not None else None
        del prompt_hidden, hidden_chunks

        yield _result(b, emitted + 1, _detok_add(b))
        emitted += 1
        if b in eos_ids:
            return
        while emitted < max_tokens:
            # Constant round geometry: same draft width every round, so the
            # collective schedule never depends on emitted-count (any
            # divergence there is a wedge). The walk budget truncates the
            # final round's emission instead.
            bs = _n_draft_width + 1
            n_draft = _n_draft_width

            import time as _t
            _t0 = _t.perf_counter()
            # rank 0 drafts; the block rides one fixed-width broadcast.
            if rank == 0:
                draft_tokens = drafter.draft_block(
                    b, hidden, drafter._cache, bs, sampler, mx.int32,
                    **greedy_kwargs,
                )
                row = [int(v) for v in draft_tokens.reshape(-1).tolist()[:n_draft]]
                row = row[:n_draft] + [0] * (_W - len(row))
                _t1 = _t.perf_counter()
                vals = _bcast_ints(rank, row, _W)
            else:
                _t1 = _t.perf_counter()
                vals = _bcast_ints(rank, None, _W)
            _t2 = _t.perf_counter()
            if vals[0] < 0:
                # peer exited; its sentinel balanced this collective.
                _exit["balanced"] = True
                return
            row = vals[:n_draft]
            draft_tokens = mx.array([row], dtype=mx.int32)

            # verify forward: identical fixed-width mini-prefill on BOTH ranks
            # (constant K+1 keeps the collective schedule round-invariant;
            # near max_tokens the walk budget just truncates the emission).
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=mx.int32), draft_tokens], axis=1
            )
            out = lm(verify_input, cache=prompt_cache,
                     capture_layer_ids=capture_ids)
            verify_hidden = (
                mx.concatenate(out.hidden_states, axis=-1)
                if getattr(out, "hidden_states", None)
                else None
            )

            # rank 0 walks acceptance; (accepted, n_new, tokens...) broadcast.
            if rank == 0:
                target_tokens = sampler(out.logits)
                accepted, new_tokens = _eagle3_walk(
                    draft_tokens, target_tokens, max_tokens - emitted
                )
                payload = [accepted, len(new_tokens)] + list(new_tokens)
                payload += [0] * (_W - len(payload))
                vals = _bcast_ints(rank, payload, _W)
            else:
                mx.eval(out.logits)
                vals = _bcast_ints(rank, None, _W)
            if vals[0] < 0:
                # peer exited between our collectives (exception path); the
                # sentinel balanced this one. Roll the whole verify block off
                # the cache (nothing was accepted) and exit.
                lm_roll = getattr(lm, "rollback_speculative_cache", None)
                if lm_roll is not None:
                    lm_roll(prompt_cache, getattr(out, "gdn_states", None),
                            -1, n_draft + 1)
                _exit["balanced"] = True
                return
            accepted, n_new = vals[0], vals[1]
            new_tokens = vals[2 : 2 + n_new]
            _t3 = _t.perf_counter()

            # both ranks roll rejected draft tokens off their caches (local).
            if accepted < n_draft:
                lm_roll = getattr(lm, "rollback_speculative_cache", None)
                if lm_roll is None:
                    raise RuntimeError("target lacks rollback_speculative_cache")
                lm_roll(
                    prompt_cache,
                    getattr(out, "gdn_states", None),
                    accepted,
                    n_draft + 1,
                )
            _t4 = _t.perf_counter()

            if rank == 0 and drafter is not None:
                accept_verified = getattr(drafter, "accept_verified_tokens", None)
                if callable(accept_verified) and verify_hidden is not None:
                    accept_verified(
                        verify_hidden, draft_tokens, accepted, new_tokens,
                        sampler, mx.int32, **greedy_kwargs,
                    )
                drafter.accept_lens.append(accepted)
                drafter.draft_lens.append(n_draft)
                # Offline-acceptance capture: dump (verify_hidden, drafts,
                # targets) for the first rounds so drafter flags/orders can
                # be iterated offline without cluster restarts.
                if _DUMP_DIR and len(drafter.draft_lens) <= 40:
                    try:
                        import numpy as _np
                        i = len(drafter.draft_lens)
                        _np.savez(
                            os.path.join(_DUMP_DIR, f"round_{i:03d}.npz"),
                            verify_hidden=_np.array(
                                verify_hidden.astype(mx.float32), copy=False
                            ),
                            draft_tokens=_np.array(row, dtype=_np.int32),
                            target_tokens=_np.array(
                                target_tokens.reshape(-1).astype(mx.int32),
                                copy=False,
                            ),
                            bonus_in=_np.int32(b),
                            accepted=_np.int32(accepted),
                        )
                    except Exception as e:
                        logger.warning("eagle3 dump failed: %s", e)
            _t5 = _t.perf_counter()
            logger.info(
                "rank %s: eagle3 round acc=%d ms draft=%.0f bcast=%.0f "
                "verify+walk=%.0f rollback=%.0f drafter_upd=%.0f",
                rank, accepted, (_t1 - _t0) * 1e3, (_t2 - _t1) * 1e3,
                (_t3 - _t2) * 1e3, (_t4 - _t3) * 1e3, (_t5 - _t4) * 1e3,
            )

            if verify_hidden is not None:
                hidden = verify_hidden[:, accepted : accepted + 1, :]
            b = new_tokens[-1] if new_tokens else b

            stop = False
            for tok in new_tokens:
                emitted += 1
                piece = _detok_add(tok)
                yield _result(tok, emitted, piece)
                if tok in eos_ids or emitted >= max_tokens:
                    stop = True
                    break
            if stop:
                return

            if emitted % 256 == 0:
                mx.clear_cache()
    finally:
        # Unconditional exit balancing: unless this rank already consumed the
        # peer's sentinel (which balanced the peer's exit), send exactly one
        # sentinel. Case table:
        #   both exit normally at the same point -> two sentinels pair;
        #   one closed/errored while peer loops   -> sentinel pairs with the
        #     peer's next W-wide collective, peer sees slot0<0, exits
        #     balanced (no re-send);
        #   both closed at the same yield         -> two sentinels pair.
        # All loop collectives share width W, so ANY pairing is wire-legal.
        if not _exit["balanced"]:
            try:
                _bcast_ints(rank, [_SENTINEL] * _W, _W)
                logger.info("rank %s: eagle3 exit sentinel sent", rank)
            except Exception as e:
                logger.warning(
                    "rank %s: eagle3 exit sentinel failed: %s", rank, e
                )


def acceptance_stats():
    """For the dashboard tile / admin endpoint (rank 0)."""
    d = _DRAFTER
    if d is None or not getattr(d, "draft_lens", None):
        return None
    drafted = sum(d.draft_lens)
    accepted = sum(d.accept_lens)
    return {
        "rounds": len(d.draft_lens),
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": round(accepted / drafted, 4) if drafted else 0.0,
        "mean_accept_len": round(
            (accepted + len(d.accept_lens)) / len(d.accept_lens), 3
        ),
    }
