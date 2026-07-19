#!/usr/bin/env python3
"""
m3_pipeline_patch.py — adds pipeline parallelism to MiniMax-M3 at runtime.

WHY
  mlx_vlm's M3 implements .shard() (tensor parallel) but NOT .pipeline().
  Tensor parallel loads the FULL model on every rank -> memory thrash/OOM
  on asymmetric machines. Pipeline parallel splits LAYERS across ranks and
  each rank reads only its own layers' weights from disk. For a 241GB model
  on 256GB+128GB machines, pipeline is the only viable strategy.

HOW
  Modeled exactly on mlx_lm's glm4_moe (the proven working example):
  - Mix PipelineMixin into MiniMaxM3Model -> adds .pipeline() + pipeline_layers
  - Patch MiniMaxM3Model.__call__ to do recv -> run own layers -> send -> all_gather
  - Mark unowned layers as None so their weights are never loaded from disk.

  rank = pipeline_size - 1 gets the FIRST layers (embeddings)
  rank = 0 gets the LAST layers (lm_head/norm) + serves the API
  (matches glm4_moe's reverse split so rank 0 produces final hidden states.)

Run this under mlx.launch so each rank gets a distributed group.
"""
from __future__ import annotations
import os
import sys

import mlx.core as mx


_PIPELINE_GROUP = None

# The eagle piggyback prints one [e3dbg] send/recv line per forward. That was a
# deadlock-hunt diagnostic; under capture-only it fires every token, so gate it
# behind an env (default off). Eagle rounds are rare enough that losing the
# line by default is fine; set MLX_M3_EAGLE3_PIPE_DEBUG=1 to restore it.
_E3_PIPE_DEBUG = os.environ.get(
    "MLX_M3_EAGLE3_PIPE_DEBUG", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Fast gate read once at import: when unset (default), the capture-only hook in
# _patched_call is skipped entirely — no m3_capture import, no per-forward work,
# byte-identical to before. Both ranks read the same whitelisted env.
_CAPTURE_ONLY_ENV = os.environ.get(
    "MLX_M3_EAGLE3_CAPTURE_ONLY", "0"
).strip().lower() in {"1", "true", "yes", "on"}


_CAPTURE_MOD = {"mod": None, "tried": False}


def _capture_module():
    """Import-and-cache m3_capture. MUST resolve on BOTH ranks so the capture
    piggyback rides symmetrically (rank1 ships even though only rank0 keeps the
    data); rank1 has no server wrapper to import it otherwise. Import is cheap
    and dependency-free (mlx + stdlib), so failure just means capture-only off."""
    if not _CAPTURE_MOD["tried"]:
        _CAPTURE_MOD["tried"] = True
        try:
            import m3_capture as _m
            _CAPTURE_MOD["mod"] = _m
        except Exception:
            _CAPTURE_MOD["mod"] = None
    return _CAPTURE_MOD["mod"]

# Chunked-prefill overlap (docs/DESIGN-chunk-overlap-native.md): while
# m3_batch_cancel's depth-2 prefill pipeline drives the model, rank1's
# per-chunk eager mx.eval(h) in _patched_call must not fire — it would
# serialize the very chunks the loop is overlapping. Decode NEVER sets this
# flag: the eager send-eval is load-bearing there (it forces the forward and
# posts the pipeline send each step).
_M3_PREFILL_OVERLAP_ACTIVE = False


def set_prefill_overlap_active(active):
    """Called by m3_batch_cancel around its overlap prefill loop."""
    global _M3_PREFILL_OVERLAP_ACTIVE
    _M3_PREFILL_OVERLAP_ACTIVE = bool(active)


def _unpadded_single_stream(h, cache) -> bool:
    """True when this forward is a single (B==1) sequence with no left padding
    -- the cluster's only generation mode. In that case a batch cache's causal
    make_mask ARRAY is identical in effect to the "causal" string, so it can be
    swapped back to keep the MSA sparse-prefill gate eligible (the gate only
    accepts None/str; a dense causal array forces the O(n^2) fallback and was
    the cause of the ~6x long-context prefill regression). Real padded batches
    (B>1 or nonzero left padding) return False and keep their explicit array."""
    if h.shape[0] != 1:
        return False
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is None:
        return True
    try:
        if left_padding.size == 0:
            return True
        return int(mx.max(mx.abs(left_padding)).item()) == 0
    except Exception:
        return False


class _PipelineMixin:
    """Layer-splitting for pipeline parallelism (from mlx_lm/models/pipeline.py)."""
    pipeline_rank = 0
    pipeline_size = 1
    start_idx = 0
    end_idx = None

    def pipeline(self, group):
        import os as _os
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        n = len(self.layers)

        # Allow an explicit layer split via env, e.g. M3_PIPELINE_LAYERS="42,18"
        # means rank 0 (last) gets 42 layers, rank 1 (first) gets 18.
        # Listed rank0-first-from-the-end. Used for asymmetric-RAM machines.
        spec = _os.environ.get("M3_PIPELINE_LAYERS")
        if spec:
            counts = [int(x) for x in spec.split(",")]
            assert len(counts) == self.pipeline_size, "M3_PIPELINE_LAYERS must list size entries"
            assert sum(counts) == n, f"M3_PIPELINE_LAYERS must sum to {n}"
            # Reverse split: rank=0 gets last `counts[0]`, rank=size-1 gets first
            # counts[size-1]. Build start/end from the right.
            end_from_left = n
            ranges = []
            for r in range(self.pipeline_size):
                c = counts[r]  # number of layers rank r owns (from the end)
                ranges.append((end_from_left - c, end_from_left))
                end_from_left -= c
            self.start_idx, self.end_idx = ranges[self.pipeline_rank]
        else:
            layers_per_rank = n // self.pipeline_size
            extra = n - layers_per_rank * self.pipeline_size
            if self.pipeline_rank < extra:
                layers_per_rank += 1
            # Reverse split: rank=size-1 gets first layers, rank=0 gets last
            self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
            self.end_idx = self.start_idx + layers_per_rank

        # Drop layers after our slice; None-out layers before (keep numbering for load)
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx

    @property
    def pipeline_layers(self):
        return [l for l in self.layers if l is not None]


def apply_pipeline_patch():
    """Monkey-patch MiniMaxM3Model to support pipeline parallelism."""
    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3Model
    from mlx_vlm.models.minimax_m3_vl.language import LanguageModel
    from mlx_vlm.models.minimax_m3_vl.language import create_attention_mask
    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache, KVCache

    cache_step = int(os.environ.get("MLX_M3_KV_CACHE_STEP", "4096"))
    layer_eval_every = int(os.environ.get("MLX_M3_LAYER_EVAL_EVERY", "0") or "0")
    disable_sparse_index = os.environ.get(
        "MLX_M3_DISABLE_SPARSE_INDEX", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    rank0_decode_owner = os.environ.get(
        "MLX_M3_RANK0_DECODE_OWNER", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    KVCache.step = cache_step
    MiniMaxM3KVCache.step = cache_step

    # Save the original __call__ so non-pipeline (size==1) path is unchanged.
    _orig_call = MiniMaxM3Model.__call__

    # Inject mixin attributes if missing
    for attr in ("pipeline_rank", "pipeline_size", "start_idx", "end_idx"):
        if not hasattr(MiniMaxM3Model, attr):
            setattr(MiniMaxM3Model, attr, getattr(_PipelineMixin, attr))
    MiniMaxM3Model.pipeline = _PipelineMixin.pipeline
    MiniMaxM3Model.pipeline_layers = property(_PipelineMixin.pipeline_layers.fget)

    def _depend_cache_on_hidden(cache_obj, h):
        """Tie final cache writes to the send op so decode graphs drain.

        GLM/DeepSeek pipeline examples do this with cache[-1].keys. MiniMax-M3
        wraps the normal KV cache inside MiniMaxM3KVCache.kv_cache and also
        mutates a sparse-attention index cache on decode, so both final key
        stores must depend on the pipeline send.
        """
        if cache_obj is None:
            return
        if hasattr(cache_obj, "keys") and cache_obj.keys is not None:
            cache_obj.keys = mx.depends(cache_obj.keys, h)
        kv_cache = getattr(cache_obj, "kv_cache", None)
        if kv_cache is not None and getattr(kv_cache, "keys", None) is not None:
            kv_cache.keys = mx.depends(kv_cache.keys, h)
        index_keys = getattr(cache_obj, "index_keys", None)
        if index_keys is not None:
            cache_obj.index_keys = mx.depends(index_keys, h)

    def _patched_call(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        capture_layer_ids=None,
        hidden_sink=None,
        position_ids=None,
    ):
        # Single-rank / no pipeline: behave exactly like the original.
        if self.pipeline_size <= 1:
            return _orig_call(
                self, inputs, inputs_embeds, mask, cache,
                capture_layer_ids, hidden_sink, position_ids,
            )

        # --- Pipeline parallel forward (mirrors glm4_moe exactly) ---
        is_first = self.pipeline_rank == self.pipeline_size - 1  # has embeddings
        is_last = self.pipeline_rank == 0  # has norm + feeds lm_head

        # CRITICAL: every rank embeds from token IDs (like glm4_moe line 272),
        # so recv_like() gets the CORRECT shape (batch, seq, hidden).
        if inputs_embeds is not None:
            h = inputs_embeds
        elif inputs is not None:
            h = self.embed_tokens(inputs)
        else:
            h = mx.zeros((1, 1, self.args.hidden_size), dtype=mx.float32)

        players = self.pipeline_layers
        if cache is None:
            cache = [None] * len(players)

        if mask is None:
            cache0 = cache[0] if cache and cache[0] is not None else None
            # A single unpadded stream's causal mask is fully described by the
            # "causal" string. BatchKVCache.make_mask returns a dense causal
            # array instead, which fails the MSA sparse-prefill eligibility gate
            # (only None/"causal" pass) and forces the dense O(n^2) fallback --
            # the root cause of the long-context prefill regression. The array
            # is a pure causal mask for B==1/no-padding, and the sparse path
            # handles causality via q_start (never consuming the mask), so emit
            # the string form to keep blockwise-sparse prefill engaged. Both
            # ranks run this identically (B==1), so the mask stays consistent.
            if _unpadded_single_stream(h, cache0):
                mask = "causal" if h.shape[1] > 1 else None
            else:
                mask = create_attention_mask(h, cache0)

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()

        # Capture-only piggyback synthesis (feature/capture-only): when eagle is
        # NOT driving this forward (no capture ids passed) but capture-only is
        # armed, synthesize the SAME capture set the eagle path would use, so the
        # boundary piggyback rides every normal forward. The decision is a pure
        # function of shared env (forward_capture_active) + the rank's layer
        # range, so both ranks derive identical _remote_caps/_ship_caps and the
        # collective schedule stays lockstep. rank0 collects the concat into a
        # private sink and hands it to m3_capture; the caller's hidden_sink stays
        # untouched (it is None on the normal decode path).
        _cap_only_sink = None
        _capmod = None
        if _CAPTURE_ONLY_ENV and not capture_set:
            _capmod = _capture_module()
            if _capmod is not None and _capmod.forward_capture_active():
                capture_set = _capmod.capture_layers_set()
                _cap_only_sink = []
        # Effective capture sink: the caller's when eagle passes one, else the
        # capture-only private sink. Leaves eagle behavior byte-identical.
        _sink = hidden_sink if hidden_sink is not None else _cap_only_sink

        # EAGLE3 capture piggyback (2026-07-09): captures for layers owned by
        # EARLIER ranks (rank1 owns 0..start_idx-1 in the 2-rank split) ride
        # the boundary send this forward already performs — the sender
        # concatenates [h, *its captured hiddens] on the feature axis and the
        # receiver splits. Zero new collectives, so the schedule stays
        # identical to the proven decode path. Both sides derive the widths
        # deterministically from (capture_set, start_idx/end_idx). Inactive
        # (empty capture_set) leaves every shape byte-identical to before.
        _remote_caps = (
            sorted(i for i in capture_set if i < self.start_idx)
            if capture_set
            else []
        )
        _ship_caps = (
            sorted(
                i for i in capture_set
                if self.start_idx <= i < (self.end_idx or self.start_idx)
            )
            if (capture_set and self.pipeline_rank != 0)
            else []
        )
        _shipped_hiddens = {}

        # Receive hidden states from the next rank (rank+1 processes earlier layers)
        if self.pipeline_rank < self.pipeline_size - 1:
            if _remote_caps:
                width = h.shape[-1]
                template = mx.zeros(
                    (h.shape[0], h.shape[1], width * (1 + len(_remote_caps))),
                    dtype=h.dtype,
                )
                if _E3_PIPE_DEBUG:
                    print(
                        f"[e3dbg] rank{self.pipeline_rank} RECV template "
                        f"shape={tuple(template.shape)} dtype={template.dtype}",
                        flush=True,
                    )
                packed = mx.distributed.recv_like(template, self.pipeline_rank + 1)
                h = packed[..., :width]
                if _sink is not None:
                    for _j in range(len(_remote_caps)):
                        _sink.append(
                            packed[..., (1 + _j) * width : (2 + _j) * width]
                        )
            else:
                h = mx.distributed.recv_like(h, self.pipeline_rank + 1)

        for local_idx, (layer, c) in enumerate(zip(players, cache)):
            h = layer(h, mask, c, position_ids=position_ids)
            global_idx = self.start_idx + local_idx
            if global_idx in capture_set:
                if global_idx in _ship_caps:
                    _shipped_hiddens[global_idx] = h
                elif _sink is not None:
                    _sink.append(h)
            if layer_eval_every > 0 and (local_idx + 1) % layer_eval_every == 0:
                # Break long lazy decode graphs into smaller Metal submissions.
                # This is slower, but prevents rank 1 from hitting macOS GPU
                # watchdog timeouts when it owns many large MiniMax layers.
                mx.eval(h)

        # Send our output to the previous rank (rank-1 processes later layers)
        if self.pipeline_rank != 0:
            if _ship_caps:
                _orig_width = h.shape[-1]
                packed = mx.concatenate(
                    [h] + [_shipped_hiddens[g] for g in _ship_caps], axis=-1
                )
                if _E3_PIPE_DEBUG:
                    print(
                        f"[e3dbg] rank{self.pipeline_rank} SEND packed "
                        f"shape={tuple(packed.shape)} dtype={packed.dtype}",
                        flush=True,
                    )
                packed = mx.distributed.send(
                    packed, (self.pipeline_rank - 1) % self.pipeline_size
                )
                # Downstream (dummy-logits return path) must see the original
                # width; slicing the SEND RESULT keeps the send dependency.
                h = packed[..., :_orig_width]
            else:
                h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            if cache:
                # Match the official MLX pipeline pattern: depend only the
                # final owned cache on the send. Depending every layer cache on
                # every token's send builds a large lazy graph and can wedge at
                # KV cache reallocation boundaries.
                _depend_cache_on_hidden(cache[-1], h)
            if rank0_decode_owner and not _M3_PREFILL_OVERLAP_ACTIVE:
                # Nonzero ranks return dummy logits in rank0-decode-owner mode,
                # so no later lm_head/all_gather operation would force this send.
                # Evaluate it here before waiting for rank 0's sampled token.
                # (Skipped while the overlap prefill loop owns evaluation —
                # see _M3_PREFILL_OVERLAP_ACTIVE at module top.)
                mx.eval(h)

        # In canonical MLX pipeline examples every rank gathers final hidden
        # state and independently samples. MiniMax-M3's huge untied lm_head makes
        # that too expensive on the worker rank. With rank0-token sync enabled,
        # rank 0 is the sole decode owner: nonzero ranks only send their hidden
        # state forward, then consume rank 0's sampled token on the next step.
        if self.pipeline_size > 1 and not rank0_decode_owner:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        if is_last or (self.pipeline_size > 1 and not rank0_decode_owner):
            h = self.norm(h)
        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)
        # Capture-only: rank0 concatenates this forward's capture layers on the
        # feature axis ([seg0,seg1,seg2] -> D_concat) and hands them to
        # m3_capture. Lazy (no eval here); m3_capture drops it if no request is
        # active or the request is over its cap.
        if _cap_only_sink and _capmod is not None:
            try:
                _capmod.push(mx.concatenate(_cap_only_sink, axis=-1))
            except Exception:
                pass
        return h

    MiniMaxM3Model.__call__ = _patched_call

    # Patch make_cache() to only build caches for owned (non-None) layers.
    # LanguageModel.make_cache is the one actually called by the generation
    # path (ar.py -> VLM Model.make_cache -> language_model.make_cache).
    def _make_cache(self):
        return [
            MiniMaxM3KVCache()
            if (layer.self_attn.has_sparse_index and not disable_sparse_index)
            else KVCache()
            for layer in self.layers
            if layer is not None
        ]
    LanguageModel.make_cache = _make_cache
    MiniMaxM3Model.make_cache = _make_cache

    print(
        "[m3_pipeline_patch] pipeline parallelism enabled for MiniMaxM3Model "
        f"(MiniMax cache+index send dependency, kv_step={cache_step}, "
        f"layer_eval_every={layer_eval_every}, "
        f"rank0_decode_owner={'on' if rank0_decode_owner else 'off'}, "
        f"sparse_index={'off' if disable_sparse_index else 'on'})"
    )


def _install_rank0_token_sync(group):
    """Force every pipeline rank to feed rank 0's sampled token.

    MLX pipeline examples assume all ranks advance generation with identical
    next-token ids. MiniMax-M3 runs the VLM generation stack independently on
    every rank, so stochastic sampling or tiny numerical drift can eventually
    make rank histories diverge. The next forward still has matching shapes, but
    the distributed decode graph is then no longer semantically lockstep and has
    been wedging after ~80-110 tokens. A tiny all_gather of the sampled token
    keeps both ranks on rank 0's token stream.
    """
    enabled = os.environ.get("MLX_M3_SYNC_SAMPLED_TOKENS", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled or group.size() <= 1:
        return
    try:
        import importlib
        ar_mod = importlib.import_module("mlx_vlm.generate.ar")
    except Exception as e:
        print(f"[m3_pipeline_patch] token sync patch unavailable: {e}")
        return

    orig = ar_mod._sample_with_positions
    if getattr(orig, "_m3_rank0_token_sync", False):
        return

    try:
        import constrained_tools as _ctools
    except Exception:
        _ctools = None
    _ct_on = _ctools is not None and _ctools.env_enabled()

    def _server_force_eos():
        # Decode-stop EOS injection (2026-07-06): the serving layer arms this
        # when a client stop arrives. Resolved lazily because this patch loads
        # before the server module finishes importing.
        import sys as _sys
        srv = _sys.modules.get("sharded_server")
        return getattr(srv, "_FORCE_EOS", None) if srv is not None else None

    def _server_next_forced_token():
        # Optional request-scoped semantic token injection (currently used to
        # close MiniMax reasoning without ending generation). Resolve lazily
        # for the same import-order reason as _server_force_eos().
        import sys as _sys
        srv = _sys.modules.get("sharded_server")
        consume = getattr(srv, "_consume_rank0_forced_token", None)
        return consume() if callable(consume) else None

    def _synced_sample_with_positions(*args, **kwargs):
        # Constrained tool decoding (rank0 only): mask logits before sampling;
        # the sampled token is folded into the automaton after the send-eval
        # below (so the load-bearing eval ordering is untouched). No-op unless
        # the env flag is on AND a per-request grammar is armed.
        con = _ctools.active() if (_ct_on and group.rank() == 0) else None
        if con is not None and len(args) >= 2:
            try:
                masked = con.mask_logits(args[1])
                if masked is not args[1]:
                    args = (args[0], masked) + tuple(args[2:])
            except Exception:
                con = None
        y = orig(*args, **kwargs)
        if group.rank() == 0:
            fe = _server_force_eos()
            if fe and fe.get("active") and fe.get("eos_id") is not None:
                # Swap rank 0's sampled token for EOS BEFORE the send: every
                # rank consumes the same stream and the generation ends
                # identically everywhere — no per-rank stop files, no extra
                # collectives, no break-point drift.
                # mx.depends is LOAD-BEARING: the eager mx.eval(sends) below
                # is what forces this rank's forward (and posts its pipeline
                # h-recv) every step. A bare mx.full constant would satisfy
                # that eval without forcing anything, leaving the peer rank
                # blocked in its h-send eval — the 20:18 stall signature.
                eos = mx.full(y.shape, fe["eos_id"], dtype=y.dtype)
                y = mx.depends(eos, y)
            else:
                forced_token_id = _server_next_forced_token()
                if forced_token_id is not None:
                    # Preserve the load-bearing forward dependency exactly as
                    # the synchronized EOS path does, but keep decode alive.
                    forced = mx.full(y.shape, forced_token_id, dtype=y.dtype)
                    y = mx.depends(forced, y)
            sends = [
                mx.distributed.send(y, dst, group=group, stream=mx.cpu)
                for dst in range(1, group.size())
            ]
            if sends:
                mx.eval(sends)
            if con is not None:
                # y is already materialized by mx.eval(sends) above; reading it
                # here adds no new eval point ahead of the send.
                try:
                    con.observe(int(y.reshape(-1)[0]))
                except Exception:
                    pass
            return y
        synced = mx.distributed.recv_like(y, 0, group=group, stream=mx.cpu)
        mx.eval(synced)
        return synced

    _synced_sample_with_positions._m3_rank0_token_sync = True
    ar_mod._sample_with_positions = _synced_sample_with_positions
    print("[m3_pipeline_patch] rank0 sampled-token sync enabled")


def _install_rank0_logits_only(group):
    """Avoid expensive lm_head work on nonzero pipeline ranks.

    In the canonical MLX-LM pipeline examples every rank computes logits after
    all_gather so each rank can independently sample the next token. MiniMax-M3
    has a very large untied lm_head (200064 x 6144), and rank 1 does not need
    real logits because sampled-token sync forces it to consume rank 0's token.
    Returning dummy logits on nonzero ranks keeps stream_generate's shape
    contract intact while removing a huge per-token worker Metal workload.
    """
    enabled = os.environ.get("MLX_M3_RANK0_ONLY_LOGITS", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled or group.size() <= 1 or group.rank() == 0:
        return
    from mlx_vlm.models.minimax_m3_vl.language import LanguageModel

    # Native mlx-vlm 0.6.4 LanguageModel produces logits inline in __call__ and
    # has no logits_from_hidden hook (unlike our old overlay). Skip the
    # optimization gracefully: rank 1 then runs its (cheap, discarded) lm_head
    # on partial hidden state, still correct because the rank0 token-sync
    # overrides rank 1's sampled token. Revisit for a native logits hook.
    orig = getattr(LanguageModel, "logits_from_hidden", None)
    if orig is None:
        print("[m3_pipeline_patch] rank0-only-logits skipped (native LanguageModel "
              "has no logits_from_hidden hook; rank1 lm_head is discarded via token sync)")
        return
    if getattr(orig, "_m3_rank0_only_logits", False):
        return

    def _dummy_logits_from_hidden(self, hidden):
        shape = (*hidden.shape[:2], int(self.args.vocab_size))
        return mx.zeros(shape, dtype=hidden.dtype)

    _dummy_logits_from_hidden._m3_rank0_only_logits = True
    _dummy_logits_from_hidden._m3_original = orig
    LanguageModel.logits_from_hidden = _dummy_logits_from_hidden
    print("[m3_pipeline_patch] nonzero rank dummy logits enabled")


def sharded_load_pipeline(repo):
    """Load the model with pipeline parallelism + per-rank weight filtering.

    Each rank only reads the weight FILES that contain its own layers,
    avoiding the full-model mmap that causes memory thrash.
    """
    import json
    from pathlib import Path
    from mlx_vlm.utils import get_model_path, load_model, load_processor, load_image_processor
    from mlx_vlm.models.minimax_m3_vl.config import ModelConfig

    apply_pipeline_patch()

    group = mx.distributed.init()
    global _PIPELINE_GROUP
    _PIPELINE_GROUP = group
    _install_rank0_token_sync(group)
    _install_rank0_logits_only(group)
    model_path = get_model_path(repo)

    # Lazy-load to get the model class, then apply pipeline split
    model = load_model(model_path, lazy=True, strict=False)

    # Determine which layers this rank owns -> which weight files it needs.
    # Read the safetensors index to map layer indices -> shard files.
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_index = json.load(f)["weight_map"]

    inner = model.language_model.model
    inner.pipeline(group)  # splits self.layers, sets start_idx/end_idx
    if os.environ.get("MLX_M3_DISABLE_SPARSE_INDEX", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        disabled = 0
        for layer in inner.pipeline_layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "has_sparse_index", False):
                attn.has_sparse_index = False
                disabled += 1
        print(f"[pipeline] rank {group.rank()}: disabled sparse index on {disabled} owned layers")

    # Build the set of files this rank actually needs, and which tensor keys.
    # A shard file can hold multiple layers, so we filter at BOTH file and tensor level.
    my_files = set()
    owned_keys = set()
    # CRITICAL: embed_tokens, lm_head, and norm are needed on EVERY rank:
    #  - embed_tokens: every rank embeds (for recv_like shape, like glm4_moe)
    #  - lm_head + norm: after all_gather broadcasts hidden states, every rank
    #    computes logits + samples the token. Without lm_head on all ranks, rank 1
    #    samples garbage tokens -> drift -> deadlock. (This is how mlx_lm pipeline
    #    works: all ranks run lm_head on the gathered hidden states.)
    for k, fname in weight_index.items():
        if ".layers." in k:
            try:
                layer_idx = int(k.split(".layers.")[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if not (inner.start_idx <= layer_idx < inner.end_idx):
                continue  # not our layer -> skip entirely
            my_files.add(fname)
            owned_keys.add(k)
        else:
            # embed_tokens, lm_head, norm, vision, etc. -> ALL ranks load these
            my_files.add(fname)
            owned_keys.add(k)

    print(f"[pipeline] rank {group.rank()}: owns layers "
          f"[{inner.start_idx}:{inner.end_idx}], needs {len(my_files)} weight files")

    # Load ONLY the needed weight files, then keep ONLY owned tensors
    from mlx_vlm.utils import _load_safetensors

    weights = {}
    for wf_name in sorted(my_files):
        wf = str(model_path / wf_name)
        file_tensors = _load_safetensors(wf)
        for k, v in file_tensors.items():
            if k in owned_keys:
                weights[k] = v

    # Strip leading "language_model." if the model expects unprefixed keys.
    # The safetensors keys are like "language_model.model.layers.N..."; the
    # model's load_weights wants the path relative to the model root.
    # mlx_vlm's load() strips via sanitize, so we keep keys as-is and let
    # strict=False skip mismatches.
    model.load_weights(list(weights.items()), strict=False)

    print(f"[pipeline] rank {group.rank()}: materializing {len(weights)} tensors")
    mx.eval(model.language_model.parameters())
    model.eval()

    # Barrier
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))

    processor = load_processor(model_path, True)
    image_processor = load_image_processor(model_path)
    if image_processor is not None:
        processor.image_processor = image_processor
    try:
        resolved = str(model_path)
        setattr(model, "_thundermlx_model_path", resolved)
        setattr(processor, "_thundermlx_model_path", resolved)
    except Exception:
        pass

    return model, processor
