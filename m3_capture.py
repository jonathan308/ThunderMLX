"""Capture-only drafter-training data collection for the 2-rank MiniMax-M3
pipeline (ThunderMLX).

Purpose: while the cluster serves NORMAL, non-speculative generation, collect
the exact inputs the EAGLE3 drafter is trained on — the concatenated
capture-layer hidden states plus the sampled token stream — so everyday usage
builds an on-distribution corpus for offline fine-tuning, WITHOUT paying
eagle's live verify/rollback speed penalty.

Design (docs mirror m3_eagle3's prompt/round dumps; consumed by
ops/eagle3_finetune.build_request_sequence):

- The concatenated capture ([seg(L0), seg(L1), seg(L2)] on the feature axis,
  18432-d for the 3-layer default) is produced by the pipeline patch's EXISTING
  eagle piggyback: rank1 ships its owned capture layer on the boundary send it
  already performs, rank0 receives it and appends its own owned capture layers.
  Zero new collectives. During normal decode the caller passes no
  capture_layer_ids, so m3_pipeline_patch synthesizes the SAME capture set from
  MLX_M3_EAGLE3_CAPTURE_LAYERS (gated on MLX_M3_EAGLE3_CAPTURE_ONLY) — the
  synthesis is a pure function of shared env + the rank's layer range, so both
  ranks ride the piggyback in lockstep.

- rank0 pushes the per-forward concat here (push()); prompt-phase prefill
  chunks and per-step decode hiddens funnel through in wire order. On request
  completion the server hands over the sampled token stream (finalize_request);
  the prompt/decode split is recovered as n_prompt = N_captured - K_tokens (see
  finalize_request), which is robust to chunked prefill and prompt caching.

Overhead discipline (target <5% decode): the piggyback adds only the capture
layers OWNED BY rank1 to the pipeline send — for the 38,22 split with capture
layers 1,29,56 that is a SINGLE 6144-d segment (layers 29,56 are rank0-local,
no wire), so the boundary send ~doubles: +~12 KB/token bf16, a few microseconds
over Thunderbolt vs a ~30-50 ms MoE decode step. Accumulation keeps mx arrays
lazy and evaluates/converts to fp16 in chunks (never a per-token GPU sync).
Default OFF => every hook is inert and behavior is byte-identical.

Env:
  MLX_M3_EAGLE3_CAPTURE_ONLY   1 to arm capture-only (default 0 = off)
  MLX_M3_EAGLE3_DUMP_DIR       corpus root (reused from the eagle dumps; empty
                               = off even if CAPTURE_ONLY=1)
  MLX_M3_EAGLE3_CAPTURE_LAYERS capture layer ids, shared with m3_eagle3
                               (default 2,30,57; production A/B'd to 1,29,56)
  MLX_M3_EAGLE3_CAPTURE_MAX_MB    per-request dump cap, abort+warn above
                                  (default 200; boot seed — live-tunable via
                                  rank0 /admin/runtime-tuning
                                  capture_max_request_bytes)
  MLX_M3_EAGLE3_CAPTURE_DIR_MAX_GB total corpus cap, stop+warn above
                                  (default 100; boot seed — live-tunable via
                                  rank0 /admin/runtime-tuning
                                  capture_max_total_bytes)
  MLX_M3_EAGLE3_CAPTURE_EVAL_EVERY chunked-eval cadence in captured positions
                                  (default 256; bounds lazy graph + memory)
"""

import os
import sys
import glob
import time
import logging
from typing import List, Optional

import mlx.core as mx

logger = logging.getLogger("m3_capture")


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


_CAPTURE_ONLY = _flag("MLX_M3_EAGLE3_CAPTURE_ONLY", "0")
_DUMP_DIR = os.environ.get("MLX_M3_EAGLE3_DUMP_DIR", "").strip()

# Capture layers: identical resolution to m3_eagle3 (_CAPTURE_OVERRIDE or the
# README default) so a capture-only dump segments EXACTLY like the eagle dumps
# the trainer was built on.
_CAPTURE_LAYERS = [
    int(x) for x in os.environ.get(
        "MLX_M3_EAGLE3_CAPTURE_LAYERS", ""
    ).replace(",", " ").split()
] or [2, 30, 57]
_CAPTURE_LAYERS_SET = set(_CAPTURE_LAYERS)

# Byte caps live in a settings dict that the enforcement sites re-read on
# every flush/finalize, so rank0's /admin/runtime-tuning can adjust them
# without a restart (set_limits below). Boot still seeds from the env vars.
_SETTINGS = {
    "max_request_bytes": int(
        float(os.environ.get("MLX_M3_EAGLE3_CAPTURE_MAX_MB", "200")) * (1 << 20)
    ),
    "max_total_bytes": int(
        float(os.environ.get("MLX_M3_EAGLE3_CAPTURE_DIR_MAX_GB", "100")) * (1 << 30)
    ),
}
_EVAL_EVERY = max(1, int(os.environ.get("MLX_M3_EAGLE3_CAPTURE_EVAL_EVERY", "256") or "256"))

# Disk/permission failures flip this: degrade the whole process to no-capture
# after one warning, never crash generation. Does NOT gate the piggyback (that
# stays env-only so the two ranks never desync mid-run).
_DISABLED = {"value": False}
_WARNED = {"dir_cap": False, "req_cap": False, "write": False}

# Single generation slot (MLX_M3_MAX_CONCURRENT_REQUESTS=1) => module-global
# per-request state is race-free, same discipline as m3_eagle3.REQUEST_ACTIVE.
_STATE = {
    "active": False,
    "materialized": [],   # list[np.ndarray] fp16, wire order, (1, T, D)
    "pending": [],        # list[mx.array] fp16, not yet evaluated
    "pending_pos": 0,     # positions buffered in `pending`
    "bytes": 0,           # materialized fp16 bytes so far this request
    "capped": False,
}

# Running estimate of corpus size so the dir cap costs one du() per process,
# not one walk per request.
_DIR_BYTES = {"value": None}


# --------------------------------------------------------------------------
# Live-tunable caps + status (rank0's /admin/runtime-tuning surface).
# --------------------------------------------------------------------------

def settings() -> dict:
    """Current live byte caps (copy; safe to expose in /health)."""
    return dict(_SETTINGS)


def set_limits(max_request_bytes=None, max_total_bytes=None) -> dict:
    """Live-tune the byte caps (values are absolute bytes; the caller —
    sharded_server's runtime-tuning endpoint — validates and clamps them).
    Re-arms the once-per-cap warnings so a later hit on the NEW cap logs.
    Returns the applied caps."""
    if max_request_bytes is not None:
        _SETTINGS["max_request_bytes"] = int(max_request_bytes)
        _WARNED["req_cap"] = False
    if max_total_bytes is not None:
        _SETTINGS["max_total_bytes"] = int(max_total_bytes)
        _WARNED["dir_cap"] = False
    return dict(_SETTINGS)


def status() -> dict:
    """Corpus usage vs caps for /health (dashboard Storage card). The first
    call walks the dump dir once; afterwards finalize keeps the estimate."""
    return {
        "capture_only": _CAPTURE_ONLY,
        "dump_dir": _DUMP_DIR,
        "armed": armed(),
        "capturing": is_capturing(),
        "degraded": _DISABLED["value"],
        "total_bytes": int(_dir_bytes()) if _DUMP_DIR else 0,
        **settings(),
    }


def forward_capture_active() -> bool:
    """Whether the pipeline patch should ride the capture piggyback on a
    no-capture-ids forward. Pure function of shared env (+ a configured dump
    dir) so BOTH ranks decide identically and the collective schedule stays in
    lockstep. Intentionally independent of per-request / disk state."""
    return _CAPTURE_ONLY and bool(_DUMP_DIR)


def capture_layers() -> List[int]:
    return list(_CAPTURE_LAYERS)


def capture_layers_set() -> set:
    return set(_CAPTURE_LAYERS_SET)


def armed() -> bool:
    """rank0-side: capture-only is on AND still writable."""
    return _CAPTURE_ONLY and bool(_DUMP_DIR) and not _DISABLED["value"]


def is_capturing() -> bool:
    return _STATE["active"] and not _STATE["capped"]


# --------------------------------------------------------------------------
# Per-request accumulation (rank0).
# --------------------------------------------------------------------------

def begin_request() -> None:
    """Arm a fresh per-request accumulator. No-op unless armed()."""
    _STATE["active"] = False
    _STATE["materialized"] = []
    _STATE["pending"] = []
    _STATE["pending_pos"] = 0
    _STATE["bytes"] = 0
    _STATE["capped"] = False
    if not armed():
        return
    if _DUMP_DIR:
        try:
            os.makedirs(_DUMP_DIR, exist_ok=True)
        except Exception as e:
            _degrade("cannot create dump dir: %s" % e)
            return
    _STATE["active"] = True


def _degrade(msg: str) -> None:
    _DISABLED["value"] = True
    _STATE["active"] = False
    if not _WARNED["write"]:
        logger.warning("capture-only disabled (degraded): %s", msg)
        _WARNED["write"] = True


def _flush_pending() -> None:
    """Evaluate the buffered mx chunks once and move them to fp16 numpy. One
    GPU sync per EVAL_EVERY positions, never per token."""
    if not _STATE["pending"]:
        return
    import numpy as _np
    try:
        mx.eval(_STATE["pending"])
        for arr in _STATE["pending"]:
            _STATE["materialized"].append(_np.array(arr, copy=False))
    except Exception as e:
        _degrade("eval/convert failed: %s" % e)
        return
    finally:
        _STATE["pending"] = []
        _STATE["pending_pos"] = 0
    _STATE["bytes"] = sum(a.nbytes for a in _STATE["materialized"])
    max_request_bytes = int(_SETTINGS["max_request_bytes"])  # re-read: live-tunable
    if _STATE["bytes"] > max_request_bytes and not _STATE["capped"]:
        _STATE["capped"] = True
        if not _WARNED["req_cap"]:
            logger.warning(
                "capture-only: request exceeded %d MB, dropping its dump "
                "(raise MLX_M3_EAGLE3_CAPTURE_MAX_MB or live-tune "
                "capture_max_request_bytes to keep long contexts)",
                max_request_bytes >> 20,
            )
            _WARNED["req_cap"] = True


def push(concat_hidden: mx.array) -> None:
    """rank0 hook, called from m3_pipeline_patch per forward with the
    concatenated capture ([seg0..segN] on the feature axis), shape (1, T, D).
    Lazy: cast to fp16 and buffer; evaluate in chunks. No-op when inactive or
    already capped for this request (keeps the piggyback wire cost but drops
    the data, so ranks never desync)."""
    if not _STATE["active"] or _STATE["capped"]:
        return
    try:
        _STATE["pending"].append(concat_hidden.astype(mx.float16))
        _STATE["pending_pos"] += int(concat_hidden.shape[1])
    except Exception as e:
        _degrade("push failed: %s" % e)
        return
    # Flush eagerly on a multi-token (prefill) chunk so a big prompt block is
    # never held lazily, and otherwise every EVAL_EVERY positions.
    if concat_hidden.shape[1] > 1 or _STATE["pending_pos"] >= _EVAL_EVERY:
        _flush_pending()


def abort_request() -> None:
    """Drop the accumulator without writing (generation error path)."""
    _STATE["active"] = False
    _STATE["materialized"] = []
    _STATE["pending"] = []
    _STATE["pending_pos"] = 0
    _STATE["bytes"] = 0
    _STATE["capped"] = False


# --------------------------------------------------------------------------
# Finalize + write (rank0).
# --------------------------------------------------------------------------

def _dir_bytes() -> int:
    if _DIR_BYTES["value"] is None:
        total = 0
        try:
            for root, _dirs, files in os.walk(_DUMP_DIR):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        except Exception:
            total = 0
        _DIR_BYTES["value"] = total
    return _DIR_BYTES["value"]


def finalize_request(prompt_ids, tokens: List[int]) -> Optional[str]:
    """Write one request's corpus files and return the request dir (or None).

    prompt_ids : the token ids fed during prefill (suffix ids under prompt
                 caching, else the full prompt), len == n_prompt.
    tokens     : the sampled decode token stream [y0, y1, ...], collected by the
                 server the SAME way it counts generation_tokens. len == K, the
                 number of decode steps (== number of decode captures).

    Split: the accumulator holds N = n_prompt + K captured positions in wire
    order, so n_prompt = N - K. prompt_hidden = first n_prompt; decode hiddens =
    the rest. Pairs are (decode_hidden[t] -> tokens[t+1]); the final decode
    hidden is unlabeled (its next token was sampled but never emitted) and is
    dropped, giving hidden[T, D] + tokens[T+1] with T = K - 1.
    """
    if not _STATE["active"]:
        abort_request()
        return None
    try:
        return _finalize_inner(prompt_ids, tokens)
    except Exception as e:
        if not _WARNED["write"]:
            logger.warning("capture-only: finalize failed: %s", e)
            _WARNED["write"] = True
        return None
    finally:
        abort_request()


def _finalize_inner(prompt_ids, tokens: List[int]) -> Optional[str]:
    import numpy as _np

    _flush_pending()
    # A flush failure degrades mid-finalize; never write a partially-converted
    # (and thus misaligned) accumulator.
    if _DISABLED["value"] or _STATE["capped"] or not _STATE["materialized"]:
        return None

    tokens = [int(t) for t in tokens]
    K = len(tokens)
    if K < 1:
        return None  # nothing decoded; the prompt->first pair alone is not useful

    full = _np.concatenate(_STATE["materialized"], axis=1)  # (1, N, D) fp16
    N = full.shape[1]
    n_prompt = N - K
    if n_prompt < 0:
        # Capture/token accounting disagreed (should not happen); skip rather
        # than write a misaligned corpus file.
        logger.warning(
            "capture-only: N=%d < K=%d, skipping request dump", N, K
        )
        return None

    # Dir cap: check before minting the request dir (re-read: live-tunable).
    max_total_bytes = int(_SETTINGS["max_total_bytes"])
    if _dir_bytes() > max_total_bytes:
        if not _WARNED["dir_cap"]:
            logger.warning(
                "capture-only: corpus exceeded %d GB, no longer capturing "
                "(raise MLX_M3_EAGLE3_CAPTURE_DIR_MAX_GB or live-tune "
                "capture_max_total_bytes)",
                max_total_bytes >> 30,
            )
            _WARNED["dir_cap"] = True
        return None

    prompt_ids = _np.asarray(prompt_ids).reshape(-1).astype(_np.int32)
    prompt_hidden = None
    if n_prompt > 0 and prompt_ids.shape[0] == n_prompt:
        prompt_hidden = full[:, :n_prompt, :]
    elif n_prompt > 0:
        # ids/hiddens disagree (e.g. tokenizer drift on the uncached path):
        # keep the decode file (self-consistent) and skip the prompt file.
        logger.warning(
            "capture-only: prompt_ids(%d) != n_prompt(%d); writing decode only",
            prompt_ids.shape[0], n_prompt,
        )

    decode_hidden = full[:, n_prompt:, :][0]     # (K, D) fp16
    T = min(decode_hidden.shape[0], K) - 1       # drop the unlabeled tail
    first_token = tokens[0]

    ts = time.time()
    req_dir = os.path.join(
        _DUMP_DIR, "req_%d_%d_%09d" % (os.getpid(), int(ts), _next_seq())
    )
    os.makedirs(req_dir, exist_ok=True)
    wrote = []
    if prompt_hidden is not None:
        write_prompt_file(
            os.path.join(req_dir, "prompt_%d.npz" % int(ts)),
            prompt_hidden, prompt_ids, first_token,
        )
        wrote.append("prompt[%d]" % n_prompt)
    if T >= 1:
        write_decode_file(
            os.path.join(req_dir, "decode_%d.npz" % int(ts)),
            decode_hidden[:T], _np.asarray(tokens[: T + 1], dtype=_np.int32),
        )
        wrote.append("decode[%d]" % T)

    if not wrote:
        try:
            os.rmdir(req_dir)
        except OSError:
            pass
        return None

    # Keep the running corpus-size estimate current without a re-walk.
    if _DIR_BYTES["value"] is not None:
        try:
            for f in os.listdir(req_dir):
                _DIR_BYTES["value"] += os.path.getsize(os.path.join(req_dir, f))
        except OSError:
            pass
    logger.info("capture-only: wrote %s -> %s", "+".join(wrote), req_dir)
    return req_dir


_SEQ = {"n": 0}


def _next_seq() -> int:
    _SEQ["n"] += 1
    return _SEQ["n"]


# --------------------------------------------------------------------------
# Pure-numpy writers (unit-testable with synthetic arrays; no model needed).
# --------------------------------------------------------------------------

def write_prompt_file(path, prompt_hidden, prompt_ids, first_token) -> None:
    """prompt_*.npz — byte-identical layout to m3_eagle3's prompt dump.
      prompt_hidden : (1, n_prompt, D) fp16   concatenated capture layers
      prompt_ids    : (n_prompt,)     int32   ids fed during prefill
      first_token   :                 int32   token sampled from the last prompt
                                              position (== decode tokens[0])
    """
    import numpy as _np
    _np.savez(
        path,
        prompt_hidden=_np.asarray(prompt_hidden, dtype=_np.float16),
        prompt_ids=_np.asarray(prompt_ids, dtype=_np.int32).reshape(-1),
        first_token=_np.int32(int(first_token)),
    )


def write_decode_file(path, hidden, tokens) -> None:
    """decode_*.npz — NEW capture-only format for native (non-speculative)
    decode. One file per request.
      hidden : (T, D)   fp16   per-step concatenated capture layers, decode
                               position t (the hidden that PREDICTED tokens[t+1])
      tokens : (T+1,)   int32  sampled stream [y0 .. yT]; tokens[0] == first_token
    Pairing: hidden[t] -> tokens[t+1], for t in 0..T-1 (teacher-forced next
    token). Consumed by ops/eagle3_finetune.build_request_sequence exactly like
    the round_*.npz decode-phase pairs.
    """
    import numpy as _np
    _np.savez(
        path,
        hidden=_np.asarray(hidden, dtype=_np.float16),
        tokens=_np.asarray(tokens, dtype=_np.int32).reshape(-1),
    )
