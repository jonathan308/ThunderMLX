#!/usr/bin/env python3
"""
Pipeline-parallel OpenAI server for MiniMax-M3 across two Macs.

WHAT THIS IS
  A 2-rank pipeline-parallel server. Rank 0 owns the last
  layers + lm_head and serves the OpenAI API; rank 1 owns
  the first layers (embeddings) and mirrors rank 0's generation in
  lockstep. Communication goes over the configured MLX/JACCL data link.

ROBUSTNESS DESIGN (the hard-won lessons)
  - Memory crashes: every generation is wrapped in try/except on BOTH
    ranks. On any error both ranks call mx.clear_cache() to release Metal
    memory back to the OS. The HTTP layer never hangs and never orphans.
  - The 4th-request crash: stream_generate runs on a module-level
    thread-local generation_stream that accumulates Metal command buffers.
    We refresh it with a FRESH stream per request (see _refresh_generation_stream).
  - Reasoning vs content routing: MiniMax-M3 emits <mm:think>...</mm:think>
    before the answer. We split reasoning/content the SAME way the official
    mlx_vlm server does (split_stream_thinking_delta), not via naive
    string-splitting. This is what makes OpenWebUI show the thinking
    dropdown correctly.

COORDINATION
  - Rank 0 broadcasts each request; both ranks run stream_generate in lockstep.
  - The model's send/recv (in __call__) + all_gather keep ranks synced.
  - thinking_mode defaults to "enabled" for OpenWebUI. Agent/coding clients
    can use the m3-agent/m3-coder/m3-no-think aliases or override per request.
  - Vision (image-to-text) is preserved via OpenAI multimodal format.
"""
import logging
import asyncio
import copy
import difflib
import gc
import hashlib
import importlib.metadata
import json
import os
import pickle
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from array import array
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager

import mlx.core as mx

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [rank ?] %(levelname)s %(message)s")
logger = logging.getLogger("sharded_server")

MODEL = os.environ.get("MLX_M3_MODEL", "mlx-community/MiniMax-M3-4bit")
MODEL_ID = os.environ.get("MLX_M3_MODEL_ID", "mlx-community/MiniMax-M3-4bit")
SHARDING_MODE = os.environ.get("M3_SHARDING_MODE", "tensor").strip().lower()
VISIBLE_THINK_MODEL_ID = "Minimax-M3"
VISIBLE_NO_THINK_MODEL_ID = "Minimax-M3-No-Think"
VISIBLE_WEB_MODEL_ID = "M3-Web"
MODEL_MODE_ALIASES = {
    "m3": "enabled",
    "minimax-m3": "enabled",
    "minimax-m3-think": "enabled",
    "minimax-m3-thinking": "enabled",
    "m3-think": "enabled",
    "m3-web": "enabled",
    "m3 web": "enabled",
    "minimax-m3-web": "enabled",
    "minimax m3 web": "enabled",
    "m3-openwebui": "enabled",
    "m3-adaptive": "adaptive",
    "minimax-m3-adaptive": "adaptive",
    "m3-agent": "disabled",
    "m3-coder": "disabled",
    "m3-no-think": "disabled",
    "m3-nothink": "disabled",
    "m3-no-thinking": "disabled",
    "minimax-m3-no-think": "disabled",
    "minimax-m3-nothink": "disabled",
    "minimax-m3-no-thinking": "disabled",
}
WEB_MODEL_ALIASES = {
    "m3-web",
    "m3 web",
    "minimax-m3-web",
    "minimax m3 web",
}
MODEL_ALIASES = tuple(
    dict.fromkeys([
        MODEL_ID,
        MODEL,
        VISIBLE_THINK_MODEL_ID,
        VISIBLE_NO_THINK_MODEL_ID,
        VISIBLE_WEB_MODEL_ID,
        *MODEL_MODE_ALIASES.keys(),
    ])
)
VISIBLE_MODEL_IDS = (
    VISIBLE_THINK_MODEL_ID,
    VISIBLE_NO_THINK_MODEL_ID,
    VISIBLE_WEB_MODEL_ID,
)
HOST = os.environ.get("MLX_M3_HOST", "0.0.0.0")
PORT = int(os.environ.get("MLX_M3_PORT", "8080"))
REQUEST_HISTORY_MAX = int(os.environ.get("MLX_M3_REQUEST_HISTORY_MAX", "32") or "32")
GENERATION_LOCK_HANDOFF_GRACE_SECONDS = max(
    0.1,
    float(os.environ.get("MLX_M3_GENERATION_LOCK_HANDOFF_GRACE_SECONDS", "5") or "5"),
)
GENERATION_LOCK_REQUEST_OWNER_GRACE_SECONDS = max(
    GENERATION_LOCK_HANDOFF_GRACE_SECONDS,
    float(os.environ.get("MLX_M3_GENERATION_LOCK_REQUEST_OWNER_GRACE_SECONDS", "30") or "30"),
)
GENERATION_LOCK_CONTROL_OWNER_GRACE_SECONDS = max(
    GENERATION_LOCK_REQUEST_OWNER_GRACE_SECONDS,
    float(os.environ.get("MLX_M3_GENERATION_LOCK_CONTROL_OWNER_GRACE_SECONDS", "120") or "120"),
)
PREFILL_STEP_SIZE = int(os.environ.get("MLX_M3_PREFILL_STEP_SIZE", "128"))
MLX_MAX_OPS_PER_BUFFER = int(os.environ.get("MLX_MAX_OPS_PER_BUFFER", "0") or "0")
MLX_MAX_MB_PER_BUFFER = int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "0") or "0")
MAX_KV_SIZE = int(os.environ.get("MLX_M3_MAX_KV_SIZE", "250000"))
ADVERTISED_MAX_MODEL_LEN = max(
    1,
    int(os.environ.get("MLX_M3_ADVERTISED_MAX_MODEL_LEN", "300000") or "300000"),
)
# Zero leaves enforcement to the model/runtime. Operators of asymmetric
# clusters can set a measured input ceiling so clients receive a compactable
# OpenAI error before the lower-memory rank reaches its local Metal limit.
HARD_MAX_INPUT_TOKENS = max(
    0,
    int(os.environ.get("MLX_M3_HARD_MAX_INPUT_TOKENS", "0") or "0"),
)
KV_QUANT_ENABLED = os.environ.get(
    "MLX_M3_KV_QUANT_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
KV_BITS = float(os.environ.get("MLX_M3_KV_BITS", "4") or "4")
KV_GROUP_SIZE = int(os.environ.get("MLX_M3_KV_GROUP_SIZE", "64") or "64")
KV_QUANT_SCHEME = os.environ.get("MLX_M3_KV_QUANT_SCHEME", "uniform").strip().lower()
if KV_QUANT_SCHEME not in {"uniform", "turboquant"}:
    logger.warning(
        "invalid MLX_M3_KV_QUANT_SCHEME=%r; falling back to uniform",
        KV_QUANT_SCHEME,
    )
    KV_QUANT_SCHEME = "uniform"


def _should_recover_generation_lock(
    *,
    lock_locked,
    active_present,
    releasing_present,
    owner_kind,
    owner_age,
    transition_age,
):
    """Return true only when an ownerless generation lock is genuinely stale."""
    if not lock_locked or active_present or releasing_present:
        return False
    if owner_kind in {"keepwarm", "control"}:
        return owner_age is not None and (
            owner_age >= GENERATION_LOCK_CONTROL_OWNER_GRACE_SECONDS
        )
    if owner_kind == "request":
        return owner_age is not None and (
            owner_age >= GENERATION_LOCK_REQUEST_OWNER_GRACE_SECONDS
        )
    if owner_kind is None:
        return transition_age is not None and (
            transition_age >= GENERATION_LOCK_HANDOFF_GRACE_SECONDS
        )
    return True
QUANTIZED_KV_START = int(os.environ.get("MLX_M3_QUANTIZED_KV_START", "5000") or "5000")
DEFAULT_MAX_TOKENS = int(os.environ.get("MLX_M3_DEFAULT_MAX_TOKENS", "4096"))
NONSTREAM_DEFAULT_MAX_TOKENS = int(
    os.environ.get("MLX_M3_NONSTREAM_DEFAULT_MAX_TOKENS", "512") or "512"
)
NONSTREAM_COALESCE_ENABLED = os.environ.get(
    "MLX_M3_NONSTREAM_COALESCE", "1"
).strip().lower() in {"1", "true", "yes", "on"}
NONSTREAM_COALESCE_GRACE_SECONDS = max(
    0.0,
    float(
        os.environ.get("MLX_M3_NONSTREAM_COALESCE_GRACE_SECONDS", "30")
        or "30"
    ),
)
NONSTREAM_DISCONNECT_GRACE_SECONDS = max(
    0.0,
    float(
        os.environ.get("MLX_M3_NONSTREAM_DISCONNECT_GRACE_SECONDS", "3")
        or "3"
    ),
)
NONSTREAM_COALESCE_MAX_ENTRIES = max(
    1,
    int(os.environ.get("MLX_M3_NONSTREAM_COALESCE_MAX_ENTRIES", "16") or "16"),
)
TITLE_DEFAULT_MAX_TOKENS = max(
    1,
    int(os.environ.get("MLX_M3_TITLE_DEFAULT_MAX_TOKENS", "256") or "256"),
)
OPENWEBUI_DEFAULT_MAX_TOKENS = int(
    os.environ.get("MLX_M3_OPENWEBUI_DEFAULT_MAX_TOKENS", "2048") or "2048"
)
DEFAULT_TEMPERATURE = float(os.environ.get("MLX_M3_DEFAULT_TEMPERATURE", "0") or "0")
DEFAULT_TOP_P = float(os.environ.get("MLX_M3_DEFAULT_TOP_P", "1.0") or "1.0")
DEFAULT_TOP_K = int(os.environ.get("MLX_M3_DEFAULT_TOP_K", "0") or "0")
DEFAULT_MIN_P = float(os.environ.get("MLX_M3_DEFAULT_MIN_P", "0.0") or "0.0")


def _nonstream_request_fingerprint(payload):
    """Hash the exact client request before server-side normalization."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"chat.completions\0" + encoded).hexdigest()


class _NonstreamRequestCoalescer:
    """Share one non-stream generation across exact client retries."""

    def __init__(
        self,
        *,
        enabled=True,
        replay_grace_seconds=30.0,
        disconnect_grace_seconds=3.0,
        max_entries=16,
    ):
        self.enabled = bool(enabled)
        self.replay_grace_seconds = max(0.0, float(replay_grace_seconds))
        self.disconnect_grace_seconds = max(
            0.0, float(disconnect_grace_seconds)
        )
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._joined_total = 0
        self._replayed_total = 0

    def _purge_locked(self, now):
        for key, entry in list(self._entries.items()):
            if not entry.get("done"):
                continue
            completed_at = float(entry.get("completed_at") or 0.0)
            if (
                not entry.get("success")
                or now - completed_at > self.replay_grace_seconds
            ):
                self._entries.pop(key, None)
        while len(self._entries) >= self.max_entries:
            removable = next(
                (
                    key
                    for key, entry in self._entries.items()
                    if entry.get("done")
                ),
                None,
            )
            if removable is None:
                break
            self._entries.pop(removable, None)

    def claim(self, key, request_id, *, now=None):
        if not self.enabled or not key:
            return None, True, False
        now = time.time() if now is None else float(now)
        with self._lock:
            self._purge_locked(now)
            entry = self._entries.get(key)
            if entry is not None:
                entry["clients"][request_id] = True
                if not entry.get("done"):
                    entry["all_disconnected_at"] = None
                entry["joined"] += 1
                self._joined_total += 1
                replayed = bool(entry.get("done"))
                if replayed:
                    entry["replayed"] += 1
                    self._replayed_total += 1
                return entry, False, replayed
            entry = {
                "key": key,
                "owner_request_id": request_id,
                "created_at": now,
                "completed_at": None,
                "event": threading.Event(),
                "done": False,
                "success": False,
                "status_code": None,
                "payload": None,
                "clients": {request_id: True},
                "all_disconnected_at": None,
                "joined": 0,
                "replayed": 0,
            }
            self._entries[key] = entry
            return entry, True, False

    def disconnect(self, entry, request_id, *, now=None):
        if entry is None:
            return 0
        now = time.time() if now is None else float(now)
        with self._lock:
            clients = entry.get("clients") or {}
            if clients.get(request_id):
                clients[request_id] = False
            connected = sum(1 for value in clients.values() if value)
            if not entry.get("done") and connected == 0:
                if entry.get("all_disconnected_at") is None:
                    entry["all_disconnected_at"] = now
            return connected

    def connected_clients(self, entry):
        if entry is None:
            return 0
        with self._lock:
            return sum(
                1 for value in (entry.get("clients") or {}).values() if value
            )

    def should_cancel(self, entry, *, now=None):
        if entry is None:
            return True
        now = time.time() if now is None else float(now)
        with self._lock:
            if entry.get("done"):
                return False
            disconnected_at = entry.get("all_disconnected_at")
            return bool(
                disconnected_at is not None
                and now - float(disconnected_at)
                >= self.disconnect_grace_seconds
            )

    def complete(self, entry, payload, *, status_code=200, now=None):
        if entry is None:
            return
        now = time.time() if now is None else float(now)
        status_code = int(status_code)
        with self._lock:
            if entry.get("done"):
                return
            entry["done"] = True
            entry["success"] = 200 <= status_code < 300
            entry["status_code"] = status_code
            entry["payload"] = copy.deepcopy(payload)
            entry["completed_at"] = now
            if not entry["success"]:
                current = self._entries.get(entry.get("key"))
                if current is entry:
                    self._entries.pop(entry.get("key"), None)
            entry["event"].set()

    def response(self, entry):
        if entry is None:
            return None
        with self._lock:
            if not entry.get("done"):
                return None
            return (
                int(entry.get("status_code") or 500),
                copy.deepcopy(entry.get("payload")),
            )

    def status(self, *, now=None):
        now = time.time() if now is None else float(now)
        with self._lock:
            self._purge_locked(now)
            entries = list(self._entries.values())
            return {
                "enabled": self.enabled,
                "active": sum(1 for entry in entries if not entry.get("done")),
                "replayable": sum(1 for entry in entries if entry.get("done")),
                "joined_total": self._joined_total,
                "replayed_total": self._replayed_total,
                "replay_grace_seconds": self.replay_grace_seconds,
                "disconnect_grace_seconds": self.disconnect_grace_seconds,
            }
TOOL_DEFAULT_TEMPERATURE = float(
    os.environ.get("MLX_M3_TOOL_DEFAULT_TEMPERATURE", "0") or "0"
)
TOOL_DEFAULT_TOP_P = float(os.environ.get("MLX_M3_TOOL_DEFAULT_TOP_P", "1.0") or "1.0")
TOOL_DEFAULT_TOP_K = int(os.environ.get("MLX_M3_TOOL_DEFAULT_TOP_K", "0") or "0")
TOOL_DEFAULT_MIN_P = float(os.environ.get("MLX_M3_TOOL_DEFAULT_MIN_P", "0.0") or "0.0")
# Approximate sparse-block reuse is an excellent prose decode optimization,
# but a live OpenCode A/B showed that it can drift inside exact structured
# arguments (tool names, absolute paths, and required fields). Tool requests
# therefore default to exact per-token block selection while ordinary chat
# keeps the independently tuned runtime value (48 on the reference cluster).
TOOL_DECODE_TOPK_REUSE_TOKENS = max(
    0,
    min(
        64,
        int(os.environ.get("MLX_M3_TOOL_DECODE_TOPK_REUSE_TOKENS", "0") or "0"),
    ),
)
TOOL_COMPAT_OVERLAY = os.environ.get(
    "MLX_M3_TOOL_COMPAT_OVERLAY", "0"
).strip().lower() in {"1", "true", "yes", "on"}
# Native-first mode still needs one bounded continuation when an explicit
# action turn ends after private reasoning without emitting a call. This is
# not a parser overlay: the retry uses the same model template, submitted tool
# schema, and native mlx-vlm parser. Successful native turns never enter it.
NATIVE_TOOL_ACTION_RETRY_ATTEMPTS = max(
    0,
    int(os.environ.get("MLX_M3_NATIVE_TOOL_ACTION_RETRY_ATTEMPTS", "1") or "0"),
)
# Retrying an unusable native tool turn against a large live KV can briefly
# retain both generations' graphs.  On the 128GB rank that exhausted wired
# memory after a 92K-token ZCode turn even though the steady-state KV fit.
# Release only the active RAM KV before this rare recovery; the validated SSD
# checkpoint remains available and is restored by the normal cache path.
NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS = max(
    0,
    int(
        os.environ.get(
            "MLX_M3_NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS",
            "65536",
        )
        or "0"
    ),
)
TOOL_PARSE_DIAGNOSTICS = os.environ.get(
    "MLX_M3_TOOL_PARSE_DIAGNOSTICS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
TOOL_THINKING_MODE = os.environ.get("MLX_M3_TOOL_THINKING_MODE", "request").strip().lower()
if TOOL_THINKING_MODE not in {"request", "enabled", "disabled", "adaptive"}:
    logger.warning(
        "invalid MLX_M3_TOOL_THINKING_MODE=%r; using request",
        TOOL_THINKING_MODE,
    )
    TOOL_THINKING_MODE = "request"
TOOL_SYSTEM_HINT_ENABLED = os.environ.get(
    "MLX_M3_TOOL_SYSTEM_HINT", "0"
).strip().lower() in {"1", "true", "yes", "on"}
INJECT_DATE_CONTEXT = os.environ.get(
    "MLX_M3_INJECT_DATE_CONTEXT", "1"
).strip().lower() in {"1", "true", "yes", "on"}
TOOL_SYSTEM_HINT_TEXT = os.environ.get(
    "MLX_M3_TOOL_SYSTEM_HINT_TEXT",
    (
        "When tools are available and a tool is needed, emit the actual tool "
        "call using the provided tool-call format. Do not merely say that you "
        "will search, inspect, run, or use a tool. After tool results are "
        "available, answer from those results as soon as you have enough "
        "evidence. Do not repeat the same file-listing or file-reading command "
        "unless the previous result clearly failed or new information is "
        "required. If tool results already answer the user's request, provide "
        "the final answer now instead of calling another tool just to list "
        "files or re-read already seen context. Only call tool names that are "
        "present in the current request's tool list."
    ),
)
TOOL_LOOP_STEER_MAX_TOOL_ONLY_TURNS = int(
    os.environ.get("MLX_M3_TOOL_LOOP_STEER_MAX_TOOL_ONLY_TURNS", "0") or "0"
)
TOOL_LOOP_STEER_MAX_REPEATED_TOOL = int(
    os.environ.get("MLX_M3_TOOL_LOOP_STEER_MAX_REPEATED_TOOL", "0") or "0"
)
TOOL_LOOP_STEER_MAX_REPEATED_COMMANDS = int(
    os.environ.get("MLX_M3_TOOL_LOOP_STEER_MAX_REPEATED_COMMANDS", "0") or "0"
)
TOOL_LOOP_FORCE_FINAL_AFTER = int(
    os.environ.get("MLX_M3_TOOL_LOOP_FORCE_FINAL_AFTER", "0") or "0"
)
TOOL_LOOP_FORCE_FINAL_REPEATED_COMMANDS = int(
    os.environ.get(
        "MLX_M3_TOOL_LOOP_FORCE_FINAL_REPEATED_COMMANDS",
        "0",
    )
    or "0"
)
# An exact command can be legitimate more than once (for example, rerunning a
# test after an edit).  The same command returning the same result repeatedly
# is a much stronger loop signal, so stop that pattern earlier without
# penalizing productive command reuse.
TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS = int(
    os.environ.get(
        "MLX_M3_TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS",
        "4",
    )
    or "0"
)
TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT = int(
    os.environ.get(
        "MLX_M3_TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT",
        "0",
    )
    or "0"
)
TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_NAMES = {
    name.strip()
    for name in os.environ.get(
        "MLX_M3_TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_NAMES",
        "",
    ).split(",")
    if name.strip()
}
TOOL_LOOP_FILTER_CONTROL_TOOLS = {
    name.strip()
    for name in os.environ.get(
        "MLX_M3_TOOL_LOOP_FILTER_CONTROL_TOOLS",
        "update_plan,create_goal,update_goal",
    ).split(",")
    if name.strip()
}
TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS = {
    name.strip()
    for name in os.environ.get(
        "MLX_M3_TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS",
        "",
    ).split(",")
    if name.strip()
}
TOOL_UNUSABLE_RETRY_ATTEMPTS = int(
    os.environ.get("MLX_M3_TOOL_UNUSABLE_RETRY_ATTEMPTS", "3") or "0"
)
TOOL_UNUSABLE_RETRY_TEMPERATURES = [
    float(part)
    for part in os.environ.get(
        "MLX_M3_TOOL_UNUSABLE_RETRY_TEMPERATURES", "0.2,0.3"
    ).split(",")
    if part.strip()
] or [0.2, 0.3]
# Hard cap on a retry's decode budget. A valid tool call stops decode early,
# so this only bounds pathological thinking rambles; long hot-suffix decodes
# are also the JACCL wedge-prone regime, so keep retries short.
TOOL_UNUSABLE_RETRY_MAX_TOKENS = int(
    os.environ.get("MLX_M3_TOOL_UNUSABLE_RETRY_MAX_TOKENS", "16384") or "0"
)
# A tool-required turn that has not even STARTED a call by this many decode
# tokens is drafting work in prose rather than executing it. Stop that attempt
# through the synchronized EOS path and let the bounded No-Think retry emit
# the call. Once a real marker begins this guard permanently disengages, so a
# large Write/Edit payload is never clipped. 0 disables.
TOOL_NO_CALL_TOKEN_BUDGET = int(
    os.environ.get("MLX_M3_TOOL_NO_CALL_TOKEN_BUDGET", "0") or "0"
)
# Later rounds of an action task may legitimately finish in prose. Give them
# more room than the mandatory first call, but do not let a fresh code draft
# run for thousands of tokens before finally emitting an empty marker.
TOOL_ACTION_NO_CALL_TOKEN_BUDGET = int(
    os.environ.get("MLX_M3_TOOL_ACTION_NO_CALL_TOKEN_BUDGET", "0") or "0"
)
# Hidden schema-repair retries already have focused recovery instructions.
# This is the no-thinking budget; thinking action retries inherit the larger
# action budget below so legitimate planning is not clipped. Every guard
# disengages as soon as a call starts, so valid Edit/Write payloads stay whole.
TOOL_RETRY_NO_CALL_TOKEN_BUDGET = int(
    os.environ.get("MLX_M3_TOOL_RETRY_NO_CALL_TOKEN_BUDGET", "384") or "0"
)
# A completed tool block normally emits EOS immediately. Closing the outer
# consumer first sends m3_batch_cancel into its GeneratorExit drain path; one
# rank occasionally never reaches that boundary and leaves rank0 Metal wired.
# Keep consuming the already synchronized batch until natural EOS instead.
BATCH_TOOL_NATURAL_DRAIN = os.environ.get(
    "MLX_M3_BATCH_TOOL_NATURAL_DRAIN", "1"
).strip().lower() in {"1", "true", "yes", "on"}
# Permit the final unusable-tool retry to use NO-THINK. Earlier retries retain
# the request template and therefore its long KV prefix; only the last resort
# pays for a template switch and cold prefill. 0 keeps every retry in the
# original mode.
TOOL_RETRY_NO_THINK = os.environ.get(
    "MLX_M3_TOOL_RETRY_NO_THINK", "1"
).strip().lower() in {"1", "true", "yes", "on"}
# Switching a thinking request to the no-thinking template invalidates its
# tokenized prefix. Keep that last-resort path for short turns only; rebuilding
# a large hot agent transcript inside a hidden retry can outlive the client's
# stream-idle budget even though both ranks remain healthy.
TOOL_RETRY_NO_THINK_MAX_PROMPT_TOKENS = int(
    os.environ.get(
        "MLX_M3_TOOL_RETRY_NO_THINK_MAX_PROMPT_TOKENS",
        "16384",
    ) or "0"
)
TOOL_WRITE_CHUNK_MAX_CHARS = int(
    os.environ.get("MLX_M3_TOOL_WRITE_CHUNK_MAX_CHARS", "0") or "0"
)
# Native Write/Edit calls are atomic and already bounded by max_tokens plus the
# incomplete-call stop. Leave their schemas and decode runway untouched by
# default; positive values opt into the legacy scaffold policy for clients that
# explicitly prefer small staged writes.
_DEFAULT_TOOL_WRITE_CHUNK_TARGET_CHARS = (
    min(49152, TOOL_WRITE_CHUNK_MAX_CHARS)
    if TOOL_WRITE_CHUNK_MAX_CHARS > 0 else 0
)
TOOL_WRITE_CHUNK_TARGET_CHARS = int(
    os.environ.get(
        "MLX_M3_TOOL_WRITE_CHUNK_TARGET_CHARS",
        str(_DEFAULT_TOOL_WRITE_CHUNK_TARGET_CHARS),
    ) or "0"
)
if TOOL_WRITE_CHUNK_MAX_CHARS > 0:
    TOOL_WRITE_CHUNK_TARGET_CHARS = max(
        1,
        min(TOOL_WRITE_CHUNK_TARGET_CHARS, TOOL_WRITE_CHUNK_MAX_CHARS),
    )
else:
    TOOL_WRITE_CHUNK_TARGET_CHARS = 0
# A real tool invocation that opens but never closes is invisible to clients
# while it burns decode tokens. Keep this far above ordinary tool payloads and
# route the stop through synchronized EOS so the existing bounded retry can
# regenerate the atomic call. This does not cap normal answers or completed
# calls. 0 disables.
TOOL_INCOMPLETE_CALL_TOKEN_BUDGET = int(
    os.environ.get("MLX_M3_TOOL_INCOMPLETE_CALL_TOKEN_BUDGET", "32768") or "0"
)
# Some MiniMax tool turns finish a complete native call, then sample control
# tokens that the detokenizer buffers without producing text. A consecutive
# empty-token budget may end only that post-completion tail. An OPEN tool call
# can legitimately contain long runs of buffered structural tokens, so it is
# governed by the much larger incomplete-call budget instead.
TOOL_DETOKENIZER_SILENT_TOKEN_BUDGET = int(
    os.environ.get(
        "MLX_M3_TOOL_DETOKENIZER_SILENT_TOKEN_BUDGET",
        "64",
    ) or "0"
)
DEFAULT_REPETITION_PENALTY = float(
    os.environ.get("MLX_M3_DEFAULT_REPETITION_PENALTY", "0") or "0"
)
# Tool turns only: a mild penalty breaks the quantized model's repetition
# spirals ("keep it nice" x100s) that burn whole tool budgets without ever
# forming a call. Chat sampling is unaffected.
TOOL_DEFAULT_REPETITION_PENALTY = float(
    os.environ.get("MLX_M3_TOOL_DEFAULT_REPETITION_PENALTY", "0") or "0"
)
# Thinking turns: a reasoning loop is a repetition spiral inside <mm:think>
# (2026-07-09 hermes: 12.5k tokens of the same reasoning, never closing).
# The same mild penalty that fixes tool spirals attacks the CAUSE, not just
# the runaway-guard aftermath; plus a small temperature floor so a
# near-greedy path (the classic 4-bit-quant loop attractor) has enough
# entropy to escape. Both apply ONLY when thinking is enabled and the client
# didn't set the value. 0 / negative disables each.
THINKING_DEFAULT_REPETITION_PENALTY = float(
    os.environ.get("MLX_M3_THINKING_DEFAULT_REPETITION_PENALTY", "1.05") or "1.05"
)
THINKING_MIN_TEMPERATURE = float(
    os.environ.get("MLX_M3_THINKING_MIN_TEMPERATURE", "0.5") or "0.5"
)
# Tools to hide from the model (comma-separated). MiniMax-4bit reliably
# invents its own patch dialects for apply_patch while its exec_command
# shell writes are dependable; hiding the patch tool routes file work
# through the shell instead of a doomed format.
TOOL_HIDE_NAMES = {
    name.strip()
    for name in os.environ.get("MLX_M3_TOOL_HIDE_NAMES", "").split(",")
    if name.strip()
}
DEFAULT_PRESENCE_PENALTY = float(
    os.environ.get("MLX_M3_DEFAULT_PRESENCE_PENALTY", "0") or "0"
)
DEFAULT_FREQUENCY_PENALTY = float(
    os.environ.get("MLX_M3_DEFAULT_FREQUENCY_PENALTY", "0") or "0"
)
IMAGE_DEFAULT_MAX_TOKENS = int(os.environ.get("MLX_M3_IMAGE_DEFAULT_MAX_TOKENS", "768"))
IMAGE_MAX_TOKENS = int(os.environ.get("MLX_M3_IMAGE_MAX_TOKENS", "0"))

def _rank_scoped_float_env(base_name, default="0"):
    rank = os.environ.get("MLX_RANK", "").strip()
    if rank:
        scoped = os.environ.get(f"{base_name}_RANK{rank}")
        if scoped not in (None, ""):
            return float(scoped)
    return float(os.environ.get(base_name, default) or default)


WIRED_LIMIT_GB = _rank_scoped_float_env("MLX_M3_WIRED_LIMIT_GB", "0")
MEMORY_LIMIT_GB = float(os.environ.get("MLX_M3_MEMORY_LIMIT_GB", "0") or "0")
CACHE_LIMIT_GB = float(os.environ.get("MLX_M3_CACHE_LIMIT_GB", "4") or "4")
STREAM_MODE = os.environ.get("MLX_M3_STREAM_MODE", "buffered").strip().lower()
if STREAM_MODE not in {"buffered"}:
    logger.warning(
        "MLX_M3_STREAM_MODE=%r is not supported in this distributed gateway; "
        "using buffered SSE",
        STREAM_MODE,
    )
    STREAM_MODE = "buffered"
MAX_CONCURRENT_REQUESTS = max(
    1, int(os.environ.get("MLX_M3_MAX_CONCURRENT_REQUESTS", "1") or "1")
)
# Distributed MiniMax-M3 generation is intentionally single-flight for now:
# both ranks share one model/prompt-cache state and must stay in lockstep.
EFFECTIVE_MAX_CONCURRENT_REQUESTS = 1
SSE_KEEPALIVE_SECONDS = float(os.environ.get("MLX_M3_SSE_KEEPALIVE_SECONDS", "5"))
# Empty OpenAI deltas keep the transport alive, but some agent clients do not
# count them as stream activity. During a buffered tool turn, periodically emit
# an explicit reasoning-channel status so a live recovery cannot be mistaken
# for a dead stream. 0 disables the client-visible pulse.
TOOL_STREAM_PROGRESS_SECONDS = float(
    os.environ.get("MLX_M3_TOOL_STREAM_PROGRESS_SECONDS", "45") or "0"
)
SSE_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


def _sse_keepalive_comment() -> str:
    """Return a protocol-level SSE heartbeat ignored by event consumers."""
    return ": keepalive\n\n"
# Native mode streams ordinary reasoning/content while retaining the short
# marker holdback below. Transaction-wide buffering remains opt-in for operators
# that deliberately enable the compatibility retry layer.
TOOL_STREAM_BUFFER_ALL = os.environ.get(
    "MLX_M3_TOOL_STREAM_BUFFER_ALL", "0"
).strip().lower() in {"1", "true", "yes", "on"}
# Stream ordinary visible content on native tool-capable turns. The marker
# holdback below prevents MiniMax tool XML from reaching the client; full-turn
# buffering remains available as an explicit compatibility mode.
TOOL_STREAM_CONTENT = os.environ.get("MLX_M3_TOOL_STREAM_CONTENT", "1") == "1"
TOOL_STREAM_HOLDBACK_CHARS = max(
    8, int(os.environ.get("MLX_M3_TOOL_STREAM_HOLDBACK_CHARS", "24") or "24")
)
# Synthesizing a `justification` argument the model never wrote puts
# server-authored text into clients' audit/approval flows. Off unless an
# operator explicitly wants the convenience (2026-07-06 audit).
TOOL_SYNTH_JUSTIFICATION = os.environ.get(
    "MLX_M3_TOOL_SYNTH_JUSTIFICATION", "0"
) == "1"
RANK1_IDLE_SLEEP_SECONDS = max(
    0.0, float(os.environ.get("MLX_M3_RANK1_IDLE_SLEEP_SECONDS", "0.01") or "0.01")
)
DEFAULT_SEED = int(os.environ.get("MLX_M3_DEFAULT_SEED", "1"))
TOOL_DEFAULT_SEED = int(os.environ.get("MLX_M3_TOOL_DEFAULT_SEED", str(DEFAULT_SEED)))
DECODE_EVAL_EVERY = int(os.environ.get("MLX_M3_DECODE_EVAL_EVERY", "0"))
DECODE_EVAL_AFTER_TOKENS = int(os.environ.get("MLX_M3_DECODE_EVAL_AFTER_TOKENS", "512"))
DECODE_EVAL_AFTER_EVERY = int(os.environ.get("MLX_M3_DECODE_EVAL_AFTER_EVERY", "1"))
THINKING_DECODE_EVAL_EVERY = int(os.environ.get("MLX_M3_THINKING_DECODE_EVAL_EVERY", "1"))
LONG_CONTEXT_DECODE_EVAL_TOKENS = int(
    os.environ.get("MLX_M3_LONG_CONTEXT_DECODE_EVAL_TOKENS", "24576") or "24576"
)
LONG_CONTEXT_DECODE_EVAL_EVERY = int(
    os.environ.get("MLX_M3_LONG_CONTEXT_DECODE_EVAL_EVERY", "3") or "3"
)
ADAPTIVE_LONG_CONTEXT_DECODE_EVAL = os.environ.get(
    "MLX_M3_ADAPTIVE_LONG_CONTEXT_DECODE_EVAL", "0"
).strip().lower() in {"1", "true", "yes", "on"}
MID_CONTEXT_DECODE_EVAL_TOKENS = int(
    os.environ.get("MLX_M3_MID_CONTEXT_DECODE_EVAL_TOKENS", "24576") or "24576"
)
MID_CONTEXT_DECODE_EVAL_EVERY = int(
    os.environ.get("MLX_M3_MID_CONTEXT_DECODE_EVAL_EVERY", "4") or "4"
)
HIGH_CONTEXT_DECODE_EVAL_TOKENS = int(
    os.environ.get("MLX_M3_HIGH_CONTEXT_DECODE_EVAL_TOKENS", "98304") or "98304"
)
HIGH_CONTEXT_DECODE_EVAL_EVERY = int(
    os.environ.get("MLX_M3_HIGH_CONTEXT_DECODE_EVAL_EVERY", "3") or "3"
)
SPARSE_TOPK_BLOCKS_OVERRIDE = int(
    os.environ.get("MLX_M3_SPARSE_TOPK_BLOCKS_OVERRIDE", "0") or "0"
)
ADAPTIVE_PREFILL_STEP_TOKENS = int(
    os.environ.get("MLX_M3_ADAPTIVE_PREFILL_STEP_TOKENS", "262144") or "262144"
)
ADAPTIVE_PREFILL_STEP_SIZE = int(
    os.environ.get("MLX_M3_ADAPTIVE_PREFILL_STEP_SIZE", "2048") or "2048"
)
ALLOW_UNSAFE_RUNTIME_TUNING = os.environ.get(
    "MLX_M3_ALLOW_UNSAFE_RUNTIME_TUNING", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_MIN_SUFFIX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_MIN_SUFFIX_TOKENS", "0") or "0"
)
PROMPT_CACHE_FAST_MIN_SUFFIX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_FAST_MIN_SUFFIX_TOKENS", "1") or "1"
)
PROMPT_CACHE_FAST_THINKING_MIN_SUFFIX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_FAST_THINKING_MIN_SUFFIX_TOKENS", "64") or "64"
)
PROMPT_CACHE_REUSE_BUCKET_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_REUSE_BUCKET_TOKENS", "0") or "0"
)
_runtime_tuning_lock = threading.RLock()
_runtime_tuning = {
    "prefill_step_size": PREFILL_STEP_SIZE,
    "long_context_decode_eval_tokens": LONG_CONTEXT_DECODE_EVAL_TOKENS,
    "long_context_decode_eval_every": LONG_CONTEXT_DECODE_EVAL_EVERY,
    "thinking_decode_eval_every": THINKING_DECODE_EVAL_EVERY,
    "adaptive_long_context_decode_eval": int(ADAPTIVE_LONG_CONTEXT_DECODE_EVAL),
    "mid_context_decode_eval_tokens": MID_CONTEXT_DECODE_EVAL_TOKENS,
    "mid_context_decode_eval_every": MID_CONTEXT_DECODE_EVAL_EVERY,
    "high_context_decode_eval_tokens": HIGH_CONTEXT_DECODE_EVAL_TOKENS,
    "high_context_decode_eval_every": HIGH_CONTEXT_DECODE_EVAL_EVERY,
    "prompt_cache_min_suffix_tokens": PROMPT_CACHE_MIN_SUFFIX_TOKENS,
    "prompt_cache_reuse_bucket_tokens": PROMPT_CACHE_REUSE_BUCKET_TOKENS,
    "visible_transcript_prewarm_min_generated": int(
        os.environ.get("MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED", "16") or "16"
    ),
    "sparse_topk_blocks": SPARSE_TOPK_BLOCKS_OVERRIDE,
    "decode_topk_reuse_tokens": int(
        os.environ.get("MLX_M3_DECODE_TOPK_REUSE_TOKENS", "0") or "0"
    ),
    "compact_decode_sort_topk": int(
        os.environ.get("MLX_M3_COMPACT_DECODE_SORT_TOPK", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    ),
}
_applied_decode_runtime = {
    "decode_topk_reuse_tokens": None,
    "compact_decode_sort_topk": None,
}


def _runtime_tuning_status():
    with _runtime_tuning_lock:
        return dict(_runtime_tuning)


def _runtime_prefill_step_size(prompt_tokens=None, suffix_tokens=None):
    with _runtime_tuning_lock:
        base = int(_runtime_tuning.get("prefill_step_size") or PREFILL_STEP_SIZE)
    if (
        ADAPTIVE_PREFILL_STEP_TOKENS > 0
        and ADAPTIVE_PREFILL_STEP_SIZE > 0
        and prompt_tokens is not None
        and int(prompt_tokens) >= ADAPTIVE_PREFILL_STEP_TOKENS
        and base > ADAPTIVE_PREFILL_STEP_SIZE
    ):
        return ADAPTIVE_PREFILL_STEP_SIZE
    return base


def _runtime_visible_transcript_prewarm_min_generated():
    with _runtime_tuning_lock:
        value = _runtime_tuning.get("visible_transcript_prewarm_min_generated")
        if value is None:
            return int(VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED)
        return int(value)


def _runtime_prompt_cache_min_suffix_tokens():
    with _runtime_tuning_lock:
        value = _runtime_tuning.get("prompt_cache_min_suffix_tokens")
        if value is None:
            return int(PROMPT_CACHE_MIN_SUFFIX_TOKENS)
        return int(value)


def _effective_prompt_cache_min_suffix_tokens(
    thinking_mode,
    session_source,
    *,
    session_id=None,
    cached_session_id=None,
    miss_reason=None,
):
    """Pick the min suffix floor for the current cache reuse path.

    A larger floor protects ambiguous visible-thinking reuse where the cached
    assistant reasoning boundary may differ across turns. Same-session reuse
    is rank-stable and should keep the fast suffix floor; otherwise short
    OpenWebUI thinking turns spend seconds re-prefilling tokens already in KV.
    """
    configured = _runtime_prompt_cache_min_suffix_tokens()
    if configured <= PROMPT_CACHE_FAST_MIN_SUFFIX_TOKENS:
        return configured
    visible_thinking = (
        _enable_thinking_for_generation(thinking_mode)
        and PROMPT_CACHE_THINKING_MODE == "visible"
    )
    if visible_thinking:
        same_session = bool(
            session_id
            and cached_session_id
            and session_id == cached_session_id
        )
        exact_or_same_prompt = miss_reason in {
            None,
            "exact_prior_transcript",
            "exact_prompt",
        }
        if same_session or exact_or_same_prompt:
            return max(
                0,
                min(configured, PROMPT_CACHE_FAST_THINKING_MIN_SUFFIX_TOKENS),
            )
        return configured
    return max(0, min(configured, PROMPT_CACHE_FAST_MIN_SUFFIX_TOKENS))


def _runtime_prompt_cache_reuse_bucket_tokens():
    with _runtime_tuning_lock:
        value = _runtime_tuning.get("prompt_cache_reuse_bucket_tokens")
        if value is None:
            return int(PROMPT_CACHE_REUSE_BUCKET_TOKENS)
        return int(value)


def _set_runtime_tuning(values, clamped_out=None):
    allowed = {
        "prefill_step_size": (128, 16384),
        "long_context_decode_eval_tokens": (0, 1_000_000),
        "long_context_decode_eval_every": (
            0,
            16 if ALLOW_UNSAFE_RUNTIME_TUNING else max(3, LONG_CONTEXT_DECODE_EVAL_EVERY),
        ),
        "thinking_decode_eval_every": (0, 16),
        "adaptive_long_context_decode_eval": (0, 1),
        "mid_context_decode_eval_tokens": (0, 1_000_000),
        "mid_context_decode_eval_every": (0, 16),
        "high_context_decode_eval_tokens": (0, 1_000_000),
        "high_context_decode_eval_every": (0, 16),
        "prompt_cache_min_suffix_tokens": (0, 4096),
        "prompt_cache_reuse_bucket_tokens": (0, 4096),
        "visible_transcript_prewarm_min_generated": (0, 16_384),
        "sparse_topk_blocks": (0, 64),
        "decode_topk_reuse_tokens": (0, 64),
        "compact_decode_sort_topk": (0, 1),
    }
    # Storage caps: same pattern, except out-of-range integers CLAMP to the
    # guard rails (reported via clamped_out) instead of erroring; garbage
    # still raises -> 400. See the _STORAGE_TUNING_KEYS notes.
    allowed.update(_STORAGE_TUNING_RANGES)
    capture = _capture_module()
    for key in ("capture_max_request_bytes", "capture_max_total_bytes"):
        if key in values and capture is None:
            raise ValueError(
                f"{key} requires the m3_capture module, which is not deployed"
            )
    changed = {}
    with _runtime_tuning_lock:
        proposed = dict(_runtime_tuning)
        for key, value in values.items():
            if key not in allowed:
                continue
            lo, hi = allowed[key]
            try:
                intval = int(value)
            except Exception as exc:
                raise ValueError(f"{key} must be an integer") from exc
            if key in _STORAGE_TUNING_KEYS:
                clamped = max(lo, min(hi, intval))
                if clamped != intval and clamped_out is not None:
                    clamped_out[key] = {"requested": intval, "applied": clamped}
                intval = clamped
            elif intval < lo or intval > hi:
                suffix = ""
                if key == "long_context_decode_eval_every" and not ALLOW_UNSAFE_RUNTIME_TUNING:
                    suffix = (
                        "; higher values reproduced a 107k decode stall/orphan path. "
                        "Set MLX_M3_ALLOW_UNSAFE_RUNTIME_TUNING=1 only for controlled A/B"
                    )
                raise ValueError(f"{key} must be between {lo} and {hi}{suffix}")
            if (
                key == "prefill_step_size"
                and not ALLOW_UNSAFE_RUNTIME_TUNING
                and intval < 4096
            ):
                raise ValueError(
                    "prefill_step_size below 4096 requires "
                    "MLX_M3_ALLOW_UNSAFE_RUNTIME_TUNING=1; "
                    "3072 reproduced a short-decode stall/orphan path"
                )
            proposed[key] = intval
            if (
                key == "high_context_decode_eval_every"
                and not ALLOW_UNSAFE_RUNTIME_TUNING
                and intval > LONG_CONTEXT_DECODE_EVAL_EVERY
            ):
                raise ValueError(
                    "high_context_decode_eval_every above the safe launch cadence "
                    "requires MLX_M3_ALLOW_UNSAFE_RUNTIME_TUNING=1"
                )
            if _runtime_tuning.get(key) != intval:
                changed[key] = intval
        if (
            not ALLOW_UNSAFE_RUNTIME_TUNING
            and int(proposed.get("adaptive_long_context_decode_eval") or 0)
            and int(proposed.get("mid_context_decode_eval_every") or 0)
            > LONG_CONTEXT_DECODE_EVAL_EVERY
        ):
            raise ValueError(
                "adaptive mid-context cadence above the safe launch cadence "
                "requires MLX_M3_ALLOW_UNSAFE_RUNTIME_TUNING=1; "
                "mid cadence 4 reproduced a 33k cached decode stall/orphan path"
            )
        for key, intval in changed.items():
            _runtime_tuning[key] = intval
    # Push capture caps into m3_capture so its per-flush/finalize checks see
    # them (capture can only be non-None here: absent module raised above).
    if capture is not None and (
        "capture_max_request_bytes" in changed or "capture_max_total_bytes" in changed
    ):
        capture.set_limits(
            max_request_bytes=changed.get("capture_max_request_bytes"),
            max_total_bytes=changed.get("capture_max_total_bytes"),
        )
    return changed


def _apply_sparse_topk_to_model(model, value):
    """Update loaded MiniMax sparse attention modules for controlled A/B tests."""
    try:
        intval = int(value or 0)
    except Exception:
        intval = 0
    if intval <= 0:
        return {"requested": intval, "updated": 0, "skipped": "non_positive"}

    seen = set()
    updated = 0

    def visit(obj, depth=0):
        nonlocal updated
        if obj is None or depth > 12:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if hasattr(obj, "sparse_topk_blocks"):
            try:
                old = int(getattr(obj, "sparse_topk_blocks"))
                if old != intval:
                    setattr(obj, "sparse_topk_blocks", intval)
                    updated += 1
            except Exception:
                pass
        if isinstance(obj, dict):
            for child in obj.values():
                visit(child, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for child in obj:
                visit(child, depth + 1)
            return
        attrs = getattr(obj, "__dict__", None)
        if not attrs:
            return
        for child in attrs.values():
            if isinstance(child, (str, bytes, int, float, bool)):
                continue
            visit(child, depth + 1)

    visit(model)
    return {"requested": intval, "updated": updated}


def _clear_decode_topk_caches(model):
    seen = set()
    cleared = 0

    def visit(obj, depth=0):
        nonlocal cleared
        if obj is None or depth > 12:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        for attr in ("_m3_decode_topk_cache", "_minimax_m3_decode_topk_cache"):
            if not hasattr(obj, attr):
                continue
            try:
                delattr(obj, attr)
                cleared += 1
            except Exception:
                pass
        if isinstance(obj, dict):
            for child in obj.values():
                visit(child, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for child in obj:
                visit(child, depth + 1)
            return
        attrs = getattr(obj, "__dict__", None)
        if not attrs:
            return
        for child in attrs.values():
            if isinstance(child, (str, bytes, int, float, bool)):
                continue
            visit(child, depth + 1)

    visit(model)
    return cleared


def _apply_decode_runtime_to_model(model, runtime):
    result = {}
    minimax_language = None
    try:
        from mlx_vlm.models.minimax_m3_vl import language as minimax_language
    except Exception as exc:
        import sys

        for name, module in sys.modules.items():
            if name.endswith("minimax_m3_vl.language"):
                minimax_language = module
                break
        if minimax_language is None:
            return {"skipped": repr(exc)}
    if "decode_topk_reuse_tokens" in runtime:
        value = int(runtime.get("decode_topk_reuse_tokens") or 0)
        current = getattr(minimax_language, "_MSA_DECODE_TOPK_REUSE_TOKENS", None)
        applied = minimax_language.set_decode_topk_reuse_tokens(value)
        previous = _applied_decode_runtime.get("decode_topk_reuse_tokens")
        baseline = previous if previous is not None else current
        changed = baseline != applied
        cleared = _clear_decode_topk_caches(model) if changed else 0
        _applied_decode_runtime["decode_topk_reuse_tokens"] = applied
        result["decode_topk_reuse_tokens"] = {
            "requested": value,
            "previous": baseline,
            "applied": applied,
            "changed": changed,
            "cleared_layer_caches": cleared,
        }
    if "compact_decode_sort_topk" in runtime:
        value = int(runtime.get("compact_decode_sort_topk") or 0)
        current = getattr(minimax_language, "_MSA_COMPACT_DECODE_SORT_TOPK", None)
        applied = minimax_language.set_compact_decode_sort_topk(value)
        previous = _applied_decode_runtime.get("compact_decode_sort_topk")
        baseline = previous if previous is not None else current
        _applied_decode_runtime["compact_decode_sort_topk"] = bool(applied)
        result["compact_decode_sort_topk"] = {
            "requested": value,
            "previous": bool(baseline) if baseline is not None else None,
            "applied": bool(applied),
            "changed": (
                bool(baseline) != bool(applied)
                if baseline is not None else False
            ),
        }
    return result


def _apply_runtime_model_tuning(model):
    runtime = _runtime_tuning_status()
    sparse_result = _apply_sparse_topk_to_model(
        model, runtime.get("sparse_topk_blocks")
    )
    decode_result = _apply_decode_runtime_to_model(model, runtime)
    if int(sparse_result.get("updated") or 0) > 0:
        decode_result["cleared_after_sparse_topk_change"] = _clear_decode_topk_caches(
            model
        )
    return {
        "sparse_topk_blocks": sparse_result,
        "decode_runtime": decode_result,
    }


def _decode_topk_language_module():
    """Return the loaded ThunderMLX MiniMax language module."""
    import sys

    candidates = []
    try:
        from mlx_vlm.models.minimax_m3_vl import language as minimax_language

        candidates.append(minimax_language)
    except Exception:
        pass
    candidates.extend(
        module
        for name, module in tuple(sys.modules.items())
        if name.endswith("minimax_m3_vl.language") and module is not None
    )
    for module in candidates:
        if callable(getattr(module, "set_decode_topk_reuse_tokens", None)):
            return module
    raise RuntimeError("patched MiniMax decode runtime module is not loaded")


def _begin_request_decode_topk_reuse(tools, rank):
    """Apply the exact structured-decode profile for one tool request.

    The server is single-flight and both ranks receive the same advertised
    tool list, so this process-local override is symmetric. The generation
    epoch invalidates layer-local selections between requests; restoring the
    configured chat value in ``finally`` cannot expose a stale selection.
    """
    if not tools:
        return None
    try:
        minimax_language = _decode_topk_language_module()

        previous = int(
            getattr(
                minimax_language,
                "_MSA_DECODE_TOPK_REUSE_TOKENS",
                _runtime_tuning_status().get("decode_topk_reuse_tokens") or 0,
            )
            or 0
        )
        target = int(TOOL_DECODE_TOPK_REUSE_TOKENS)
        if previous != target:
            minimax_language.set_decode_topk_reuse_tokens(target)
            logger.info(
                "rank %s: tool decode top-k reuse override %d -> %d",
                rank,
                previous,
                target,
            )
        return minimax_language, previous
    except Exception as exc:
        logger.warning(
            "rank %s: tool decode top-k reuse override unavailable: %s",
            rank,
            exc,
        )
        return None


def _restore_request_decode_topk_reuse(state, rank):
    if state is None:
        return
    minimax_language, previous = state
    try:
        current = int(
            getattr(
                minimax_language,
                "_MSA_DECODE_TOPK_REUSE_TOKENS",
                previous,
            )
            or 0
        )
        if current != previous:
            minimax_language.set_decode_topk_reuse_tokens(previous)
            logger.info(
                "rank %s: restored chat decode top-k reuse %d",
                rank,
                previous,
            )
    except Exception as exc:
        logger.warning(
            "rank %s: failed to restore decode top-k reuse to %d: %s",
            rank,
            previous,
            exc,
        )


REFRESH_GENERATION_STREAM = os.environ.get(
    "MLX_M3_REFRESH_GENERATION_STREAM", "0"
).strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_THINKING_BUDGET = int(os.environ.get("MLX_M3_THINKING_BUDGET", "0"))
MIN_THINKING_BUDGET = int(os.environ.get("MLX_M3_MIN_THINKING_BUDGET", "16"))
ALLOW_THINKING_BUDGET = os.environ.get(
    "MLX_M3_ALLOW_THINKING_BUDGET", "0"
).strip().lower() in {"1", "true", "yes", "on"}
VALID_THINKING_MODES = {"enabled", "disabled", "adaptive"}
DEFAULT_THINKING_MODE = os.environ.get("MLX_M3_THINKING_MODE", "enabled").strip().lower()
if DEFAULT_THINKING_MODE not in VALID_THINKING_MODES:
    logger.warning(
        "invalid MLX_M3_THINKING_MODE=%r; falling back to enabled",
        DEFAULT_THINKING_MODE,
    )
    DEFAULT_THINKING_MODE = "enabled"
GEN_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "repetition_penalty",
    "repetition_context_size",
    "presence_penalty",
    "presence_context_size",
    "frequency_penalty",
    "frequency_context_size",
    "logit_bias",
    "thinking_budget",
    "thinking_start_token",
    "thinking_end_token",
    "resize_shape",
    "max_long_side_pixel",
    "skip_special_tokens",
)
_SHUTTING_DOWN = False
_WATCHDOG_TICK = None
_WATCHDOG_PREFILL_BUDGET = None  # set by run_with_watchdog; sizes prefill stall window (fix A)
_METAL_LIMITS = {}
_DECODE_EVAL_CONTEXT = threading.local()

# Cross-request prompt cache (KV reuse across turns). Gated by env so the
# proven no-cache path remains the fallback.
PROMPT_CACHE_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_PROMPT_CACHE_THINKING_FLAG = os.environ.get(
    "MLX_M3_PROMPT_CACHE_THINKING", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_THINKING_MODE = os.environ.get(
    "MLX_M3_PROMPT_CACHE_THINKING_MODE",
    "full" if _PROMPT_CACHE_THINKING_FLAG else "off",
).strip().lower()
if PROMPT_CACHE_THINKING_MODE not in {"off", "visible", "full"}:
    logger.warning(
        "invalid MLX_M3_PROMPT_CACHE_THINKING_MODE=%r; using off",
        PROMPT_CACHE_THINKING_MODE,
    )
    PROMPT_CACHE_THINKING_MODE = "off"
PROMPT_CACHE_THINKING_ENABLED = PROMPT_CACHE_THINKING_MODE != "off"
PROMPT_CACHE_DIRECT_SUFFIX_IDS = os.environ.get(
    "MLX_M3_PROMPT_CACHE_DIRECT_SUFFIX_IDS", "1"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_MIN_REUSE = int(os.environ.get("MLX_M3_PROMPT_CACHE_MIN_REUSE", "32"))
PROMPT_CACHE_TTL_SECONDS = int(os.environ.get("MLX_M3_PROMPT_CACHE_TTL_SECONDS", "10800"))
PROMPT_CACHE_MAX_TOKENS = int(os.environ.get("MLX_M3_PROMPT_CACHE_MAX_TOKENS", "0"))
PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS", "4096") or "4096"
)
# 2026-07-09: keep the INPUT-prefix KV when a stream is cancelled instead of
# dropping the whole cache. Agent clients (codex goals) retry the same
# conversation after a client-side timeout; the old reset forced a full
# re-prefill of the entire context on every retry (145s at 41k = the codex
# "looping, no output" death spiral).
PROMPT_CACHE_KEEP_ON_CANCEL = os.environ.get(
    "MLX_M3_KEEP_CACHE_ON_CANCEL", "1"
).strip().lower() not in {"0", "false", "no", "off"}
PROMPT_CACHE_PROTECT_LARGE_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_PROTECT_LARGE", "1"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_PROTECT_MIN_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_PROTECT_MIN_TOKENS", "32768") or "32768"
)
PROMPT_CACHE_PROTECT_BYPASS_MAX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_PROTECT_BYPASS_MAX_TOKENS", "8192") or "8192"
)
PROMPT_CACHE_SESSION_PROTECT_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SESSION_PROTECT", "1"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS", "256") or "256"
)
PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS", "8192") or "8192"
)
PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS", "512") or "512"
)
PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_REUSE_RATIO = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_REUSE_RATIO", "0.915") or "0.915"
)
PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_TOKENS", "0") or "0"
)
PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_SUFFIX_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_SUFFIX_TOKENS", "0") or "0"
)
PROMPT_CACHE_KEEPWARM_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_KEEPWARM", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_KEEPWARM_MODE = os.environ.get(
    "MLX_M3_PROMPT_CACHE_KEEPWARM_MODE", "metal"
).strip().lower()
if PROMPT_CACHE_KEEPWARM_MODE not in {"metal", "prewarm"}:
    logger.warning(
        "invalid MLX_M3_PROMPT_CACHE_KEEPWARM_MODE=%r; using metal",
        PROMPT_CACHE_KEEPWARM_MODE,
    )
    PROMPT_CACHE_KEEPWARM_MODE = "metal"
PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS", "15") or "15"
)
PROMPT_CACHE_KEEPWARM_IDLE_AFTER_SECONDS = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_KEEPWARM_IDLE_AFTER_SECONDS", "10") or "10"
)
PROMPT_CACHE_KEEPWARM_MATRIX_SIZE = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_KEEPWARM_MATRIX_SIZE", "1") or "1"
)
PROMPT_CACHE_KEEPWARM_LARGE_CACHE_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_KEEPWARM_LARGE_CACHE_TOKENS", "8192") or "8192"
)
PROMPT_CACHE_KEEPWARM_LARGE_INTERVAL_SECONDS = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_KEEPWARM_LARGE_INTERVAL_SECONDS", "60") or "60"
)
PROMPT_CACHE_KEEPWARM_SLOW_BACKOFF_SECONDS = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_KEEPWARM_SLOW_BACKOFF_SECONDS", "60") or "60"
)
PROMPT_CACHE_REQUEST_START_KEEPWARM_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_REQUEST_START_KEEPWARM", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_REQUEST_START_KEEPWARM_IDLE_SECONDS = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_REQUEST_START_KEEPWARM_IDLE_SECONDS", "2") or "2"
)
PROMPT_CACHE_REQUEST_START_KEEPWARM_MATRIX_SIZE = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_REQUEST_START_KEEPWARM_MATRIX_SIZE", "128") or "128"
)
PROMPT_CACHE_REQUEST_START_KEEPWARM_REPEATS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_REQUEST_START_KEEPWARM_REPEATS", "1") or "1"
)
PROMPT_CACHE_POST_RESPONSE_KEEPWARM_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_POST_RESPONSE_KEEPWARM", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_POST_RESPONSE_KEEPWARM_DELAY_SECONDS = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_POST_RESPONSE_KEEPWARM_DELAY_SECONDS", "5") or "5"
)
PROMPT_CACHE_POST_RESPONSE_KEEPWARM_MATRIX_SIZE = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_POST_RESPONSE_KEEPWARM_MATRIX_SIZE", "128") or "128"
)
PROMPT_CACHE_POST_RESPONSE_KEEPWARM_REPEATS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_POST_RESPONSE_KEEPWARM_REPEATS", "1") or "1"
)
CLEAR_CACHE_AFTER_REQUEST = os.environ.get(
    "MLX_M3_CLEAR_CACHE_AFTER_REQUEST", "0"
).strip().lower() in {"1", "true", "yes", "on"}
CLEAR_CACHE_AFTER_ERROR = os.environ.get(
    "MLX_M3_CLEAR_CACHE_AFTER_ERROR", "1"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SESSION_MAP_MAX = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SESSION_MAP_MAX", "16") or "16"
)
PROMPT_CACHE_RESIDENT_SLOTS = max(
    1,
    int(os.environ.get("MLX_M3_PROMPT_CACHE_RESIDENT_SLOTS", "2") or "2"),
)
def _env_int_rank_aware(name, default):
    """Per-rank env override: NAME_RANK<n> beats NAME. The ranks have very
    different headroom (256GB vs 128GB); a shared budget sized for rank 0
    paged rank 1 into a crawl on 2026-07-06 (the 'frozen decode' stall and
    the earlier 0.69 t/s slow-turn signature)."""
    rank = os.environ.get("MLX_RANK", "").strip()
    if rank:
        v = os.environ.get(f"{name}_RANK{rank}")
        if v is not None and str(v).strip():
            return int(v)
    return int(os.environ.get(name, str(default)) or str(default))


PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS = _env_int_rank_aware(
    "MLX_M3_PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS", 0
)
PROMPT_CACHE_SESSION_MANIFEST_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST", "1"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SESSION_MANIFEST_MAX = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SESSION_MANIFEST_MAX", "64") or "64"
)
PROMPT_CACHE_SESSION_MANIFEST_PATH = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST_PATH",
    os.path.join(
        os.environ.get("M3_LOG_DIR", os.path.dirname(os.path.abspath(__file__))),
        "prompt_cache_sessions.json",
    ),
)
PROMPT_CACHE_SSD_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SSD_RESTORE_ENABLED = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_RESTORE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SSD_THINKING_BOUNDARY_RESTORE = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_THINKING_BOUNDARY_RESTORE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SSD_AUTO_SAVE = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS = max(
    0,
    int(
        os.environ.get(
            "MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS", "8192"
        )
        or "8192"
    ),
)
# SSD artifacts are cold backing, not a license to reserve the request's full
# output ceiling in unified memory. A 45k-token restore previously inherited a
# 32k-token max-output reserve on every layer and pushed the 128GB rank into a
# local Metal OOM. Keep only a small, rank-aware hot append window; normal KV
# growth takes over if a response actually exceeds it.
PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS = max(
    0,
    _env_int_rank_aware("MLX_M3_PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS", 4096),
)
# Persist logical cache contents by default. Spare backing is cheap to recreate
# during restore and expensive to serialize forever, especially after a large
# request temporarily grew the live cache.
PROMPT_CACHE_SSD_SAVE_RESERVE_TOKENS = max(
    0,
    _env_int_rank_aware("MLX_M3_PROMPT_CACHE_SSD_SAVE_RESERVE_TOKENS", 0),
)
PROMPT_CACHE_SSD_DIR = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "thundermlx", "prompt-kv"),
)
PROMPT_CACHE_SSD_DIR_RANK0 = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_DIR_RANK0", ""
).strip()
PROMPT_CACHE_SSD_DIR_RANK1 = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_DIR_RANK1", ""
).strip()
PROMPT_CACHE_SSD_TTL_SECONDS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SSD_TTL_SECONDS", "432000") or "432000"
)
PROMPT_CACHE_SSD_MAX_BYTES = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SSD_MAX_BYTES", "429496729600") or "429496729600"
)
PROMPT_CACHE_SSD_MIN_TOKENS = int(
    os.environ.get("MLX_M3_PROMPT_CACHE_SSD_MIN_TOKENS", "8192") or "8192"
)
PROMPT_CACHE_SSD_SAVE_REASONING = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_SAVE_REASONING", "0"
).strip().lower() in {"1", "true", "yes", "on"}
PROMPT_CACHE_SSD_PRIVACY = os.environ.get(
    "MLX_M3_PROMPT_CACHE_SSD_PRIVACY", "local"
).strip().lower()
PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS = float(
    os.environ.get("MLX_M3_PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS", "10")
    or "10"
)

# ---- Live-tunable storage caps (dashboard "Storage" card) ------------------
# rank0's /admin/runtime-tuning adjusts these without a restart. Boot seeds
# from the env vars; the ENFORCEMENT sites re-read the mutables every time
# (_prompt_cache_ssd_prune_unlocked here; m3_capture re-reads its settings
# dict per flush/finalize). Out-of-range integers CLAMP to these guard rails
# instead of erroring; garbage still 400s. Intentionally rank0-local: the
# tuning broadcast skips these keys, so rank1's SSD prune keeps its
# env-seeded cap (its disk budget differs) and capture-corpus writes only
# ever happen on rank0.
_STORAGE_TUNING_KEYS = frozenset({
    "prompt_cache_ssd_max_bytes",
    "capture_max_request_bytes",
    "capture_max_total_bytes",
})
_STORAGE_TUNING_RANGES = {
    "prompt_cache_ssd_max_bytes": (50 << 30, 400 << 30),  # 50-400 GiB
    "capture_max_request_bytes": (50 << 20, 1 << 30),     # 50 MiB-1 GiB
    "capture_max_total_bytes": (10 << 30, 200 << 30),     # 10-200 GiB
}


def _capture_module():
    """m3_capture is optional (the golden tree does not ship it); resolve it
    lazily so the capture tunables cleanly report "not deployed" without it."""
    try:
        import m3_capture
        return m3_capture
    except ImportError:
        return None


_capture_boot = _capture_module()
with _runtime_tuning_lock:
    _runtime_tuning["prompt_cache_ssd_max_bytes"] = int(PROMPT_CACHE_SSD_MAX_BYTES)
    if _capture_boot is not None:
        for _tuning_key, _cap_key in (
            ("capture_max_request_bytes", "max_request_bytes"),
            ("capture_max_total_bytes", "max_total_bytes"),
        ):
            _runtime_tuning[_tuning_key] = int(_capture_boot.settings()[_cap_key])
del _capture_boot


def _runtime_prompt_cache_ssd_max_bytes():
    with _runtime_tuning_lock:
        value = _runtime_tuning.get("prompt_cache_ssd_max_bytes")
        if value is None:
            return int(PROMPT_CACHE_SSD_MAX_BYTES)
        return int(value)


def _capture_corpus_status():
    """Capture-corpus usage + caps for /health (same role as the SSD stats,
    sourced from m3_capture's module state). Absent module => not deployed."""
    capture = _capture_module()
    if capture is None:
        return {"deployed": False}
    try:
        return {"deployed": True, **capture.status()}
    except Exception as e:
        return {"deployed": True, "error": str(e)}


VISIBLE_TRANSCRIPT_PREWARM_ENABLED = os.environ.get(
    "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM", "1"
).strip().lower() in {"1", "true", "yes", "on"}
VISIBLE_TRANSCRIPT_PREWARM_BLOCKING = os.environ.get(
    "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_BLOCKING", "0"
).strip().lower() in {"1", "true", "yes", "on"}
VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED = int(
    os.environ.get("MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED", "16") or "16"
)
VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS = int(
    os.environ.get("MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS", "8192") or "8192"
)
VISIBLE_TRANSCRIPT_PREWARM_MAX_SUFFIX_TOKENS = int(
    os.environ.get("MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_SUFFIX_TOKENS", "4096") or "4096"
)
VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS = int(
    os.environ.get("MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS", "512") or "512"
)
REASONING_RECALL_ENABLED = os.environ.get(
    "MLX_M3_REASONING_RECALL", "1"
).strip().lower() in {"1", "true", "yes", "on"}
# Default ON: the thinking model surfaces reasoning on tool turns exactly
# like oMLX. Callers who want a silent agent lane use the non-thinking
# model endpoint instead of suppressing reasoning one layer down here.
EMIT_TOOL_REASONING = os.environ.get(
    "MLX_M3_EMIT_TOOL_REASONING", "1"
).strip().lower() in {"1", "true", "yes", "on"}
REASONING_RECALL_MAX_SESSIONS = int(
    os.environ.get("MLX_M3_REASONING_RECALL_MAX_SESSIONS", "16") or "16"
)
REASONING_RECALL_MAX_ITEMS = int(
    os.environ.get("MLX_M3_REASONING_RECALL_MAX_ITEMS", "64") or "64"
)
REASONING_RECALL_MAX_CHARS = int(
    os.environ.get("MLX_M3_REASONING_RECALL_MAX_CHARS", "65536") or "65536"
)
MAX_TOKENS_CEILING = int(os.environ.get("MLX_M3_MAX_TOKENS_CEILING", "16384"))
OMLX_MINIMAX_OVERLAY = os.environ.get(
    "MLX_M3_OMLX_MINIMAX_OVERLAY", "0"
).strip().lower() in {"1", "true", "yes", "on"}
USE_DIRECT_DECODE_KERNEL = os.environ.get(
    "MLX_M3_USE_DIRECT_DECODE_KERNEL", "0"
).strip().lower() in {"1", "true", "yes", "on"}
DIRECT_DECODE_EVAL_MODE = os.environ.get(
    "MLX_M3_DIRECT_DECODE_EVAL_MODE", "small"
).strip().lower()
# HARD-DISABLED 2026-07-06 (dead-code audit): the per-token cross-stream
# all_sum stop-check was THE historical wedge root cause (racing the model's
# collectives on the same QP/CQ). No env flag may resurrect it. The safe
# replacement is the nonce-coordinated file stop (MLX_M3_SAFE_DECODE_STOP).
UNSAFE_INFLIGHT_STOP = False
# Decode-phase coordinated stop (nonce file-stop). Default OFF: the 2026-07-06
# live iterations desynced ranks twice (stale rank1 code, then re-boot races).
# The mechanism is sound on paper and both loops carry the check; it re-enables
# via env after passing a dedicated offline acceptance rig.
SAFE_DECODE_STOP = os.environ.get("MLX_M3_SAFE_DECODE_STOP", "0") == "1"
STOP_ON_CLIENT_DISCONNECT = os.environ.get(
    "MLX_M3_STOP_ON_CLIENT_DISCONNECT", "0"
).strip().lower() in {"1", "true", "yes", "on"}
STOP_CHECK_EVERY = max(
    1,
    int(os.environ.get("MLX_M3_STOP_CHECK_EVERY", "4") or "4"),
)
PREFILL_STOP_CHECK_EVERY = max(
    0,
    int(os.environ.get("MLX_M3_PREFILL_STOP_CHECK_EVERY", "0") or "0"),
)
PREFILL_STOP_FILE = os.path.expanduser(
    os.environ.get(
        "MLX_M3_PREFILL_STOP_FILE",
        "/private/tmp/minimax_m3_prefill_stop_requested",
    )
)
RANK1_SSH = os.environ.get("MLX_M3_RANK1_SSH", "").strip()
THINKING_RAW_SILENT_LIMIT = int(
    os.environ.get("MLX_M3_THINKING_RAW_SILENT_LIMIT", "64") or "64"
)

# Thinking-runaway guard (2026-07-09 hermes 486s/12.5k-token pure-think
# runaway): a Think turn deep in a long conversation can loop inside
# <mm:think> and never emit </mm:think>, decoding to the 32k ceiling with
# zero visible content. When a turn is STILL in thinking past this many
# generated tokens, arm the proven EOS-injection stop (same path /v1/stop
# uses, batch-cancel certified). Only pure-think runaways trip it — once
# thinking closes and the answer starts, the guard disengages, so legit
# long answers are never clipped. 0 disables.
THINKING_RUNAWAY_TOKEN_BUDGET = int(
    os.environ.get("MLX_M3_THINKING_RUNAWAY_TOKEN_BUDGET", "8192") or "8192"
)
# A tool-bearing turn that remains inside <mm:think> this long is not making
# progress toward a call. End that attempt so the existing no-thinking retry
# can emit the tool call without spending the full prose-thinking budget.
TOOL_THINKING_RUNAWAY_TOKEN_BUDGET = int(
    os.environ.get("MLX_M3_TOOL_THINKING_RUNAWAY_TOKEN_BUDGET", "0")
    or "0"
)
# Flavor-agnostic degenerate-repetition guard (2026-07-10 zcode copy-spiral:
# the model locked onto `]<]minimax[>[ grep -n '...'` and re-emitted it
# hundreds of times — a bare-marker+shell-command shape no tag/JSON detector
# catches, and token-level repetition_penalty=1.05 doesn't break a whole-
# phrase loop). This guard doesn't care WHAT repeats: when the decode tail is
# a tight cycle (a short substring repeated many times), force-stop. 0 = off.
DECODE_REPETITION_GUARD_TOKENS = int(
    os.environ.get("MLX_M3_DECODE_REPETITION_GUARD_TOKENS", "12") or "12"
)

# In-flight generation stop. Set via POST /v1/stop or client disconnect.
_STOP_FLAG = threading.Event()
_STOP_KIND_LOCK = threading.Lock()
_STOP_KIND = None
_STOP_REQUEST_ID = None
# Request id + nonce are kept outside the public active-request payload. The
# pair lets a dashboard cancel target the request it actually displayed even
# if that request finishes while the stop call is crossing to rank 1.
_ACTIVE_STOP_TARGET = {"request_id": None, "nonce": None}


def _clear_prefill_stop_file(reason="new request"):
    if not PREFILL_STOP_FILE:
        return
    try:
        os.unlink(PREFILL_STOP_FILE)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug("prefill stop file clear failed (%s): %s", reason, e)


def _prefill_stop_at_from_active(active_info):
    """Choose a future prefill chunk where both ranks should stop.

    The stop file is observed locally by each rank after a prefill chunk
    completes. Picking a future chunk avoids the race where one rank has just
    passed the current callback while the other has not yet reached it.
    """
    if not active_info:
        return None
    try:
        total_tokens = int(
            active_info.get("prefill_total_tokens")
            or (active_info.get("request_shape") or {}).get("full_prompt_tokens")
            or 0
        )
        processed_tokens = int(active_info.get("prefill_processed_tokens") or 0)
    except Exception:
        return None
    if total_tokens <= 0:
        return None
    step = max(1, _runtime_prefill_step_size(total_tokens))
    stop_at = processed_tokens + (2 * step)
    if processed_tokens <= 0:
        stop_at = 2 * step
    return max(step, min(total_tokens, stop_at))


# Per-generation stop nonce: rank 0 mints it, ships it to rank 1 inside the
# generation broadcast, and a decode stop file is honored only when its nonce
# matches the current generation's. Clock-free (cross-machine wall-clock
# freshness checks desynced the ranks — 2026-07-06), request-scoped, and
# stale-proof. Single-flight serving makes the module global safe.

def _thinking_template_kwargs(config, *, enable_thinking=False, thinking_mode=None):
    """Compat shim: mlx-vlm 0.6.4 removed prompt_utils.thinking_template_kwargs
    (the 0.6.3 helper) and folded its logic into apply_chat_template. This
    replicates the 0.6.3 return exactly so our explicit thinking_mode routing
    works across versions. MiniMax nuance preserved: omit enable_thinking only
    when the model is MiniMax AND nothing was explicitly requested (adaptive).
    """
    kwargs = {}
    model_type = str(getattr(config, "model_type", "") or "")
    is_minimax = "minimax" in model_type.lower()
    if enable_thinking or thinking_mode is not None or not is_minimax:
        kwargs["enable_thinking"] = enable_thinking
    if thinking_mode is not None:
        kwargs["thinking_mode"] = thinking_mode
    return kwargs

_STOP_NONCE = {"value": None}
# Decode-stop via EOS injection (2026-07-06 redesign): the file-based decode
# stop desynced ranks — the stop file exists only on rank 0's filesystem, so
# rank 0 broke while rank 1 kept decoding into a collective mismatch (caught
# by the offline acceptance rig). Instead, when a stop is requested rank 0
# swaps its next SAMPLED token for EOS inside the existing sampled-token
# sync; both ranks receive EOS through the collective they already run and
# end the generation identically, with zero new collectives.
_FORCE_EOS = {"active": False, "eos_id": None}


def _arm_rank0_semantic_eos(rank, reason, token_index):
    """Request a decode stop without letting local parser state split ranks.

    Tool/reasoning parsers run independently in each process and can observe a
    completed or degenerate fragment on different iterations even though the
    sampled token IDs are synchronized. Only rank 0 may turn that semantic
    observation into control flow. Its sampler replaces the next token with
    EOS, and the existing token synchronization delivers that same EOS to all
    ranks at one shared decode boundary.
    """
    if rank != 0:
        return False
    if not _BATCH_PATH_ACTIVE.get("value"):
        logger.warning(
            "rank 0: semantic decode stop (%s) deferred at token %d because "
            "the request is using the upstream generator; allowing natural "
            "EOS preserves distributed lockstep",
            reason,
            token_index,
        )
        return False
    if _FORCE_EOS.get("eos_id") is None:
        logger.error(
            "rank 0: cannot arm semantic decode stop (%s) at token %d: "
            "EOS id is unavailable",
            reason,
            token_index,
        )
        return False
    if not _FORCE_EOS.get("active"):
        logger.info(
            "rank 0: arming synchronized EOS for %s at token %d",
            reason,
            token_index,
        )
        _FORCE_EOS["active"] = True
    return True

# Rank-0 op-channel mutex: _bcast frames each op as bare size+payload
# all_sums with no interleaving protection, so two rank-0 threads running
# broadcast transactions concurrently shred the words and rank 1 wedges in
# recv (30-min hang -> guard exit-75 — 2026-07-06 P4 photograph). Hold this
# for one COMPLETE op transaction at a time: the op _bcast plus every
# follow-up collective until the op finishes on both ranks. RLock so the
# request transaction can re-enter (request-start keepwarm runs inside it).
# Lock order: generation_lock FIRST, then this mutex — never wait on
# generation_lock while holding it.
_RANK0_OP_MUTEX = threading.RLock()


def _prefill_stop_payload(
    reason="stop",
    stop_at_tokens=None,
    phase=None,
    nonce=None,
    decode_stop_at_tokens=None,
    request_id=None,
):
    payload = {
        "version": 1,
        "at": round(time.time(), 6),
        "reason": str(reason),
    }
    if stop_at_tokens is not None:
        try:
            payload["stop_at_tokens"] = int(stop_at_tokens)
        except Exception:
            pass
    if decode_stop_at_tokens is not None:
        try:
            payload["decode_stop_at_tokens"] = int(decode_stop_at_tokens)
        except Exception:
            pass
    if phase:
        payload["phase"] = str(phase)
    if nonce:
        payload["nonce"] = str(nonce)
    if request_id:
        payload["request_id"] = str(request_id)
    return payload


def _read_prefill_stop_file():
    if not PREFILL_STOP_FILE:
        return None
    try:
        with open(PREFILL_STOP_FILE, "r", encoding="utf-8") as fh:
            raw = fh.read(4096).strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug("prefill stop file read failed: %s", e)
        return {"version": 0, "reason": "unreadable"}
    if not raw:
        return {"version": 0, "reason": "empty"}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {"version": 0}
    except Exception:
        return {"version": 0, "reason": raw}


def _touch_local_prefill_stop_file(
    reason="stop",
    stop_at_tokens=None,
    phase=None,
    nonce=None,
    decode_stop_at_tokens=None,
    request_id=None,
):
    if not PREFILL_STOP_FILE:
        return False
    payload = _prefill_stop_payload(
        reason,
        stop_at_tokens,
        phase,
        nonce,
        decode_stop_at_tokens,
        request_id,
    )
    try:
        parent = os.path.dirname(PREFILL_STOP_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(PREFILL_STOP_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.write("\n")
        return True
    except Exception as e:
        logger.warning("prefill stop file touch failed (%s): %s", reason, e)
        return False


def _touch_remote_prefill_stop_file(
    reason="stop",
    stop_at_tokens=None,
    phase=None,
    nonce=None,
    decode_stop_at_tokens=None,
    request_id=None,
):
    if not RANK1_SSH or not PREFILL_STOP_FILE:
        return None
    quoted_path = shlex.quote(PREFILL_STOP_FILE)
    payload = json.dumps(
        _prefill_stop_payload(
            reason,
            stop_at_tokens,
            phase,
            nonce,
            decode_stop_at_tokens,
            request_id,
        ),
        separators=(",", ":"),
    )
    quoted_payload = shlex.quote(payload)
    command = (
        f"mkdir -p {shlex.quote(os.path.dirname(PREFILL_STOP_FILE) or '/tmp')} "
        f"&& printf '%s\\n' {quoted_payload} > {quoted_path}"
    )
    try:
        completed = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=2",
                "-o", "ConnectionAttempts=1",
                RANK1_SSH,
                command,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        ok = completed.returncode == 0
        if not ok:
            logger.warning(
                "remote prefill stop file touch failed on %s (rc=%s)",
                RANK1_SSH,
                completed.returncode,
            )
        return ok
    except Exception as e:
        logger.warning("remote prefill stop file touch failed on %s: %s", RANK1_SSH, e)
        return False


def _set_stop_request(kind="user", request_id=None):
    global _STOP_KIND, _STOP_REQUEST_ID
    with _STOP_KIND_LOCK:
        _STOP_KIND = str(kind or "user")
        _STOP_REQUEST_ID = str(request_id).strip() if request_id else None
        _STOP_FLAG.set()


def _clear_stop_request():
    global _STOP_KIND, _STOP_REQUEST_ID
    with _STOP_KIND_LOCK:
        _STOP_FLAG.clear()
        _STOP_KIND = None
        _STOP_REQUEST_ID = None


def _local_stop_kind():
    if not _STOP_FLAG.is_set():
        return None
    with _STOP_KIND_LOCK:
        return _STOP_KIND or "user"


def _user_stop_requested(request_id=None):
    kind = _local_stop_kind()
    if not kind or kind == "tool":
        return False
    if request_id:
        with _STOP_KIND_LOCK:
            target = _STOP_REQUEST_ID
        if target and target != str(request_id):
            return False
    return True


def _stop_request_target(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("request_id", "expected_request_id", "id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _stop_request_matches_active(payload, active_info):
    expected = _stop_request_target(payload)
    if expected is None:
        return True
    return bool(active_info and str(active_info.get("id") or "") == expected)


def _stop_boundaries_from_active(active_info=None):
    try:
        emitted = int((active_info or {}).get("tokens_emitted") or 0)
    except Exception:
        emitted = 0
    return {
        "prefill_stop_at_tokens": _prefill_stop_at_from_active(active_info),
        "decode_stop_at_tokens": max(16, emitted + 16),
    }


def _request_inflight_stop(
    reason="stop",
    active_info=None,
    request_id=None,
    stop_nonce=None,
):
    request_id = request_id or (active_info or {}).get("id")
    _set_stop_request("user", request_id=request_id)
    # Carry independent future-safe boundaries for BOTH phases. A stop can
    # arrive after the request slot is visible but before prefill/decode metrics
    # are published. The old single-phase payload classified that race as
    # prefill-only; if a short prefill completed before its next callback, the
    # subsequent decode never observed the stop and ran to the token ceiling.
    boundaries = _stop_boundaries_from_active(active_info)
    decode_stop_at_tokens = boundaries["decode_stop_at_tokens"]
    stop_at_tokens = boundaries["prefill_stop_at_tokens"]
    phase = "any"
    nonce = stop_nonce or _STOP_NONCE.get("value")
    remote_ok = _touch_remote_prefill_stop_file(
        reason,
        stop_at_tokens,
        phase,
        nonce,
        decode_stop_at_tokens,
        request_id,
    )
    local_ok = False
    if remote_ok is not False:
        local_ok = _touch_local_prefill_stop_file(
            reason,
            stop_at_tokens,
            phase,
            nonce,
            decode_stop_at_tokens,
            request_id,
        )
    else:
        logger.warning(
            "prefill stop file not armed locally because rank1 propagation failed; "
            "decode token-boundary stop remains armed"
        )
    return {
        "prefill_stop_local": local_ok,
        "prefill_stop_remote": remote_ok,
        "prefill_stop_at_tokens": stop_at_tokens,
        "decode_stop_at_tokens": decode_stop_at_tokens,
        "request_id": request_id,
    }


def _gb_to_bytes(value):
    return int(float(value) * 1024**3)


def _bytes_to_gb(value):
    return round(int(value) / 1024**3, 2)


# ---------------------------------------------------------------------------
# Cross-request prompt cache (single slot — the cluster is single-flight)
# ---------------------------------------------------------------------------
# Both ranks maintain their own KV cache but apply IDENTICAL logic: rank 0
# tokenizes the prompt and broadcasts token_ids to rank 1, so both compute the
# same longest-common-prefix against their (identically-populated) cache. This
# keeps the two ranks in lockstep while skipping already-processed context.
import threading as _threading

_prompt_cache_lock = _threading.RLock()
_tokenizer_runtime_lock = _threading.RLock()
_reasoning_recall_lock = _threading.RLock()
_prompt_cache_holder = {
    "cache": None,        # list of KV cache objects (reused across requests)
    "token_ids": [],      # full token sequence currently held in the cache
    "cache_len": 0,       # actual KV length: prompt tokens + generated tokens
    "last_input_tokens": 0,
    "last_generated_tokens": 0,
    "last_exact_generated_ids": False,
    "last_suffix_ids": None,
    "prompt": None,
    "session_id": None,
    "session_source": None,
    "max_kv_size": MAX_KV_SIZE,
    "last_event": None,   # most recent cache decision/update for health/debug
    "last_prepare_event": None,
    "last_update_event": None,
    "last_keepwarm_event": None,
    "last_keepwarm_at": None,
    "keepwarm_count": 0,
    "created_at": None,
    "last_access_at": None,
    "in_use": False,
    "in_use_started_at": None,
}
_prompt_cache_session_map = OrderedDict()
_prompt_cache_resident_slots = OrderedDict()
_reasoning_recall_sessions = OrderedDict()
_prompt_cache_session_manifest_state = {
    "loaded_entries": 0,
    "entry_count": 0,
    "last_loaded_at": None,
    "last_written_at": None,
    "last_cleared_at": None,
    "last_error": None,
}
_prompt_cache_ssd_state = {
    "last_save_at": None,
    "last_restore_at": None,
    "last_prune_at": None,
    "last_clear_at": None,
    "last_scan_at": None,
    "last_error": None,
    "last_restore_miss_reason": None,
    "last_restore_attempt_reason": None,
    "last_saved_session": None,
    "last_restored_session": None,
    "saved_sessions": 0,
    "restored_sessions": 0,
    "pruned_sessions": 0,
    "last_saved_tokens": 0,
    "last_saved_bytes": 0,
    "last_restored_tokens": 0,
    "last_restore_target_capacity": 0,
    "last_restore_requested_append_reserve_tokens": 0,
    "last_restore_append_reserve_tokens": 0,
    "last_saved_capacity": 0,
    "last_saved_spare_tokens": 0,
    "last_auto_save_deferred_at": None,
    "last_auto_save_deferred_reason": None,
    "auto_save_deferred_count": 0,
}
_prompt_cache_ssd_autosave_anchors = OrderedDict()
_prompt_cache_ssd_scan_cache = {"at": None, "scan": None}
_metal_warmup_lock = _threading.RLock()
_metal_warmup_last_event = None

PROMPT_CACHE_SSD_SCHEMA_VERSION = 3
PROMPT_CACHE_SSD_RECENT_ENTRIES = 12


def _prompt_cache_session_key(session_id=None, session_source=None):
    if session_id:
        return str(session_id)
    return "__default__"


def _prompt_cache_current_session_key_unlocked():
    return _prompt_cache_session_key(
        _prompt_cache_holder.get("session_id"),
        _prompt_cache_holder.get("session_source"),
    )


def _prompt_cache_resident_total_tokens_unlocked():
    total = 0
    if _prompt_cache_holder.get("cache") is not None:
        total += int(_prompt_cache_holder.get("cache_len") or 0)
    current_key = _prompt_cache_current_session_key_unlocked()
    for key, entry in _prompt_cache_resident_slots.items():
        if key == current_key:
            continue
        total += int(entry.get("cache_len") or 0)
    return total


def _prompt_cache_stash_current_unlocked(reason="stash"):
    """Keep the current live KV slot available for a later same-session turn."""
    if PROMPT_CACHE_RESIDENT_SLOTS <= 1:
        return False
    holder = _prompt_cache_holder
    cache = holder.get("cache")
    token_ids = list(holder.get("token_ids") or [])
    if cache is None or not token_ids:
        return False
    key = _prompt_cache_current_session_key_unlocked()
    if not key:
        return False
    _prompt_cache_resident_slots[key] = {
        "cache": cache,
        "token_ids": token_ids,
        "prompt": holder.get("prompt"),
        "cache_len": int(holder.get("cache_len") or 0),
        "last_input_tokens": int(holder.get("last_input_tokens") or 0),
        "last_generated_tokens": int(holder.get("last_generated_tokens") or 0),
        "last_exact_generated_ids": bool(holder.get("last_exact_generated_ids")),
        "session_id": holder.get("session_id"),
        "session_source": holder.get("session_source"),
        "created_at": holder.get("created_at"),
        "last_access_at": round(time.time(), 3),
        "stashed_at": round(time.time(), 3),
        "stash_reason": reason,
    }
    _prompt_cache_resident_slots.move_to_end(key)
    while len(_prompt_cache_resident_slots) > max(0, PROMPT_CACHE_RESIDENT_SLOTS - 1):
        evicted_key, evicted = _prompt_cache_resident_slots.popitem(last=False)
        logger.info(
            "prompt-cache resident slot evicted: %s (%s tokens)",
            evicted_key,
            evicted.get("cache_len"),
        )
    if PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS > 0:
        while (
            _prompt_cache_resident_slots
            and _prompt_cache_resident_total_tokens_unlocked()
            > PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS
        ):
            evicted_key, evicted = _prompt_cache_resident_slots.popitem(last=False)
            logger.info(
                "prompt-cache resident slot evicted by token cap: %s (%s tokens)",
                evicted_key,
                evicted.get("cache_len"),
            )
    return True


def _prompt_cache_restore_resident_unlocked(session_id=None, session_source=None):
    """Swap in a previously stashed live KV slot for the requested session."""
    if PROMPT_CACHE_RESIDENT_SLOTS <= 1 or not session_id:
        return None
    key = _prompt_cache_session_key(session_id, session_source)
    slot = _prompt_cache_resident_slots.pop(key, None)
    if not slot:
        return None
    current_key = _prompt_cache_current_session_key_unlocked()
    if current_key != key:
        _prompt_cache_stash_current_unlocked(reason=f"restore:{key}")
    _prompt_cache_holder["cache"] = slot.get("cache")
    _prompt_cache_holder["token_ids"] = list(slot.get("token_ids") or [])
    _prompt_cache_holder["prompt"] = slot.get("prompt")
    _prompt_cache_holder["cache_len"] = int(slot.get("cache_len") or 0)
    _prompt_cache_holder["last_input_tokens"] = int(slot.get("last_input_tokens") or 0)
    _prompt_cache_holder["last_generated_tokens"] = int(slot.get("last_generated_tokens") or 0)
    _prompt_cache_holder["last_exact_generated_ids"] = bool(slot.get("last_exact_generated_ids"))
    _prompt_cache_holder["session_id"] = slot.get("session_id")
    _prompt_cache_holder["session_source"] = slot.get("session_source")
    _prompt_cache_holder["created_at"] = slot.get("created_at")
    _prompt_cache_holder["last_access_at"] = round(time.time(), 3)
    logger.info(
        "prompt-cache resident slot restored: %s (%s tokens)",
        key,
        _prompt_cache_holder["cache_len"],
    )
    return {
        "restored_key": key,
        "restored_cache_len": _prompt_cache_holder["cache_len"],
        "restored_stashed_at": slot.get("stashed_at"),
        "restored_reason": slot.get("stash_reason"),
    }


def _session_manifest_path():
    return os.path.expanduser(PROMPT_CACHE_SESSION_MANIFEST_PATH)


def _sanitize_prompt_cache_session_entry(key, entry):
    """Persist only cache metadata, never raw prompt text or token ids."""
    allowed = {
        "session_id",
        "session_source",
        "first_seen_at",
        "requests",
        "updates",
        "bypasses_preserved",
        "last_prompt_tokens",
        "last_reuse_tokens",
        "last_suffix_tokens",
        "last_missed_tokens",
        "cache_len",
        "key_tokens",
        "protected_cache_tokens",
        "last_reuse_ratio",
        "last_miss_reason",
        "last_action",
        "last_phase",
        "last_at",
        "ssd_rehydratable",
        "ssd_saved_at",
        "ssd_cache_bytes",
        "ssd_key_tokens",
        "ssd_session_hash",
    }
    out = {"key": str(key)}
    for field in allowed:
        value = entry.get(field)
        if value is None:
            continue
        if field.endswith("_tokens") or field in {
            "cache_len",
            "key_tokens",
            "protected_cache_tokens",
            "requests",
            "updates",
            "bypasses_preserved",
            "ssd_cache_bytes",
            "ssd_key_tokens",
        }:
            try:
                value = int(value)
            except Exception:
                continue
        elif field in {"first_seen_at", "last_at", "last_reuse_ratio", "ssd_saved_at"}:
            try:
                value = round(float(value), 6)
            except Exception:
                continue
        elif field == "ssd_rehydratable":
            value = bool(value)
        else:
            value = str(value)
        out[field] = value
    out["metadata_only"] = True
    out["rehydratable"] = bool(out.get("ssd_rehydratable"))
    return out


def _load_prompt_cache_session_manifest_unlocked():
    if (
        not PROMPT_CACHE_SESSION_MANIFEST_ENABLED
        or PROMPT_CACHE_SESSION_MAP_MAX <= 0
        or PROMPT_CACHE_SESSION_MANIFEST_MAX <= 0
    ):
        return
    path = _session_manifest_path()
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        entries = payload.get("entries") if isinstance(payload, dict) else []
        entry_count = len(entries) if isinstance(entries, list) else 0
        loaded = 0
        for raw in entries[-PROMPT_CACHE_SESSION_MAP_MAX:]:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key") or _prompt_cache_session_key(raw.get("session_id")))
            entry = {
                k: v for k, v in raw.items()
                if k not in {"key", "resident"}
            }
            entry["metadata_only"] = True
            entry["rehydratable"] = bool(entry.get("ssd_rehydratable"))
            entry["loaded_from_manifest"] = True
            _prompt_cache_session_map[key] = entry
            loaded += 1
        while len(_prompt_cache_session_map) > PROMPT_CACHE_SESSION_MAP_MAX:
            _prompt_cache_session_map.popitem(last=False)
        _prompt_cache_session_manifest_state.update({
            "loaded_entries": loaded,
            "entry_count": entry_count,
            "last_loaded_at": round(time.time(), 3),
            "last_error": None,
        })
        if loaded:
            logger.info("prompt-cache session manifest loaded: %s entries", loaded)
    except Exception as e:
        _prompt_cache_session_manifest_state["last_error"] = str(e)
        logger.warning("prompt-cache session manifest load failed: %s", e)


def _write_prompt_cache_session_manifest_unlocked():
    if (
        not PROMPT_CACHE_SESSION_MANIFEST_ENABLED
        or PROMPT_CACHE_SESSION_MAP_MAX <= 0
        or PROMPT_CACHE_SESSION_MANIFEST_MAX <= 0
    ):
        return
    path = _session_manifest_path()
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        merged = OrderedDict()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                for raw in existing.get("entries") or []:
                    if not isinstance(raw, dict):
                        continue
                    key = str(raw.get("key") or _prompt_cache_session_key(raw.get("session_id")))
                    raw["metadata_only"] = True
                    raw["rehydratable"] = bool(raw.get("ssd_rehydratable"))
                    merged[key] = raw
            except Exception as e:
                logger.debug("prompt-cache session manifest merge read failed: %s", e)
        for key, entry in _prompt_cache_session_map.items():
            merged[str(key)] = _sanitize_prompt_cache_session_entry(key, entry)
        while len(merged) > PROMPT_CACHE_SESSION_MANIFEST_MAX:
            merged.popitem(last=False)
        entries = list(merged.values())
        payload = {
            "schema": 1,
            "model_id": MODEL_ID,
            "updated_at": round(time.time(), 3),
            "entries": entries,
        }
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp, path)
        _prompt_cache_session_manifest_state.update({
            "entry_count": len(entries),
            "last_written_at": payload["updated_at"],
            "last_error": None,
        })
    except Exception as e:
        _prompt_cache_session_manifest_state["last_error"] = str(e)
        logger.debug("prompt-cache session manifest write failed: %s", e)


def _prompt_cache_session_manifest_status_unlocked():
    path = _session_manifest_path()
    exists = os.path.exists(path)
    size_bytes = None
    if exists:
        try:
            size_bytes = os.path.getsize(path)
        except Exception:
            size_bytes = None
    return {
        "enabled": PROMPT_CACHE_SESSION_MANIFEST_ENABLED,
        "path": path,
        "max_entries": PROMPT_CACHE_SESSION_MANIFEST_MAX,
        "exists": exists,
        "size_bytes": size_bytes,
        **_prompt_cache_session_manifest_state,
    }


def _sha256_text(value):
    if value is None:
        value = ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


def _sha256_file(path, limit_bytes=None):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            remaining = int(limit_bytes) if limit_bytes else None
            while True:
                if remaining is not None:
                    if remaining <= 0:
                        break
                    chunk = f.read(min(1024 * 1024, remaining))
                    remaining -= len(chunk)
                else:
                    chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _token_ids_sha256(token_ids):
    h = hashlib.sha256()
    ids = list(token_ids or [])
    h.update(len(ids).to_bytes(8, "little", signed=False))
    chunk = array("I")
    for tok in ids:
        try:
            chunk.append(int(tok) & 0xFFFFFFFF)
        except Exception:
            chunk.append(0)
        if len(chunk) >= 16384:
            if sys.byteorder != "little":
                chunk.byteswap()
            h.update(chunk.tobytes())
            chunk = array("I")
    if chunk:
        if sys.byteorder != "little":
            chunk.byteswap()
        h.update(chunk.tobytes())
    return h.hexdigest()


def _prompt_cache_ssd_root(rank=None):
    if rank is None:
        rank, _ = _prompt_cache_ssd_current_rank_world()
    scoped = ""
    if int(rank) == 0:
        scoped = PROMPT_CACHE_SSD_DIR_RANK0
    elif int(rank) == 1:
        scoped = PROMPT_CACHE_SSD_DIR_RANK1
    return os.path.expanduser(scoped or PROMPT_CACHE_SSD_DIR)


def _prompt_cache_ssd_session_hash(session_key):
    return hashlib.sha256(str(session_key or "__default__").encode("utf-8")).hexdigest()


def _prompt_cache_ssd_current_rank_world():
    env_rank = os.environ.get("MLX_RANK", "").strip()
    if env_rank:
        try:
            group = mx.distributed.init()
            return int(env_rank), int(group.size())
        except Exception:
            world = int(os.environ.get("MLX_WORLD_SIZE", "1") or "1")
            return int(env_rank), world
    try:
        group = mx.distributed.init()
        return int(group.rank()), int(group.size())
    except Exception:
        rank = int(os.environ.get("MLX_RANK", "0") or "0")
        world = int(os.environ.get("MLX_WORLD_SIZE", "1") or "1")
        return rank, world


def _prompt_cache_ssd_rank_dir(session_hash, rank=None):
    if rank is None:
        rank, _ = _prompt_cache_ssd_current_rank_world()
    return os.path.join(_prompt_cache_ssd_root(), session_hash, f"rank{int(rank)}")


def _prompt_cache_ssd_session_label(session_key):
    text = str(session_key or "__default__")
    if len(text) <= 96:
        return text
    return f"{text[:48]}...{text[-24:]}"


def _prompt_cache_ssd_file_fingerprint():
    root = os.path.dirname(os.path.abspath(__file__))
    files = [
        os.path.join(root, "sharded_server.py"),
        os.path.join(root, "m3_batch_cancel.py"),
        os.path.join(root, "m3_pipeline_patch.py"),
        os.path.join(root, "m3_eagle3.py"),
        os.path.join(root, "MSA Support", "mlx_vlm", "models", "minimax_m3_vl", "language.py"),
        os.path.join(root, "MSA Support", "mlx_vlm", "models", "minimax_m3_vl", "msa.py"),
    ]
    parts = []
    for path in files:
        digest = _sha256_file(path)
        if digest:
            parts.append({"path": os.path.relpath(path, root), "sha256": digest})
    return {
        "files": parts,
        "hash": _sha256_text(parts),
    }


def _safe_version(package):
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def _processor_fingerprint(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(processor, "chat_template", None)
    if not template and tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)
    fields = {
        "processor_class": processor.__class__.__module__ + "." + processor.__class__.__name__,
        "tokenizer_class": (
            tokenizer.__class__.__module__ + "." + tokenizer.__class__.__name__
            if tokenizer is not None else None
        ),
        "chat_template_hash": _sha256_text(template or ""),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    fields["hash"] = _sha256_text(fields)
    return fields


def _model_path_fingerprint(model=None, processor=None):
    resolved = (
        getattr(model, "_thundermlx_model_path", None)
        or getattr(processor, "_thundermlx_model_path", None)
        or getattr(getattr(processor, "tokenizer", None), "name_or_path", None)
    )
    raw = os.path.expanduser(str(resolved or MODEL or ""))
    exists = os.path.exists(raw)
    stat = None
    if exists:
        try:
            st = os.stat(raw)
            stat = {
                "path": os.path.abspath(raw),
                "is_dir": os.path.isdir(raw),
                "mtime_ns": int(st.st_mtime_ns),
                "size": int(st.st_size),
            }
        except Exception:
            stat = {"path": os.path.abspath(raw), "stat_error": True}
    payload = {
        "model": MODEL,
        "model_id": MODEL_ID,
        "resolved_path": str(resolved or ""),
        "path_stat": stat,
    }
    payload["hash"] = _sha256_text(payload)
    return payload


def _cache_class_names(cache):
    return [
        c.__class__.__module__ + "." + c.__class__.__name__
        for c in (cache or [])
    ]


def _prompt_cache_ssd_runtime_fingerprint(model, processor, cache=None):
    payload = {
        "schema": PROMPT_CACHE_SSD_SCHEMA_VERSION,
        "model": _model_path_fingerprint(model, processor),
        "processor": _processor_fingerprint(processor),
        "sharding": {
            "mode": SHARDING_MODE,
            "pipeline_layers": os.environ.get("M3_PIPELINE_LAYERS", ""),
            "backend": os.environ.get("M3_MLX_BACKEND", ""),
        },
        "rank_count": _prompt_cache_ssd_current_rank_world()[1],
        "kv": {
            "max_kv_size": MAX_KV_SIZE,
            "quant_enabled": KV_QUANT_ENABLED,
            "bits": KV_BITS,
            "group_size": KV_GROUP_SIZE,
            "scheme": KV_QUANT_SCHEME,
            "quantized_start": QUANTIZED_KV_START,
        },
        "prompt_cache": {
            "thinking_mode": PROMPT_CACHE_THINKING_MODE,
            "generated_reuse_max_tokens": PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS,
            "max_tokens": PROMPT_CACHE_MAX_TOKENS,
        },
        "versions": {
            "mlx": _safe_version("mlx"),
            "mlx-lm": _safe_version("mlx-lm"),
            "mlx-vlm": _safe_version("mlx-vlm"),
        },
        "cache_impl": _prompt_cache_ssd_file_fingerprint(),
        "cache_classes": _cache_class_names(cache),
    }
    payload["hash"] = _sha256_text(payload)
    return payload


def _is_mx_array(value):
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "nbytes")
        and value.__class__.__module__.startswith("mlx.")
    )


def _flatten_cache_state(value, arrays, prefix="state"):
    if value is None:
        return {"type": "none"}
    if _is_mx_array(value):
        name = f"a{len(arrays):04d}"
        try:
            arrays[name] = mx.contiguous(value)
        except Exception:
            arrays[name] = value
        return {
            "type": "array",
            "name": name,
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
            "nbytes": int(value.nbytes),
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _flatten_cache_state(v, arrays, f"{prefix}_{i}")
                for i, v in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _flatten_cache_state(v, arrays, f"{prefix}_{i}")
                for i, v in enumerate(value)
            ],
        }
    if isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"unsupported cache state value at {prefix}: {type(value)!r}")


def _restore_cache_state(structure, arrays):
    typ = structure.get("type") if isinstance(structure, dict) else None
    if typ == "none":
        return None
    if typ == "scalar":
        return structure.get("value")
    if typ == "array":
        name = structure.get("name")
        if name not in arrays:
            raise ValueError(f"missing array {name}")
        arr = arrays[name]
        expected_shape = tuple(int(v) for v in structure.get("shape") or [])
        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"array {name} shape mismatch {tuple(arr.shape)} != {expected_shape}"
            )
        expected_dtype = str(structure.get("dtype") or "")
        if expected_dtype and str(arr.dtype) != expected_dtype:
            raise ValueError(
                f"array {name} dtype mismatch {arr.dtype} != {expected_dtype}"
            )
        return arr
    if typ == "tuple":
        return tuple(_restore_cache_state(v, arrays) for v in structure.get("items") or [])
    if typ == "list":
        return [_restore_cache_state(v, arrays) for v in structure.get("items") or []]
    raise ValueError(f"unsupported cache state structure: {typ!r}")


def _cache_state_arrays(value, out=None):
    if out is None:
        out = []
    if _is_mx_array(value):
        out.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _cache_state_arrays(item, out)
    return out


def _prompt_cache_ssd_round_capacity(logical_tokens, reserve_tokens, step):
    logical_tokens = max(0, int(logical_tokens or 0))
    reserve_tokens = max(0, int(reserve_tokens or 0))
    step = max(1, int(step or 1))
    target = logical_tokens + reserve_tokens
    return ((target + step - 1) // step) * step


def _prompt_cache_ssd_bound_sequence(value, logical_tokens, reserve_tokens, step):
    if not _is_mx_array(value) or len(value.shape) < 3:
        return value
    capacity = int(value.shape[2])
    target = min(
        capacity,
        _prompt_cache_ssd_round_capacity(logical_tokens, reserve_tokens, step),
    )
    if target >= capacity:
        return value
    return value[..., :target, :]


def _prompt_cache_ssd_backing_state(layer_cache, *, max_spare_tokens=None):
    """Return cache tensors and explicit logical offsets for persistence.

    Schema v3 can retain a deliberately bounded append window, but defaults to
    logical cache contents only. Restore recreates a small hot append reserve.
    This prevents one large output allowance from becoming permanent capacity
    on every layer and rank.
    """
    nested = getattr(layer_cache, "kv_cache", None)
    if nested is not None:
        keys = getattr(nested, "keys", None)
        values = getattr(nested, "values", None)
        if _is_mx_array(keys) and _is_mx_array(values):
            kv_offset = int(getattr(nested, "offset", keys.shape[2]) or 0)
            step = max(1, int(getattr(layer_cache, "step", 1) or 1))
            index_keys = getattr(layer_cache, "index_keys", None)
            if index_keys is not None and not _is_mx_array(index_keys):
                index_keys = None
            index_offset = int(getattr(layer_cache, "index_offset", 0) or 0)
            if max_spare_tokens is not None:
                keys = _prompt_cache_ssd_bound_sequence(
                    keys, kv_offset, max_spare_tokens, step
                )
                values = _prompt_cache_ssd_bound_sequence(
                    values, kv_offset, max_spare_tokens, step
                )
                if index_keys is not None:
                    index_keys = _prompt_cache_ssd_bound_sequence(
                        index_keys, index_offset, max_spare_tokens, step
                    )
            return (
                ((keys, values), index_keys),
                {
                    "layout": "nested_kv_backing_v2",
                    "kv_offset": kv_offset,
                    "kv_capacity": int(keys.shape[2]),
                    "index_offset": index_offset,
                    "index_capacity": (
                        int(index_keys.shape[2]) if index_keys is not None else 0
                    ),
                    "saved_spare_tokens": max(0, int(keys.shape[2]) - kv_offset),
                },
            )

    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)
    if _is_mx_array(keys) and _is_mx_array(values):
        offset = int(getattr(layer_cache, "offset", keys.shape[2]) or 0)
        step = max(1, int(getattr(layer_cache, "step", 1) or 1))
        if max_spare_tokens is not None:
            keys = _prompt_cache_ssd_bound_sequence(
                keys, offset, max_spare_tokens, step
            )
            values = _prompt_cache_ssd_bound_sequence(
                values, offset, max_spare_tokens, step
            )
        return (
            (keys, values),
            {
                "layout": "kv_backing_v2",
                "offset": offset,
                "capacity": int(keys.shape[2]),
                "saved_spare_tokens": max(0, int(keys.shape[2]) - offset),
            },
        )

    return layer_cache.state, {"layout": "state_v1"}


def _prompt_cache_ssd_pad_sequence_capacity(value, target_capacity):
    if not _is_mx_array(value) or len(value.shape) < 3:
        return value
    current = int(value.shape[2])
    target = max(current, int(target_capacity or 0))
    if target <= current:
        return value
    padding = [(0, 0)] * len(value.shape)
    padding[2] = (0, target - current)
    return mx.pad(value, padding)


def _prompt_cache_ssd_restore_backing_state(
    layer_cache,
    state,
    storage,
    *,
    target_capacity=0,
):
    """Install schema-v3 state with a bounded request append capacity."""
    storage = storage or {}
    layout = str(storage.get("layout") or "")

    if layout in {"nested_kv_backing_v1", "nested_kv_backing_v2"}:
        kv_state, index_state = state
        if not isinstance(kv_state, (tuple, list)) or len(kv_state) != 2:
            raise ValueError("invalid nested KV backing state")
        keys, values = kv_state
        kv_offset = int(storage.get("kv_offset") or 0)
        index_offset = int(storage.get("index_offset") or 0)
        step = max(1, int(getattr(layer_cache, "step", 1) or 1))
        requested_capacity = _prompt_cache_ssd_round_capacity(
            max(kv_offset, index_offset, int(target_capacity or 0)),
            0,
            step,
        )
        # Do not inherit oversized spare capacity from an artifact. Crop before
        # installing the state, then pad only to this request's bounded target.
        keys = _prompt_cache_ssd_bound_sequence(
            keys, requested_capacity, 0, step
        )
        values = _prompt_cache_ssd_bound_sequence(
            values, requested_capacity, 0, step
        )
        if index_state is not None:
            index_state = _prompt_cache_ssd_bound_sequence(
                index_state, requested_capacity, 0, step
            )
        layer_cache.state = ((keys, values), index_state)
        nested = getattr(layer_cache, "kv_cache", None)
        if nested is None:
            raise ValueError("nested KV backing metadata on non-nested cache")
        keys = getattr(nested, "keys", None)
        index_keys = getattr(layer_cache, "index_keys", None)
        kv_capacity = int(keys.shape[2]) if keys is not None else 0
        index_capacity = int(index_keys.shape[2]) if index_keys is not None else 0
        if kv_offset < 0 or kv_offset > kv_capacity:
            raise ValueError(
                f"invalid restored KV offset {kv_offset}/{kv_capacity}"
            )
        if index_offset < 0 or index_offset > index_capacity:
            raise ValueError(
                f"invalid restored index offset {index_offset}/{index_capacity}"
            )
        if requested_capacity > kv_capacity:
            nested.keys = _prompt_cache_ssd_pad_sequence_capacity(
                nested.keys, requested_capacity
            )
            nested.values = _prompt_cache_ssd_pad_sequence_capacity(
                nested.values, requested_capacity
            )
        if index_keys is not None and requested_capacity > index_capacity:
            layer_cache.index_keys = _prompt_cache_ssd_pad_sequence_capacity(
                index_keys, requested_capacity
            )
        nested.offset = kv_offset
        layer_cache.index_offset = index_offset
        return {
            "layout": layout,
            "offset": kv_offset,
            "capacity": int(nested.keys.shape[2]),
            "index_offset": index_offset,
            "index_capacity": (
                int(layer_cache.index_keys.shape[2])
                if layer_cache.index_keys is not None
                else 0
            ),
        }

    if layout in {"kv_backing_v1", "kv_backing_v2"}:
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("invalid KV backing state")
        keys, values = state
        offset = int(storage.get("offset") or 0)
        step = max(1, int(getattr(layer_cache, "step", 1) or 1))
        requested_capacity = _prompt_cache_ssd_round_capacity(
            max(offset, int(target_capacity or 0)),
            0,
            step,
        )
        keys = _prompt_cache_ssd_bound_sequence(
            keys, requested_capacity, 0, step
        )
        values = _prompt_cache_ssd_bound_sequence(
            values, requested_capacity, 0, step
        )
        layer_cache.state = (keys, values)
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        capacity = int(keys.shape[2]) if keys is not None else 0
        if offset < 0 or offset > capacity:
            raise ValueError(f"invalid restored KV offset {offset}/{capacity}")
        if requested_capacity > capacity:
            layer_cache.keys = _prompt_cache_ssd_pad_sequence_capacity(
                keys, requested_capacity
            )
            layer_cache.values = _prompt_cache_ssd_pad_sequence_capacity(
                values, requested_capacity
            )
        layer_cache.offset = offset
        return {
            "layout": layout,
            "offset": offset,
            "capacity": int(layer_cache.keys.shape[2]),
        }

    if layout != "state_v1":
        raise ValueError(f"unsupported SSD cache backing layout: {layout!r}")
    layer_cache.state = state
    return {"layout": layout, "offset": None, "capacity": None}


def _write_json_atomic(path, payload):
    tmp = f"{path}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
    os.replace(tmp, path)


def _dir_size_bytes(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _prompt_cache_ssd_scan_unlocked():
    root = _prompt_cache_ssd_root()
    rank, _ = _prompt_cache_ssd_current_rank_world()
    entries = []
    total_bytes = 0
    if not os.path.isdir(root):
        return {"entries": entries, "entry_count": 0, "total_bytes": 0}
    for session_hash in sorted(os.listdir(root)):
        if session_hash.startswith(".") or session_hash == "manifest.json":
            continue
        session_dir = os.path.join(root, session_hash)
        if not os.path.isdir(session_dir):
            continue
        entry_bytes = _dir_size_bytes(session_dir)
        total_bytes += entry_bytes
        rank_dir = os.path.join(session_dir, f"rank{rank}")
        meta_path = os.path.join(rank_dir, "session.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                meta = {"read_error": str(e)}
        saved_at = meta.get("saved_at")
        if saved_at is None:
            try:
                saved_at = os.path.getmtime(session_dir)
            except Exception:
                saved_at = None
        entries.append({
            "session_hash": session_hash,
            "rank": rank,
            "session_key": meta.get("session_key_label"),
            "session_source": meta.get("session_source"),
            "key_tokens": meta.get("key_tokens"),
            "cache_len": meta.get("cache_len"),
            "complete": bool(meta.get("complete")),
            "saved_at": saved_at,
            "last_access_at": meta.get("last_access_at") or saved_at,
            "bytes": entry_bytes,
            "rank_exists": os.path.isdir(rank_dir),
            "restore_ready": bool(meta.get("complete")),
            "runtime_hash": meta.get("runtime_hash"),
            "token_ids_hash": meta.get("token_ids_hash"),
            "error": meta.get("read_error"),
        })
    entries.sort(key=lambda row: float(row.get("last_access_at") or 0), reverse=True)
    return {
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": total_bytes,
    }


def _prompt_cache_ssd_invalidate_scan_cache_unlocked():
    _prompt_cache_ssd_scan_cache["at"] = None
    _prompt_cache_ssd_scan_cache["scan"] = None


def _prompt_cache_ssd_scan_cached_unlocked(force=False):
    now = time.time()
    cached_at = _prompt_cache_ssd_scan_cache.get("at")
    cached_scan = _prompt_cache_ssd_scan_cache.get("scan")
    if (
        not force
        and cached_scan is not None
        and cached_at is not None
        and PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS > 0
        and now - float(cached_at) < PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS
    ):
        return cached_scan
    scan = _prompt_cache_ssd_scan_unlocked()
    _prompt_cache_ssd_scan_cache["at"] = now
    _prompt_cache_ssd_scan_cache["scan"] = scan
    _prompt_cache_ssd_state["last_scan_at"] = round(now, 3)
    return scan


def _prompt_cache_ssd_write_manifest_unlocked(scan=None):
    root = _prompt_cache_ssd_root()
    os.makedirs(root, exist_ok=True)
    scan = scan or _prompt_cache_ssd_scan_unlocked()
    payload = {
        "schema": PROMPT_CACHE_SSD_SCHEMA_VERSION,
        "updated_at": round(time.time(), 3),
        "entry_count": scan.get("entry_count", 0),
        "total_bytes": scan.get("total_bytes", 0),
        "entries": scan.get("entries", [])[:PROMPT_CACHE_SSD_RECENT_ENTRIES],
    }
    _write_json_atomic(os.path.join(root, "manifest.json"), payload)


def _prompt_cache_ssd_prune_unlocked(reason="prune"):
    if not PROMPT_CACHE_SSD_ENABLED:
        return {"ok": True, "enabled": False, "removed": 0, "bytes_removed": 0}
    root = _prompt_cache_ssd_root()
    scan = _prompt_cache_ssd_scan_unlocked()
    now = time.time()
    entries = list(scan.get("entries") or [])
    remove_hashes = set()
    for row in entries:
        saved_at = float(row.get("saved_at") or 0.0)
        if (
            PROMPT_CACHE_SSD_TTL_SECONDS > 0
            and saved_at > 0
            and now - saved_at > PROMPT_CACHE_SSD_TTL_SECONDS
        ):
            remove_hashes.add(row["session_hash"])
    remaining = [row for row in entries if row["session_hash"] not in remove_hashes]
    total = sum(int(row.get("bytes") or 0) for row in remaining)
    # Re-read the live-tunable cap on every prune (env-seeded at boot; rank0's
    # runtime-tuning endpoint can move it, rank1 keeps the env default).
    ssd_max_bytes = _runtime_prompt_cache_ssd_max_bytes()
    if ssd_max_bytes > 0 and total > ssd_max_bytes:
        for row in sorted(remaining, key=lambda r: float(r.get("last_access_at") or 0.0)):
            if total <= ssd_max_bytes:
                break
            remove_hashes.add(row["session_hash"])
            total -= int(row.get("bytes") or 0)
    removed = 0
    bytes_removed = 0
    for session_hash in remove_hashes:
        path = os.path.join(root, session_hash)
        bytes_removed += _dir_size_bytes(path)
        try:
            shutil.rmtree(path)
            removed += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            _prompt_cache_ssd_state["last_error"] = str(e)
            logger.warning("prompt-cache SSD prune failed for %s: %s", session_hash, e)
    now_rounded = round(time.time(), 3)
    _prompt_cache_ssd_state.update({
        "last_prune_at": now_rounded,
        "pruned_sessions": int(_prompt_cache_ssd_state.get("pruned_sessions") or 0) + removed,
    })
    _prompt_cache_ssd_invalidate_scan_cache_unlocked()
    try:
        _prompt_cache_ssd_write_manifest_unlocked()
    except Exception as e:
        _prompt_cache_ssd_state["last_error"] = str(e)
    return {
        "ok": True,
        "enabled": True,
        "reason": reason,
        "removed": removed,
        "bytes_removed": bytes_removed,
    }


def _prompt_cache_ssd_clear_unlocked(reason="clear"):
    root = _prompt_cache_ssd_root()
    removed_bytes = _dir_size_bytes(root)
    try:
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)
        now = round(time.time(), 3)
        _prompt_cache_ssd_state.update({
            "last_clear_at": now,
            "last_error": None,
            "last_restore_miss_reason": f"cleared:{reason}",
        })
        _prompt_cache_ssd_invalidate_scan_cache_unlocked()
        return {"ok": True, "removed_bytes": removed_bytes, "path": root}
    except Exception as e:
        _prompt_cache_ssd_state["last_error"] = str(e)
        return {"ok": False, "error": str(e), "path": root}


def _prompt_cache_ssd_layer_path(rank_dir, index):
    return os.path.join(rank_dir, f"layer-{int(index):03d}.safetensors")


def _prompt_cache_ssd_record_autosave_anchor_unlocked(
    session_key, token_ids, runtime_hash=None
):
    key = str(session_key or "")
    if not key:
        return
    ids = list(token_ids or [])
    _prompt_cache_ssd_autosave_anchors[key] = {
        "tokens": len(ids),
        "token_hash": _token_ids_sha256(ids),
        "runtime_hash": str(runtime_hash or ""),
    }
    _prompt_cache_ssd_autosave_anchors.move_to_end(key)
    while len(_prompt_cache_ssd_autosave_anchors) > max(
        8, PROMPT_CACHE_SESSION_MAP_MAX
    ):
        _prompt_cache_ssd_autosave_anchors.popitem(last=False)


def _prompt_cache_ssd_autosave_due_unlocked(model, processor):
    """Return whether this rank should persist the current completed KV state.

    A 256k two-rank checkpoint is roughly 33 GiB. Rewriting it after every
    tiny follow-up adds seconds after the final token and unnecessary SSD
    wear. Anchors are advanced only after a successful save or coordinated
    restore, so both ranks make the same deterministic token-delta decision.
    """
    holder = _prompt_cache_holder
    token_ids = list(holder.get("token_ids") or [])
    session_id = holder.get("session_id")
    session_source = holder.get("session_source")
    if not PROMPT_CACHE_SSD_ENABLED or not PROMPT_CACHE_SSD_AUTO_SAVE:
        return False, "disabled"
    if not session_id:
        return True, "no_session_anchor"
    if len(token_ids) < PROMPT_CACHE_SSD_MIN_TOKENS:
        return True, "below_min_tokens"
    session_key = _prompt_cache_session_key(session_id, session_source)
    anchor = _prompt_cache_ssd_autosave_anchors.get(session_key)
    if not anchor:
        return True, "first_checkpoint"

    current_tokens = len(token_ids)
    saved_tokens = int(anchor.get("tokens") or 0)
    if current_tokens < saved_tokens:
        return True, "cache_rewound"
    if current_tokens == saved_tokens:
        current_hash = _token_ids_sha256(token_ids)
        if current_hash != str(anchor.get("token_hash") or ""):
            return True, "same_length_branch_changed"
        return False, "unchanged"

    delta = current_tokens - saved_tokens
    if delta < PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS:
        return False, f"delta_below_threshold:{delta}"
    return True, f"delta_threshold_reached:{delta}"


def _prompt_cache_ssd_maybe_autosave_unlocked(
    model, processor, *, prompt=None, reason="update"
):
    due, decision = _prompt_cache_ssd_autosave_due_unlocked(model, processor)
    if not due:
        now = round(time.time(), 3)
        _prompt_cache_ssd_state.update({
            "last_auto_save_deferred_at": now,
            "last_auto_save_deferred_reason": decision,
            "auto_save_deferred_count": int(
                _prompt_cache_ssd_state.get("auto_save_deferred_count") or 0
            ) + 1,
        })
        logger.info("prompt-cache SSD autosave deferred (%s)", decision)
        return False
    if prompt is not None:
        _prompt_cache_make_ssd_checkpoint_unlocked(prompt=prompt, reason=reason)
    return _prompt_cache_ssd_save_current_unlocked(
        model,
        processor,
        reason=f"{reason}:{decision}",
    )


def _prompt_cache_ssd_save_current_unlocked(model, processor, *, reason="update"):
    if not PROMPT_CACHE_SSD_ENABLED:
        return False
    holder = _prompt_cache_holder
    cache = holder.get("cache")
    token_ids = list(holder.get("token_ids") or [])
    session_id = holder.get("session_id")
    session_source = holder.get("session_source")
    cache_len = int(holder.get("cache_len") or 0)
    if cache is None:
        return False
    if not session_id:
        _prompt_cache_ssd_state["last_restore_miss_reason"] = "save_skipped:no_session_id"
        return False
    if len(token_ids) < PROMPT_CACHE_SSD_MIN_TOKENS:
        _prompt_cache_ssd_state["last_restore_miss_reason"] = "save_skipped:below_min_tokens"
        return False
    if cache_len <= 0 or len(token_ids) != cache_len:
        _prompt_cache_ssd_state["last_restore_miss_reason"] = "save_skipped:key_cache_len_mismatch"
        return False
    counted = _cache_token_count(cache)
    if counted and counted != cache_len:
        _prompt_cache_ssd_state["last_restore_miss_reason"] = "save_skipped:cache_offset_mismatch"
        return False
    generated = int(holder.get("last_generated_tokens") or 0)
    if generated > 0 and not holder.get("last_exact_generated_ids"):
        _prompt_cache_ssd_state["last_restore_miss_reason"] = "save_skipped:generated_ids_not_exact"
        return False

    rank, world = _prompt_cache_ssd_current_rank_world()
    session_key = _prompt_cache_session_key(session_id, session_source)
    session_hash = _prompt_cache_ssd_session_hash(session_key)
    root = _prompt_cache_ssd_root()
    rank_dir = _prompt_cache_ssd_rank_dir(session_hash, rank)
    tmp_dir = f"{rank_dir}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    try:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        runtime = _prompt_cache_ssd_runtime_fingerprint(model, processor, cache)
        token_hash = _token_ids_sha256(token_ids)
        tail_hash = _token_ids_sha256(token_ids[-min(len(token_ids), 4096):])
        layer_meta = []
        saved_capacities = []
        saved_spares = []
        # Admin/manual save runs from an HTTP thread, not necessarily the MLX
        # generation thread. Give MLX an explicit stream before evaluating and
        # serializing tensors from the cached KV state.
        with mx.stream(mx.default_device()):
            mx.save_safetensors(
                os.path.join(tmp_dir, "tokens.safetensors"),
                {"token_ids": mx.array(token_ids, dtype=mx.int32)},
            )
            for idx, layer_cache in enumerate(cache):
                arrays = {}
                backing_state, storage = _prompt_cache_ssd_backing_state(
                    layer_cache,
                    max_spare_tokens=PROMPT_CACHE_SSD_SAVE_RESERVE_TOKENS,
                )
                structure = _flatten_cache_state(backing_state, arrays, f"layer_{idx}")
                if arrays:
                    mx.eval(*arrays.values())
                layer_path = _prompt_cache_ssd_layer_path(tmp_dir, idx)
                mx.save_safetensors(layer_path, arrays)
                capacity = storage.get("kv_capacity", storage.get("capacity"))
                if capacity is not None:
                    saved_capacities.append(int(capacity))
                    saved_spares.append(max(0, int(capacity) - cache_len))
                layer_meta.append({
                    "index": idx,
                    "class": layer_cache.__class__.__module__ + "." + layer_cache.__class__.__name__,
                    "file": os.path.basename(layer_path),
                    "storage": storage,
                    "structure": structure,
                    "arrays": {
                        name: {
                            "shape": meta.get("shape"),
                            "dtype": meta.get("dtype"),
                            "nbytes": meta.get("nbytes"),
                        }
                        for name, meta in _flatten_arrays_from_structure(structure).items()
                    },
                })
        total_bytes = _dir_size_bytes(tmp_dir)
        now = round(time.time(), 3)
        metadata = {
            "schema": PROMPT_CACHE_SSD_SCHEMA_VERSION,
            "complete": True,
            "saved_at": now,
            "last_access_at": now,
            "reason": reason,
            "rank": rank,
            "rank_count": world,
            "session_hash": session_hash,
            "session_key_label": _prompt_cache_ssd_session_label(session_key),
            "session_id": session_id,
            "session_source": session_source,
            "model_id": MODEL_ID,
            "model": MODEL,
            "runtime": runtime,
            "runtime_hash": runtime.get("hash"),
            "key_tokens": len(token_ids),
            "cache_len": cache_len,
            "last_input_tokens": int(holder.get("last_input_tokens") or 0),
            "last_generated_tokens": generated,
            "last_exact_generated_ids": bool(holder.get("last_exact_generated_ids")),
            "token_ids_hash": token_hash,
            "token_ids_tail_hash": tail_hash,
            "token_ids_dtype": "int32",
            "layers": layer_meta,
            "layer_count": len(layer_meta),
            "save_reserve_tokens": PROMPT_CACHE_SSD_SAVE_RESERVE_TOKENS,
            "saved_min_capacity": (
                min(saved_capacities) if saved_capacities else cache_len
            ),
            "saved_max_capacity": (
                max(saved_capacities) if saved_capacities else cache_len
            ),
            "saved_max_spare_tokens": max(saved_spares) if saved_spares else 0,
            "bytes": total_bytes,
            "privacy": {
                "raw_prompt_text": False,
                "stores_token_ids": True,
                "stores_kv_tensors": True,
                "note": "Token ids and KV tensors can still reveal prompt content; keep this directory local.",
            },
        }
        _write_json_atomic(os.path.join(tmp_dir, "session.json"), metadata)
        os.makedirs(os.path.dirname(rank_dir), exist_ok=True)
        if os.path.isdir(rank_dir):
            shutil.rmtree(rank_dir)
        os.replace(tmp_dir, rank_dir)
        total_bytes = _dir_size_bytes(rank_dir)
        metadata["bytes"] = total_bytes
        _write_json_atomic(os.path.join(rank_dir, "session.json"), metadata)
        try:
            _prompt_cache_ssd_write_manifest_unlocked()
        except Exception as e:
            logger.debug("prompt-cache SSD manifest write failed: %s", e)
        _prompt_cache_ssd_state.update({
            "last_save_at": now,
            "last_error": None,
            "last_saved_session": metadata["session_key_label"],
            "saved_sessions": int(_prompt_cache_ssd_state.get("saved_sessions") or 0) + 1,
            "last_saved_tokens": len(token_ids),
            "last_saved_bytes": total_bytes,
            "last_saved_capacity": (
                min(saved_capacities) if saved_capacities else cache_len
            ),
            "last_saved_spare_tokens": max(saved_spares) if saved_spares else 0,
            "last_auto_save_deferred_reason": None,
        })
        _prompt_cache_ssd_record_autosave_anchor_unlocked(
            session_key,
            token_ids,
            runtime.get("hash"),
        )
        _prompt_cache_ssd_invalidate_scan_cache_unlocked()
        entry = _prompt_cache_session_map.get(session_key)
        if entry is not None:
            entry["ssd_rehydratable"] = True
            entry["ssd_saved_at"] = now
            entry["ssd_cache_bytes"] = total_bytes
            entry["ssd_key_tokens"] = len(token_ids)
            entry["ssd_session_hash"] = session_hash[:16]
            _write_prompt_cache_session_manifest_unlocked()
        logger.info(
            "prompt-cache SSD saved %s rank%s (%d tokens, %.2f GiB)",
            session_key,
            rank,
            len(token_ids),
            total_bytes / 1024**3,
        )
        _prompt_cache_ssd_prune_unlocked(reason="after_save")
        return True
    except Exception as e:
        _prompt_cache_ssd_state["last_error"] = str(e)
        logger.warning("prompt-cache SSD save failed: %s", e)
        try:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return False


def _flatten_arrays_from_structure(structure, out=None):
    if out is None:
        out = {}
    if not isinstance(structure, dict):
        return out
    if structure.get("type") == "array":
        out[structure["name"]] = structure
    for item in structure.get("items") or []:
        _flatten_arrays_from_structure(item, out)
    return out


def _prompt_cache_ssd_thinking_boundary_restore_safe(
    processor, stored_ids, token_ids, restore_tokens
):
    """Allow only the harmless trailing MiniMax thinking-start marker crop."""
    stored_ids = [int(token) for token in (stored_ids or [])]
    token_ids = [int(token) for token in (token_ids or [])]
    restore_tokens = int(restore_tokens or 0)
    if len(token_ids) <= len(stored_ids):
        return False
    if restore_tokens != len(stored_ids) - 1:
        return False
    if stored_ids[:restore_tokens] != token_ids[:restore_tokens]:
        return False
    try:
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        with _tokenizer_runtime_lock:
            marker_ids = tok.encode("<mm:think>", add_special_tokens=False)
        marker_ids = [int(token) for token in marker_ids]
    except Exception:
        return False
    return len(marker_ids) == 1 and stored_ids[restore_tokens:] == marker_ids


def _prompt_cache_ssd_load_candidate_unlocked(model, processor, token_ids,
                                              session_id, session_source,
                                              allow_partial_restore=True,
                                              allow_thinking_boundary_restore=False,
                                              append_reserve_tokens=0):
    if not PROMPT_CACHE_SSD_ENABLED:
        return None, None, None, "disabled"
    if not PROMPT_CACHE_SSD_RESTORE_ENABLED:
        return None, None, None, "restore_disabled"
    if not session_id:
        return None, None, None, "no_session_id"
    token_ids = list(token_ids or [])
    if not token_ids:
        return None, None, None, "no_token_ids"

    rank, world = _prompt_cache_ssd_current_rank_world()
    session_key = _prompt_cache_session_key(session_id, session_source)
    session_hash = _prompt_cache_ssd_session_hash(session_key)
    rank_dir = _prompt_cache_ssd_rank_dir(session_hash, rank)
    meta_path = os.path.join(rank_dir, "session.json")
    if not os.path.exists(meta_path):
        return None, None, None, "artifact_missing"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("schema") or 0) != PROMPT_CACHE_SSD_SCHEMA_VERSION:
            return None, None, None, "schema_mismatch"
        if not meta.get("complete"):
            return None, None, None, "incomplete_artifact"
        artifact_rank = meta.get("rank")
        if artifact_rank is None:
            artifact_rank = -1
        if int(artifact_rank) != rank:
            return None, None, None, (
                f"rank_mismatch:current={rank}:artifact={artifact_rank}:"
                f"path={os.path.basename(rank_dir)}"
            )
        if int(meta.get("rank_count") or 0) != world:
            return None, None, None, (
                f"rank_count_mismatch:current={world}:artifact={meta.get('rank_count')}"
            )
        key_tokens = int(meta.get("key_tokens") or 0)
        cache_len = int(meta.get("cache_len") or 0)
        if key_tokens <= 0 or cache_len != key_tokens:
            return None, None, None, "unsafe_key_cache_len"
        tokens_payload = mx.load(os.path.join(rank_dir, "tokens.safetensors"))
        stored_ids_array = tokens_payload.get("token_ids")
        if stored_ids_array is None:
            return None, None, None, "token_ids_missing"
        stored_ids = [int(v) for v in stored_ids_array.tolist()]
        if len(stored_ids) != key_tokens:
            return None, None, None, "token_ids_length_mismatch"
        if _token_ids_sha256(stored_ids) != meta.get("token_ids_hash"):
            return None, None, None, "stored_token_hash_mismatch"
        restore_tokens = key_tokens
        partial_restore = False
        thinking_boundary_restore = False
        if len(token_ids) < key_tokens:
            reuse = _common_prefix_len(stored_ids, token_ids)
            partial_restore = True
            restore_tokens = reuse
        else:
            prefix_hash = _token_ids_sha256(token_ids[:key_tokens])
            if prefix_hash != meta.get("token_ids_hash"):
                reuse = _common_prefix_len(stored_ids, token_ids)
                partial_restore = True
                restore_tokens = reuse
        if partial_restore:
            if not allow_partial_restore:
                thinking_boundary_restore = (
                    allow_thinking_boundary_restore
                    and _prompt_cache_ssd_thinking_boundary_restore_safe(
                        processor,
                        stored_ids,
                        token_ids,
                        restore_tokens,
                    )
                )
                if not thinking_boundary_restore:
                    return None, None, None, (
                        f"partial_restore_disabled:prefix={restore_tokens}:"
                        f"stored={key_tokens}"
                    )
                logger.info(
                    "prompt-cache SSD accepted trailing <mm:think> boundary "
                    "crop (%d/%d tokens)",
                    restore_tokens,
                    key_tokens,
                )
            partial_min = max(
                PROMPT_CACHE_MIN_REUSE,
                min(PROMPT_CACHE_SSD_MIN_TOKENS, key_tokens),
            )
            partial_ratio = (restore_tokens / key_tokens) if key_tokens else 0.0
            if restore_tokens < partial_min or partial_ratio < 0.95:
                return None, None, None, (
                    f"token_hash_mismatch:prefix={restore_tokens}:"
                    f"stored={key_tokens}:ratio={partial_ratio:.4f}"
                )

        requested_append_reserve_tokens = max(
            0, int(append_reserve_tokens or 0)
        )
        append_reserve_tokens = min(
            requested_append_reserve_tokens,
            PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS,
        )
        requested_capacity = len(token_ids) + append_reserve_tokens
        if MAX_KV_SIZE > 0:
            requested_capacity = min(requested_capacity, MAX_KV_SIZE)
        restore_target_capacity = max(key_tokens, requested_capacity)

        from mlx_vlm.models import cache as _cache_mod

        # Do not reuse or mutate the holder's current object for restore
        # staging. Commit only after both ranks validate and agree.
        new_cache = _cache_mod.make_prompt_cache(
            model.language_model, max_kv_size=MAX_KV_SIZE
        )
        runtime = _prompt_cache_ssd_runtime_fingerprint(model, processor, new_cache)
        if runtime.get("hash") != meta.get("runtime_hash"):
            return None, None, None, "runtime_hash_mismatch"
        layers = meta.get("layers") or []
        if len(layers) != len(new_cache):
            return None, None, None, "layer_count_mismatch"
        restored_capacities = []
        for layer_meta, layer_cache in zip(layers, new_cache):
            expected_class = layer_cache.__class__.__module__ + "." + layer_cache.__class__.__name__
            if layer_meta.get("class") != expected_class:
                return None, None, None, "cache_class_mismatch"
            arrays = mx.load(os.path.join(rank_dir, layer_meta.get("file")))
            state = _restore_cache_state(layer_meta.get("structure"), arrays)
            restored_storage = _prompt_cache_ssd_restore_backing_state(
                layer_cache,
                state,
                layer_meta.get("storage"),
                target_capacity=restore_target_capacity,
            )
            layer_arrays = _cache_state_arrays(
                _prompt_cache_ssd_backing_state(layer_cache)[0]
            )
            if layer_arrays:
                # Materialize one layer at a time. At 350k, retaining every
                # loaded source tensor until one final mx.eval doubles the
                # transient KV footprint and can exhaust rank-0 headroom.
                mx.eval(*layer_arrays)
            capacity = restored_storage.get("capacity")
            if capacity is not None:
                restored_capacities.append(int(capacity))
            del arrays, state, layer_arrays
        if restore_tokens < key_tokens:
            trim_tokens = key_tokens - restore_tokens
            if not _trim_prompt_cache_in_place(new_cache, trim_tokens):
                return None, None, None, "partial_restore_trim_failed"
        if restore_tokens < key_tokens:
            stored_ids = stored_ids[:restore_tokens]
            meta["_partial_restore"] = True
            meta["_stored_key_tokens"] = key_tokens
            meta["_thinking_boundary_restore"] = thinking_boundary_restore
        meta["_restore_tokens"] = restore_tokens
        meta["_restore_target_capacity"] = restore_target_capacity
        meta["_restore_capacity"] = (
            min(restored_capacities) if restored_capacities else restore_tokens
        )
        meta["_restore_requested_append_reserve_tokens"] = (
            requested_append_reserve_tokens
        )
        meta["_restore_append_reserve_tokens"] = append_reserve_tokens
        meta["last_access_at"] = round(time.time(), 3)
        try:
            _write_json_atomic(
                meta_path,
                {key: value for key, value in meta.items() if not key.startswith("_")},
            )
        except Exception:
            pass
        return new_cache, stored_ids, meta, None
    except Exception as e:
        return None, None, None, f"restore_error:{e}"


def _prompt_cache_ssd_try_restore_unlocked(model, processor, token_ids,
                                           session_id=None, session_source=None,
                                           allow_partial_restore=True,
                                           allow_thinking_boundary_restore=False,
                                           append_reserve_tokens=0):
    if not PROMPT_CACHE_SSD_ENABLED:
        return None
    if not PROMPT_CACHE_SSD_RESTORE_ENABLED:
        _prompt_cache_ssd_state["last_restore_miss_reason"] = "restore_disabled"
        return None
    cache, stored_ids, meta, miss_reason = _prompt_cache_ssd_load_candidate_unlocked(
        model, processor, token_ids, session_id, session_source,
        allow_partial_restore=allow_partial_restore,
        allow_thinking_boundary_restore=allow_thinking_boundary_restore,
        append_reserve_tokens=append_reserve_tokens,
    )
    rank, world = _prompt_cache_ssd_current_rank_world()
    local_ok = 1 if cache is not None and stored_ids and meta else 0
    local_cache_len = int((meta or {}).get("_restore_tokens") or 0) if local_ok else 0
    local_key_tokens = local_cache_len
    try:
        ok_total = mx.distributed.all_sum(mx.array(local_ok, dtype=mx.int32))
        len_total = mx.distributed.all_sum(mx.array(local_cache_len, dtype=mx.int64))
        key_total = mx.distributed.all_sum(mx.array(local_key_tokens, dtype=mx.int64))
        mx.eval(ok_total, len_total, key_total)
        all_ok = int(ok_total.item()) == world
        same_len = int(len_total.item()) == local_cache_len * world
        same_key = int(key_total.item()) == local_key_tokens * world
    except Exception as e:
        all_ok = False
        same_len = False
        same_key = False
        miss_reason = f"restore_sync_failed:{e}"
    if not (all_ok and same_len and same_key and local_ok):
        reason = miss_reason or "rank_restore_mismatch"
        _prompt_cache_ssd_state["last_restore_miss_reason"] = reason
        _prompt_cache_ssd_state["last_error"] = None
        if cache is not None:
            del cache
            gc.collect()
        return None

    session_key = _prompt_cache_session_key(session_id, session_source)
    current_key = _prompt_cache_current_session_key_unlocked()
    if current_key != session_key:
        _prompt_cache_stash_current_unlocked(reason=f"ssd_restore:{session_key}")
    _prompt_cache_holder["cache"] = cache
    _prompt_cache_holder["token_ids"] = stored_ids
    _prompt_cache_holder["prompt"] = None
    _prompt_cache_holder["cache_len"] = local_cache_len
    _prompt_cache_holder["last_input_tokens"] = min(
        int(meta.get("last_input_tokens") or local_key_tokens),
        local_key_tokens,
    )
    _prompt_cache_holder["last_generated_tokens"] = 0
    _prompt_cache_holder["last_exact_generated_ids"] = False
    _prompt_cache_holder["session_id"] = session_id
    _prompt_cache_holder["session_source"] = session_source
    _prompt_cache_holder["created_at"] = meta.get("saved_at") or round(time.time(), 3)
    _prompt_cache_holder["last_access_at"] = round(time.time(), 3)
    _prompt_cache_ssd_state.update({
        "last_restore_at": _prompt_cache_holder["last_access_at"],
        "last_restore_miss_reason": None,
        "last_restored_session": meta.get("session_key_label") or session_key,
        "restored_sessions": int(_prompt_cache_ssd_state.get("restored_sessions") or 0) + 1,
        "last_restored_tokens": local_cache_len,
        "last_restore_target_capacity": int(
            meta.get("_restore_capacity") or local_cache_len
        ),
        "last_restore_requested_append_reserve_tokens": int(
            meta.get("_restore_requested_append_reserve_tokens") or 0
        ),
        "last_restore_append_reserve_tokens": int(
            meta.get("_restore_append_reserve_tokens") or 0
        ),
        "last_restore_thinking_boundary": bool(
            meta.get("_thinking_boundary_restore")
        ),
    })
    _prompt_cache_ssd_record_autosave_anchor_unlocked(
        session_key,
        stored_ids,
        meta.get("runtime_hash"),
    )
    _set_prompt_cache_event(
        "ssd_restore",
        prompt_tokens=len(token_ids or []),
        reuse_tokens=local_key_tokens,
        suffix_tokens=max(0, len(token_ids or []) - local_key_tokens),
        cache_len=local_cache_len,
        session_id=session_id,
        session_source=session_source,
        ssd_session_hash=(meta.get("session_hash") or "")[:16],
        ssd_rank=rank,
        ssd_restore_capacity=int(meta.get("_restore_capacity") or local_cache_len),
        ssd_requested_append_reserve_tokens=int(
            meta.get("_restore_requested_append_reserve_tokens") or 0
        ),
        ssd_append_reserve_tokens=int(
            meta.get("_restore_append_reserve_tokens") or 0
        ),
        ssd_thinking_boundary_restore=bool(
            meta.get("_thinking_boundary_restore")
        ),
        **_prompt_cache_match_fields(len(token_ids or []), local_key_tokens),
    )
    logger.info(
        "prompt-cache SSD restored %s rank%s (%d/%d tokens)",
        session_key,
        rank,
        local_key_tokens,
        len(token_ids or []),
    )
    return {
        "restored_ssd": True,
        "ssd_session_hash": (meta.get("session_hash") or "")[:16],
        "ssd_saved_at": meta.get("saved_at"),
        "restored_cache_len": local_cache_len,
        "ssd_restore_capacity": int(
            meta.get("_restore_capacity") or local_cache_len
        ),
        "ssd_requested_append_reserve_tokens": int(
            meta.get("_restore_requested_append_reserve_tokens") or 0
        ),
        "ssd_append_reserve_tokens": int(
            meta.get("_restore_append_reserve_tokens") or 0
        ),
        "ssd_partial_restore": bool(meta.get("_partial_restore")),
        "ssd_thinking_boundary_restore": bool(
            meta.get("_thinking_boundary_restore")
        ),
        "ssd_stored_key_tokens": int(meta.get("_stored_key_tokens") or local_key_tokens),
    }


def _prompt_cache_ssd_restore_eligible_unlocked(token_ids, session_id):
    if not PROMPT_CACHE_SSD_ENABLED or not PROMPT_CACHE_SSD_RESTORE_ENABLED:
        return False
    if not session_id:
        return False
    if len(token_ids or []) < PROMPT_CACHE_SSD_MIN_TOKENS:
        return False
    return True


def _prompt_cache_ssd_maybe_restore_unlocked(model, processor, token_ids,
                                             session_id=None, session_source=None,
                                             reason="miss",
                                             allow_partial_restore=True,
                                             allow_thinking_boundary_restore=False,
                                             append_reserve_tokens=0):
    """Try durable restore only when RAM/live reuse has already missed.

    This helper intentionally does not inspect local artifact existence before
    calling the distributed restore path. All ranks must make the same decision;
    the restore path itself synchronizes success/failure safely.
    """
    if not _prompt_cache_ssd_restore_eligible_unlocked(token_ids, session_id):
        return None
    _prompt_cache_ssd_state["last_restore_attempt_reason"] = reason
    return _prompt_cache_ssd_try_restore_unlocked(
        model,
        processor,
        token_ids,
        session_id=session_id,
        session_source=session_source,
        allow_partial_restore=allow_partial_restore,
        allow_thinking_boundary_restore=allow_thinking_boundary_restore,
        append_reserve_tokens=append_reserve_tokens,
    )


def _prompt_cache_make_ssd_checkpoint_unlocked(prompt=None, reason="manual_save"):
    """Trim unsafe generated tails only for explicit durable-save operations."""
    if not PROMPT_CACHE_SSD_ENABLED:
        return False
    holder = _prompt_cache_holder
    cache_obj = holder.get("cache")
    key_ids = list(holder.get("token_ids") or [])
    generated_count = int(holder.get("last_generated_tokens") or 0)
    cache_len = int(holder.get("cache_len") or 0)
    if (
        cache_obj is None
        or generated_count <= 0
        or holder.get("last_exact_generated_ids")
        or not key_ids
        or cache_len <= len(key_ids)
    ):
        return False
    trim_tokens = cache_len - len(key_ids)
    if trim_tokens != generated_count:
        _prompt_cache_ssd_state["last_restore_miss_reason"] = (
            "save_skipped:checkpoint_len_mismatch"
        )
        return False
    if not _trim_prompt_cache_in_place(cache_obj, trim_tokens):
        _prompt_cache_ssd_state["last_restore_miss_reason"] = (
            "save_skipped:prompt_prefix_trim_failed"
        )
        return False
    with mx.stream(mx.default_device()):
        mx.eval([c.state for c in cache_obj])
    holder["cache_len"] = len(key_ids)
    holder["last_generated_tokens"] = 0
    holder["last_exact_generated_ids"] = False
    holder["prompt"] = prompt if isinstance(prompt, str) else None
    _set_prompt_cache_event(
        "updated_prompt_prefix_checkpoint",
        phase="update",
        reason=reason,
        prompt_tokens=int(holder.get("last_input_tokens") or len(key_ids)),
        trimmed_generated_tokens=trim_tokens,
        key_tokens=len(key_ids),
        cache_len=len(key_ids),
        session_id=holder.get("session_id"),
        session_source=holder.get("session_source"),
        exact_generated_ids=False,
        generated_reuse_allowed=False,
        durable_checkpoint="prompt_prefix",
        **_prompt_cache_match_fields(len(key_ids), len(key_ids)),
    )
    return True


def _prompt_cache_ssd_status_unlocked(force_scan=False):
    rank, world = _prompt_cache_ssd_current_rank_world()
    base_path = os.path.expanduser(PROMPT_CACHE_SSD_DIR)
    rank0_path = os.path.expanduser(PROMPT_CACHE_SSD_DIR_RANK0 or PROMPT_CACHE_SSD_DIR)
    rank1_path = os.path.expanduser(PROMPT_CACHE_SSD_DIR_RANK1 or PROMPT_CACHE_SSD_DIR)
    path = _prompt_cache_ssd_root(rank)
    stat_path = path
    while stat_path and not os.path.exists(stat_path):
        parent = os.path.dirname(stat_path)
        if parent == stat_path:
            break
        stat_path = parent
    if not stat_path:
        stat_path = os.path.expanduser("~")
    free_bytes = None
    scan = {"entries": [], "entry_count": None, "total_bytes": None}
    try:
        scan = _prompt_cache_ssd_scan_cached_unlocked(force=force_scan)
        stat = os.statvfs(stat_path)
        free_bytes = int(stat.f_bavail * stat.f_frsize)
    except Exception as e:
        _prompt_cache_ssd_state["last_error"] = str(e)
    return {
        "enabled": PROMPT_CACHE_SSD_ENABLED,
        "restore_enabled": PROMPT_CACHE_SSD_RESTORE_ENABLED,
        "thinking_boundary_restore_enabled": (
            PROMPT_CACHE_SSD_THINKING_BOUNDARY_RESTORE
        ),
        "auto_save": PROMPT_CACHE_SSD_AUTO_SAVE,
        "auto_save_min_delta_tokens": (
            PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS
        ),
        "append_reserve_tokens": PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS,
        "save_reserve_tokens": PROMPT_CACHE_SSD_SAVE_RESERVE_TOKENS,
        "schema_version": PROMPT_CACHE_SSD_SCHEMA_VERSION,
        "mode": (
            "restore+autosave"
            if (
                PROMPT_CACHE_SSD_ENABLED
                and PROMPT_CACHE_SSD_RESTORE_ENABLED
                and PROMPT_CACHE_SSD_AUTO_SAVE
            )
            else (
                "restore"
                if PROMPT_CACHE_SSD_ENABLED and PROMPT_CACHE_SSD_RESTORE_ENABLED
                else (
                    "autosave"
                    if PROMPT_CACHE_SSD_ENABLED and PROMPT_CACHE_SSD_AUTO_SAVE
                    else ("manual-save" if PROMPT_CACHE_SSD_ENABLED else "off")
                )
            )
        ),
        "path": path,
        "base_path": base_path,
        "rank": rank,
        "rank_count": world,
        "rank0_path": rank0_path,
        "rank1_path": rank1_path,
        "path_mode": (
            "rank-specific"
            if PROMPT_CACHE_SSD_DIR_RANK0 or PROMPT_CACHE_SSD_DIR_RANK1
            else "shared-default"
        ),
        "exists": os.path.isdir(path),
        "entry_count": scan.get("entry_count"),
        "total_bytes": scan.get("total_bytes"),
        "entries": (scan.get("entries") or [])[:PROMPT_CACHE_SSD_RECENT_ENTRIES],
        "ttl_seconds": PROMPT_CACHE_SSD_TTL_SECONDS,
        "max_bytes": _runtime_prompt_cache_ssd_max_bytes(),
        "min_tokens": PROMPT_CACHE_SSD_MIN_TOKENS,
        "status_scan_interval_seconds": PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS,
        "save_reasoning": PROMPT_CACHE_SSD_SAVE_REASONING,
        "privacy": PROMPT_CACHE_SSD_PRIVACY,
        "free_bytes": free_bytes,
        **_prompt_cache_ssd_state,
    }


def _clear_prompt_cache_session_manifest_unlocked():
    path = _session_manifest_path()
    try:
        if os.path.exists(path):
            os.remove(path)
        now = round(time.time(), 3)
        _prompt_cache_session_manifest_state.update({
            "loaded_entries": 0,
            "entry_count": 0,
            "last_cleared_at": now,
            "last_written_at": None,
            "last_error": None,
        })
        return True
    except Exception as e:
        _prompt_cache_session_manifest_state["last_error"] = str(e)
        logger.warning("prompt-cache session manifest clear failed: %s", e)
        return False


def _record_prompt_cache_session_event(action, phase, fields):
    if PROMPT_CACHE_SESSION_MAP_MAX <= 0:
        return
    try:
        session_id = (
            fields.get("session_id")
            or fields.get("request_session_id")
            or fields.get("protected_session_id")
            or _prompt_cache_holder.get("session_id")
        )
        session_source = (
            fields.get("session_source")
            or fields.get("request_session_source")
            or fields.get("protected_session_source")
            or _prompt_cache_holder.get("session_source")
        )
        key = _prompt_cache_session_key(session_id, session_source)
        now = float(fields.get("at") or time.time())
        entry = _prompt_cache_session_map.get(key) or {
            "session_id": session_id,
            "session_source": session_source,
            "first_seen_at": round(now, 3),
            "requests": 0,
            "updates": 0,
            "bypasses_preserved": 0,
        }
        for field in ("prompt_tokens", "reuse_tokens", "suffix_tokens", "missed_tokens"):
            if fields.get(field) is not None:
                entry[f"last_{field}"] = int(fields.get(field) or 0)
        for field in ("cache_len", "key_tokens", "protected_cache_tokens"):
            if fields.get(field) is not None:
                entry[field] = int(fields.get(field) or 0)
        if fields.get("reuse_ratio") is not None:
            entry["last_reuse_ratio"] = fields.get("reuse_ratio")
        if fields.get("miss_reason"):
            entry["last_miss_reason"] = fields.get("miss_reason")
        if action in {
            "cold",
            "reuse",
            "low_reuse",
            "bypass_preserve_large_cache",
            "bypass_preserve_session_cache",
        }:
            entry["requests"] = int(entry.get("requests") or 0) + 1
        if phase == "update":
            entry["updates"] = int(entry.get("updates") or 0) + 1
        if action in {"bypass_preserve_large_cache", "bypass_preserve_session_cache"}:
            entry["bypasses_preserved"] = int(entry.get("bypasses_preserved") or 0) + 1
        entry["last_action"] = action
        entry["last_phase"] = phase
        entry["last_at"] = round(now, 3)
        if session_id:
            entry["session_id"] = session_id
        if session_source:
            entry["session_source"] = session_source
        _prompt_cache_session_map[key] = entry
        _prompt_cache_session_map.move_to_end(key)
        while len(_prompt_cache_session_map) > PROMPT_CACHE_SESSION_MAP_MAX:
            _prompt_cache_session_map.popitem(last=False)
        _write_prompt_cache_session_manifest_unlocked()
    except Exception as e:
        logger.debug("prompt-cache session-map update failed: %s", e)


def _prompt_cache_session_map_status_unlocked():
    current_key = _prompt_cache_current_session_key_unlocked()
    loaded = _prompt_cache_holder.get("cache") is not None
    resident_slot_keys = set(_prompt_cache_resident_slots.keys())
    entries = []
    for key, entry in reversed(_prompt_cache_session_map.items()):
        row = dict(entry)
        row["key"] = key
        row["resident"] = bool(loaded and key == current_key)
        row["resident_slot"] = bool(key in resident_slot_keys)
        row["ssd_rehydratable"] = bool(row.get("ssd_rehydratable"))
        # The distributed cache map currently has one safe resident KV slot.
        # Extra in-memory slots are restorable without reprocessing, but the
        # SSD tier can also make a metadata row rehydratable after restart.
        row["metadata_only"] = not (row["resident"] or row["resident_slot"])
        row["rehydratable"] = bool(row["resident_slot"] or row["ssd_rehydratable"])
        if row["resident"]:
            row["cache_len"] = int(_prompt_cache_holder.get("cache_len") or row.get("cache_len") or 0)
            row["key_tokens"] = len(_prompt_cache_holder.get("token_ids") or [])
        elif row["resident_slot"]:
            slot = _prompt_cache_resident_slots.get(key) or {}
            row["cache_len"] = int(slot.get("cache_len") or row.get("cache_len") or 0)
            row["key_tokens"] = len(slot.get("token_ids") or [])
        elif row["ssd_rehydratable"]:
            row["cache_len"] = int(row.get("cache_len") or row.get("ssd_key_tokens") or 0)
            row["key_tokens"] = int(row.get("key_tokens") or row.get("ssd_key_tokens") or 0)
        entries.append(row)
    for key, slot in reversed(_prompt_cache_resident_slots.items()):
        if any(row.get("key") == key for row in entries):
            continue
        entries.append({
            "key": key,
            "resident": False,
            "resident_slot": True,
            "metadata_only": False,
            "rehydratable": True,
            "session_id": slot.get("session_id"),
            "session_source": slot.get("session_source"),
            "cache_len": int(slot.get("cache_len") or 0),
            "key_tokens": len(slot.get("token_ids") or []),
            "last_at": slot.get("last_access_at"),
            "last_action": "resident_slot",
        })
    return {
        "max_entries": PROMPT_CACHE_SESSION_MAP_MAX,
        "resident_slots_max": PROMPT_CACHE_RESIDENT_SLOTS,
        "resident_slots": [
            {
                "key": key,
                "session_id": slot.get("session_id"),
                "session_source": slot.get("session_source"),
                "cache_len": int(slot.get("cache_len") or 0),
                "key_tokens": len(slot.get("token_ids") or []),
                "stashed_at": slot.get("stashed_at"),
                "stash_reason": slot.get("stash_reason"),
            }
            for key, slot in reversed(_prompt_cache_resident_slots.items())
        ],
        "resident_total_tokens": _prompt_cache_resident_total_tokens_unlocked(),
        "resident_total_max_tokens": PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS,
        "resident_key": current_key if loaded else None,
        "entries": entries,
    }


def _set_prompt_cache_event(action, *, phase="prepare", **fields):
    event = {"action": action, "at": round(time.time(), 3)}
    event.update(fields)
    _prompt_cache_holder["last_event"] = event
    _prompt_cache_holder["last_access_at"] = event["at"]
    if phase == "update":
        _prompt_cache_holder["last_update_event"] = event
    else:
        _prompt_cache_holder["last_prepare_event"] = event
    _record_prompt_cache_session_event(action, phase, event)


def _clear_prompt_cache_key_state_unlocked(holder):
    holder["token_ids"] = []
    holder["cache_len"] = 0
    holder["last_input_tokens"] = 0
    holder["last_generated_tokens"] = 0
    holder["last_exact_generated_ids"] = False
    holder["last_suffix_ids"] = None
    holder["prompt"] = None
    holder["session_id"] = None
    holder["session_source"] = None


def _drop_prompt_cache_unlocked(reason="reset", clear_manifest=False,
                                clear_resident=True, **fields):
    """Drop the cached KV state. Caller must hold _prompt_cache_lock.

    clear_resident=False scopes the drop to the live cache only: per-request
    resets (stop/disconnect/error on ONE stream) must not destroy the stashed
    resident slots of every other session on the server (2026-07-06 audit).
    """
    _prompt_cache_holder["cache"] = None
    _clear_prompt_cache_key_state_unlocked(_prompt_cache_holder)
    if clear_resident:
        _prompt_cache_resident_slots.clear()
    _prompt_cache_holder["created_at"] = None
    _prompt_cache_holder["last_keepwarm_event"] = None
    _prompt_cache_holder["last_keepwarm_at"] = None
    _prompt_cache_holder["keepwarm_count"] = 0
    _prompt_cache_holder["in_use"] = False
    _prompt_cache_holder["in_use_started_at"] = None
    if clear_manifest:
        _prompt_cache_session_map.clear()
    else:
        for entry in _prompt_cache_session_map.values():
            entry["metadata_only"] = True
            entry["rehydratable"] = bool(entry.get("ssd_rehydratable"))
            entry["loaded_from_manifest"] = bool(entry.get("loaded_from_manifest"))
    _set_prompt_cache_event(reason, prompt_tokens=0, reuse_tokens=0, **fields)
    if clear_manifest:
        _clear_prompt_cache_session_manifest_unlocked()


def _mark_prompt_cache_in_use(in_use):
    if not PROMPT_CACHE_ENABLED:
        return
    with _prompt_cache_lock:
        _prompt_cache_holder["in_use"] = bool(in_use)
        if in_use:
            now = round(time.time(), 3)
            _prompt_cache_holder["last_access_at"] = now
            _prompt_cache_holder["in_use_started_at"] = now
        else:
            _prompt_cache_holder["in_use_started_at"] = None
    if in_use and PROMPT_CACHE_KEEPWARM_ENABLED:
        # Barrier: wait out any keepwarm Metal submission that passed its idle
        # check before we flipped in_use, so generation never starts while a
        # keepwarm matmul is in flight on this rank.
        with _keepwarm_submit_lock:
            pass


def _recover_stale_prompt_cache_in_use(reason="idle recovery", max_age=90.0):
    """Clear an abandoned in_use flag only from known-idle server paths."""
    if not PROMPT_CACHE_ENABLED:
        return False
    now = time.time()
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        if not holder.get("in_use"):
            return False
        started = float(holder.get("in_use_started_at") or 0.0)
        age = now - started if started > 0 else max_age + 1.0
        if age < max_age:
            return False
        holder["in_use"] = False
        holder["in_use_started_at"] = None
        holder["last_keepwarm_event"] = {
            "ok": False,
            "action": "stale_in_use_recovered",
            "at": round(now, 3),
            "age_seconds": round(age, 3),
            "reason": reason,
        }
        logger.warning(
            "prompt-cache: recovered stale in_use flag after %.1fs (%s)",
            age,
            reason,
        )
        return True


def _expire_idle_prompt_cache():
    """Release an idle prompt cache after the configured TTL."""
    if not PROMPT_CACHE_ENABLED or PROMPT_CACHE_TTL_SECONDS <= 0:
        return False
    expired = False
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        if holder.get("cache") is None or holder.get("in_use"):
            return False
        last_access = float(holder.get("last_access_at") or 0.0)
        if last_access <= 0:
            return False
        idle = time.time() - last_access
        if idle >= PROMPT_CACHE_TTL_SECONDS:
            _drop_prompt_cache_unlocked(
                "ttl_expired",
                idle_seconds=round(idle, 3),
                ttl_seconds=PROMPT_CACHE_TTL_SECONDS,
            )
            expired = True
    if expired:
        mx.clear_cache()
        gc.collect()
        logger.info("prompt-cache ttl expired; cache released")
    return expired


def _enforce_prompt_cache_size_limit():
    if not PROMPT_CACHE_ENABLED or PROMPT_CACHE_MAX_TOKENS <= 0:
        return False
    dropped = False
    with _prompt_cache_lock:
        cache_len = int(_prompt_cache_holder.get("cache_len") or 0)
        if _prompt_cache_holder.get("cache") is not None and cache_len > PROMPT_CACHE_MAX_TOKENS:
            _drop_prompt_cache_unlocked(
                "max_tokens_exceeded",
                cache_len=cache_len,
                max_tokens=PROMPT_CACHE_MAX_TOKENS,
            )
            dropped = True
    if dropped:
        mx.clear_cache()
        gc.collect()
        logger.info("prompt-cache max token limit exceeded; cache released")
    return dropped


def _start_prompt_cache_janitor():
    if not PROMPT_CACHE_ENABLED or PROMPT_CACHE_TTL_SECONDS <= 0:
        return

    def _run():
        interval = max(5.0, min(60.0, PROMPT_CACHE_TTL_SECONDS / 4))
        while True:
            time.sleep(interval)
            try:
                _expire_idle_prompt_cache()
            except Exception as e:
                logger.debug("prompt-cache janitor failed: %s", e)

    t = threading.Thread(target=_run, name="prompt-cache-janitor", daemon=True)
    t.start()
    logger.info("prompt-cache janitor armed (ttl=%ss)", PROMPT_CACHE_TTL_SECONDS)


def _common_prefix_len(a, b):
    """Length of the longest common prefix of two token-id lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _prompt_cache_match_fields(prompt_tokens, reuse_tokens, *, previous_key_tokens=None):
    prompt_tokens = int(prompt_tokens or 0)
    reuse_tokens = int(reuse_tokens or 0)
    missed_tokens = max(0, prompt_tokens - reuse_tokens)
    fields = {
        "reuse_ratio": round(reuse_tokens / prompt_tokens, 4) if prompt_tokens else 0.0,
        "missed_tokens": missed_tokens,
    }
    if previous_key_tokens is not None:
        fields["previous_key_tokens"] = int(previous_key_tokens or 0)
    return fields


def _decode_prompt_cache_token_window(processor, token_ids, center, radius=18):
    """Decode a tiny token window around a cache mismatch for diagnostics."""
    try:
        if token_ids is None:
            return None
        center = int(center)
        token_ids = list(token_ids)
        start = max(0, center - int(radius))
        end = min(len(token_ids), center + int(radius) + 1)
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return {
            "start": start,
            "end": end,
            "token_ids": [int(t) for t in token_ids[start:end]],
            "text": tok.decode(token_ids[start:end]),
        }
    except Exception as e:
        return {"error": str(e)}


def _add_prompt_cache_mismatch_windows(reuse_diag, processor, previous_ids, prompt_ids):
    if not isinstance(reuse_diag, dict):
        return reuse_diag
    mi = reuse_diag.get("mismatch_index")
    if not isinstance(mi, int):
        return reuse_diag
    reuse_diag = dict(reuse_diag)
    reuse_diag["previous_mismatch_window"] = _decode_prompt_cache_token_window(
        processor, previous_ids, mi
    )
    reuse_diag["prompt_mismatch_window"] = _decode_prompt_cache_token_window(
        processor, prompt_ids, mi
    )
    return reuse_diag


def _prompt_cache_reuse_diagnostics(
    holder, prompt_tokens, reuse_tokens, candidate_token_ids=None
):
    """Classify where a prompt-cache match diverged.

    This is especially useful for OpenAI-compatible UIs that do not send
    previous assistant reasoning back. Overall reuse can look healthy while the
    cache still trims at the prior assistant boundary and has to prefill a long
    response again.
    """
    previous_key_tokens = len(holder.get("token_ids") or [])
    previous_input_tokens = int(holder.get("last_input_tokens") or 0)
    previous_generated_tokens = int(holder.get("last_generated_tokens") or 0)
    reuse_tokens = int(reuse_tokens or 0)
    prompt_tokens = int(prompt_tokens or 0)
    reused_generated_tokens = max(0, reuse_tokens - previous_input_tokens)
    generated_reuse_ratio = (
        round(reused_generated_tokens / previous_generated_tokens, 4)
        if previous_generated_tokens > 0 else None
    )
    if previous_key_tokens <= 0:
        reason = "cold"
    elif reuse_tokens >= previous_key_tokens:
        reason = "exact_prior_transcript"
    elif previous_input_tokens > 0 and reuse_tokens < previous_input_tokens:
        reason = "history_prefix_mismatch"
    elif previous_generated_tokens > 0 and reuse_tokens == previous_input_tokens:
        reason = "previous_assistant_start_mismatch"
    elif previous_generated_tokens > 0 and reuse_tokens < previous_key_tokens:
        reason = "previous_assistant_partial_mismatch"
    else:
        reason = "new_suffix_only"
    mismatch = {}
    if 0 <= reuse_tokens < min(previous_key_tokens, prompt_tokens):
        previous_ids = holder.get("token_ids") or []
        prompt_ids = candidate_token_ids or []
        mismatch["mismatch_index"] = reuse_tokens
        mismatch["mismatch_region"] = (
            "generated_tail"
            if previous_input_tokens > 0 and reuse_tokens >= previous_input_tokens
            else "prompt_prefix"
        )
        mismatch["generated_tail_offset"] = (
            reuse_tokens - previous_input_tokens
            if previous_input_tokens > 0 and reuse_tokens >= previous_input_tokens
            else None
        )
        try:
            mismatch["previous_token_at_mismatch"] = int(previous_ids[reuse_tokens])
        except Exception:
            mismatch["previous_token_at_mismatch"] = None
        try:
            mismatch["prompt_token_at_mismatch"] = int(prompt_ids[reuse_tokens])
        except Exception:
            mismatch["prompt_token_at_mismatch"] = None
    return {
        "miss_reason": reason,
        "previous_input_tokens": previous_input_tokens,
        "previous_generated_tokens": previous_generated_tokens,
        "previous_key_tokens": previous_key_tokens,
        "reused_generated_tokens": reused_generated_tokens,
        "generated_reuse_ratio": generated_reuse_ratio,
        "would_reprocess_tokens": max(0, prompt_tokens - reuse_tokens),
        "previous_exact_generated_ids": bool(holder.get("last_exact_generated_ids")),
        **mismatch,
    }


def _normalize_cache_session(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:128]


def _is_auto_cache_session_source(session_source):
    return str(session_source or "").startswith("auto.")


def _auto_cache_reuse_requires_rebuild(
    holder, token_ids, reuse_tokens, session_source, current_session_id=None
):
    """Return a reason when auto-session cache reuse could cross chat boundaries.

    Auto sessions are a convenience for OpenAI-compatible clients that do not
    send a real chat id. They are necessarily heuristic. To prevent independent
    chats with similar openers from sharing KV state, only force a rebuild when
    the derived auto-session id changes. Same-id auto sessions are allowed to use
    the normal prefix/backtrack path because OpenWebUI can shift the serialized
    history while still sending the same continuing chat.
    """
    if not _is_auto_cache_session_source(session_source):
        return None
    previous_session_id = holder.get("session_id")
    if previous_session_id and current_session_id and previous_session_id == current_session_id:
        return None
    previous_input_tokens = int(holder.get("last_input_tokens") or 0)
    if previous_input_tokens <= 0:
        return None
    prompt_tokens = len(token_ids or [])
    reuse_tokens = int(reuse_tokens or 0)
    if prompt_tokens > previous_input_tokens and reuse_tokens < previous_input_tokens:
        return "auto_session_history_prefix_mismatch"
    if prompt_tokens <= previous_input_tokens and reuse_tokens < prompt_tokens:
        return "auto_session_prompt_prefix_mismatch"
    return None


def _auto_cache_session_from_request(request):
    """Derive a privacy-safe session key when the client omits one.

    OpenWebUI and many agent clients send full conversation history but not a
    stable chat id. Hashing a few conversation-shape anchors lets cache
    protection distinguish independent chats without exposing prompt text.
    """
    if not isinstance(request, dict):
        return None, None
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, None
    anchors = []
    image_parts = 0
    image_anchors = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content", "")
        text_parts = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("text", "input_text"):
                    text_parts.append(str(part.get("text") or part.get("content") or ""))
                elif _extract_image_source(part):
                    image_parts += 1
                    image_anchors.append(_short_hash(_extract_image_source(part)))
        elif isinstance(content, str):
            text_parts.append(content)
        text = "\n".join(t for t in text_parts if t)
        if role == "system" and text:
            anchors.append(("system", _short_hash(text)))
        elif role == "user" and text:
            anchors.append(("user", _short_hash(text)))
    user_anchors = [h for role, h in anchors if role == "user"]
    if not user_anchors:
        return None, None
    tools = request.get("tools")
    # 2026-07-06 cache audit: hashing the full tool JSON made the auto id flip
    # whenever a client mutated schemas/descriptions between turns (gateways
    # prune tool lists per turn, opencode injects cwd into descriptions).
    # Sorted names keep "different agent = different session" while surviving
    # cosmetic churn; token-level prefix checks still guard actual reuse.
    tool_names = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
                name = str(fn.get("name") or tool.get("name") or "")
                if name:
                    tool_names.append(name)
    tools_hash = _short_hash(",".join(sorted(tool_names))) if tool_names else "-"
    requested_model = _normalize_cache_session(request.get("model")) or "-"
    material = "|".join([
        f"model={requested_model}",
        # OpenWebUI can mutate its system prompt between turns with transient
        # tool/status/context text. Keep the auto session tied to stable chat
        # anchors so normal follow-ups reuse KV instead of stashing each turn.
        "system=-",
        f"first_user={user_anchors[0]}",
        f"tools_hash={tools_hash}",
        f"images={image_parts}",
        f"image_hashes={','.join(image_anchors[:8])}",
    ])
    return f"auto:{_short_hash(material)}", "auto.conversation_fingerprint"


def _request_cache_session(request):
    """Return a privacy-safe cache session id from OpenAI-compatible fields."""
    metadata = request.get("metadata") if isinstance(request, dict) else None
    if isinstance(metadata, dict):
        for key in (
            "session_id",
            "conversation_id",
            "chat_id",
            "thread_id",
            "agent_session_id",
            "cache_session_id",
        ):
            sid = _normalize_cache_session(metadata.get(key))
            if sid:
                return sid, f"metadata.{key}"
    for key in ("session_id", "conversation_id", "chat_id", "thread_id"):
        sid = _normalize_cache_session(request.get(key)) if isinstance(request, dict) else None
        if sid:
            return sid, key
    return _auto_cache_session_from_request(request)


def _tokenize_prompt(processor, prompt):
    """Tokenize the rendered prompt to token ids (the cache key)."""
    try:
        with _tokenizer_runtime_lock:
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            ids = tok.encode(prompt, add_special_tokens=False)
        return list(ids)
    except Exception as e:
        logger.debug(f"prompt-cache tokenize failed: {e}")
        return None


def _get_or_build_prompt_cache(model):
    """Return the reusable prompt cache, building it fresh if needed."""
    with _prompt_cache_lock:
        return _get_or_build_prompt_cache_unlocked(model)


def _get_or_build_prompt_cache_unlocked(model):
    holder = _prompt_cache_holder
    if holder["cache"] is None:
        from mlx_vlm.models import cache as _cache_mod

        holder["cache"] = _cache_mod.make_prompt_cache(
            model.language_model, max_kv_size=holder["max_kv_size"]
        )
        _clear_prompt_cache_key_state_unlocked(holder)
        holder["created_at"] = round(time.time(), 3)
    return holder["cache"]


def _cache_token_count(cache):
    """Best-effort count of tokens currently held in the prompt cache."""
    if not cache:
        return 0
    try:
        # KVCache-like objects expose an offset/keys shape; use the first layer.
        c0 = cache[0]
        for attr in ("offset", "_offset", "idx"):
            v = getattr(c0, attr, None)
            if isinstance(v, int):
                return v
        kc = getattr(c0, "kv_cache", None)
        if kc is not None:
            keys = getattr(kc, "keys", None)
            if keys is not None:
                return int(keys.shape[2])
            for attr in ("offset", "_offset", "idx"):
                v = getattr(kc, attr, None)
                if isinstance(v, int):
                    return v
    except Exception:
        pass
    return 0


def _cache_capacity_status(cache):
    """Return first-layer logical/capacity counters without evaluating MLX."""
    status = {
        "cache_physical_tokens": 0,
        "cache_capacity_tokens": 0,
        "cache_spare_tokens": 0,
        "index_capacity_tokens": 0,
    }
    if not cache:
        return status
    try:
        layer = cache[0]
        nested = getattr(layer, "kv_cache", None)
        if nested is not None:
            keys = getattr(nested, "keys", None)
            offset = int(getattr(nested, "offset", 0) or 0)
            capacity = int(keys.shape[2]) if keys is not None else 0
            index_keys = getattr(layer, "index_keys", None)
            index_capacity = (
                int(index_keys.shape[2]) if index_keys is not None else 0
            )
        else:
            keys = getattr(layer, "keys", None)
            offset = int(getattr(layer, "offset", 0) or 0)
            capacity = int(keys.shape[2]) if keys is not None else 0
            index_capacity = 0
        status.update({
            "cache_physical_tokens": offset,
            "cache_capacity_tokens": capacity,
            "cache_spare_tokens": max(0, capacity - offset),
            "index_capacity_tokens": index_capacity,
        })
    except Exception:
        pass
    return status


def _trim_prompt_cache_in_place(cache, n):
    """Trim n tokens from every layer cache; return True only if all agree."""
    if not cache or n <= 0:
        return True
    trimmed = []
    for c in cache:
        if c is None:
            continue
        if hasattr(c, "is_trimmable") and not c.is_trimmable():
            return False
    for c in cache:
        if c is None:
            continue
        if not hasattr(c, "trim"):
            return False
        trimmed.append(int(c.trim(n)))
    return bool(trimmed) and all(t == n for t in trimmed)


def _prompt_cache_status():
    with _prompt_cache_lock:
        now = time.time()
        last_access = _prompt_cache_holder.get("last_access_at")
        idle = round(now - float(last_access), 3) if last_access else None
        expires = None
        if (
            PROMPT_CACHE_TTL_SECONDS > 0
            and idle is not None
            and _prompt_cache_holder.get("cache") is not None
            and not _prompt_cache_holder.get("in_use")
        ):
            expires = max(0.0, round(PROMPT_CACHE_TTL_SECONDS - idle, 3))
        capacity_status = _cache_capacity_status(_prompt_cache_holder.get("cache"))
        return {
            "enabled": PROMPT_CACHE_ENABLED,
            "thinking_enabled": PROMPT_CACHE_THINKING_ENABLED,
            "thinking_mode": PROMPT_CACHE_THINKING_MODE,
            "min_reuse": PROMPT_CACHE_MIN_REUSE,
            "ttl_seconds": PROMPT_CACHE_TTL_SECONDS,
            "max_tokens": PROMPT_CACHE_MAX_TOKENS,
            "generated_reuse_max_tokens": PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS,
            "protect_large": PROMPT_CACHE_PROTECT_LARGE_ENABLED,
            "protect_min_tokens": PROMPT_CACHE_PROTECT_MIN_TOKENS,
            "protect_bypass_max_tokens": PROMPT_CACHE_PROTECT_BYPASS_MAX_TOKENS,
            "session_protect": PROMPT_CACHE_SESSION_PROTECT_ENABLED,
            "session_protect_min_tokens": PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS,
            "session_protect_bypass_max_tokens": PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS,
            "session_id": _prompt_cache_holder.get("session_id"),
            "session_source": _prompt_cache_holder.get("session_source"),
            "key_tokens": len(_prompt_cache_holder.get("token_ids") or []),
            "cache_len": int(_prompt_cache_holder.get("cache_len") or 0),
            **capacity_status,
            "last_input_tokens": int(_prompt_cache_holder.get("last_input_tokens") or 0),
            "last_generated_tokens": int(_prompt_cache_holder.get("last_generated_tokens") or 0),
            "last_exact_generated_ids": bool(_prompt_cache_holder.get("last_exact_generated_ids")),
            "loaded": _prompt_cache_holder.get("cache") is not None,
            "in_use": bool(_prompt_cache_holder.get("in_use")),
            "in_use_started_at": _prompt_cache_holder.get("in_use_started_at"),
            "in_use_age_seconds": (
                round(now - float(_prompt_cache_holder.get("in_use_started_at")), 3)
                if _prompt_cache_holder.get("in_use_started_at")
                else None
            ),
            "idle_seconds": idle,
            "expires_in_seconds": expires,
            "last_event": _prompt_cache_holder.get("last_event"),
            "last_prepare_event": _prompt_cache_holder.get("last_prepare_event"),
            "last_update_event": _prompt_cache_holder.get("last_update_event"),
            "last_keepwarm_event": _prompt_cache_holder.get("last_keepwarm_event"),
            "keepwarm": {
                "enabled": PROMPT_CACHE_KEEPWARM_ENABLED,
                "mode": PROMPT_CACHE_KEEPWARM_MODE,
                "interval_seconds": PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS,
                "idle_after_seconds": PROMPT_CACHE_KEEPWARM_IDLE_AFTER_SECONDS,
                "matrix_size": PROMPT_CACHE_KEEPWARM_MATRIX_SIZE,
                "large_cache_tokens": PROMPT_CACHE_KEEPWARM_LARGE_CACHE_TOKENS,
                "large_interval_seconds": PROMPT_CACHE_KEEPWARM_LARGE_INTERVAL_SECONDS,
                "slow_backoff_seconds": PROMPT_CACHE_KEEPWARM_SLOW_BACKOFF_SECONDS,
                "request_start_enabled": PROMPT_CACHE_REQUEST_START_KEEPWARM_ENABLED,
                "request_start_idle_seconds": PROMPT_CACHE_REQUEST_START_KEEPWARM_IDLE_SECONDS,
                "request_start_matrix_size": PROMPT_CACHE_REQUEST_START_KEEPWARM_MATRIX_SIZE,
                "request_start_repeats": PROMPT_CACHE_REQUEST_START_KEEPWARM_REPEATS,
                "post_response_enabled": PROMPT_CACHE_POST_RESPONSE_KEEPWARM_ENABLED,
                "post_response_delay_seconds": PROMPT_CACHE_POST_RESPONSE_KEEPWARM_DELAY_SECONDS,
                "post_response_matrix_size": PROMPT_CACHE_POST_RESPONSE_KEEPWARM_MATRIX_SIZE,
                "post_response_repeats": PROMPT_CACHE_POST_RESPONSE_KEEPWARM_REPEATS,
                "count": int(_prompt_cache_holder.get("keepwarm_count") or 0),
                "last_at": _prompt_cache_holder.get("last_keepwarm_at"),
            },
            "session_map": _prompt_cache_session_map_status_unlocked(),
            "session_manifest": _prompt_cache_session_manifest_status_unlocked(),
            "ssd": _prompt_cache_ssd_status_unlocked(),
        }


def _prompt_cache_last_prepare_action():
    with _prompt_cache_lock:
        event = _prompt_cache_holder.get("last_prepare_event") or {}
        return event.get("action")


def _prompt_cache_last_suffix_ids():
    with _prompt_cache_lock:
        suffix_ids = _prompt_cache_holder.get("last_suffix_ids")
        return list(suffix_ids) if suffix_ids else None


def _prompt_cache_current_prompt_snapshot():
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        prompt = holder.get("prompt")
        token_ids = list(holder.get("token_ids") or [])
        if not isinstance(prompt, str) or not prompt or not token_ids:
            return None
        # If the key includes generated ids, the saved prompt no longer maps
        # one-to-one to token_ids. Skip idle prewarm in that case rather than
        # rebuilding a mismatched cache.
        if int(holder.get("last_input_tokens") or 0) != len(token_ids):
            return None
        return {
            "prompt": prompt,
            "token_ids": token_ids,
            "session_id": holder.get("session_id"),
            "session_source": holder.get("session_source"),
            "cache_len": int(holder.get("cache_len") or 0),
        }


def _prompt_cache_request_start_keepwarm_candidate(min_idle_seconds=None):
    if not PROMPT_CACHE_ENABLED:
        return None
    now = time.time()
    min_idle = (
        PROMPT_CACHE_REQUEST_START_KEEPWARM_IDLE_SECONDS
        if min_idle_seconds is None else float(min_idle_seconds or 0.0)
    )
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        if holder.get("cache") is None or holder.get("in_use"):
            return None
        cache_len = int(holder.get("cache_len") or 0)
        if cache_len <= 0:
            return None
        last_access = float(holder.get("last_access_at") or 0.0)
        idle = now - last_access if last_access > 0 else 0.0
        if idle < min_idle:
            return None
        return {
            "cache_len": cache_len,
            "idle_seconds": round(idle, 3),
            "session_id": holder.get("session_id"),
            "session_source": holder.get("session_source"),
        }


def _prompt_cache_prepare_preserves_existing_cache(action=None):
    if action is None:
        action = _prompt_cache_last_prepare_action()
    return action in {
        "bypass_preserve_large_cache",
        "bypass_preserve_session_cache",
    }


# Serializes keepwarm Metal submissions against generation start. The idle
# check in _touch_prompt_cache_keepwarm runs under _prompt_cache_lock, but the
# warmup matmul itself runs outside it; without this lock a generation could
# begin between the check and the submission and interleave GPU work with the
# first distributed ops of the request.
_keepwarm_submit_lock = threading.Lock()


def _touch_prompt_cache_keepwarm(warmup_cb=None):
    if not PROMPT_CACHE_ENABLED or not PROMPT_CACHE_KEEPWARM_ENABLED:
        return False
    started = time.time()
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        cache = holder.get("cache")
        if cache is None or holder.get("in_use"):
            return False
        cache_len = int(holder.get("cache_len") or 0)
        if cache_len <= 0:
            return False
        last_access = float(holder.get("last_access_at") or 0.0)
        idle = time.time() - last_access if last_access > 0 else 0.0
        if idle < PROMPT_CACHE_KEEPWARM_IDLE_AFTER_SECONDS:
            return False
        last_keepwarm = float(holder.get("last_keepwarm_at") or 0.0)
        if (
            PROMPT_CACHE_KEEPWARM_LARGE_CACHE_TOKENS > 0
            and cache_len >= PROMPT_CACHE_KEEPWARM_LARGE_CACHE_TOKENS
            and PROMPT_CACHE_KEEPWARM_LARGE_INTERVAL_SECONDS > 0
            and last_keepwarm > 0
            and time.time() - last_keepwarm < PROMPT_CACHE_KEEPWARM_LARGE_INTERVAL_SECONDS
        ):
            return False
        last_event = holder.get("last_keepwarm_event") or {}
        last_elapsed_ms = float(last_event.get("elapsed_ms") or 0.0)
        if (
            PROMPT_CACHE_KEEPWARM_SLOW_BACKOFF_SECONDS > 0
            and last_keepwarm > 0
            and last_elapsed_ms >= 1000.0
            and time.time() - last_keepwarm < PROMPT_CACHE_KEEPWARM_SLOW_BACKOFF_SECONDS
        ):
            return False
    try:
        # Do not evaluate the KV cache object itself here. Those arrays can be
        # tied to the generation thread's MLX stream, and touching them from
        # the keepwarm thread raises "no Stream(...) in current thread". Use
        # the same bounded matmul warmup as the admin endpoint so the Metal
        # path used by prefill/decode stays warm during OpenWebUI think time.
        with _keepwarm_submit_lock:
            # Re-check under the submit lock: a generation may have started
            # between the idle check above and this submission.
            with _prompt_cache_lock:
                if _prompt_cache_holder.get("in_use"):
                    return False
            if warmup_cb is not None:
                event = warmup_cb(
                    size=PROMPT_CACHE_KEEPWARM_MATRIX_SIZE,
                    repeats=1,
                    reason="prompt-cache keepwarm",
                )
                if not event or event.get("skipped"):
                    return False
            else:
                event = _metal_warmup_touch(
                    size=PROMPT_CACHE_KEEPWARM_MATRIX_SIZE,
                    repeats=1,
                    reason="prompt-cache keepwarm",
                )
        if not event.get("ok"):
            raise RuntimeError(event.get("error") or "metal warmup failed")
    except Exception as e:
        with _prompt_cache_lock:
            holder = _prompt_cache_holder
            holder["last_keepwarm_event"] = {
                "action": "error",
                "at": round(time.time(), 3),
                "error": str(e),
                "cache_len": cache_len,
            }
        logger.debug("prompt-cache keepwarm failed: %s", e)
        return False
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        if holder.get("cache") is None or holder.get("in_use"):
            return False
        try:
            cache_len = int(holder.get("cache_len") or cache_len)
            last_access = float(holder.get("last_access_at") or last_access)
            idle = time.time() - last_access if last_access > 0 else idle
        except Exception:
            pass
        holder["keepwarm_count"] = int(holder.get("keepwarm_count") or 0) + 1
        event = {
            "action": "metal_touch",
            "at": round(time.time(), 3),
            "cache_len": cache_len,
            "idle_seconds": round(idle, 3),
            "elapsed_ms": round((time.time() - started) * 1000, 3),
            "count": holder["keepwarm_count"],
        }
        holder["last_keepwarm_event"] = event
        holder["last_keepwarm_at"] = event["at"]
    return True


def _start_prompt_cache_keepwarm(warmup_cb=None):
    if (
        not PROMPT_CACHE_ENABLED
        or not PROMPT_CACHE_KEEPWARM_ENABLED
        or PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS <= 0
    ):
        return

    def _run():
        interval = max(0.25, PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS)
        while True:
            time.sleep(interval)
            try:
                _touch_prompt_cache_keepwarm(warmup_cb=warmup_cb)
            except Exception as e:
                logger.debug("prompt-cache keepwarm loop failed: %s", e)

    t = threading.Thread(target=_run, name="prompt-cache-keepwarm", daemon=True)
    t.start()
    logger.info(
        "prompt-cache keepwarm armed (interval=%.2fs idle_after=%.2fs)",
        PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS,
        PROMPT_CACHE_KEEPWARM_IDLE_AFTER_SECONDS,
    )


def _metal_warmup_touch(size=128, repeats=2, reason="manual"):
    """Run a bounded independent GPU touch without altering prompt-cache state."""
    global _metal_warmup_last_event
    started = time.time()
    size = max(16, min(1024, int(size or 128)))
    repeats = max(1, min(16, int(repeats or 2)))
    try:
        with mx.stream(mx.default_device()):
            acc = None
            for i in range(repeats):
                a = mx.ones((size, size), dtype=mx.float16) * (i + 1)
                b = mx.ones((size, size), dtype=mx.float16)
                value = mx.sum(a @ b)
                acc = value if acc is None else acc + value
            mx.eval(acc)
        event = {
            "ok": True,
            "action": "metal_warmup",
            "reason": str(reason or "manual")[:128],
            "at": round(time.time(), 3),
            "matrix_size": size,
            "repeats": repeats,
            "elapsed_ms": round((time.time() - started) * 1000, 3),
        }
    except Exception as e:
        event = {
            "ok": False,
            "action": "metal_warmup_error",
            "reason": str(reason or "manual")[:128],
            "at": round(time.time(), 3),
            "matrix_size": size,
            "repeats": repeats,
            "elapsed_ms": round((time.time() - started) * 1000, 3),
            "error": str(e),
        }
        logger.debug("metal warmup failed: %s", e)
    with _metal_warmup_lock:
        _metal_warmup_last_event = event
    return event


def _metal_warmup_status():
    with _metal_warmup_lock:
        return dict(_metal_warmup_last_event) if _metal_warmup_last_event else None



def _prefix_plan_consensus(rank, prompt, prompt_to_send, cached_prompt_cache,
                           cached_suffix_ids):
    """Rank-coherence gate for the prefill plan (2026-07-07; hardened 2026-07-08).

    Both ranks compute cache reuse independently and are ASSUMED to agree.
    After retries/stops their holder states can drift, and a mismatched plan
    sends different-shaped pipeline messages (rank1 suffix=64 vs rank0 full
    7974 re-prefill) -> IBV_WC_LOC_LEN_ERR (status=1 wr_id=0x20001, five
    captures) -> frozen step. 2026-07-08 (goals-crash 19:16:24 photograph):
    comparing the PHYSICAL offset alone missed a divergence where both ranks
    drained to the same lockstep boundary (physicals equal at 18344) while
    the cache_len counters that drive suffix planning differed (rank0 kept
    375 generated tokens, rank1's consumer closed at 290) — plans 254 vs 339
    sailed through and froze the pipeline with zero progress. So exchange
    the physical offset, the planned counter, AND the suffix size; ANY
    mismatch (or a local counter/physical split) makes BOTH ranks drop to an
    identical full prefill (fresh cache) — correctness over the rare
    re-prefill. The all_sum runs UNCONDITIONALLY on both ranks before any
    decision: a one-rank early return would leave the peer's collective
    unpaired, which is the same freeze this gate exists to prevent.
    Runs in the sequential prepare phase: no model collectives in flight.
    """
    try:
        my_phys = int(cached_prompt_cache[0].offset) if cached_prompt_cache else 0
    except Exception:
        my_phys = -1
    my_counter = 0
    my_incoherent = 0
    if cached_prompt_cache is not None:
        try:
            with _prompt_cache_lock:
                my_counter = int(_prompt_cache_holder.get("cache_len") or 0)
        except Exception:
            my_counter = -1
        if my_counter != my_phys:
            my_incoherent = 1
    if cached_suffix_ids:
        my_suffix = len(cached_suffix_ids)
    else:
        my_suffix = len(prompt_to_send or "")
    try:
        group = mx.distributed.init()
        if group.size() <= 1:
            return prompt_to_send, cached_prompt_cache, cached_suffix_ids
        logger.info(
            "prefix-plan consensus: phys=%d plan_len=%d suffix=%d incoherent=%d",
            my_phys, my_counter, my_suffix, my_incoherent,
        )
        mine = (my_phys, my_counter, my_suffix, my_incoherent)
        total = mx.distributed.all_sum(mx.array(mine, dtype=mx.int32))
        mx.eval(total)
        peer = [int(t) - m for t, m in zip(total.tolist(), mine)]
    except Exception as e:
        logger.warning("prefix-plan consensus unavailable (%s); keeping local plan", e)
        return prompt_to_send, cached_prompt_cache, cached_suffix_ids
    if (
        peer[0] == my_phys
        and peer[1] == my_counter
        and peer[2] == my_suffix
        and my_incoherent == 0
        and peer[3] == 0
    ):
        return prompt_to_send, cached_prompt_cache, cached_suffix_ids
    logger.warning(
        "rank %s: PREFIX PLAN DIVERGENCE (phys mine=%d peer=%d | plan_len "
        "mine=%d peer=%d | suffix mine=%d peer=%d | incoherent mine=%d "
        "peer=%d) — both ranks rebuilding with a full identical prefill",
        rank, my_phys, peer[0], my_counter, peer[1], my_suffix, peer[2],
        my_incoherent, peer[3],
    )
    _reset_prompt_cache("prefix plan divergence", clear_resident=False)
    return prompt, None, None


def _prewarm_plan_consensus(my_skip, prompt_to_send, cached_prompt_cache):
    """Symmetric go/no-go gate for the prewarm's 1-token pipeline generation.

    The prewarm runs full pipeline collectives, so a rank-local skip (or a
    diverged reuse plan) leaves the peer rank alone in prefill send/recv —
    the freeze class photographed adjacent to prewarms. One all_sum runs
    UNCONDITIONALLY on both ranks; returns "ok" | "skip" | "diverged" with
    the same verdict on every rank: any rank skipping means all skip.
    """
    try:
        my_phys = int(cached_prompt_cache[0].offset) if cached_prompt_cache else 0
    except Exception:
        my_phys = -1
    my_suffix = len(prompt_to_send or "")
    try:
        group = mx.distributed.init()
        if group.size() <= 1:
            return "skip" if my_skip else "ok"
        size = group.size()
        mine = (1 if my_skip else 0, my_phys, my_suffix)
        total = mx.distributed.all_sum(mx.array(mine, dtype=mx.int32))
        mx.eval(total)
        skip_total, phys_total, suffix_total = [int(v) for v in total.tolist()]
    except Exception as e:
        logger.warning("prewarm consensus unavailable (%s); skipping prewarm", e)
        return "skip"
    if skip_total > 0:
        return "skip"
    if phys_total != size * my_phys or suffix_total != size * my_suffix:
        return "diverged"
    return "ok"


def _prepare_cached_prompt(model, processor, prompt, token_ids,
                           session_id=None, session_source=None,
                           thinking_mode="adaptive",
                           append_reserve_tokens=0):
    """Compute (suffix_prompt, prompt_cache) for prefix-aware generation.

    Returns (prompt_to_send, prompt_cache_or_None). If a long prefix is reused,
    the cache is trimmed to that prefix and only the suffix prompt is sent. If
    reuse is short/none, a fresh cache is built and the full prompt is sent.
    Both ranks call this with IDENTICAL token_ids (broadcast by rank 0), so
    their resulting cache state matches and the distributed generation stays
    in lockstep.

    Cache-state tracking: after a generation, the KV cache holds INPUT + the
    generated tokens. We store the INPUT token_ids as the key (the part shared
    with the next turn's prompt) and use the cache's OWN offset to know its
    true length when trimming. This keeps the prefix math correct even though
    generated tokens extend the cache beyond the stored key.
    """
    if not PROMPT_CACHE_ENABLED or token_ids is None:
        return prompt, None

    _expire_idle_prompt_cache()
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        restored_slot = None
        restored_ssd = None
        # Thinking prompts include control tokens and hidden/visible routing
        # boundaries. General partial durable restores remain disabled. The one
        # safe exception is an append-only continuation where the stored cache
        # differs solely by MiniMax's trailing <mm:think> generation marker.
        thinking_enabled = _enable_thinking_for_generation(thinking_mode)
        allow_partial_ssd_restore = not thinking_enabled
        allow_thinking_boundary_restore = (
            thinking_enabled and PROMPT_CACHE_SSD_THINKING_BOUNDARY_RESTORE
        )
        if (
            session_id
            and holder.get("cache") is not None
            and _prompt_cache_current_session_key_unlocked()
            != _prompt_cache_session_key(session_id, session_source)
        ):
            if PROMPT_CACHE_RESIDENT_SLOTS > 1:
                restored_slot = _prompt_cache_restore_resident_unlocked(
                    session_id=session_id,
                    session_source=session_source,
                )
            elif not _is_auto_cache_session_source(session_source):
                # A one-slot runtime cannot retain two live KV trees. Persist
                # the outgoing explicit session, then release it before the
                # normal cold/SSD path selects the requested session. This is
                # both the isolation boundary and the memory-safe alternative
                # to staging two 200k+ caches during a durable restore.
                outgoing_session_id = holder.get("session_id")
                outgoing_cache_len = int(holder.get("cache_len") or 0)
                _prompt_cache_ssd_maybe_autosave_unlocked(
                    model,
                    processor,
                    prompt=holder.get("prompt"),
                    reason=f"one_slot_switch:{session_id}",
                )
                holder["cache"] = None
                _clear_prompt_cache_key_state_unlocked(holder)
                gc.collect()
                logger.info(
                    "prompt-cache: released one-slot %s-session %d-token cache "
                    "before switch to %s-session",
                    outgoing_session_id,
                    outgoing_cache_len,
                    session_id,
                )
        cached_ids = holder["token_ids"]
        cache = holder["cache"]

        if cache is None or not cached_ids:
            restored_ssd = _prompt_cache_ssd_maybe_restore_unlocked(
                model,
                processor,
                token_ids,
                session_id=session_id,
                session_source=session_source,
                reason="cold_or_empty_ram",
                allow_partial_restore=allow_partial_ssd_restore,
                allow_thinking_boundary_restore=allow_thinking_boundary_restore,
                append_reserve_tokens=append_reserve_tokens,
            )
            if restored_ssd:
                cached_ids = holder["token_ids"]
                cache = holder["cache"]
            else:
                # First request, or cache was reset: full prompt, fresh cache.
                holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
                _clear_prompt_cache_key_state_unlocked(holder)
                holder["session_id"] = session_id
                holder["session_source"] = session_source
                holder["last_suffix_ids"] = list(token_ids)
                _set_prompt_cache_event(
                    "cold",
                    prompt_tokens=len(token_ids),
                    reuse_tokens=0,
                    suffix_tokens=len(token_ids),
                    session_id=session_id,
                    session_source=session_source,
                    restored_resident_slot=bool(restored_slot),
                    **(restored_slot or {}),
                    restored_ssd_cache=False,
                    **_prompt_cache_match_fields(len(token_ids), 0),
                )
                return prompt, holder["cache"]

        # Common prefix between the stored INPUT ids and the new prompt ids.
        L = _common_prefix_len(cached_ids, token_ids)
        reuse = L
        reuse_diag = _prompt_cache_reuse_diagnostics(
            holder, len(token_ids), reuse, token_ids
        )
        reuse_diag = _add_prompt_cache_mismatch_windows(
            reuse_diag, processor, cached_ids, token_ids
        )

        protected_cache_tokens = int(holder.get("cache_len") or len(cached_ids) or 0)
        cached_session_id = holder.get("session_id")
        cached_session_source = holder.get("session_source")
        auto_rebuild_reason = _auto_cache_reuse_requires_rebuild(
            holder,
            token_ids,
            reuse,
            session_source or cached_session_source,
            current_session_id=session_id,
        )
        if not auto_rebuild_reason and (
            _enable_thinking_for_generation(thinking_mode)
            and PROMPT_CACHE_THINKING_MODE == "visible"
            and _is_auto_cache_session_source(session_source or cached_session_source)
            and cached_session_id
            and session_id
            and cached_session_id != session_id
            and reuse_diag.get("miss_reason") == "history_prefix_mismatch"
        ):
            auto_rebuild_reason = "visible_thinking_history_prefix_mismatch"
        if auto_rebuild_reason:
            # 2026-07-06 cache audit: isolation rebuilds used to nuke the live
            # KV un-stashed. Auto-session ids flip on client tool-list churn
            # and on interleaved tool-less probes, so each flip cost the active
            # agent its entire cache (the dominant zero-reuse leak). Isolation
            # only requires that the OTHER chat's KV is not reused for THIS
            # request — a cacheless full prefill satisfies that without
            # destroying anything. Preserve first, destroy last:
            #   1. short prompt vs protected cache -> cacheless bypass
            #   2. otherwise stash to a resident slot, try SSD restore
            #   3. only then rebuild fresh
            if (
                protected_cache_tokens >= PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS
                and len(token_ids) <= PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS
            ):
                _set_prompt_cache_event(
                    "bypass_preserve_auto_isolation",
                    prompt_tokens=len(token_ids),
                    reuse_tokens=0,
                    suffix_tokens=len(token_ids),
                    protected_cache_tokens=protected_cache_tokens,
                    protected_session_id=cached_session_id,
                    protected_session_source=cached_session_source,
                    request_session_id=session_id,
                    request_session_source=session_source,
                    isolation_reason=auto_rebuild_reason,
                    **_prompt_cache_match_fields(len(token_ids), 0),
                    **reuse_diag,
                )
                holder["last_suffix_ids"] = None
                logger.info(
                    "prompt-cache: bypassing %s-session isolation prompt (%d tokens, "
                    "reason=%s) to preserve %s-session %d-token cache",
                    session_id,
                    len(token_ids),
                    auto_rebuild_reason,
                    cached_session_id,
                    protected_cache_tokens,
                )
                return prompt, None
            stashed = False
            if (
                PROMPT_CACHE_RESIDENT_SLOTS > 1
                and protected_cache_tokens >= PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS
            ):
                stashed = _prompt_cache_stash_current_unlocked(
                    reason=f"auto_isolation:{session_id or '__default__'}"
                )
                restored_ssd = _prompt_cache_ssd_maybe_restore_unlocked(
                    model,
                    processor,
                    token_ids,
                    session_id=session_id,
                    session_source=session_source,
                    reason="auto_isolation_after_stash",
                    allow_partial_restore=allow_partial_ssd_restore,
                    allow_thinking_boundary_restore=allow_thinking_boundary_restore,
                    append_reserve_tokens=append_reserve_tokens,
                )
                if restored_ssd:
                    return _prepare_cached_prompt(
                        model,
                        processor,
                        prompt,
                        token_ids,
                        session_id=session_id,
                        session_source=session_source,
                        thinking_mode=thinking_mode,
                        append_reserve_tokens=append_reserve_tokens,
                    )
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "auto_session_isolation_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=0,
                suffix_tokens=len(token_ids),
                previous_reuse_tokens=reuse,
                previous_session_id=cached_session_id,
                previous_session_source=cached_session_source,
                request_session_id=session_id,
                request_session_source=session_source,
                protected_cache_tokens=protected_cache_tokens,
                stashed_previous_session=bool(stashed),
                isolation_reason=auto_rebuild_reason,
                **_prompt_cache_match_fields(len(token_ids), 0),
                **reuse_diag,
            )
            logger.info(
                "prompt-cache: rebuilding auto-session prompt to prevent cross-chat reuse "
                "(reason=%s, previous=%s/%s, next=%s/%s, reuse=%d/%d, stashed=%s)",
                auto_rebuild_reason,
                cached_session_id,
                cached_session_source,
                session_id,
                session_source,
                reuse,
                len(token_ids),
                bool(stashed),
            )
            return prompt, holder["cache"]
        session_mismatch = bool(
            PROMPT_CACHE_SESSION_PROTECT_ENABLED
            and session_id
            and cached_session_id
            and session_id != cached_session_id
        )
        reuse_ratio = (reuse / len(token_ids)) if token_ids else 0.0
        small_static_prefix_switch = bool(
            session_mismatch
            and reuse >= PROMPT_CACHE_MIN_REUSE
            and len(token_ids) <= PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS
            and protected_cache_tokens <= max(
                len(token_ids) + PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS,
                PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS * 4,
            )
            and reuse_ratio >= max(
                0.90,
                PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_REUSE_RATIO,
            )
        )
        if (
            session_mismatch
            and protected_cache_tokens >= PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS
            and not small_static_prefix_switch
        ):
            stashed = False
            if PROMPT_CACHE_RESIDENT_SLOTS > 1:
                stashed = _prompt_cache_stash_current_unlocked(
                    reason=f"session_switch:{session_id or '__default__'}"
                )
            restored_ssd = _prompt_cache_ssd_maybe_restore_unlocked(
                model,
                processor,
                token_ids,
                session_id=session_id,
                session_source=session_source,
                reason="session_switch_after_stash",
                allow_partial_restore=allow_partial_ssd_restore,
                allow_thinking_boundary_restore=allow_thinking_boundary_restore,
                append_reserve_tokens=append_reserve_tokens,
            )
            if restored_ssd:
                return _prepare_cached_prompt(
                    model,
                    processor,
                    prompt,
                    token_ids,
                    session_id=session_id,
                    session_source=session_source,
                    thinking_mode=thinking_mode,
                    append_reserve_tokens=append_reserve_tokens,
                )
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "session_switch_stash_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=0,
                suffix_tokens=len(token_ids),
                previous_reuse_tokens=reuse,
                previous_session_id=cached_session_id,
                previous_session_source=cached_session_source,
                stashed_previous_session=bool(stashed),
                protected_cache_tokens=protected_cache_tokens,
                request_session_id=session_id,
                request_session_source=session_source,
                session_id=session_id,
                session_source=session_source,
                **_prompt_cache_match_fields(len(token_ids), 0),
                **reuse_diag,
            )
            logger.info(
                "prompt-cache: stashed %s-session %d-token cache before switch to %s-session (%d tokens)",
                cached_session_id,
                protected_cache_tokens,
                session_id,
                len(token_ids),
            )
            return prompt, holder["cache"]

        if (
            session_mismatch
            and protected_cache_tokens >= PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS
            and len(token_ids) <= PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS
            and not small_static_prefix_switch
        ):
            _set_prompt_cache_event(
                "bypass_preserve_session_cache",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                suffix_tokens=len(token_ids),
                protected_cache_tokens=protected_cache_tokens,
                protected_session_id=cached_session_id,
                protected_session_source=cached_session_source,
                request_session_id=session_id,
                request_session_source=session_source,
                session_protect_min_tokens=PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS,
                session_bypass_max_tokens=PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS,
                protected_cache_reuse_ratio=round(
                    reuse / protected_cache_tokens, 4
                ) if protected_cache_tokens else 0.0,
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            holder["last_suffix_ids"] = None
            logger.info(
                "prompt-cache: bypassing %s-session short prompt (%d tokens) to preserve %s-session %d-token cache",
                session_id,
                len(token_ids),
                cached_session_id,
                protected_cache_tokens,
            )
            return prompt, None
        if (
            PROMPT_CACHE_PROTECT_LARGE_ENABLED
            and protected_cache_tokens >= PROMPT_CACHE_PROTECT_MIN_TOKENS
            and len(token_ids) <= PROMPT_CACHE_PROTECT_BYPASS_MAX_TOKENS
            and reuse < PROMPT_CACHE_PROTECT_MIN_TOKENS
        ):
            _set_prompt_cache_event(
                "bypass_preserve_large_cache",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                suffix_tokens=len(token_ids),
                protected_cache_tokens=protected_cache_tokens,
                session_id=cached_session_id,
                session_source=cached_session_source,
                request_session_id=session_id,
                request_session_source=session_source,
                protect_min_tokens=PROMPT_CACHE_PROTECT_MIN_TOKENS,
                bypass_max_tokens=PROMPT_CACHE_PROTECT_BYPASS_MAX_TOKENS,
                protected_cache_reuse_ratio=round(
                    reuse / protected_cache_tokens, 4
                ) if protected_cache_tokens else 0.0,
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            holder["last_suffix_ids"] = None
            logger.info(
                "prompt-cache: bypassing short prompt (%d tokens, reuse=%d/%d protected) to preserve %d-token cache",
                len(token_ids),
                reuse,
                PROMPT_CACHE_PROTECT_MIN_TOKENS,
                protected_cache_tokens,
            )
            return prompt, None

        if (
            session_mismatch
            and protected_cache_tokens < PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS
            and PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS > 0
            and len(token_ids) <= PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS
            and reuse >= PROMPT_CACHE_MIN_REUSE
            and reuse_ratio <= PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_REUSE_RATIO
        ):
            # Different short chats often share only MiniMax's static template
            # prefix. Reusing that tiny suffix path can be slower than a fresh
            # small prefill on the distributed runtime, while offering no real
            # multi-turn cache value. Rebuild fresh so the new short session can
            # become hot without disturbing protected long resident slots.
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "static_prefix_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                suffix_tokens=len(token_ids),
                previous_session_id=cached_session_id,
                previous_session_source=cached_session_source,
                protected_cache_tokens=protected_cache_tokens,
                static_prefix_rebuild_max_tokens=PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS,
                static_prefix_rebuild_max_reuse_ratio=(
                    PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_REUSE_RATIO
                ),
                session_id=session_id,
                session_source=session_source,
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            logger.info(
                "prompt-cache: rebuilding small cross-session static-prefix hit "
                "(reuse=%d/%d ratio=%.3f, previous=%s, next=%s)",
                reuse,
                len(token_ids),
                reuse_ratio,
                cached_session_id,
                session_id,
            )
            return prompt, holder["cache"]

        if reuse < PROMPT_CACHE_MIN_REUSE:
            restored_ssd = _prompt_cache_ssd_maybe_restore_unlocked(
                model,
                processor,
                token_ids,
                session_id=session_id,
                session_source=session_source,
                reason="low_reuse_before_rebuild",
                allow_partial_restore=allow_partial_ssd_restore,
                allow_thinking_boundary_restore=allow_thinking_boundary_restore,
                append_reserve_tokens=append_reserve_tokens,
            )
            if restored_ssd:
                return _prepare_cached_prompt(
                    model,
                    processor,
                    prompt,
                    token_ids,
                    session_id=session_id,
                    session_source=session_source,
                    thinking_mode=thinking_mode,
                    append_reserve_tokens=append_reserve_tokens,
                )
            # Not enough overlap to bother: reset and process full prompt.
            _prompt_cache_stash_current_unlocked(reason=f"replace:{session_id or '__default__'}")
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "low_reuse",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                suffix_tokens=len(token_ids),
                session_id=session_id,
                session_source=session_source,
                restored_resident_slot=bool(restored_slot),
                **(restored_slot or {}),
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            logger.info("prompt-cache: low reuse (%d/%d); full prefill", reuse, len(token_ids))
            return prompt, holder["cache"]

        # Trim the cache back to the shared prefix. Track the KV length
        # explicitly instead of querying layer internals; every rank updates
        # this counter with the same prompt length + generated-token count.
        actual_len = int(holder.get("cache_len") or 0)
        if actual_len < reuse:
            logger.warning(
                "prompt-cache length mismatch (cache_len=%d < reuse=%d); rebuilding fresh",
                actual_len, reuse,
            )
            _prompt_cache_stash_current_unlocked(reason=f"length_mismatch:{session_id or '__default__'}")
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "length_mismatch_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                cache_len=actual_len,
                restored_resident_slot=bool(restored_slot),
                **(restored_slot or {}),
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            return prompt, holder["cache"]
        trim_n = actual_len - reuse
        if trim_n > 0 and not _trim_prompt_cache_in_place(cache, trim_n):
            logger.warning("prompt-cache trim failed; rebuilding fresh")
            _prompt_cache_stash_current_unlocked(reason=f"trim_failed:{session_id or '__default__'}")
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "trim_failed_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                cache_was=actual_len,
                session_id=session_id,
                session_source=session_source,
                restored_resident_slot=bool(restored_slot),
                **(restored_slot or {}),
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            return prompt, holder["cache"]
        holder["cache_len"] = reuse

        # Reuse: send only the suffix tokens as the prompt.
        suffix_ids = token_ids[L:]
        min_suffix_tokens = _effective_prompt_cache_min_suffix_tokens(
            thinking_mode,
            session_source or cached_session_source,
            session_id=session_id,
            cached_session_id=cached_session_id,
            miss_reason=reuse_diag.get("miss_reason"),
        )
        bucket_tokens = _runtime_prompt_cache_reuse_bucket_tokens()
        reuse_diag["effective_min_suffix_tokens"] = min_suffix_tokens
        reuse_diag["configured_min_suffix_tokens"] = (
            _runtime_prompt_cache_min_suffix_tokens()
        )
        if not suffix_ids and reuse > 0:
            # Generation needs at least one input token. Back the cache up one
            # token and send that final token through the normal prefill path.
            if not _trim_prompt_cache_in_place(cache, 1):
                logger.warning("prompt-cache exact-hit backtrack failed; full prefill")
                _prompt_cache_stash_current_unlocked(reason=f"exact_hit_backtrack_failed:{session_id or '__default__'}")
                holder["cache"] = None
                holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
                _clear_prompt_cache_key_state_unlocked(holder)
                holder["session_id"] = session_id
                holder["session_source"] = session_source
                holder["last_suffix_ids"] = list(token_ids)
                _set_prompt_cache_event(
                    "exact_hit_backtrack_failed",
                    prompt_tokens=len(token_ids),
                    reuse_tokens=reuse,
                    session_id=session_id,
                    session_source=session_source,
                    restored_resident_slot=bool(restored_slot),
                    **(restored_slot or {}),
                    **_prompt_cache_match_fields(len(token_ids), reuse),
                    **reuse_diag,
                )
                return prompt, holder["cache"]
            reuse -= 1
            holder["cache_len"] = reuse
            suffix_ids = token_ids[reuse:]
            reuse_diag = _prompt_cache_reuse_diagnostics(
                holder, len(token_ids), reuse, token_ids
            )
            reuse_diag = _add_prompt_cache_mismatch_windows(
                reuse_diag, processor, holder.get("token_ids") or [], token_ids
            )
        if (
            suffix_ids
            and bucket_tokens > 0
            and reuse > PROMPT_CACHE_MIN_REUSE
        ):
            bucketed_reuse = (reuse // bucket_tokens) * bucket_tokens
            if bucketed_reuse < PROMPT_CACHE_MIN_REUSE:
                bucketed_reuse = reuse
            if bucketed_reuse < reuse:
                backtrack = reuse - bucketed_reuse
                if not _trim_prompt_cache_in_place(cache, backtrack):
                    logger.warning("prompt-cache reuse-bucket backtrack failed; full prefill")
                    _prompt_cache_stash_current_unlocked(reason=f"reuse_bucket_backtrack_failed:{session_id or '__default__'}")
                    holder["cache"] = None
                    holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
                    _clear_prompt_cache_key_state_unlocked(holder)
                    holder["session_id"] = session_id
                    holder["session_source"] = session_source
                    holder["last_suffix_ids"] = list(token_ids)
                    _set_prompt_cache_event(
                        "reuse_bucket_backtrack_failed",
                        prompt_tokens=len(token_ids),
                        reuse_tokens=reuse,
                        session_id=session_id,
                        session_source=session_source,
                        restored_resident_slot=bool(restored_slot),
                        **(restored_slot or {}),
                        **_prompt_cache_match_fields(len(token_ids), reuse),
                        **reuse_diag,
                    )
                    return prompt, holder["cache"]
                original_reuse = reuse
                reuse = bucketed_reuse
                holder["cache_len"] = reuse
                suffix_ids = token_ids[reuse:]
                reuse_diag = _prompt_cache_reuse_diagnostics(
                    holder, len(token_ids), reuse, token_ids
                )
                reuse_diag = _add_prompt_cache_mismatch_windows(
                    reuse_diag, processor, holder.get("token_ids") or [], token_ids
                )
                reuse_diag["reuse_bucket_tokens"] = bucket_tokens
                reuse_diag["reuse_bucket_original_reuse_tokens"] = original_reuse
                reuse_diag["reuse_bucket_backtrack_tokens"] = backtrack
        if (
            suffix_ids
            and min_suffix_tokens > 0
            and len(suffix_ids) < min_suffix_tokens
            and reuse > PROMPT_CACHE_MIN_REUSE
        ):
            # Very small suffix prefills (often 10-20 tokens in OpenWebUI
            # follow-ups) can underutilize the distributed MLX path and take
            # several seconds before the first decode token. Backtrack a small
            # amount so the prefill runs as a healthier batch while still
            # reusing the overwhelming majority of the prompt.
            backtrack = min(
                reuse,
                max(0, min_suffix_tokens - len(suffix_ids)),
            )
            if backtrack > 0:
                if not _trim_prompt_cache_in_place(cache, backtrack):
                    logger.warning("prompt-cache min-suffix backtrack failed; full prefill")
                    _prompt_cache_stash_current_unlocked(reason=f"min_suffix_backtrack_failed:{session_id or '__default__'}")
                    holder["cache"] = None
                    holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
                    _clear_prompt_cache_key_state_unlocked(holder)
                    holder["session_id"] = session_id
                    holder["session_source"] = session_source
                    holder["last_suffix_ids"] = list(token_ids)
                    _set_prompt_cache_event(
                        "min_suffix_backtrack_failed",
                        prompt_tokens=len(token_ids),
                        reuse_tokens=reuse,
                        session_id=session_id,
                        session_source=session_source,
                        restored_resident_slot=bool(restored_slot),
                        **(restored_slot or {}),
                        **_prompt_cache_match_fields(len(token_ids), reuse),
                        **reuse_diag,
                    )
                    return prompt, holder["cache"]
                original_reuse = reuse
                reuse -= backtrack
                holder["cache_len"] = reuse
                suffix_ids = token_ids[reuse:]
                reuse_diag = _prompt_cache_reuse_diagnostics(
                    holder, len(token_ids), reuse, token_ids
                )
                reuse_diag = _add_prompt_cache_mismatch_windows(
                    reuse_diag, processor, holder.get("token_ids") or [], token_ids
                )
                reuse_diag["min_suffix_original_reuse_tokens"] = original_reuse
                reuse_diag["min_suffix_backtrack_tokens"] = backtrack
                reuse_diag["min_suffix_target_tokens"] = min_suffix_tokens
        if (
            _enable_thinking_for_generation(thinking_mode)
            and PROMPT_CACHE_THINKING_MODE == "visible"
            and _is_auto_cache_session_source(session_source or cached_session_source)
            and cached_session_id
            and session_id
            and cached_session_id != session_id
            and reuse_diag.get("miss_reason") == "history_prefix_mismatch"
        ):
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "visible_thinking_partial_reuse_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=0,
                suffix_tokens=len(token_ids),
                previous_reuse_tokens=reuse,
                previous_session_id=cached_session_id,
                previous_session_source=cached_session_source,
                request_session_id=session_id,
                request_session_source=session_source,
                protected_cache_tokens=protected_cache_tokens,
                **_prompt_cache_match_fields(len(token_ids), 0),
                **reuse_diag,
            )
            logger.info(
                "prompt-cache: rebuilding visible-thinking auto-session after "
                "partial history-prefix reuse became unsafe "
                "(previous=%s/%s, next=%s/%s, reuse=%d/%d)",
                cached_session_id,
                cached_session_source,
                session_id,
                session_source,
                reuse,
                len(token_ids),
            )
            return prompt, holder["cache"]
        if (
            _enable_thinking_for_generation(thinking_mode)
            and PROMPT_CACHE_THINKING_MODE == "visible"
            and PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_TOKENS > 0
            and PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_SUFFIX_TOKENS > 0
            and len(token_ids) <= PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_TOKENS
            and 0 < len(suffix_ids) <= PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_SUFFIX_TOKENS
            and protected_cache_tokens < PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS
        ):
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "small_visible_thinking_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=0,
                suffix_tokens=len(token_ids),
                previous_reuse_tokens=reuse,
                previous_suffix_tokens=len(suffix_ids),
                previous_session_id=cached_session_id,
                previous_session_source=cached_session_source,
                request_session_id=session_id,
                request_session_source=session_source,
                small_rebuild_max_tokens=PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_TOKENS,
                small_rebuild_max_suffix_tokens=PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_SUFFIX_TOKENS,
                protected_cache_tokens=protected_cache_tokens,
                **_prompt_cache_match_fields(len(token_ids), 0),
                **reuse_diag,
            )
            logger.info(
                "prompt-cache: rebuilding small visible-thinking prompt "
                "(prompt=%d, suffix=%d, previous_reuse=%d) because tiny "
                "cached suffixes are slower after idle",
                len(token_ids),
                len(suffix_ids),
                reuse,
            )
            return prompt, holder["cache"]
        try:
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            suffix_prompt = tok.decode(suffix_ids)
            holder["last_suffix_ids"] = list(suffix_ids)
            logger.info(
                "prompt-cache: reusing %d/%d tokens (suffix=%d tokens, cache_was=%d)",
                reuse, len(token_ids), len(suffix_ids), actual_len,
            )
            _set_prompt_cache_event(
                "reuse",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                suffix_tokens=len(suffix_ids),
                cache_was=actual_len,
                session_id=session_id or cached_session_id,
                session_source=session_source or cached_session_source,
                restored_resident_slot=bool(restored_slot),
                **(restored_slot or {}),
                restored_ssd_cache=bool(restored_ssd),
                **(restored_ssd or {}),
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            return suffix_prompt, cache
        except Exception as e:
            logger.warning(f"prompt-cache suffix decode failed: {e}; full prefill")
            _prompt_cache_stash_current_unlocked(reason=f"suffix_decode_failed:{session_id or '__default__'}")
            holder["cache"] = None
            holder["cache"] = _get_or_build_prompt_cache_unlocked(model)
            _clear_prompt_cache_key_state_unlocked(holder)
            holder["session_id"] = session_id
            holder["session_source"] = session_source
            holder["last_suffix_ids"] = list(token_ids)
            _set_prompt_cache_event(
                "suffix_decode_failed_rebuild",
                prompt_tokens=len(token_ids),
                reuse_tokens=reuse,
                error=str(e),
                session_id=session_id,
                session_source=session_source,
                restored_resident_slot=bool(restored_slot),
                **(restored_slot or {}),
                **_prompt_cache_match_fields(len(token_ids), reuse),
                **reuse_diag,
            )
            return prompt, holder["cache"]


def _prompt_cache_mode_for_request(thinking_mode, token_ids):
    if not PROMPT_CACHE_ENABLED or token_ids is None:
        return "off"
    if not _enable_thinking_for_generation(thinking_mode):
        return "full"
    return PROMPT_CACHE_THINKING_MODE


def _prompt_cache_allowed_for_request(thinking_mode, token_ids):
    mode = _prompt_cache_mode_for_request(thinking_mode, token_ids)
    if mode == "off":
        if PROMPT_CACHE_ENABLED and token_ids is not None:
            with _prompt_cache_lock:
                _set_prompt_cache_event(
                    "thinking_cache_bypass",
                    prompt_tokens=len(token_ids),
                    reuse_tokens=0,
                    reason=f"MLX_M3_PROMPT_CACHE_THINKING_MODE={PROMPT_CACHE_THINKING_MODE}",
                    **_prompt_cache_match_fields(len(token_ids), 0),
                )
        return False
    return True


def _prompt_cache_allowed_for_generation(thinking_mode, token_ids, image):
    """Keep text KV reuse away from image-bearing VLM generations.

    MLX-VLM expands image placeholders into feature tokens during prefill. A
    text-prefix cache can trim those placeholders while the caller still
    supplies image features, yielding ``Image features and Image tokens do not
    match`` on a retry or follow-up. Until multimodal cache metadata tracks
    those feature positions explicitly, a full image-bearing prefill is the
    only shape-safe path. Text-only sessions keep their normal RAM/SSD reuse.
    """
    if image is not None:
        return False
    return _prompt_cache_allowed_for_request(thinking_mode, token_ids)


def _should_tokenize_prompt_for_cache(thinking_mode):
    if not PROMPT_CACHE_ENABLED:
        return False
    if _enable_thinking_for_generation(thinking_mode):
        return PROMPT_CACHE_THINKING_MODE != "off"
    return True


def _thinking_generation_hit_limit(thinking_mode, generated_tokens, max_tokens):
    if not _enable_thinking_for_generation(thinking_mode):
        return False
    try:
        limit = int(max_tokens or 0)
    except Exception:
        limit = 0
    if limit <= 0:
        return False
    return int(generated_tokens or 0) >= limit


def _update_prompt_cache_after_generation(token_ids, generated_token_ids=None,
                                          generated_tokens=0,
                                          include_generated_ids=True,
                                          session_id=None,
                                          session_source=None,
                                          prompt=None,
                                          model=None,
                                          processor=None,
                                          save_reason="generation"):
    """Record the full token sequence now held in the cache.

    `stream_generate` mutates the prompt cache in place as it decodes. Earlier
    versions only stored the input prompt as the cache key while tracking
    cache_len as input+generated. That made the next OpenAI chat turn trim away
    the assistant response and re-prefill it. Store exact generated token ids
    when available so follow-up turns can reuse through the previous assistant
    message if the client sends it back unchanged.
    """
    if not PROMPT_CACHE_ENABLED or token_ids is None:
        return
    generated_ids = []
    if generated_token_ids:
        for tok in generated_token_ids:
            try:
                generated_ids.append(int(tok.item() if hasattr(tok, "item") else tok))
            except Exception:
                continue
    elif generated_tokens:
        # Last-resort accounting if an older generator did not expose token ids.
        generated_ids = [None] * int(max(0, generated_tokens))
    with _prompt_cache_lock:
        key_ids = list(token_ids)
        generated_count = len(generated_ids) if generated_ids else int(max(0, generated_tokens))
        generated_key_limit = (
            generated_count
            if PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS <= 0
            else min(generated_count, PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS)
        )
        generated_key_ids = []
        generated_reuse_allowed = (
            include_generated_ids
            and generated_ids
            and all(tok is not None for tok in generated_ids)
            and generated_key_limit > 0
        )
        if generated_reuse_allowed:
            generated_key_ids = generated_ids[:generated_key_limit]
            key_ids.extend(generated_key_ids)
        exact_generated_ids = bool(
            generated_reuse_allowed and len(generated_key_ids) == generated_count
        )
        effective_session_id = session_id
        effective_session_source = session_source
        if effective_session_id is None:
            effective_session_id = _prompt_cache_holder.get("session_id")
            effective_session_source = _prompt_cache_holder.get("session_source")
        _prompt_cache_holder["token_ids"] = key_ids
        _prompt_cache_holder["prompt"] = (
            prompt
            if isinstance(prompt, str) and len(key_ids) == len(token_ids)
            else None
        )
        counted_len = int(len(token_ids) + generated_count)
        cache_len = counted_len
        cache_obj = _prompt_cache_holder.get("cache")
        try:
            physical_len = int(cache_obj[0].offset) if cache_obj else None
        except Exception:
            physical_len = None
        if physical_len is not None and physical_len != counted_len:
            # 2026-07-08 goals-crash fix: per-rank token accounting under-
            # counts when a synchronized stop drains past the consumer's exit
            # (rank1 kept 290 of 375 lockstep-drained tokens), splitting this
            # counter from the physical KV and making the next turn's suffix
            # plans diverge across ranks into a pipeline freeze. The KV cache
            # itself is the ground truth both ranks share through the
            # lockstep drain, so record ITS length and clamp the reuse key.
            logger.warning(
                "prompt-cache: generation accounting (%d) != physical KV "
                "length (%d); trusting the physical cache",
                counted_len, physical_len,
            )
            cache_len = physical_len
            if len(key_ids) > cache_len:
                key_ids = key_ids[:cache_len]
                _prompt_cache_holder["token_ids"] = key_ids
        _prompt_cache_holder["cache_len"] = cache_len
        _prompt_cache_holder["last_input_tokens"] = int(len(token_ids))
        _prompt_cache_holder["last_generated_tokens"] = int(generated_count)
        _prompt_cache_holder["last_exact_generated_ids"] = exact_generated_ids
        _prompt_cache_holder["session_id"] = effective_session_id
        _prompt_cache_holder["session_source"] = effective_session_source
        _set_prompt_cache_event(
            "updated",
            phase="update",
            prompt_tokens=len(token_ids),
            generated_tokens=generated_count,
            key_tokens=len(key_ids),
            cache_len=_prompt_cache_holder["cache_len"],
            session_id=effective_session_id,
            session_source=effective_session_source,
            exact_generated_ids=exact_generated_ids,
            generated_reuse_allowed=bool(generated_reuse_allowed),
            generated_key_tokens=len(generated_key_ids),
            generated_key_truncated=bool(
                generated_reuse_allowed and len(generated_key_ids) < generated_count
            ),
            generated_reuse_max_tokens=PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS,
            **_prompt_cache_match_fields(len(key_ids), len(token_ids)),
        )
        if (
            model is not None
            and processor is not None
            and PROMPT_CACHE_SSD_AUTO_SAVE
        ):
            _prompt_cache_ssd_maybe_autosave_unlocked(
                model,
                processor,
                prompt=prompt,
                reason=save_reason,
            )
    _enforce_prompt_cache_size_limit()


def _keep_prompt_prefix_after_cancel(token_ids, reason="cancelled"):
    """Trim this rank's cache to the request's input prefix.

    Cancelled streams used to _reset_prompt_cache, but agent clients retry
    the SAME conversation after a client-side timeout, and the reset forced
    a full re-prefill of the whole context on every retry. This helper is a
    rank-local best effort only: a distributed caller must not assume both
    ranks retained the same physical cache merely because rank 0 succeeded.
    User/client cancellation therefore follows this local trim with a
    coordinated two-rank reset at the HTTP transaction boundary. Returns
    True when this rank retained the prefix; local callers may otherwise
    fall back to a full reset."""
    if (
        not PROMPT_CACHE_KEEP_ON_CANCEL
        or not PROMPT_CACHE_ENABLED
        or not token_ids
    ):
        return False
    # A bypassed request never touched the holder cache — trimming it to
    # THIS request's input would corrupt the preserved session.
    try:
        if _prompt_cache_prepare_preserves_existing_cache():
            return False
    except Exception:
        return False
    n_input = len(token_ids)
    with _prompt_cache_lock:
        holder = _prompt_cache_holder
        cache = holder.get("cache")
        if cache is None:
            return False
        try:
            phys = int(cache[0].offset)
        except Exception:
            return False
        if phys < n_input:
            return False
        trim_n = phys - n_input
        if trim_n > 0 and not _trim_prompt_cache_in_place(cache, trim_n):
            return False
        holder["token_ids"] = list(token_ids)
        holder["prompt"] = None
        holder["cache_len"] = n_input
        holder["last_suffix_ids"] = None
        holder["last_input_tokens"] = n_input
        holder["last_generated_tokens"] = 0
        holder["last_exact_generated_ids"] = False
        _set_prompt_cache_event(
            "cancel_kept_prefix",
            phase="update",
            reason=reason,
            prompt_tokens=n_input,
            trimmed_generated_tokens=trim_n,
            cache_len=n_input,
        )
    logger.info(
        "prompt-cache: kept %d-token input prefix after %s (trimmed %d generated)",
        n_input, reason, trim_n,
    )
    return True


def _reset_prompt_cache(reason="reset", *, clear_manifest=False,
                        clear_resident=True):
    """Drop the cached KV (e.g. on error or model change)."""
    with _prompt_cache_lock:
        _drop_prompt_cache_unlocked(reason, clear_manifest=clear_manifest,
                                    clear_resident=clear_resident)
    logger.info(f"prompt-cache {reason}")


def _reset_prompt_cache_and_clear_memory(reason="reset", *, clear_manifest=False,
                                         clear_resident=True):
    """Drop cached prompt KV and release idle MLX buffers."""
    _reset_prompt_cache(reason, clear_manifest=clear_manifest,
                        clear_resident=clear_resident)
    _clear_mlx_memory(f"prompt-cache {reason}")


def _reset_prompt_cache_on_all_ranks(rank, reason="reset", *, clear_memory=False,
                                     clear_manifest=False, clear_resident=True):
    """Reset the distributed RAM prompt cache after a post-decode rank-0 verdict."""
    if rank == 0:
        try:
            _bcast({
                "op": "reset_prompt_cache",
                "reason": reason,
                "clear_memory": bool(clear_memory),
                "clear_manifest": bool(clear_manifest),
                "clear_resident": bool(clear_resident),
            }, rank)
        except Exception as e:
            logger.warning("prompt-cache reset broadcast failed (%s): %s", reason, e)
    if clear_memory:
        _reset_prompt_cache_and_clear_memory(reason, clear_manifest=clear_manifest,
                                             clear_resident=clear_resident)
    else:
        _reset_prompt_cache(reason, clear_manifest=clear_manifest,
                            clear_resident=clear_resident)


def _prewarm_prompt_cache(model, processor, prompt, token_ids, *,
                          reason="prewarm", session_id=None,
                          session_source=None,
                          reset_on_failure=True):
    """Replace the shared prompt cache with a prefetched prompt prefix.

    MLX-VLM does not expose a public "prefill only" helper for this path, so
    we run a one-token generation against a fresh cache and trim the sampled
    token back off. Both distributed ranks do the same work via a mirror op.
    """
    if not PROMPT_CACHE_ENABLED or token_ids is None:
        return False
    token_ids = list(token_ids)
    if not token_ids:
        return False
    # A one-token visible-transcript prewarm still evaluates attention over the
    # entire resident KV. At very large context this optional post-response
    # pass can wedge inside a Metal event after the client has already received
    # its answer. The limit is rank-invariant because both peers receive the
    # same token_ids, so both can safely skip before entering collectives while
    # preserving the completed request's existing cache state.
    if (
        VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS > 0
        and len(token_ids) > VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS
    ):
        _set_prompt_cache_event(
            "prewarm_skipped_context_limit",
            reason=reason,
            prompt_tokens=len(token_ids),
            max_tokens=VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS,
            session_id=session_id,
            session_source=session_source,
        )
        logger.info(
            "prompt-cache visible prewarm skipped above hard context limit "
            "(%d > %d tokens)",
            len(token_ids),
            VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS,
        )
        return False
    existing_reuse = 0
    existing_suffix = len(token_ids)
    with _prompt_cache_lock:
        existing_ids = list(_prompt_cache_holder.get("token_ids") or [])
    if existing_ids:
        existing_reuse = _common_prefix_len(existing_ids, token_ids)
        existing_suffix = max(0, len(token_ids) - existing_reuse)
    my_skip_too_large = False
    if (
        VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS > 0
        and len(token_ids) > VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS
        and (
            existing_reuse < PROMPT_CACHE_MIN_REUSE
            or existing_suffix > VISIBLE_TRANSCRIPT_PREWARM_MAX_SUFFIX_TOKENS
        )
    ):
        _set_prompt_cache_event(
            "prewarm_skipped_too_large",
            reason=reason,
            prompt_tokens=len(token_ids),
            reuse_tokens=existing_reuse,
            suffix_tokens=existing_suffix,
            max_tokens=VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS,
            max_suffix_tokens=VISIBLE_TRANSCRIPT_PREWARM_MAX_SUFFIX_TOKENS,
        )
        # 2026-07-08: do NOT return yet — the too-large decision reads the
        # rank-LOCAL holder (existing_reuse), so ranks can disagree; a lone
        # early return here strands the peer in the prewarm's pipeline
        # collectives. Carry the skip into the consensus below instead.
        my_skip_too_large = True

    from mlx_vlm.generate import stream_generate

    _refresh_generation_stream()
    _mark_prompt_cache_in_use(True)
    generated = 0
    cache = None
    prompt_to_send = prompt
    try:
        my_skip = my_skip_too_large
        prepare_event = {}
        if not my_skip_too_large:
            prompt_to_send, cache = _prepare_cached_prompt(
                model, processor, prompt, token_ids,
                session_id=session_id,
                session_source=session_source,
                append_reserve_tokens=1,
            )
            prepare_event = _prompt_cache_status().get("last_prepare_event") or {}
            my_skip = bool(
                cache is None
                and _prompt_cache_prepare_preserves_existing_cache(
                    prepare_event.get("action")
                )
            )
        verdict = _prewarm_plan_consensus(my_skip, prompt_to_send, cache)
        if verdict == "skip":
            if not my_skip_too_large:
                logger.info(
                    "prompt-cache visible prewarm skipped to preserve %s-token cache",
                    prepare_event.get("protected_cache_tokens"),
                )
            return False
        if verdict == "diverged":
            logger.warning(
                "prompt-cache prewarm plan diverged across ranks; resetting "
                "cache and skipping prewarm"
            )
            if reset_on_failure:
                _reset_prompt_cache("prewarm plan divergence")
            return False
        _set_prompt_cache_event(
            "prewarm_start",
            reason=reason,
            prompt_tokens=len(token_ids),
            reuse_tokens=int(prepare_event.get("reuse_tokens") or 0),
            suffix_tokens=int(prepare_event.get("suffix_tokens") or len(token_ids)),
            reuse_action=prepare_event.get("action"),
            cache_was=prepare_event.get("cache_was"),
            session_id=session_id,
            session_source=session_source,
        )

        gen_kwargs = dict(
            model=model,
            processor=processor,
            prompt=prompt_to_send,
            max_tokens=1,
            enable_thinking=False,
            prefill_step_size=_runtime_prefill_step_size(len(token_ids)),
            max_kv_size=MAX_KV_SIZE,
            prompt_cache=cache,
            temperature=0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
        )
        gen_kwargs.update(_kv_quant_kwargs())
        # BATCH PATH (2026-07-07): this prewarm was the LAST caller of the
        # stream generator on the pipeline — the pre-building geometry that
        # deadlocks ranks when one side closes early (the exact disease the
        # batch-cancel path was built to remove from requests). Every freeze
        # photograph since 21:50 sits adjacent to a prewarm; the recurring
        # wire error (status=1 wr_id=0x20001) matches an abandoned pre-built
        # step's unmatched send/recv. Same cure as requests: _generation_iter.
        _pw_rank, _pw_world = _prompt_cache_ssd_current_rank_world()
        with _tokenizer_runtime_lock:
            for response in _generation_iter(_pw_rank, gen_kwargs):
                generation_tokens = int(
                    getattr(response, "generation_tokens", 0) or 0
                )
                generated = max(generated, generation_tokens)

        if generated > 0 and not _trim_prompt_cache_in_place(cache, generated):
            logger.warning("prompt-cache visible prewarm trim failed; dropping cache")
            if reset_on_failure:
                _reset_prompt_cache("prewarm trim failed")
            return False
        mx.eval([c.state for c in cache])
        with _prompt_cache_lock:
            _prompt_cache_holder["token_ids"] = token_ids
            _prompt_cache_holder["prompt"] = prompt
            _prompt_cache_holder["cache_len"] = len(token_ids)
            _prompt_cache_holder["last_input_tokens"] = len(token_ids)
            _prompt_cache_holder["last_generated_tokens"] = 0
            _prompt_cache_holder["last_exact_generated_ids"] = False
            _prompt_cache_holder["session_id"] = session_id
            _prompt_cache_holder["session_source"] = session_source
            _set_prompt_cache_event(
                "prewarm_visible_transcript",
                phase="update",
                reason=reason,
                prompt_tokens=len(token_ids),
                prefill_tokens=len(_tokenize_prompt(processor, prompt_to_send) or []),
                generated_tokens=0,
                key_tokens=len(token_ids),
                cache_len=len(token_ids),
                session_id=session_id,
                session_source=session_source,
                trimmed_generated_tokens=generated,
                **_prompt_cache_match_fields(len(token_ids), len(token_ids)),
            )
            if PROMPT_CACHE_SSD_AUTO_SAVE:
                _prompt_cache_ssd_maybe_autosave_unlocked(
                    model,
                    processor,
                    reason=f"visible_transcript_prewarm:{reason}",
                )
        logger.info(
            "prompt-cache visible prewarm complete (%d tokens, trimmed=%d)",
            len(token_ids), generated,
        )
        return True
    except Exception as e:
        logger.warning("prompt-cache visible prewarm failed: %s", e)
        if _prompt_cache_prepare_preserves_existing_cache():
            logger.info("prompt-cache: preserved large cache after bypassed prewarm failure")
            return False
        if reset_on_failure:
            _reset_prompt_cache("prewarm failed")
        return False
    finally:
        _mark_prompt_cache_in_use(False)


def _render_visible_transcript_prompt(model, processor, processed_messages,
                                      assistant_content, *, num_images=0,
                                      thinking_mode=None, tools=None,
                                      assistant_reasoning=None):
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        return None
    if num_images:
        return None
    from mlx_vlm.prompt_utils import apply_chat_template

    mode = thinking_mode if thinking_mode in VALID_THINKING_MODES else DEFAULT_THINKING_MODE
    thinking_enabled = _enable_thinking_for_generation(mode)
    visible_messages = [dict(m) for m in processed_messages]
    assistant_message = {
        "role": "assistant",
        "content": _sanitize_inbound_message_content("assistant", assistant_content),
    }
    if isinstance(assistant_reasoning, str) and assistant_reasoning.strip():
        assistant_message["reasoning_content"] = assistant_reasoning
        assistant_message["content"] = _assistant_content_for_template(
            assistant_message,
            assistant_message["content"],
        )
    elif thinking_enabled and PROMPT_CACHE_THINKING_MODE == "visible":
        assistant_message["reasoning_content"] = " "
    visible_messages.append(assistant_message)
    tk = _thinking_template_kwargs(
        model.config,
        enable_thinking=(mode == "enabled"),
        thinking_mode=mode,
    )
    if tools:
        tk["tools"] = tools
    return apply_chat_template(
        processor,
        model.config,
        visible_messages,
        add_generation_prompt=False,
        num_images=0,
        **tk,
    )


def _maybe_prewarm_visible_transcript(model, processor, rank, processed_messages,
                                      raw_output, *, thinking_mode, generated_tokens,
                                      num_images=0, tools=None,
                                      visible_output=None,
                                      session_id=None,
                                      session_source=None,
                                      preserve_reasoning=False):
    thinking_enabled = _enable_thinking_for_generation(thinking_mode)
    min_generated = _runtime_visible_transcript_prewarm_min_generated()
    if (
        not VISIBLE_TRANSCRIPT_PREWARM_ENABLED
        or not PROMPT_CACHE_ENABLED
        or rank != 0
        or tools
        or num_images
        or int(generated_tokens or 0) < min_generated
    ):
        return False
    if (
        VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS > 0
        and int(generated_tokens or 0) > VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS
    ):
        _set_prompt_cache_event(
            "prewarm_skipped_generated_too_large",
            reason="visible_transcript_after_response",
            generated_tokens=int(generated_tokens or 0),
            max_generated_tokens=VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS,
            thinking_mode=thinking_mode,
            session_id=session_id,
            session_source=session_source,
        )
        return False
    if thinking_enabled and PROMPT_CACHE_THINKING_MODE != "visible":
        return False
    content = visible_output if isinstance(visible_output, str) else None
    reasoning = None
    if content is None:
        reasoning, content = split_thinking_text(
            raw_output or "",
            assume_in_thinking=thinking_enabled,
        )
        if thinking_enabled and not reasoning:
            return False
    elif thinking_enabled:
        reasoning, _ = split_thinking_text(
            raw_output or "",
            assume_in_thinking=True,
        )
    if not content:
        return False
    try:
        prompt = _render_visible_transcript_prompt(
            model, processor, processed_messages, content,
            num_images=num_images,
            thinking_mode=thinking_mode,
            assistant_reasoning=(
                reasoning
                if thinking_enabled and preserve_reasoning
                else None
            ),
        )
        token_ids = _tokenize_prompt(processor, prompt) if prompt else None
        if not prompt or not token_ids:
            return False
        _bcast({
            "op": "prewarm_prompt_cache",
            "prompt": prompt,
            "token_ids": token_ids,
            "reason": "visible_transcript_after_response",
            "visible_source": "stream_delta" if visible_output is not None else "raw_split",
            "thinking_mode": thinking_mode,
            "session_id": session_id,
            "session_source": session_source,
        }, rank)
        return _prewarm_prompt_cache(
            model,
            processor,
            prompt,
            token_ids,
            reason=(
                "visible_transcript_after_response:"
                + ("stream_delta" if visible_output is not None else "raw_split")
                + f":{thinking_mode}"
            ),
            session_id=session_id,
            session_source=session_source,
        )
    except Exception as e:
        logger.warning("visible transcript prewarm setup failed: %s", e)
        return False


def _install_omlx_minimax_overlay():
    """Prefer the vendored oMLX MiniMax-M3 model subtree when enabled."""
    if not OMLX_MINIMAX_OVERLAY:
        return False

    overlay_models = (
        os.path.dirname(os.path.abspath(__file__))
        + "/MSA Support/mlx_vlm/models"
    )
    if not os.path.isdir(overlay_models):
        logger.warning(
            "MiniMax MSA support requested but missing: %s", overlay_models
        )
        return False

    try:
        import importlib
        import mlx_vlm.models as vlm_models

        package_path = getattr(vlm_models, "__path__", None)
        if package_path is None:
            logger.warning("MiniMax MSA support: mlx_vlm.models has no __path__")
            return False
        if overlay_models in package_path:
            package_path.remove(overlay_models)
        package_path.insert(0, overlay_models)

        for name in list(sys.modules):
            if name.startswith("mlx_vlm.models.minimax_m3_vl"):
                del sys.modules[name]

        lang = importlib.import_module("mlx_vlm.models.minimax_m3_vl.language")
        has_msa = hasattr(
            getattr(lang, "MiniMaxAttention", object), "_msa_prefill_attention"
        )
        logger.info(
            "MiniMax MSA support installed from %s (msa_prefill=%s, language=%s)",
            overlay_models, has_msa, getattr(lang, "__file__", "?"),
        )
        return bool(has_msa)
    except Exception as e:
        logger.warning("MiniMax MSA support failed: %s", e)
        return False


def _stop_requested():
    """True if an in-flight stop was requested locally."""
    return _STOP_FLAG.is_set()


class GenerationCancelled(Exception):
    """Expected control-flow exception for synchronized in-flight cancellation."""


def _stop_requested_synced_reason(rank, token_index):
    """REMOVED 2026-07-06 (dead-code audit). This function issued a per-token
    distributed all_sum on stream=mx.cpu, concurrent with the model's own
    collectives on the same QP/CQ — a cross-stream ibverbs race that silently
    lost completions: the root cause of every historical 10k-decode wedge.
    Inert tombstone: always reports "no stop". Safe replacement: the
    nonce-coordinated stop file (see _request_inflight_stop / the decode
    file-stop checks in both generation loops)."""
    return False, None


def _stop_requested_synced(rank, token_index):
    requested, _reason = _stop_requested_synced_reason(rank, token_index)
    return requested


def _check_prefill_stop(rank, processed_tokens, total_tokens):
    """Abort prefill at a synchronized chunk boundary when stop was requested."""
    step = max(1, _runtime_prefill_step_size(total_tokens))
    chunk_index = max(1, int((int(processed_tokens or 0) + step - 1) // step))
    stop_payload = _read_prefill_stop_file()
    if stop_payload is not None:
        payload_nonce = str(stop_payload.get("nonce") or "")
        active_nonce = str(_STOP_NONCE.get("value") or "")
        if payload_nonce and active_nonce and payload_nonce != active_nonce:
            logger.debug(
                "ignoring stale prefill stop nonce %s for active nonce %s",
                payload_nonce[:8],
                active_nonce[:8],
            )
            return
        _set_stop_request(
            "user",
            request_id=stop_payload.get("request_id"),
        )
        stop_at = stop_payload.get("prefill_stop_at_tokens")
        if stop_at is None:
            stop_at = stop_payload.get("stop_at_tokens")
        try:
            stop_at = int(stop_at) if stop_at is not None else None
        except Exception:
            stop_at = None
        if stop_at is None or int(processed_tokens or 0) >= stop_at:
            raise GenerationCancelled(
                f"stop requested during prefill at {int(processed_tokens or 0)}/"
                f"{int(total_tokens or 0)} tokens"
            )
    if PREFILL_STOP_CHECK_EVERY <= 0:
        return
    if (
        chunk_index % PREFILL_STOP_CHECK_EVERY != 0
        and int(processed_tokens or 0) < int(total_tokens or 0)
    ):
        return
    if _stop_requested_synced(rank, chunk_index * STOP_CHECK_EVERY):
        raise GenerationCancelled(
            f"stop requested during prefill at {int(processed_tokens or 0)}/"
            f"{int(total_tokens or 0)} tokens"
        )



def _configure_metal_memory_limits():
    """Match official MLX servers: set wired limit before model load.

    MLX-LM's server sets the wired limit to Apple's recommended working-set
    size at startup. MiniMax-M3 tensor sharding can need a higher explicit
    limit on smaller-memory worker machines, so the launch script can override it.
    """
    global _METAL_LIMITS
    if not getattr(mx, "metal", None) or not mx.metal.is_available():
        _METAL_LIMITS = {"available": False}
        return

    info = mx.device_info()
    recommended = int(info.get("max_recommended_working_set_size", 0) or 0)
    max_working_set = int(info.get("max_working_set_size", 0) or 0)
    limits = {
        "available": True,
        "recommended_gb": _bytes_to_gb(recommended) if recommended else None,
        "max_working_set_gb": _bytes_to_gb(max_working_set) if max_working_set else None,
        "wired_limit_gb": None,
        "memory_limit_gb": None,
        "cache_limit_gb": None,
    }

    wired_limit = _gb_to_bytes(WIRED_LIMIT_GB) if WIRED_LIMIT_GB > 0 else recommended
    if max_working_set and wired_limit > max_working_set:
        logger.warning(
            "requested MLX_M3_WIRED_LIMIT_GB=%s exceeds device max %.2f GB; clamping",
            WIRED_LIMIT_GB,
            _bytes_to_gb(max_working_set),
        )
        wired_limit = max_working_set
    if wired_limit:
        try:
            old = mx.set_wired_limit(wired_limit)
        except ValueError:
            if recommended and wired_limit != recommended:
                logger.warning(
                    "wired limit %.2f GB rejected; falling back to recommended %.2f GB",
                    _bytes_to_gb(wired_limit),
                    _bytes_to_gb(recommended),
                )
                wired_limit = recommended
                old = mx.set_wired_limit(wired_limit)
            else:
                raise
        limits["wired_limit_gb"] = _bytes_to_gb(wired_limit)
        limits["old_wired_limit_gb"] = _bytes_to_gb(old)

    if MEMORY_LIMIT_GB > 0:
        memory_limit = _gb_to_bytes(MEMORY_LIMIT_GB)
        old = mx.set_memory_limit(memory_limit)
        limits["memory_limit_gb"] = _bytes_to_gb(memory_limit)
        limits["old_memory_limit_gb"] = _bytes_to_gb(old)

    if CACHE_LIMIT_GB >= 0:
        cache_limit = _gb_to_bytes(CACHE_LIMIT_GB)
        old = mx.set_cache_limit(cache_limit)
        limits["cache_limit_gb"] = _bytes_to_gb(cache_limit)
        limits["old_cache_limit_gb"] = _bytes_to_gb(old)

    _METAL_LIMITS = limits
    logger.info("Metal memory limits: %s", limits)


def _watchdog_tick(progress=True):
    tick = _WATCHDOG_TICK
    if tick is not None:
        try:
            tick(progress=progress)
        except Exception:
            pass


def _watchdog_note_prefill(tokens):
    """Tell the watchdog how many tokens this turn will prefill so it can size
    the stall window to the work — a large prefill blocks in the jaccl recv for
    the whole chunk and would otherwise trip the fixed 240s window (fix A)."""
    hook = _WATCHDOG_PREFILL_BUDGET
    if hook is not None:
        try:
            hook(tokens)
        except Exception:
            pass


def _clear_mlx_memory(reason):
    """Best-effort MLX/Metal cleanup before ordinary process exits."""
    try:
        logger.info(f"clearing MLX/Metal cache ({reason})")
        # Drop Python references first so their MLX buffers enter the allocator
        # cache before we flush it.  Clearing in the opposite order leaves the
        # just-collected buffers wired until a later request.
        gc.collect()
        mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        logger.info(f"MLX/Metal cache cleared ({reason})")
    except Exception as e:
        logger.warning(f"MLX/Metal cache clear failed during {reason}: {e}")


def _clear_transient_mlx_memory(reason):
    """Release completed-request graphs without touching live prompt KV."""
    try:
        collected = gc.collect()
        mx.clear_cache()
        logger.debug(
            "cleared transient MLX allocations (%s, collected=%d)",
            reason,
            collected,
        )
    except Exception as e:
        logger.debug("transient MLX cleanup failed during %s: %s", reason, e)


def _install_shutdown_handlers():
    def _handle_signal(signum, _frame):
        global _SHUTTING_DOWN
        if _SHUTTING_DOWN:
            os._exit(128 + signum)
        _SHUTTING_DOWN = True
        name = signal.Signals(signum).name
        logger.warning(f"received {name}; releasing MLX/Metal memory before exit")
        _clear_mlx_memory(name)
        raise SystemExit(128 + signum)

    for sig_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Per-request generation stream refresh
# ---------------------------------------------------------------------------
def _refresh_generation_stream():
    """Create a FRESH MLX stream for the generation pipeline, replacing the
    module-level generation_stream in all mlx_vlm generate submodules.

    ROOT CAUSE of the 4th-request crash: the module-level generation_stream
    (created once at import via mx.new_thread_local_stream) accumulates Metal
    command buffers across requests. After ~3 requests the GPU queue depth
    exceeds the driver timeout -> SIGABRT -> orphaned wired memory -> reboot.
    A fresh stream per request means each generation starts with an empty
    queue -> no accumulation -> no crash.

    NOTE: in mlx_vlm 0.6.3 the submodules are imported directly (NOT via
    `from mlx_vlm.generate import common` — that resolves to the generate
    FUNCTION, not the package, and raises ImportError). This bug was silently
    breaking every streaming request before.
    """
    if not REFRESH_GENERATION_STREAM:
        return
    _fresh = mx.new_thread_local_stream(mx.default_device())
    import importlib
    for mod_name in (
        "mlx_vlm.generate.common",
        "mlx_vlm.generate.dispatch",
        "mlx_vlm.generate.ar",
    ):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "generation_stream"):
                mod.generation_stream = _fresh
        except Exception as e:
            logger.debug(f"could not refresh generation_stream in {mod_name}: {e}")


def _install_decode_eval_patch():
    """Force periodic decode materialization in mlx_vlm's AR generator.

    The stock MLX-VLM decode loop uses async_eval/decode-ahead and only clears
    cache every 256 generated tokens. That is good for single-process models,
    but our MiniMax-M3 pipeline adds distributed send/recv/all_gather plus
    MiniMax sparse-index cache mutation on every token. Letting that lazy graph
    run too far ahead has been reproducibly wedging around 66-68 tokens.

    Evaluating the yielded token/logprobs at a configurable cadence gives the
    distributed graph a hard drain point. Early decode can use a looser cadence
    for throughput, then long generations can switch to a tighter cadence before
    the lazy graph grows enough to wedge the distributed pipeline.
    """
    if DECODE_EVAL_EVERY <= 0:
        logger.info("decode eval patch disabled")
        return
    import importlib
    ar_mod = importlib.import_module("mlx_vlm.generate.ar")
    dispatch_mod = importlib.import_module("mlx_vlm.generate.dispatch")

    orig = ar_mod.generate_step
    if getattr(orig, "_m3_decode_eval_patched", False):
        return

    def _patched_generate_step(*args, **kwargs):
        for idx, item in enumerate(orig(*args, **kwargs), start=1):
            forced = int(getattr(_DECODE_EVAL_CONTEXT, "force_every", 0) or 0)
            if forced > 0:
                cadence = forced
            else:
                cadence = DECODE_EVAL_EVERY
                if (
                    DECODE_EVAL_AFTER_TOKENS > 0
                    and idx >= DECODE_EVAL_AFTER_TOKENS
                    and DECODE_EVAL_AFTER_EVERY > 0
                ):
                    cadence = DECODE_EVAL_AFTER_EVERY
            if cadence > 0 and idx % cadence == 0:
                try:
                    token, logprobs = item
                    mx.eval(token, logprobs)
                    _watchdog_tick(progress=False)
                except Exception:
                    pass
            yield item

    _patched_generate_step._m3_decode_eval_patched = True
    ar_mod.generate_step = _patched_generate_step
    dispatch_mod.generate_step = _patched_generate_step
    logger.info(
        "installed decode eval patch (every %s token(s), after %s -> every %s)",
        DECODE_EVAL_EVERY,
        DECODE_EVAL_AFTER_TOKENS,
        DECODE_EVAL_AFTER_EVERY,
    )


@contextmanager
def _decode_eval_context(force_every=0):
    previous = int(getattr(_DECODE_EVAL_CONTEXT, "force_every", 0) or 0)
    _DECODE_EVAL_CONTEXT.force_every = int(force_every or 0)
    try:
        yield
    finally:
        _DECODE_EVAL_CONTEXT.force_every = previous


def _decode_eval_force_for_thinking(thinking_mode):
    if _enable_thinking_for_generation(thinking_mode):
        with _runtime_tuning_lock:
            return int(_runtime_tuning.get("thinking_decode_eval_every") or 0)
    return 0


def _decode_eval_force_for_request(thinking_mode, token_ids):
    force = _decode_eval_force_for_thinking(thinking_mode)
    token_count = len(token_ids) if token_ids is not None else 0
    with _runtime_tuning_lock:
        long_context_tokens = int(
            _runtime_tuning.get("long_context_decode_eval_tokens") or 0
        )
        long_context_every = int(
            _runtime_tuning.get("long_context_decode_eval_every") or 0
        )
        adaptive_enabled = bool(
            int(_runtime_tuning.get("adaptive_long_context_decode_eval") or 0)
        )
        mid_context_tokens = int(
            _runtime_tuning.get("mid_context_decode_eval_tokens") or 0
        )
        mid_context_every = int(
            _runtime_tuning.get("mid_context_decode_eval_every") or 0
        )
        high_context_tokens = int(
            _runtime_tuning.get("high_context_decode_eval_tokens") or 0
        )
        high_context_every = int(
            _runtime_tuning.get("high_context_decode_eval_every") or 0
        )
    if adaptive_enabled:
        tier_every = 0
        if (
            high_context_tokens > 0
            and high_context_every > 0
            and token_count >= high_context_tokens
        ):
            tier_every = high_context_every
        elif (
            mid_context_tokens > 0
            and mid_context_every > 0
            and token_count >= mid_context_tokens
        ):
            tier_every = mid_context_every
        elif (
            long_context_tokens > 0
            and long_context_every > 0
            and token_count >= long_context_tokens
        ):
            tier_every = long_context_every
        if tier_every > 0:
            if force > 0:
                return min(force, tier_every)
            return tier_every
        return force
    if (
        long_context_tokens > 0
        and long_context_every > 0
        and token_count >= long_context_tokens
    ):
        if force > 0:
            return min(force, long_context_every)
        return long_context_every
    return force


# ---------------------------------------------------------------------------
# Reasoning/content splitting (ported from mlx_vlm/server/app.py)
# ---------------------------------------------------------------------------
# These marker pairs are exactly what the official mlx_vlm server uses.
# <mm:think>...</mm:think> is MiniMax-M3's native thinking format.
_DEFAULT_MARKER_PAIRS = [
    ("<|channel>analysis", "<channel|>"),
    ("<|channel>thought", "<channel|>"),
    ("<|channel>thinking", "<channel|>"),
    ("<think>", "</think>"),
    ("<mm:think>", "</mm:think>"),
]


def _strip_thinking_control_markers(text):
    if not text:
        return text
    for start_marker, end_marker in _DEFAULT_MARKER_PAIRS:
        text = text.replace(start_marker, "")
        text = text.replace(end_marker, "")
    text = re.sub(r"<\|(?:channel|message|end|start)[^>]*>", "", text)
    return text


def _find_earliest_marker(text, markers):
    marker_pos, found_marker = -1, ""
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0 and (marker_pos < 0 or pos < marker_pos):
            marker_pos, found_marker = pos, marker
    return marker_pos, found_marker


def _partial_marker_suffix(text, markers):
    suffix = ""
    for marker in markers:
        max_len = min(len(marker) - 1, len(text))
        for length in range(1, max_len + 1):
            candidate = text[-length:]
            if marker.startswith(candidate) and len(candidate) > len(suffix):
                suffix = candidate
    return suffix


def split_stream_thinking_delta(accumulated, delta, in_thinking,
                                *, at_response_start=False,
                                thinking_start_token=None,
                                thinking_end_token=None):
    """Split a streamed token into reasoning/content deltas.

    Direct port of mlx_vlm/server/app.py::_split_stream_thinking_delta so our
    reasoning routing matches the official server exactly. Returns:
      (in_thinking, accumulated, at_response_start, delta_reasoning, delta_content)
    """
    delta_reasoning = None
    delta_content = None
    marker_pairs = list(_DEFAULT_MARKER_PAIRS)
    if thinking_start_token and thinking_end_token:
        marker_pairs.insert(0, (thinking_start_token, thinking_end_token))
    start_markers = tuple(dict.fromkeys(s for s, _ in marker_pairs))
    end_markers = tuple(dict.fromkeys(e for _, e in marker_pairs))

    if at_response_start and not in_thinking:
        leading_text = accumulated.lstrip()
        if not leading_text:
            return in_thinking, accumulated, at_response_start, None, None
        for end_marker in end_markers:
            if leading_text.startswith(end_marker):
                accumulated = leading_text[len(end_marker):]
                at_response_start = False
                return (in_thinking, "", at_response_start, None,
                        accumulated or None)
            if end_marker.startswith(leading_text):
                return in_thinking, accumulated, at_response_start, None, None
        at_response_start = False
        delta = accumulated

    if in_thinking:
        marker_pos, start_marker = _find_earliest_marker(accumulated, start_markers)
        if marker_pos >= 0:
            accumulated = (accumulated[:marker_pos]
                           + accumulated[marker_pos + len(start_marker):].lstrip("\n"))
        marker_pos, end_marker = _find_earliest_marker(accumulated, end_markers)
        if marker_pos >= 0:
            reasoning = accumulated[:marker_pos]
            content = accumulated[marker_pos + len(end_marker):]
            content = _strip_thinking_control_markers(content)
            return (False, "", False, reasoning or None, content or None)
        pending_suffix = _partial_marker_suffix(accumulated, end_markers)
        if pending_suffix:
            reasoning = accumulated[:-len(pending_suffix)]
            return (in_thinking, pending_suffix, False, reasoning or None, None)
        return in_thinking, "", False, accumulated or None, None

    marker_pos, start_marker = _find_earliest_marker(accumulated, start_markers)
    if not in_thinking and marker_pos >= 0:
        in_thinking = True
        prefix = accumulated[:marker_pos]
        accumulated = accumulated[marker_pos + len(start_marker):].lstrip("\n")
        delta_content = prefix or None
    elif not in_thinking and (
        "<think" in accumulated
        or "<mm:think" in accumulated
        or _partial_marker_suffix(accumulated, start_markers)
    ):
        pass  # hold until the start marker is complete
    else:
        delta_content = _strip_thinking_control_markers(delta)
        accumulated = ""

    return in_thinking, accumulated, at_response_start, delta_reasoning, delta_content


def split_thinking_text(text, *, assume_in_thinking=False,
                        thinking_start_token=None, thinking_end_token=None):
    """Non-streaming split of a complete response into (reasoning, content)."""
    marker_pairs = list(_DEFAULT_MARKER_PAIRS)
    if thinking_start_token and thinking_end_token:
        marker_pairs.insert(0, (thinking_start_token, thinking_end_token))
    end_markers = tuple(dict.fromkeys(e for _, e in marker_pairs))
    start_markers = tuple(dict.fromkeys(s for s, _ in marker_pairs))

    in_thinking = assume_in_thinking
    # Strip any leading start marker if we begin inside a think block
    if in_thinking:
        for sm in start_markers:
            if text.lstrip().startswith(sm):
                text = text.lstrip()[len(sm):]
                break

    # Find the end marker that closes the thinking block
    pos, end_marker = _find_earliest_marker(text, end_markers)
    if in_thinking and pos >= 0:
        reasoning = text[:pos]
        content = text[pos + len(end_marker):]
        content = _strip_thinking_control_markers(content)
        return (reasoning.strip() or None, content.strip() or None)

    if in_thinking:
        return (text.strip() or None, None)

    # Adaptive MiniMax may choose to emit a full thinking block even though the
    # prompt did not open one. Strip that block from content and return it as
    # reasoning instead of leaking raw <mm:think> tags into OpenAI content.
    start_pos, start_marker = _find_earliest_marker(text, start_markers)
    if start_pos >= 0:
        prefix = text[:start_pos]
        rest = text[start_pos + len(start_marker):].lstrip("\n")
        end_pos, end_marker = _find_earliest_marker(rest, end_markers)
        if end_pos >= 0:
            reasoning = rest[:end_pos]
            suffix = rest[end_pos + len(end_marker):]
            content = (prefix + suffix).strip()
            content = _strip_thinking_control_markers(content)
            return (reasoning.strip() or None, content or None)
        return (rest.strip() or None, prefix.strip() or None)

    # Disabled thinking can produce a leading close marker. Treat it as control
    # text, not user-visible content.
    leading = text.lstrip()
    for end_marker in end_markers:
        if leading.startswith(end_marker):
            content = _strip_thinking_control_markers(leading[len(end_marker):])
            return (None, content.strip() or None)

    return (None, _strip_thinking_control_markers(text).strip() or None)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _request_generation_params(request, tools=None):
    params = {k: request[k] for k in GEN_PARAM_KEYS if k in request and request[k] is not None}
    if "thinking_budget" in params and not ALLOW_THINKING_BUDGET:
        logger.warning(
            "ignoring request thinking_budget=%r: MiniMax-M3 distributed budgeted "
            "thinking is disabled by MLX_M3_ALLOW_THINKING_BUDGET=0 after stall tests",
            params.get("thinking_budget"),
        )
        params.pop("thinking_budget", None)
    has_tools = bool(tools)
    defaults = {
        "temperature": TOOL_DEFAULT_TEMPERATURE if has_tools else DEFAULT_TEMPERATURE,
        "top_p": TOOL_DEFAULT_TOP_P if has_tools else DEFAULT_TOP_P,
        "top_k": TOOL_DEFAULT_TOP_K if has_tools else DEFAULT_TOP_K,
        "min_p": TOOL_DEFAULT_MIN_P if has_tools else DEFAULT_MIN_P,
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    if has_tools and TOOL_DEFAULT_REPETITION_PENALTY > 0:
        params.setdefault("repetition_penalty", TOOL_DEFAULT_REPETITION_PENALTY)
    if DEFAULT_REPETITION_PENALTY > 0:
        params.setdefault("repetition_penalty", DEFAULT_REPETITION_PENALTY)
    # Prose-thinking sampling must not overwrite tool sampling. Previously the
    # 0.5 thinking floor replaced the 0.2 tool default on thinking+tools turns,
    # which made the quantized model drift into long reasoning instead of
    # emitting a call. Explicit client values still win through setdefault.
    if not has_tools and _resolve_thinking_mode(request) == "enabled":
        if THINKING_DEFAULT_REPETITION_PENALTY > 0:
            params.setdefault("repetition_penalty",
                              THINKING_DEFAULT_REPETITION_PENALTY)
        if (THINKING_MIN_TEMPERATURE > 0
                and "temperature" not in request  # client didn't ask
                and float(params.get("temperature", 0.0) or 0.0)
                < THINKING_MIN_TEMPERATURE):
            params["temperature"] = THINKING_MIN_TEMPERATURE
    if DEFAULT_PRESENCE_PENALTY != 0:
        params.setdefault("presence_penalty", DEFAULT_PRESENCE_PENALTY)
    if DEFAULT_FREQUENCY_PENALTY != 0:
        params.setdefault("frequency_penalty", DEFAULT_FREQUENCY_PENALTY)
    default_seed = TOOL_DEFAULT_SEED if has_tools else DEFAULT_SEED
    if "seed" not in params and default_seed >= 0:
        params["seed"] = default_seed
    return params


def _generation_defaults_status():
    runtime_tuning = _runtime_tuning_status()
    return {
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "min_p": DEFAULT_MIN_P,
        "tool_temperature": TOOL_DEFAULT_TEMPERATURE,
        "tool_top_p": TOOL_DEFAULT_TOP_P,
        "tool_top_k": TOOL_DEFAULT_TOP_K,
        "tool_min_p": TOOL_DEFAULT_MIN_P,
        "tool_decode_topk_reuse_tokens": TOOL_DECODE_TOPK_REUSE_TOKENS,
        "tool_seed": TOOL_DEFAULT_SEED,
        "tool_thinking_mode": TOOL_THINKING_MODE,
        "tool_compat_overlay": TOOL_COMPAT_OVERLAY,
        "native_tool_action_retry_attempts": (
            NATIVE_TOOL_ACTION_RETRY_ATTEMPTS
        ),
        "native_tool_action_retry_ram_reset_tokens": (
            NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS
        ),
        "inject_date_context": INJECT_DATE_CONTEXT,
        "tool_no_call_token_budget": TOOL_NO_CALL_TOKEN_BUDGET,
        "tool_action_no_call_token_budget": (
            TOOL_ACTION_NO_CALL_TOKEN_BUDGET
        ),
        "tool_retry_no_call_token_budget": (
            TOOL_RETRY_NO_CALL_TOKEN_BUDGET
        ),
        "tool_retry_no_think": TOOL_RETRY_NO_THINK,
        "tool_retry_no_think_max_prompt_tokens": (
            TOOL_RETRY_NO_THINK_MAX_PROMPT_TOKENS
        ),
        "tool_stream_progress_seconds": TOOL_STREAM_PROGRESS_SECONDS,
        "tool_thinking_runaway_token_budget": (
            TOOL_THINKING_RUNAWAY_TOKEN_BUDGET
        ),
        "tool_unusable_retry_attempts": TOOL_UNUSABLE_RETRY_ATTEMPTS,
        "tool_unusable_retry_max_tokens": TOOL_UNUSABLE_RETRY_MAX_TOKENS,
        "tool_stream_buffer_all": TOOL_STREAM_BUFFER_ALL,
        "tool_write_chunk_max_chars": TOOL_WRITE_CHUNK_MAX_CHARS,
        "tool_write_chunk_target_chars": TOOL_WRITE_CHUNK_TARGET_CHARS,
        "tool_write_scaffold_threshold_chars": _tool_write_early_stop_chars(),
        "tool_incomplete_call_token_budget": (
            TOOL_INCOMPLETE_CALL_TOKEN_BUDGET
        ),
        "tool_detokenizer_silent_token_budget": (
            TOOL_DETOKENIZER_SILENT_TOKEN_BUDGET
        ),
        "tool_loop_force_final_repeated_tool_count": (
            TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT
        ),
        "tool_loop_force_final_identical_command_results": (
            TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS
        ),
        "tool_loop_force_final_repeated_tool_names": sorted(
            TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_NAMES
        ),
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "nonstream_default_max_tokens": NONSTREAM_DEFAULT_MAX_TOKENS,
        "openwebui_default_max_tokens": OPENWEBUI_DEFAULT_MAX_TOKENS,
        "max_tokens_ceiling": MAX_TOKENS_CEILING,
        "max_kv_size": MAX_KV_SIZE,
        "advertised_max_model_len": ADVERTISED_MAX_MODEL_LEN,
        "hard_max_input_tokens": HARD_MAX_INPUT_TOKENS,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "effective_prefill_step_size": runtime_tuning.get("prefill_step_size"),
        "adaptive_prefill_step_tokens": ADAPTIVE_PREFILL_STEP_TOKENS,
        "adaptive_prefill_step_size": ADAPTIVE_PREFILL_STEP_SIZE,
        "mlx_max_ops_per_buffer": MLX_MAX_OPS_PER_BUFFER,
        "mlx_max_mb_per_buffer": MLX_MAX_MB_PER_BUFFER,
        "decode_eval_every": DECODE_EVAL_EVERY,
        "decode_eval_after_tokens": DECODE_EVAL_AFTER_TOKENS,
        "decode_eval_after_every": DECODE_EVAL_AFTER_EVERY,
        "thinking_decode_eval_every": THINKING_DECODE_EVAL_EVERY,
        "thinking_raw_silent_limit": THINKING_RAW_SILENT_LIMIT,
        "long_context_decode_eval_tokens": LONG_CONTEXT_DECODE_EVAL_TOKENS,
        "long_context_decode_eval_every": LONG_CONTEXT_DECODE_EVAL_EVERY,
        "adaptive_long_context_decode_eval": ADAPTIVE_LONG_CONTEXT_DECODE_EVAL,
        "mid_context_decode_eval_tokens": MID_CONTEXT_DECODE_EVAL_TOKENS,
        "mid_context_decode_eval_every": MID_CONTEXT_DECODE_EVAL_EVERY,
        "high_context_decode_eval_tokens": HIGH_CONTEXT_DECODE_EVAL_TOKENS,
        "high_context_decode_eval_every": HIGH_CONTEXT_DECODE_EVAL_EVERY,
        "effective_thinking_decode_eval_every": runtime_tuning.get("thinking_decode_eval_every"),
        "effective_long_context_decode_eval_tokens": runtime_tuning.get("long_context_decode_eval_tokens"),
        "effective_long_context_decode_eval_every": runtime_tuning.get("long_context_decode_eval_every"),
        "runtime_tuning": runtime_tuning,
        "unsafe_runtime_tuning_allowed": ALLOW_UNSAFE_RUNTIME_TUNING,
        "thinking_budget_allowed": ALLOW_THINKING_BUDGET,
        "default_thinking_budget": DEFAULT_THINKING_BUDGET,
        "min_thinking_budget": MIN_THINKING_BUDGET,
        "prompt_cache_thinking_enabled": PROMPT_CACHE_THINKING_ENABLED,
        "prompt_cache_thinking_mode": PROMPT_CACHE_THINKING_MODE,
        "prompt_cache_direct_suffix_ids": PROMPT_CACHE_DIRECT_SUFFIX_IDS,
        "prompt_cache_min_suffix_tokens": PROMPT_CACHE_MIN_SUFFIX_TOKENS,
        "effective_prompt_cache_min_suffix_tokens": runtime_tuning.get(
            "prompt_cache_min_suffix_tokens"
        ),
        "prompt_cache_fast_min_suffix_tokens": PROMPT_CACHE_FAST_MIN_SUFFIX_TOKENS,
        "prompt_cache_fast_thinking_min_suffix_tokens": (
            PROMPT_CACHE_FAST_THINKING_MIN_SUFFIX_TOKENS
        ),
        "prompt_cache_reuse_bucket_tokens": PROMPT_CACHE_REUSE_BUCKET_TOKENS,
        "effective_prompt_cache_reuse_bucket_tokens": runtime_tuning.get(
            "prompt_cache_reuse_bucket_tokens"
        ),
        "clear_cache_after_request": CLEAR_CACHE_AFTER_REQUEST,
        "clear_cache_after_error": CLEAR_CACHE_AFTER_ERROR,
        "visible_transcript_prewarm_blocking": VISIBLE_TRANSCRIPT_PREWARM_BLOCKING,
        "refresh_generation_stream": REFRESH_GENERATION_STREAM,
        "kv_quant_enabled": KV_QUANT_ENABLED,
        "kv_bits": KV_BITS,
        "kv_group_size": KV_GROUP_SIZE,
        "kv_quant_scheme": KV_QUANT_SCHEME,
        "quantized_kv_start": QUANTIZED_KV_START,
        "direct_decode_kernel": USE_DIRECT_DECODE_KERNEL,
        "direct_decode_eval_mode": DIRECT_DECODE_EVAL_MODE,
        "compact_decode_sort_topk": bool(
            runtime_tuning.get("compact_decode_sort_topk")
        ),
        "decode_topk_reuse_tokens": runtime_tuning.get("decode_topk_reuse_tokens"),
        "sparse_topk_blocks_override": SPARSE_TOPK_BLOCKS_OVERRIDE,
        "effective_sparse_topk_blocks": runtime_tuning.get("sparse_topk_blocks"),
        "unsafe_inflight_stop": UNSAFE_INFLIGHT_STOP,
        "stop_on_client_disconnect": STOP_ON_CLIENT_DISCONNECT,
        "stop_check_every": STOP_CHECK_EVERY,
        "prefill_stop_check_every": PREFILL_STOP_CHECK_EVERY,
        "msa_k1_impl": os.environ.get("MLX_M3_MSA_K1_IMPL", "auto"),
        "msa_prefill_blockwise_topk_min_kv_len": int(
            os.environ.get("MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_MIN_KV_LEN", "32768")
            or "32768"
        ),
        "msa_prefill_blockwise_topk_block_chunk": int(
            os.environ.get("MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_BLOCK_CHUNK", "2") or "2"
        ),
        "msa_prefill_long_k_small_l_max_l": int(
            os.environ.get("MLX_M3_MSA_PREFILL_LONG_K_SMALL_L_MAX_L", "0") or "0"
        ),
        "msa_prefill_long_k_small_l_min_kv": int(
            os.environ.get("MLX_M3_MSA_PREFILL_LONG_K_SMALL_L_MIN_KV", "32768")
            or "32768"
        ),
        "visible_transcript_prewarm": VISIBLE_TRANSCRIPT_PREWARM_ENABLED,
        "visible_transcript_prewarm_min_generated": VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED,
        "effective_visible_transcript_prewarm_min_generated": (
            runtime_tuning.get("visible_transcript_prewarm_min_generated")
            if runtime_tuning.get("visible_transcript_prewarm_min_generated") is not None
            else VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED
        ),
        "visible_transcript_prewarm_max_tokens": VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS,
        "visible_transcript_prewarm_max_suffix_tokens": VISIBLE_TRANSCRIPT_PREWARM_MAX_SUFFIX_TOKENS,
        "visible_transcript_prewarm_max_generated_tokens": VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS,
        "reasoning_recall": REASONING_RECALL_ENABLED,
        "reasoning_recall_max_sessions": REASONING_RECALL_MAX_SESSIONS,
        "reasoning_recall_max_items": REASONING_RECALL_MAX_ITEMS,
        "reasoning_recall_max_chars": REASONING_RECALL_MAX_CHARS,
        "prompt_cache_session_protect": PROMPT_CACHE_SESSION_PROTECT_ENABLED,
        "prompt_cache_session_protect_min_tokens": PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS,
        "prompt_cache_session_protect_bypass_max_tokens": PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS,
        "prompt_cache_static_prefix_rebuild_max_tokens": PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_TOKENS,
        "prompt_cache_static_prefix_rebuild_max_reuse_ratio": PROMPT_CACHE_STATIC_PREFIX_REBUILD_MAX_REUSE_RATIO,
        "prompt_cache_session_manifest": PROMPT_CACHE_SESSION_MANIFEST_ENABLED,
        "prompt_cache_session_manifest_max": PROMPT_CACHE_SESSION_MANIFEST_MAX,
        "prompt_cache_resident_slots": PROMPT_CACHE_RESIDENT_SLOTS,
        "prompt_cache_resident_max_total_tokens": PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS,
        "prompt_cache_ssd": PROMPT_CACHE_SSD_ENABLED,
        "prompt_cache_ssd_restore": PROMPT_CACHE_SSD_RESTORE_ENABLED,
        "prompt_cache_ssd_thinking_boundary_restore": (
            PROMPT_CACHE_SSD_THINKING_BOUNDARY_RESTORE
        ),
        "prompt_cache_ssd_auto_save": PROMPT_CACHE_SSD_AUTO_SAVE,
        "prompt_cache_ssd_auto_save_min_delta_tokens": (
            PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS
        ),
        "prompt_cache_ssd_append_reserve_tokens": (
            PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS
        ),
        "prompt_cache_ssd_save_reserve_tokens": (
            PROMPT_CACHE_SSD_SAVE_RESERVE_TOKENS
        ),
        "batch_append_reserve_tokens": _env_int_rank_aware(
            "MLX_M3_BATCH_APPEND_RESERVE_TOKENS",
            PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS,
        ),
        "prompt_cache_ssd_dir": os.path.expanduser(PROMPT_CACHE_SSD_DIR),
        "prompt_cache_ssd_dir_rank0": os.path.expanduser(
            PROMPT_CACHE_SSD_DIR_RANK0 or PROMPT_CACHE_SSD_DIR
        ),
        "prompt_cache_ssd_dir_rank1": os.path.expanduser(
            PROMPT_CACHE_SSD_DIR_RANK1 or PROMPT_CACHE_SSD_DIR
        ),
        "prompt_cache_ssd_ttl_seconds": PROMPT_CACHE_SSD_TTL_SECONDS,
        "prompt_cache_ssd_max_bytes": PROMPT_CACHE_SSD_MAX_BYTES,
        "effective_prompt_cache_ssd_max_bytes": runtime_tuning.get(
            "prompt_cache_ssd_max_bytes", PROMPT_CACHE_SSD_MAX_BYTES
        ),
        "prompt_cache_ssd_min_tokens": PROMPT_CACHE_SSD_MIN_TOKENS,
    }


def _kernel_stats_status():
    try:
        from mlx_vlm.models.minimax_m3_vl.msa import get_kernel_stats

        return get_kernel_stats()
    except Exception as e:
        return {"available": False, "error": str(e)}


def _model_key(model_id):
    return str(model_id or "").strip().lower()


def _short_hash(value):
    if value is None:
        value = ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]


def _reasoning_recall_content_key(content):
    if not isinstance(content, str):
        return None
    visible = _sanitize_inbound_message_content("assistant", content)
    if not isinstance(visible, str) or not visible.strip():
        return None
    return _short_hash(visible.strip())


def _reasoning_recall_tool_calls_key(tool_calls):
    if not tool_calls:
        return None
    normalized = []
    for tool_call in _normalize_tool_calls(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments", {})
        if not name:
            continue
        normalized.append({
            "id": str(tool_call.get("id") or ""),
            "type": str(tool_call.get("type") or "function"),
            "name": name,
            "arguments": arguments if isinstance(arguments, dict) else {},
        })
    if not normalized:
        return None
    return "tool:" + _short_hash(normalized)


def _reasoning_recall_key(visible_content=None, tool_calls=None):
    tool_key = _reasoning_recall_tool_calls_key(tool_calls)
    if tool_key:
        return tool_key
    content_key = _reasoning_recall_content_key(visible_content)
    return ("content:" + content_key) if content_key else None


def _recall_assistant_reasoning(session_id, visible_content, *, tool_calls=None,
                                session_source=None):
    if (
        not REASONING_RECALL_ENABLED
        or not session_id
        or _is_auto_cache_session_source(session_source)
    ):
        return None
    key = _reasoning_recall_key(visible_content, tool_calls)
    if not key:
        return None
    session_key = _prompt_cache_session_key(session_id)
    with _reasoning_recall_lock:
        entries = _reasoning_recall_sessions.get(session_key)
        if not entries:
            return None
        entry = entries.get(key)
        if not entry:
            return None
        entries.move_to_end(key)
        _reasoning_recall_sessions.move_to_end(session_key)
        reasoning = entry.get("reasoning")
    return reasoning if isinstance(reasoning, str) and reasoning.strip() else None


def _remember_assistant_reasoning(session_id, visible_content, raw_output, *,
                                  thinking_mode=None, session_source=None,
                                  tool_calls=None):
    if (
        not REASONING_RECALL_ENABLED
        or not session_id
        or _is_auto_cache_session_source(session_source)
    ):
        return False
    if not _enable_thinking_for_generation(thinking_mode):
        return False
    key = _reasoning_recall_key(visible_content, tool_calls)
    if not key:
        return False
    try:
        reasoning, content = split_thinking_text(
            raw_output or "",
            assume_in_thinking=True,
        )
    except Exception:
        reasoning, content = None, None
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False
    reasoning = _strip_thinking_control_markers(reasoning).strip()
    if not reasoning:
        return False
    if REASONING_RECALL_MAX_CHARS > 0 and len(reasoning) > REASONING_RECALL_MAX_CHARS:
        logger.info(
            "reasoning-recall: skipped %s-char reasoning for session %s",
            len(reasoning),
            session_id,
        )
        return False
    session_key = _prompt_cache_session_key(session_id)
    with _reasoning_recall_lock:
        entries = _reasoning_recall_sessions.get(session_key)
        if entries is None:
            entries = OrderedDict()
            _reasoning_recall_sessions[session_key] = entries
        entries[key] = {
            "reasoning": reasoning,
            "key_kind": "tool_calls" if _reasoning_recall_tool_calls_key(tool_calls) else "content",
            "visible_chars": len(visible_content or ""),
            "tool_calls": len(tool_calls or []),
            "reasoning_chars": len(reasoning),
            "at": round(time.time(), 3),
        }
        entries.move_to_end(key)
        while REASONING_RECALL_MAX_ITEMS > 0 and len(entries) > REASONING_RECALL_MAX_ITEMS:
            entries.popitem(last=False)
        _reasoning_recall_sessions.move_to_end(session_key)
        while (
            REASONING_RECALL_MAX_SESSIONS > 0
            and len(_reasoning_recall_sessions) > REASONING_RECALL_MAX_SESSIONS
        ):
            _reasoning_recall_sessions.popitem(last=False)
    return True


def _message_shape(message):
    message = message if isinstance(message, dict) else {}
    role = str(message.get("role", "user"))
    content = message.get("content", "")
    text_parts = []
    image_parts = 0
    content_kind = type(content).__name__
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text"):
                text_parts.append(part.get("text", "") or part.get("content", ""))
            elif _extract_image_source(part):
                image_parts += 1
    elif isinstance(content, str):
        text_parts.append(content)
    elif content is not None:
        text_parts.append(str(content))
    text = "\n".join(t for t in text_parts if t)
    extra_keys = sorted(
        k for k in message.keys()
        if k not in {"role", "content", "reasoning", "reasoning_content", "thinking"}
    )
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
        or ""
    )
    return {
        "role": role,
        "content_kind": content_kind,
        "text_len": len(text),
        "text_hash": _short_hash(text),
        "image_parts": image_parts,
        "reasoning_len": len(reasoning) if isinstance(reasoning, str) else 0,
        "reasoning_hash": _short_hash(reasoning) if isinstance(reasoning, str) and reasoning else None,
        "extra_keys": extra_keys,
    }


def _request_shape_summary(request, processed_messages, prompt, token_ids, *,
                           thinking_mode, response_model, max_tokens, stream,
                           image_count, tools, max_tokens_source=None,
                           gen_params=None):
    raw_messages = request.get("messages") or []
    processed_shape = []
    for message in processed_messages:
        content = message.get("content", "")
        processed_shape.append({
            "role": message.get("role"),
            "content_len": len(content) if isinstance(content, str) else 0,
            "content_hash": _short_hash(content) if isinstance(content, str) else None,
            "has_tool_calls": bool(message.get("tool_calls")),
            "extra_keys": sorted(k for k in message.keys() if k not in {"role", "content"}),
        })
    tool_names = sorted(_tool_names_from_schema(tools or []))
    return {
        "requested_model": request.get("model"),
        "response_model": response_model,
        "top_level_keys": sorted(request.keys()),
        "date_context_injected": bool(request.get("_date_context_injected")),
        "cache_session_id": _request_cache_session(request)[0],
        "cache_session_source": _request_cache_session(request)[1],
        "thinking_mode": thinking_mode,
        "stream": bool(stream),
        "requested_max_tokens": request.get("max_tokens", request.get("max_completion_tokens")),
        "effective_max_tokens": int(max_tokens),
        "max_tokens_source": max_tokens_source,
        "message_count": len(raw_messages),
        "raw_messages": [_message_shape(m) for m in raw_messages],
        "processed_messages": processed_shape,
        "image_count": int(image_count),
        "tools_count": len(tools or []),
        "tool_names": tool_names[:160],
        "tool_names_truncated": len(tool_names) > 160,
        "tool_name_hashes": [_short_hash(name) for name in tool_names[:160]],
        "email_like_tools": [
            name for name in tool_names
            if any(part in name.lower() for part in ("email", "mail", "gmail", "smtp"))
        ][:32],
        "functions_count": len(request.get("functions") or []),
        "tool_source": request.get("_tool_source", "tools" if tools else "none"),
        "tool_loop_steering": request.get("_tool_loop_steering"),
        "tool_choice": request.get("tool_choice", request.get("function_call")),
        "tool_default_sampling": bool(tools),
        "effective_sampling": {
            k: gen_params.get(k)
            for k in ("temperature", "top_p", "top_k", "min_p", "seed",
                      "repetition_penalty")
            if isinstance(gen_params, dict) and k in gen_params
        },
        "prompt_chars": len(prompt or ""),
        "prompt_hash": _short_hash(prompt or ""),
        "full_prompt_tokens": len(token_ids) if token_ids is not None else None,
    }


def _request_looks_like_openwebui(request, processed_messages=None):
    """Detect OpenWebUI's dynamic environment prompt without needing headers."""
    # OpenWebUI's OpenAI-compatible pipeline often includes stream_options and
    # our gateway-side _tool_source marker even when the optional date/system
    # prompt is disabled. Treat that shape as OpenWebUI so no-max-token chat
    # requests use MLX_M3_OPENWEBUI_DEFAULT_MAX_TOKENS instead of the much larger
    # agent-friendly text default.
    if "stream_options" in request and "_tool_source" in request:
        return True
    texts = []
    for message in (processed_messages or request.get("messages") or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") not in {"system", "developer"}:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
    text = "\n".join(texts)
    markers = (
        "Use this date/time context",
        "Full current datetime",
        "Current timezone",
        "running locally",
    )
    return any(marker in text for marker in markers)


def _request_looks_like_title_metadata(request, messages=None, tools=None):
    """Identify short client-side chat-title jobs without matching user prose."""
    if bool(request.get("stream", False)):
        return False
    if request.get("max_tokens", request.get("max_completion_tokens")) is not None:
        return False
    if tools or request.get("tools") or request.get("functions"):
        return False

    messages = messages if messages is not None else request.get("messages") or []
    if not isinstance(messages, list) or not (1 <= len(messages) <= 4):
        return False

    # Hermes keeps the title instruction stable while embedding different chat
    # excerpts in the user message. Retain the semantic checks below for other
    # clients, and recognize this observed system template exactly so wording
    # drift in the embedded excerpt cannot reopen the 32k sidecar path.
    known_title_system_hashes = {"45a5d81d42c4"}
    if any(
        isinstance(message, dict)
        and message.get("role") in {"system", "developer"}
        and isinstance(message.get("content"), str)
        and _short_hash(message["content"]) in known_title_system_hashes
        for message in messages
    ):
        return True

    text_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
    text = " ".join(" ".join(text_parts).lower().split())
    if not text or len(text) > 8_192:
        return False

    title_markers = (
        "generate a title",
        "generate the title",
        "generate a concise title",
        "generate a short title",
        "create a title for this conversation",
        "create a concise title",
        "create a short title",
        "title for this conversation",
        "title for the conversation",
        "conversation title",
        "chat title",
    )
    output_markers = (
        "respond only with",
        "return only",
        "output only",
        "do not include",
        "no quotation marks",
        "under 50 characters",
        "under 80 characters",
    )
    has_system_context = any(
        isinstance(message, dict)
        and message.get("role") in {"system", "developer"}
        for message in messages
    )
    return any(marker in text for marker in title_markers) and (
        any(marker in text for marker in output_markers) or has_system_context
    )


_AUTHORITATIVE_DATE_CONTEXT_RE = re.compile(
    r"(?i)(?:use this date/time context|full current datetime|current timezone|"
    r"current date\s*[:=]|today(?:'s)? date\s*[:=]|"
    r"the current date is\s+\d{4}-\d{2}-\d{2})"
)


def _current_date_context_text():
    """Return a daily-stable local date context for model system prompts."""
    now = time.localtime()
    date_text = time.strftime("%Y-%m-%d (%A)", now)
    zone_name = time.strftime("%Z", now).strip()
    raw_offset = time.strftime("%z", now).strip()
    if re.fullmatch(r"[+-]\d{4}", raw_offset):
        raw_offset = f"{raw_offset[:3]}:{raw_offset[3:]}"
    zone_parts = [part for part in (zone_name, raw_offset) if part]
    zone_text = f" ({', '.join(zone_parts)})" if zone_parts else ""
    return (
        f"Current date: {date_text}. Local timezone{zone_text}. "
        "Treat this date as authoritative when interpreting relative dates "
        "such as today, tomorrow, or yesterday."
    )


_DATE_CONTEXT_SESSION_LOCK = threading.Lock()
_DATE_CONTEXT_BY_SESSION = {}
_DATE_CONTEXT_SESSION_MAX = 512


def _date_context_for_session(session_id, current_text=None):
    """Keep the injected date stable while a prompt-cache session is active.

    A date change near the beginning of the rendered prompt invalidates every
    later KV entry. Pinning only for the cache session avoids an 80k+ re-prefill
    at midnight; new or idle-expired sessions still receive today's date.
    """
    text = current_text or _current_date_context_text()
    key = str(session_id or "").strip()
    if not key:
        return text
    now = time.monotonic()
    idle_ttl = max(60, int(PROMPT_CACHE_TTL_SECONDS or 0))
    with _DATE_CONTEXT_SESSION_LOCK:
        stale = [
            item_key
            for item_key, item in _DATE_CONTEXT_BY_SESSION.items()
            if now - float(item.get("last_seen") or 0.0) > idle_ttl
        ]
        for item_key in stale:
            _DATE_CONTEXT_BY_SESSION.pop(item_key, None)
        item = _DATE_CONTEXT_BY_SESSION.get(key)
        if item:
            item["last_seen"] = now
            return item["text"]
        if len(_DATE_CONTEXT_BY_SESSION) >= _DATE_CONTEXT_SESSION_MAX:
            oldest = min(
                _DATE_CONTEXT_BY_SESSION,
                key=lambda item_key: float(
                    _DATE_CONTEXT_BY_SESSION[item_key].get("last_seen") or 0.0
                ),
            )
            _DATE_CONTEXT_BY_SESSION.pop(oldest, None)
        _DATE_CONTEXT_BY_SESSION[key] = {"text": text, "last_seen": now}
        return text


def _add_date_system_context(processed_messages, *, session_id=None):
    """Inject today's date unless the client already supplied date context."""
    messages = list(processed_messages or [])
    if not INJECT_DATE_CONTEXT:
        return messages, False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "developer"
        }:
            continue
        content = message.get("content")
        if isinstance(content, str) and _AUTHORITATIVE_DATE_CONTEXT_RE.search(content):
            return messages, False
    return [
        {
            "role": "system",
            "content": _date_context_for_session(session_id),
        },
        *messages,
    ], True


def _response_model_id(requested_model):
    key = _model_key(requested_model)
    if key in WEB_MODEL_ALIASES:
        return VISIBLE_WEB_MODEL_ID
    mode = MODEL_MODE_ALIASES.get(key)
    if mode == "disabled":
        return VISIBLE_NO_THINK_MODEL_ID
    return VISIBLE_THINK_MODEL_ID


def _kv_quant_kwargs():
    if not KV_QUANT_ENABLED:
        return {}
    return {
        "kv_bits": KV_BITS,
        "kv_group_size": KV_GROUP_SIZE,
        "kv_quant_scheme": KV_QUANT_SCHEME,
        "quantized_kv_start": QUANTIZED_KV_START,
    }


def _apply_default_thinking_budget(gen_params, thinking_mode, max_tokens):
    if (
        thinking_mode != "enabled"
        or not ALLOW_THINKING_BUDGET
        or DEFAULT_THINKING_BUDGET <= 0
        or "thinking_budget" in gen_params
    ):
        return
    answer_room = max(1, max_tokens - 8)
    dynamic_budget = max(MIN_THINKING_BUDGET, max_tokens // 2)
    gen_params["thinking_budget"] = min(
        DEFAULT_THINKING_BUDGET,
        dynamic_budget,
        answer_room,
    )


def _resolve_thinking_mode(request):
    if "thinking_mode" in request and request["thinking_mode"] is not None:
        mode = request["thinking_mode"]
    elif "enable_thinking" in request:
        mode = "enabled" if request.get("enable_thinking") else "disabled"
    elif request.get("reasoning_effort") is not None:
        effort = str(request.get("reasoning_effort")).strip().lower()
        if effort in ("none", "off", "disabled", "disable", "false", "0"):
            mode = "disabled"
        else:
            mode = "enabled"
    elif _model_key(request.get("model")) in MODEL_MODE_ALIASES:
        mode = MODEL_MODE_ALIASES[_model_key(request.get("model"))]
    else:
        mode = DEFAULT_THINKING_MODE

    if isinstance(mode, bool):
        return "enabled" if mode else "disabled"
    mode = str(mode).strip().lower()
    if mode not in VALID_THINKING_MODES:
        logger.warning("invalid thinking_mode=%r; using %s", mode, DEFAULT_THINKING_MODE)
        return DEFAULT_THINKING_MODE
    return mode


def _enable_thinking_for_generation(thinking_mode):
    return thinking_mode == "enabled"


def _normalize_tool_calls(tool_calls):
    normalized = []
    for tc in tool_calls or []:
        tc = dict(tc) if isinstance(tc, dict) else tc
        if isinstance(tc, dict) and "function" in tc:
            fn = dict(tc["function"])
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    fn["arguments"] = {}
            tc["function"] = fn
        normalized.append(tc)
    return normalized


def _tool_choice_required_name(request):
    """OpenAI tool_choice forcing: 'required' or a named function.

    Returns (required: bool, name: str|None). Accepts the OpenAI shapes
    {"type": "function", "function": {"name": N}} and legacy
    function_call={"name": N}; "auto"/"none"/absent -> (False, None).
    """
    choice = request.get("tool_choice", request.get("function_call"))
    if isinstance(choice, str):
        return (choice == "required", None)
    if isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else choice
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            return (True, str(name))
    return (False, None)


def _tool_choice_validation_error(request, tools):
    """Return an OpenAI-compatible validation error for ``tool_choice``."""
    has_tool_choice = "tool_choice" in request
    choice = request.get("tool_choice", request.get("function_call"))
    if choice is None:
        return None
    if isinstance(choice, str):
        normalized = choice.strip().lower()
        if normalized not in {"none", "auto", "required"}:
            return (
                "Invalid tool_choice. Expected 'none', 'auto', 'required', "
                "or a specific function."
            )
        if normalized == "required" and not tools:
            return "tool_choice 'required' requires at least one tool."
        return None
    if not isinstance(choice, dict):
        return "Invalid tool_choice."

    function = choice.get("function")
    if has_tool_choice:
        if choice.get("type") != "function" or not isinstance(function, dict):
            return (
                "A specific tool_choice must be "
                "{'type':'function','function':{'name':'...'}}."
            )
        name = function.get("name")
    else:
        name = function.get("name") if isinstance(function, dict) else choice.get("name")
    if not isinstance(name, str) or not name.strip():
        return "A specific function choice requires a non-empty function name."
    if not any(_tool_function_name(tool) == name for tool in (tools or [])):
        return f"tool_choice references unknown function {name!r}."
    return None


def _apply_tool_choice_instruction(messages, request):
    """Place explicit tool forcing where MiniMax's template can see it.

    MiniMax's chat template consumes an initial system/developer message and
    user messages, but ignores a trailing system role. Put the instruction in
    the same visible locations used by mlx-vlm's OpenAI server so named and
    required choices cannot silently degrade to prose.
    """
    required, name = _tool_choice_required_name(request)
    if not required:
        return list(messages or [])
    instruction = (
        f"You must call the function {name!r} to answer this request. "
        "Do not call any other function and do not answer directly."
        if name
        else (
            "You must call one or more of the available functions to answer "
            "this request. Do not answer directly without calling a function."
        )
    )
    patched = [dict(message) if isinstance(message, dict) else message
               for message in (messages or [])]

    first_has_instruction = False
    if patched and isinstance(patched[0], dict):
        first = patched[0]
        if first.get("role") in {"root", "system", "developer"}:
            content = first.get("content")
            if isinstance(content, str):
                first["content"] = f"{content}\n\n{instruction}".strip()
                first_has_instruction = True

    user_has_instruction = False
    for message in reversed(patched):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = f"{content}\n\n{instruction}".strip()
            user_has_instruction = True
        elif isinstance(content, list):
            message["content"] = list(content) + [
                {"type": "text", "text": instruction}
            ]
            user_has_instruction = True
        break
    if not first_has_instruction and not user_has_instruction:
        patched.insert(0, {"role": "system", "content": instruction})
    return patched


def _tools_from_request(request):
    """Return OpenAI tool schemas, accepting legacy `functions` clients too."""
    if _tool_choice_disables_tools(request):
        # OpenAI semantics: tool_choice "none" means the model must not call
        # tools this turn, so do not advertise them to the template/parser.
        request["_tool_source"] = "tool_choice_none"
        return None
    tools = request.get("tools")
    # OpenAI named forcing: advertise ONLY the requested function so the
    # model cannot pick another; the required flag then drives the ladder.
    _req, _name = _tool_choice_required_name(request)
    if _req:
        request["_tool_choice_required"] = True
    if _name and tools:
        named = [t for t in tools if _tool_function_name(t) == _name]
        if named:
            request["_tool_source"] = "tool_choice_named"
            return named
    if tools and TOOL_HIDE_NAMES:
        filtered = [
            tool for tool in tools
            if _tool_function_name(tool) not in TOOL_HIDE_NAMES
        ]
        if filtered and len(filtered) < len(tools):
            request["_tool_source"] = "tools_hidden_names"
            return filtered
    if tools:
        request["_tool_source"] = "tools"
        return tools
    functions = request.get("functions")
    if isinstance(functions, list) and functions:
        converted = []
        for fn in functions:
            if not isinstance(fn, dict):
                continue
            converted.append({"type": "function", "function": dict(fn)})
        if converted:
            request["_tool_source"] = "functions"
            return converted
    request["_tool_source"] = "none"
    return None


def _tool_choice_disables_tools(request):
    choice = request.get("tool_choice", request.get("function_call"))
    if choice is None:
        return False
    if isinstance(choice, str):
        return choice.strip().lower() in {"none", "no", "false", "0"}
    if isinstance(choice, dict):
        return str(choice.get("type", "")).strip().lower() == "none"
    return False


def _tool_call_arguments_dict(tool_call):
    if not isinstance(tool_call, dict):
        return {}
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return {"input": args}
    return args if isinstance(args, dict) else {}


def _tool_call_name_for_loop(tool_call):
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
    return str(function.get("name") or tool_call.get("name") or "").strip()


def _tool_call_command_fingerprint(tool_call):
    args = _tool_call_arguments_dict(tool_call)
    for key in ("cmd", "command", "input", "script", "code"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value.strip())[:500]
    edit_parts = []
    for key in (
        "patch",
        "diff",
        "file_path",
        "path",
        "old_string",
        "old_text",
        "new_string",
        "new_text",
        "content",
    ):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            edit_parts.append(f"{key}={value.strip()}")
    if edit_parts:
        return re.sub(r"\s+", " ", "\n".join(edit_parts))[:500]
    return ""


def _tool_loop_steering_diag(messages, tools):
    if not tools or TOOL_LOOP_STEER_MAX_TOOL_ONLY_TURNS <= 0:
        return None
    last_user_index = -1
    for index, message in enumerate(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = index
    if last_user_index < 0:
        return None

    assistant_tool_turns = 0
    tool_results = 0
    visible_assistant_turns = 0
    tool_fallback_turns = 0
    command_counts = {}
    command_result_counts = {}
    pending_commands_by_id = {}
    pending_commands_without_id = []
    tool_name_counts = {}
    user_content_counts = {}
    user_content_indices = {}
    for message_index, message in enumerate(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = re.sub(r"\s+", " ", content.strip())
        # Agent shims sometimes retry by appending the same long user/tool
        # instruction as a fresh user turn, so a "since latest user" loop
        # detector cannot see the earlier failed attempts. Count only chunky
        # repeated prompts so normal short chat like "continue" is unaffected.
        if len(normalized) >= 500:
            key = _short_hash(normalized)
            user_content_counts[key] = user_content_counts.get(key, 0) + 1
            user_content_indices.setdefault(key, []).append(message_index)
    for message in (messages or [])[last_user_index + 1:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if (
                isinstance(content, str)
                and content.strip()
                and _looks_like_tool_compat_fallback_content(content)
            ):
                tool_fallback_turns += 1
            elif isinstance(content, str) and content.strip():
                visible_assistant_turns += 1
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                assistant_tool_turns += 1
                for tool_call in tool_calls:
                    tool_name = _tool_call_name_for_loop(tool_call)
                    if tool_name:
                        tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
                    fingerprint = _tool_call_command_fingerprint(tool_call)
                    if fingerprint:
                        command_counts[fingerprint] = command_counts.get(fingerprint, 0) + 1
                        call_id = str(tool_call.get("id") or "").strip()
                        if call_id:
                            pending_commands_by_id[call_id] = fingerprint
                        else:
                            pending_commands_without_id.append(fingerprint)
        elif role == "tool":
            tool_results += 1
            call_id = str(message.get("tool_call_id") or "").strip()
            command = pending_commands_by_id.pop(call_id, "") if call_id else ""
            if not command and pending_commands_without_id:
                command = pending_commands_without_id.pop(0)
            if command:
                result = message.get("content", "")
                if not isinstance(result, str):
                    try:
                        result = json.dumps(
                            result,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                    except (TypeError, ValueError):
                        result = str(result)
                normalized_result = re.sub(r"\s+", " ", result.strip())
                pair = f"{command}\nresult={_short_hash(normalized_result)}"
                command_result_counts[pair] = (
                    command_result_counts.get(pair, 0) + 1
                )

    if visible_assistant_turns:
        return None
    repeated_command, repeated_command_count = "", 0
    if command_counts:
        repeated_command, repeated_command_count = max(
            command_counts.items(),
            key=lambda item: item[1],
        )
    repeated_command_result, repeated_command_result_count = "", 0
    if command_result_counts:
        repeated_command_result, repeated_command_result_count = max(
            command_result_counts.items(),
            key=lambda item: item[1],
        )
    repeated_tool, repeated_tool_count = "", 0
    if tool_name_counts:
        repeated_tool, repeated_tool_count = max(
            tool_name_counts.items(),
            key=lambda item: item[1],
        )
    reasons = []
    if (
        assistant_tool_turns >= TOOL_LOOP_STEER_MAX_TOOL_ONLY_TURNS
        and tool_results >= TOOL_LOOP_STEER_MAX_TOOL_ONLY_TURNS
    ):
        reasons.append("many_tool_only_turns")
    # Repeated shell/apply_patch calls are normal for coding agents. Treat
    # exact repeated commands as diagnostics only; hard loop protection below
    # is reserved for fallback-text loops and whole-prompt replay storms.
    if (
        TOOL_LOOP_STEER_MAX_REPEATED_COMMANDS > 0
        and repeated_command_count >= TOOL_LOOP_STEER_MAX_REPEATED_COMMANDS
    ):
        reasons.append("repeated_command")
    if (
        TOOL_LOOP_STEER_MAX_REPEATED_TOOL > 0
        and repeated_tool_count >= TOOL_LOOP_STEER_MAX_REPEATED_TOOL
    ):
        reasons.append("repeated_tool")
    if (
        TOOL_LOOP_FORCE_FINAL_AFTER > 0
        and assistant_tool_turns >= TOOL_LOOP_FORCE_FINAL_AFTER
        and tool_results >= TOOL_LOOP_FORCE_FINAL_AFTER
        and (
            "many_tool_only_turns" in reasons
            or "repeated_command" in reasons
            or "repeated_tool" in reasons
        )
    ):
        reasons.append("force_final")
    if (
        TOOL_LOOP_FORCE_FINAL_REPEATED_COMMANDS > 0
        and repeated_command_count >= TOOL_LOOP_FORCE_FINAL_REPEATED_COMMANDS
        and assistant_tool_turns >= repeated_command_count
        and tool_results >= repeated_command_count
        and "force_final" not in reasons
    ):
        reasons.append("force_final")
    if (
        TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS > 0
        and repeated_command_result_count
        >= TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS
        and "force_final" not in reasons
    ):
        reasons.append("identical_command_result")
        reasons.append("force_final")
    if (
        TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT > 0
        and repeated_tool_count >= TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT
        and repeated_tool in TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_NAMES
        and assistant_tool_turns >= repeated_tool_count
        and tool_results >= repeated_tool_count
        and "force_final" not in reasons
    ):
        reasons.append("repeated_tool_limit")
        reasons.append("force_final")
    if tool_fallback_turns >= 2:
        reasons.append("tool_fallback_loop")
        if "force_final" not in reasons:
            reasons.append("force_final")
    repeated_user_prompt_key = max(
        user_content_counts,
        key=user_content_counts.get,
        default=None,
    )
    repeated_user_prompt_count = (
        user_content_counts.get(repeated_user_prompt_key, 0)
        if repeated_user_prompt_key else 0
    )
    repeated_user_stalled_intervals = 0
    repeated_indices = user_content_indices.get(repeated_user_prompt_key, [])
    for left, right in zip(repeated_indices, repeated_indices[1:]):
        made_progress = False
        for message in (messages or [])[left + 1:right]:
            if not isinstance(message, dict):
                continue
            if message.get("role") in {"tool", "function"}:
                made_progress = True
                break
            if message.get("role") != "assistant":
                continue
            if message.get("tool_calls"):
                made_progress = True
                break
            content = message.get("content")
            if (
                isinstance(content, str)
                and content.strip()
                and not _looks_like_tool_compat_fallback_content(content)
            ):
                made_progress = True
                break
        if not made_progress:
            repeated_user_stalled_intervals += 1
    if (
        repeated_user_prompt_count >= 3
        and repeated_user_stalled_intervals >= 2
    ):
        reasons.append("repeated_user_tool_prompt")
        if "force_final" not in reasons:
            reasons.append("force_final")
    if not reasons:
        return None
    return {
        "triggered": True,
        "reasons": reasons,
        "assistant_tool_turns": assistant_tool_turns,
        "tool_results": tool_results,
        "tool_fallback_turns": tool_fallback_turns,
        "repeated_user_prompt_count": repeated_user_prompt_count,
        "repeated_user_stalled_intervals": repeated_user_stalled_intervals,
        "repeated_tool": repeated_tool,
        "repeated_tool_count": repeated_tool_count,
        "repeated_command_count": repeated_command_count,
        "repeated_command_hash": _short_hash(repeated_command) if repeated_command else None,
        "repeated_command_result_count": repeated_command_result_count,
        "repeated_command_result_hash": (
            _short_hash(repeated_command_result)
            if repeated_command_result else None
        ),
    }


def _tool_loop_steering_text(diag):
    if not diag or not diag.get("triggered"):
        return ""
    reasons = set(diag.get("reasons") or [])
    if "force_final" in reasons:
        return (
            "Tool loop breaker: stop calling tools for this single turn because "
            "the recent transcript already contains repeated tool calls and "
            "tool results without a visible assistant answer. Do not request "
            "another tool. Provide the final "
            "answer now using the gathered tool results, and if the task could "
            "not be completed, state exactly what is missing. Do not mention "
            "tool availability or server/tooling state; this is only a "
            "one-turn loop recovery instruction."
        )
    repeated_tool = diag.get("repeated_tool") or "the same tool"
    filtered = diag.get("filtered_tools") or []
    filtered_text = ""
    if filtered:
        filtered_text = (
            " The exact repeated action has already been attempted; do not "
            f"repeat {', '.join(filtered)} with the same arguments on this "
            "turn. Use a different available tool if more work is needed, or "
            "answer from the gathered results."
        )
    return (
        "Long agent-loop steering: the recent transcript already contains "
        "multiple tool-call turns and tool results after the latest user "
        "request, with repeated tool activity. Tools remain available. "
        f"Do not call {repeated_tool} again unless it is genuinely the next "
        "new action. Prefer a different concrete tool if more evidence is "
        "needed, or provide the final answer if the gathered tool output is "
        "already enough. Do not emit raw tool markup or describe a tool call "
        "in prose; use the provided tool-call format for any tool."
        f"{filtered_text}"
    )


def _tool_loop_forced_final_fallback(diag):
    count = int((diag or {}).get("repeated_command_result_count") or 0)
    if count:
        return (
            "The same action returned the same result repeatedly, so I "
            "stopped retrying it. This step is still incomplete and needs a "
            "different action before the task can continue."
        )
    return (
        "The recent actions stopped making progress, so I ended the repeated "
        "attempt. The task is still incomplete and needs a different next "
        "step."
    )


def _tool_function_name(tool):
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(function.get("name") or "").strip()


def _filter_looping_control_tools(tools, diag):
    if not tools or not diag or not diag.get("triggered"):
        return tools, []
    repeated_tool = str(diag.get("repeated_tool") or "").strip()
    reasons = set(diag.get("reasons") or [])
    filter_names = set()
    if repeated_tool in TOOL_LOOP_FILTER_CONTROL_TOOLS and "repeated_tool" in reasons:
        filter_names.add(repeated_tool)
    # Keep normal work tools stable by default. Changing the advertised tool
    # schema mid-agent-loop can bust prompt-cache reuse and makes some Codex /
    # ZCoder shims think a tool disappeared. Operators can opt in via
    # MLX_M3_TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS for special clients.
    if (
        repeated_tool in TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS
        and "repeated_command" in reasons
    ):
        filter_names.add(repeated_tool)
    if not filter_names:
        return tools, []
    filtered_tools = [
        tool for tool in tools
        if _tool_function_name(tool) not in filter_names
    ]
    if len(filtered_tools) == len(tools) or not filtered_tools:
        return tools, []
    return filtered_tools, sorted(filter_names)


def _require_alternate_work_tool(request, tools, filtered_names):
    """Require one different tool after filtering an exact repeated action.

    Leaving ``tool_choice=auto`` lets the model answer that no write tool is
    available even when Edit/Bash remain in the native client schema. Only an
    evidence-gated repeated *work* tool enters this path; ordinary turns and
    control-tool loop filtering keep their original OpenAI semantics.
    """
    if not isinstance(request, dict) or not tools or not filtered_names:
        return False
    filtered_work_tools = (
        set(filtered_names) & TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS
    )
    if not filtered_work_tools:
        return False
    request["tool_choice"] = "required"
    request["_tool_loop_required_alternate"] = True
    return True


def _tool_working_directory_from_messages(processed_messages):
    """Extract an explicit client-provided working directory, if present.

    Agent clients commonly put this in an <env> block. MiniMax can otherwise
    substitute a memorized example home directory in an otherwise valid tool
    call. Only anchored metadata is accepted; arbitrary prose paths are not.
    """
    labels = ("Working directory", "Current working directory")
    for message in processed_messages or []:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "developer"
        }:
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        env_sections = re.findall(
            r"<env>\s*(.*?)\s*</env>", content, flags=re.IGNORECASE | re.DOTALL
        )
        sections = env_sections or [content]
        for section in sections:
            for label in labels:
                match = re.search(
                    rf"(?mi)^\s*{re.escape(label)}\s*:\s*(/[^\r\n]+?)\s*$",
                    section,
                )
                if match:
                    candidate = os.path.normpath(match.group(1).strip())
                    if (
                        os.path.isabs(candidate)
                        and candidate != "/"
                        and len(candidate) <= 1024
                        and not any(ord(ch) < 32 for ch in candidate)
                    ):
                        return candidate
        cwd_match = re.search(
            r"<cwd>\s*(/[^<\r\n]+?)\s*</cwd>",
            content,
            flags=re.IGNORECASE,
        )
        if cwd_match:
            candidate = os.path.normpath(cwd_match.group(1).strip())
            if (
                os.path.isabs(candidate)
                and candidate != "/"
                and len(candidate) <= 1024
                and not any(ord(ch) < 32 for ch in candidate)
            ):
                return candidate
    # ZCode can place the task root in the user's first instruction instead of
    # its system <env> block.  Accept only an imperative, directory-labeled
    # form ("Work only in the existing /abs/path directory"); a bare path in
    # ordinary prose is not authority for later mutation rewrites.
    user_workspace_re = re.compile(
        r"(?is)\bwork\s+only\s+in\s+(?:the\s+existing\s+)?"
        r"(?:[`\"'])?(?P<path>/[^\r\n`\"']+?)(?:[`\"'])?\s+"
        r"(?:directory|folder)\b"
    )
    for message in reversed(processed_messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        match = user_workspace_re.search(content)
        if not match:
            continue
        candidate = os.path.normpath(match.group("path").strip())
        if (
            os.path.isabs(candidate)
            and candidate != "/"
            and len(candidate) <= 1024
            and not any(ord(ch) < 32 for ch in candidate)
        ):
            return candidate
    return ""


def _tool_write_early_stop_chars():
    """Payload size where an open Write is more useful as a scaffold."""
    if TOOL_WRITE_CHUNK_MAX_CHARS <= 0:
        return 0
    if TOOL_WRITE_CHUNK_TARGET_CHARS <= 0:
        return TOOL_WRITE_CHUNK_MAX_CHARS
    # Leave room for path/argument markup and a prompt closing tag while still
    # stopping a giant file body well before the parser's hard ceiling.
    return min(
        TOOL_WRITE_CHUNK_MAX_CHARS,
        TOOL_WRITE_CHUNK_TARGET_CHARS + 1024,
    )


def _model_facing_tool_schemas(tools):
    """Annotate bounded file payloads without mutating client tool schemas."""
    if not tools or TOOL_WRITE_CHUNK_TARGET_CHARS <= 0:
        return tools
    bounded = copy.deepcopy(tools)
    for tool in bounded:
        if not isinstance(tool, dict):
            continue
        function = (
            tool.get("function")
            if isinstance(tool.get("function"), dict)
            else tool
        )
        if not isinstance(function, dict):
            continue
        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(function.get("name") or "").lower(),
        )
        if normalized in _WRITE_FILE_TOOL_NAMES:
            payload_keys = ("content", "contents", "text", "data")
        elif normalized == "applypatch":
            # Codex exposes apply_patch as a free-form string in `input`.
            # Without a model-facing bound MiniMax can spend the whole output
            # budget drafting one unterminated patch before validation sees it.
            payload_keys = (
                "input",
                "patch",
                "patch_text",
                "patchText",
                "content",
                "diff",
            )
        elif normalized in _MUTATING_FILE_TOOL_NAMES:
            payload_keys = (
                "new_string",
                "newString",
                "new_text",
                "newText",
                "replacement",
            )
        else:
            continue
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties")
            if isinstance(parameters, dict)
            else None
        )
        if not isinstance(properties, dict):
            continue
        if normalized in _WRITE_FILE_TOOL_NAMES:
            # OpenCode's Write schema can advertise content before filePath.
            # MiniMax usually follows that order, so an interrupted large body
            # may never reveal the destination needed for safe scaffolding.
            # Reorder only the model-facing copy; the client's schema and
            # argument names remain untouched.
            path_keys = [
                key for key in _FILE_PATH_ARGUMENT_KEYS if key in properties
            ]
            if path_keys:
                ordered_keys = path_keys + [
                    key for key in properties if key not in path_keys
                ]
                parameters["properties"] = {
                    key: properties[key] for key in ordered_keys
                }
                properties = parameters["properties"]
        for key in payload_keys:
            spec = properties.get(key)
            if not isinstance(spec, dict):
                continue
            existing = spec.get("maxLength")
            if isinstance(existing, int) and existing > 0:
                spec["maxLength"] = min(
                    existing,
                    TOOL_WRITE_CHUNK_TARGET_CHARS,
                )
            else:
                spec["maxLength"] = TOOL_WRITE_CHUNK_TARGET_CHARS
            note = (
                f"Maximum {TOOL_WRITE_CHUNK_TARGET_CHARS} characters per "
                "call. For larger files, create a small scaffold and use "
                "focused follow-up Edit calls."
            )
            description = str(spec.get("description") or "").strip()
            if note not in description:
                spec["description"] = (
                    f"{description} {note}".strip()
                )
    return bounded


def _file_write_chunk_hint(tools):
    if TOOL_WRITE_CHUNK_MAX_CHARS <= 0:
        return ""
    has_write = any(
        re.sub(r"[^a-z0-9]", "", _tool_function_name(tool).lower())
        in _MUTATING_FILE_TOOL_NAMES
        for tool in (tools or [])
    )
    if not has_write:
        return ""
    return (
        "Large-file rule: target each file-write `content` payload at or "
        f"below {TOOL_WRITE_CHUNK_TARGET_CHARS} characters; the hard parser "
        f"ceiling is {TOOL_WRITE_CHUNK_MAX_CHARS}. For a larger file, first "
        "write a small valid working scaffold, then continue in later turns "
        "with focused Edit calls or bounded append operations. Never attempt "
        "the entire large file in one Write call. Do not bypass this limit "
        "with a large Bash heredoc, printf, tee, base64, or shell-embedded "
        "file body; use the provided Write and Edit tools in bounded stages. "
        "Never pass bare Python or other source code as a Bash command."
    )


def _add_tool_system_hint_if_needed(processed_messages, request, tools, tool_loop_diag=None):
    force_final = bool(
        tool_loop_diag
        and "force_final" in set(tool_loop_diag.get("reasons") or [])
    )
    if (not tools and not force_final) or not TOOL_SYSTEM_HINT_ENABLED or _tool_choice_disables_tools(request):
        return processed_messages
    # The static tool primer can become planning fuel inside <mm:think> and
    # encourage the model to draft work instead of calling a tool. Keep the
    # original thinking prompt shape, but retain dynamic steering once a real
    # loop is detected so long-running agents can recover.
    thinking_enabled = _resolve_thinking_mode(request) == "enabled"
    if tools and thinking_enabled:
        working_directory = _tool_working_directory_from_messages(
            processed_messages
        )
        hint_parts = [
            "Tool availability rule: use a provided tool when the answer "
            "depends on current, external, or user-specific information that "
            "is not already present in the conversation. If existing tool "
            "results already provide the needed evidence, answer from them; "
            "otherwise answer directly when no tool is needed."
        ]
        # Keep this primer byte-for-byte stable across conversational and
        # action-looking turns. Conditionally adding the execution clause near
        # the front of a long agent prompt invalidates every later KV entry
        # whenever that lightweight classifier changes its answer.
        hint_parts.append(
            "Tool execution rule for this task: if more work is needed, "
            "keep reasoning brief and emit exactly one focused function "
            "call using the provided tool format; if the task is complete, "
            "answer normally. Do not draft code, commands, arguments, or "
            "tool payloads inside <mm:think> or plain prose."
        )
        write_hint = _file_write_chunk_hint(tools)
        if write_hint:
            hint_parts.append(write_hint)
        if working_directory:
            hint_parts.append(
                "Tool path anchor: the client explicitly set the working "
                f"directory to {json.dumps(working_directory)}. Resolve "
                "relative file names inside that directory and never "
                "substitute a remembered example home, Downloads, or "
                "project path."
            )
        hint = "\n\n".join(hint_parts)
    else:
        hint_parts = [
            (TOOL_SYSTEM_HINT_TEXT or "").strip()
        ] if tools else []
        write_hint = _file_write_chunk_hint(tools)
        if write_hint:
            hint_parts.append(write_hint)
        working_directory = _tool_working_directory_from_messages(
            processed_messages
        ) if tools else ""
        if working_directory:
            hint_parts.append(
                "Tool path anchor: the client explicitly set the working "
                f"directory to {json.dumps(working_directory)}. Resolve "
                "relative file names inside that directory and never "
                "substitute a remembered example home, Downloads, or "
                "project path."
            )
        hint = "\n\n".join(part for part in hint_parts if part)
    steer = _tool_loop_steering_text(tool_loop_diag)
    if not hint and not steer:
        return processed_messages
    patched = list(processed_messages)
    for message in patched:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            continue
        content = message.get("content")
        if hint and isinstance(content, str) and hint in content:
            hint = ""
        if steer and isinstance(content, str) and steer in content:
            steer = ""
    if hint:
        patched.insert(0, {"role": "system", "content": hint})
    if steer:
        # A loop hint prepended before a long agent transcript is too remote:
        # MiniMax can keep copying the latest successful command even though
        # the instruction is technically present. Keep the static primer at
        # the front for cache stability, but put dynamic recovery immediately
        # before generation so the next action changes without removing tools.
        patched.append({"role": "system", "content": steer})
    return patched


def _load_tool_parser(processor):
    try:
        from mlx_vlm.tool_parsers import _infer_tool_parser, load_tool_module

        template = getattr(processor, "chat_template", None)
        if not template and getattr(processor, "tokenizer", None) is not None:
            template = getattr(processor.tokenizer, "chat_template", None)
        parser_type = _infer_tool_parser(template)
        return load_tool_module(parser_type) if parser_type else None
    except Exception as e:
        logger.warning(f"tool parser unavailable: {e}")
        return None


def _tool_call_markers(tool_module):
    start = getattr(tool_module, "tool_call_start", "") if tool_module else ""
    end = getattr(tool_module, "tool_call_end", "") if tool_module else ""
    return start, end


# Display-style tool call: "[Tool call: name]" then (optionally after the
# MiniMax namespace marker) a JSON args object. The JSON is what separates
# real emitted calls from prose that merely mentions the phrase.
_DISPLAY_TOOL_CALL_RE = re.compile(
    r"\[Tool call:\s*[A-Za-z_][\w:.-]*\s*\]\s*(?:\]<\]minimax\[>\[\s*)?\{"
)


def _strip_raw_tool_blocks(text, tool_module):
    """Remove raw MiniMax tool markup so invalid blocks never leak as content."""
    result = _strip_raw_tool_blocks_inner(text, tool_module)
    # 2026-07-10 ca0f2748: BARE namespace markers (no <tool_call> after them)
    # survived both branches below and 3.8k tokens of marker spam shipped as
    # visible content alongside a recovered call. Two or more bare markers is
    # never prose (single occurrences stay: prose legitimately quoting the
    # syntax once was the 2026-07-06 truncation regression) — cut at the
    # first marker.
    if result and text:
        start, _ = _tool_call_markers(tool_module)
        ns_token = start.removesuffix("<tool_call>") if start else ""
        if ns_token and result.count(ns_token) >= 2:
            logger.warning(
                "stripping %d bare tool markers from visible content",
                result.count(ns_token),
            )
            result = result[:result.find(ns_token)].strip()
    return result


def _strip_raw_tool_blocks_inner(text, tool_module):
    if not text:
        return text
    start, end = _tool_call_markers(tool_module)
    if not start or start not in text:
        ns_token = start.removesuffix("<tool_call>") if start else ""
        if ns_token and ns_token in text and _looks_like_raw_tool_fragment(text, tool_module):
            return text[:text.find(ns_token)].strip()
        # 2026-07-06 audit: only treat "[Tool call:" as markup when it is the
        # actual display format — bracketed name followed by a JSON object.
        # Bare prose mentioning the phrase ("the log shows [Tool call: x]
        # fired") was being truncated at the substring.
        display = _DISPLAY_TOOL_CALL_RE.search(text)
        if display:
            return text[:display.start()].strip()
        return text
    if end:
        pattern = re.compile(
            f"{re.escape(start)}.*?{re.escape(end)}",
            flags=re.DOTALL,
        )
        stripped = re.sub(pattern, " ", text).strip()
        # If the model started a malformed block and never closed it, drop from
        # the start marker onward instead of exposing tool syntax to clients.
        if start in stripped:
            stripped = stripped[:stripped.find(start)].strip()
        return stripped
    return re.sub(f"{re.escape(start)}.*?(?:\n|$)", " ", text, flags=re.DOTALL).strip()


def _looks_like_raw_tool_fragment(text, tool_module):
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if not text or not ns_token or ns_token not in text:
        return False
    fragment = text[text.find(ns_token):]
    if any(marker in fragment for marker in (
        "<tool_call",
        "</tool_call",
        "<invoke",
        "invoke name",
        "</invoke",
        "[Tool call:",
    )):
        return True
    # Bare-name flavor (2026-07-10 zcode): the model emits the namespace
    # marker + tool name + JSON args with NO <tool_call>/<invoke> structure:
    #   ]<]minimax[>[ Bash {"command": "...", "description": "..."}
    # Every rung above keys on the tag markers, so this shipped verbatim as
    # visible content. Marker followed by an identifier and an opening brace
    # is tool intent, never prose.
    rest = fragment[len(ns_token):]
    # identifier optional: 2026-07-10 second flavor is marker + self-describing
    # JSON directly: ]<]minimax[>[ {"name": "Write", "arguments": {...}}
    return bool(re.match(r"\s*(<?[A-Za-z_][\w.-]{0,40}>?\s*)?\{", rest))


def _has_empty_native_invoke(text):
    """Recognize MiniMax's name-less native invocation failure."""
    if not isinstance(text, str) or "invoke" not in text.lower():
        return False
    compact = text.replace("]<]minimax[>[", "")
    compact = re.sub(r"\s+", " ", compact)
    return bool(re.search(
        r"(?:<invoke\s*>|(?<![A-Za-z0-9_])invoke\s*:?)\s*</invoke>",
        compact,
        flags=re.IGNORECASE,
    ))


def _looks_like_degenerate_repetition(text, min_cycles=10):
    """Flavor-agnostic copy-spiral detector on the decode tail.

    True when the last stretch of output is a unit repeated many times
    back-to-back — the quantized-decode loop shape, whatever the unit is
    (marker spam, a repeated shell command, a word, a whole paragraph).
    Two bands:
      * short periods (3-120 chars): >=10 tight repeats — conservative so
        numbered lists / a few repeated code lines never trip; real text
        varies within the period, a spiral is byte-identical.
      * long periods (121-1200 chars): >=5 verbatim repeats. 2026-07-10
        zcode: a No-Think retry looped a ~300-char analysis paragraph
        ("Wait, I see at line 367... Hmm, I don't see this...") ~30 times
        and sailed under the old 120-char cap. Five byte-identical copies
        of a >120-char block back-to-back does not occur in real prose.
    Early-exit slicing keeps the common (non-looping) case ~free.
    """
    if not text:
        return False
    tail = text[-4800:]
    n = len(tail)
    for period in range(3, 1201):
        cycles = min_cycles if period <= 120 else 5
        span = period * cycles
        if span > n:
            if period > 120:
                break
            continue
        window = tail[-span:]
        unit = window[:period]
        if unit.strip() and all(
            window[i:i + period] == unit for i in range(0, span, period)
        ):
            return True
    return False


def _tool_fragment_looks_degenerate(text, tool_module):
    """Marker-spam detector for the silent-fragment decode guard.

    A legitimate MiniMax tool block uses a handful of namespace markers no
    matter how long its payload is; degenerate loops re-emit the marker almost
    every token. Only the spam shape may trip the guard — counting any
    in-progress block truncates long payloads (patches, multi-line commands)
    and manufactures the malformed calls the recovery paths exist to repair.
    """
    if not _looks_like_raw_tool_fragment(text, tool_module):
        return False
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    fragment = text[text.find(ns_token):]
    open_pos = fragment.rfind("<tool_call>")
    if open_pos >= 0 and "</tool_call" in fragment[open_pos:]:
        # Last block is closed; the final parse judges it, not the guard.
        return False
    # A Todo/plan array prefixes every nested field tag with the MiniMax
    # namespace. Eight ordinary items easily exceed 32 markers before the
    # closing tag, so marker count alone truncated valid ZCode TodoWrite calls
    # mid-item and manufactured the retry loop it was meant to prevent.
    if re.search(
        rf"{re.escape(ns_token)}<(?P<array>todos|plan|items|tasks)>.*?"
        rf"{re.escape(ns_token)}<item>",
        fragment,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return False
    # 32, not 16: a legitimate many-parameter loose call can pass 16 prefixed
    # segments while still open; true spam re-emits the marker per token and
    # blows past 32 within the same silent window (2026-07-06 audit).
    return fragment.count(ns_token) >= 32


def _tool_names_from_schema(tools):
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _tool_name_map_from_schema(tools):
    return {name.lower(): name for name in _tool_names_from_schema(tools)}


def _canonical_tool_name(name, name_map):
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered in name_map:
        return name_map[lowered]
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    if compact:
        for key, value in name_map.items():
            if re.sub(r"[^a-z0-9]", "", key) == compact:
                return value
    aliases = {
        "terminal": ("bash", "shell", "run_command", "execute_command", "exec_command", "invoke_command", "terminal"),
        "command": ("bash", "shell", "run_command", "execute_command", "exec_command", "invoke_command", "terminal"),
        "exec": ("bash", "shell", "run_command", "execute_command", "exec_command", "invoke_command", "terminal"),
        "todo": ("todowrite", "update_plan"),
        "todos": ("todowrite", "update_plan"),
        "cmd": ("bash", "shell", "run_command", "execute_command", "exec_command", "invoke_command", "terminal"),
        "invoke_command": ("bash", "shell", "run_command", "execute_command", "exec_command", "terminal"),
        "exec_command": ("bash", "shell", "run_command", "execute_command", "invoke_command", "terminal"),
        "read": ("read_file", "readfile", "open_file", "view_file", "read"),
        "read_file": ("read", "readfile", "open_file", "view_file"),
        "readfile": ("read", "read_file", "open_file", "view_file"),
        "write": ("write_file", "writefile", "edit_file", "write"),
        "write_file": ("write", "writefile", "edit_file"),
        "writefile": ("write", "write_file", "edit_file"),
        "exec_stdin": ("write_stdin", "stdin", "send_stdin", "send_input", "write_input"),
        "stdin": ("write_stdin", "exec_stdin", "send_stdin", "send_input", "write_input"),
        "send_stdin": ("write_stdin", "exec_stdin", "stdin", "send_input", "write_input"),
        "write_stdin": ("exec_stdin", "stdin", "send_stdin", "send_input", "write_input"),
    }
    for alias in aliases.get(lowered, ()):
        if alias in name_map:
            return name_map[alias]
        alias_compact = re.sub(r"[^a-z0-9]", "", alias)
        for key, value in name_map.items():
            if re.sub(r"[^a-z0-9]", "", key) == alias_compact:
                return value
    return stripped


def _tool_parameters_for_name(tools, name):
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        params = function.get("parameters")
        return params if isinstance(params, dict) else {}
    return {}


def _tool_schema_expects_arguments(tools, name):
    """Whether an empty object violates the advertised JSON schema.

    A schema may expose optional properties while accepting ``{}``. Treating
    any ``properties`` entry as mandatory rejected legitimate parameterless
    Hermes calls such as ``skills_list`` and forced the retry ladder. Required
    fields are the authoritative boundary; their individual names are checked
    again before a call is returned.
    """
    params = _tool_parameters_for_name(tools, name)
    required = params.get("required") if isinstance(params, dict) else None
    return bool(required) if isinstance(required, list) else False


def _tool_schema_property_names(tools, name):
    params = _tool_parameters_for_name(tools, name)
    props = params.get("properties") if isinstance(params, dict) else None
    return list(props.keys()) if isinstance(props, dict) else []


def _tool_schema_property_specs(tools, name):
    params = _tool_parameters_for_name(tools, name)
    props = params.get("properties") if isinstance(params, dict) else None
    return props if isinstance(props, dict) else {}


def _tool_schema_type_mismatches(arguments, tools, name):
    """Return arguments whose value or one array item violates JSON Schema."""
    if not isinstance(arguments, dict):
        return []
    specs = _tool_schema_property_specs(tools, name)
    type_map = {
        "array": lambda value: isinstance(value, list),
        "object": lambda value: isinstance(value, dict),
        "string": lambda value: isinstance(value, str),
        "boolean": lambda value: isinstance(value, bool),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        "null": lambda value: value is None,
    }
    mismatches = []
    for key, value in arguments.items():
        spec = specs.get(key)
        expected = spec.get("type") if isinstance(spec, dict) else None
        expected_types = expected if isinstance(expected, list) else [expected]
        checks = [type_map[item] for item in expected_types if item in type_map]
        if checks and not any(check(value) for check in checks):
            mismatches.append(key)
            continue
        if not isinstance(value, list) or not isinstance(spec, dict):
            continue
        item_spec = spec.get("items")
        if not isinstance(item_spec, dict) or item_spec.get("type") != "object":
            continue
        required = item_spec.get("required")
        required = required if isinstance(required, list) else []
        for item_index, item in enumerate(value):
            if not isinstance(item, dict):
                mismatches.append(f"{key}[{item_index}]")
                continue
            for child_key in required:
                if (
                    isinstance(child_key, str)
                    and (
                        child_key not in item
                        or item.get(child_key) in (None, "")
                    )
                ):
                    mismatches.append(
                        f"{key}[{item_index}].{child_key}"
                    )
    return mismatches


def _tool_schema_required_names(tools, name):
    params = _tool_parameters_for_name(tools, name)
    required = params.get("required") if isinstance(params, dict) else None
    return [x for x in required if isinstance(x, str)] if isinstance(required, list) else []


def _tool_schema_allows_additional_properties(tools, name):
    """True when the advertised schema explicitly permits unknown argument keys.

    Absent additionalProperties keeps the historical drop behavior: strict
    agent shims (zod validators) reject unknown keys even though JSON Schema
    defaults to permissive, so passthrough must be opt-in via the schema.
    """
    params = _tool_parameters_for_name(tools, name)
    extra = params.get("additionalProperties")
    return bool(extra) if not isinstance(extra, dict) else True


def _tool_names_with_property(tools, property_name):
    matches = []
    prop = str(property_name or "").lower()
    if not prop:
        return matches
    for name in _tool_names_from_schema(tools):
        props = {p.lower() for p in _tool_schema_property_names(tools, name)}
        if prop in props:
            matches.append(name)
    return matches


def _infer_tool_name_from_body(attrs, body, tools, name_map):
    """Infer a missing malformed MiniMax <invoke ...> name from schema + args."""
    haystack = f"{attrs or ''}\n{body or ''}"
    # OpenCode thinking output can lose only the name attribute and leave the
    # legacy alias as the first body token, e.g. ``<invoke>read_file>``. Name
    # that explicit schema alias before path-like text is mistaken for Bash.
    body_name = re.match(
        r"(?is)^\s*(?P<name>[A-Za-z_$][\w:.$-]*)\s*>",
        body or "",
    )
    if body_name:
        candidate = _canonical_tool_name(body_name.group("name"), name_map)
        if candidate in _tool_names_from_schema(tools):
            return candidate
    if re.search(r"</?(?:todos|todo)\b", haystack, flags=re.IGNORECASE):
        todo_tools = _tool_names_with_property(tools, "todos")
        if len(todo_tools) == 1:
            return todo_tools[0]
    if re.search(r"<question\b", haystack, flags=re.IGNORECASE):
        question_tools = _tool_names_with_property(tools, "questions")
        if len(question_tools) == 1:
            return question_tools[0]
    for match in re.finditer(r'\{\s*"name"\s*:\s*"(?P<name>(?:\\.|[^"])*)"', haystack):
        candidate = _canonical_tool_name(_loads_json_string_fragment(match.group("name")), name_map)
        if candidate:
            return candidate
    tagless = re.sub(r"<[^>]+>", " ", haystack)
    loose_shell = any(
        _looks_like_shell_command(segment)
        for segment in re.split(r"[\r\n]+", tagless)
    )
    if (
        _looks_like_shell_command(haystack)
        or loose_shell
        or re.search(r"<(?:cmd|command|input)\b", haystack, flags=re.IGNORECASE)
    ):
        preferred = [
            "bash", "shell", "terminal", "run_command", "execute_command",
            "exec_command", "invoke_command", "Bash", "Shell",
        ]
        for candidate in preferred:
            canonical = _canonical_tool_name(candidate, name_map)
            if canonical in _tool_names_from_schema(tools):
                return canonical
        command_tools = []
        for prop in ("command", "cmd", "input"):
            command_tools.extend(_tool_names_with_property(tools, prop))
        command_tools = list(dict.fromkeys(command_tools))
        if len(command_tools) == 1:
            return command_tools[0]
    for prop in ("file_path", "path"):
        if re.search(rf'"{re.escape(prop)}"\s*:', haystack):
            path_tools = _tool_names_with_property(tools, prop)
            if len(path_tools) == 1:
                return path_tools[0]
    return None


def _openai_tool_call(name, arguments, index):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if isinstance(arguments, dict) else {},
                               ensure_ascii=False)
    return {
        "type": "function",
        "index": index,
        "id": str(uuid.uuid4()),
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _canonicalize_tool_argument_keys(arguments, tools, name):
    """Shape arguments to the exact property names the client advertised."""
    if not isinstance(arguments, dict):
        return {}
    props = _tool_schema_property_names(tools, name)
    required = _tool_schema_required_names(tools, name)
    if not props:
        return dict(arguments)

    prop_by_lower = {prop.lower(): prop for prop in props}
    prop_by_compact = {
        re.sub(r"[^a-z0-9]", "", prop.lower()): prop for prop in props
    }
    aliases = {
        "cmd": (
            "cmd", "command", "input", "shell_command", "shellCommand",
            "command_string", "commandString", "script", "code",
        ),
        "command": (
            "command", "cmd", "input", "shell_command", "shellCommand",
            "command_string", "commandString", "script", "code",
        ),
        "input": (
            "input", "cmd", "command", "query", "text", "content",
        ),
        "query": ("query", "search", "search_query", "searchQuery", "input", "text"),
        "url": ("url", "uri", "link", "href", "input", "query", "text"),
        "prompt": (
            "prompt", "query", "question", "instruction", "instructions",
            "description", "task", "objective", "goal", "input", "text",
            "content",
        ),
        "question": ("question", "prompt", "query", "input", "text", "content"),
        "options": ("options", "choices", "items", "values"),
        "description": (
            "description", "prompt", "objective", "goal", "task",
            "instructions", "instruction", "input", "text", "content",
        ),
        "subagent_type": (
            "subagent_type", "subagentType", "agent_type", "agentType",
            "type", "name", "agent",
        ),
        "objective": (
            "objective", "goal", "task", "request", "prompt", "description",
            "instructions", "instruction", "input", "text", "content",
        ),
        "goal": (
            "goal", "objective", "task", "request", "prompt", "description",
            "instructions", "instruction", "input", "text", "content",
        ),
        "path": ("path", "file", "file_path", "filePath", "filename", "target"),
        "file_path": ("file_path", "filePath", "path", "file", "filename", "target"),
        "old_string": (
            "old_string", "oldString", "old_text", "oldText", "before",
            "from", "find", "search", "target",
        ),
        "new_string": (
            "new_string", "newString", "new_text", "newText", "after",
            "new_value", "newValue", "to", "replace", "replacement",
        ),
        "old_text": (
            "old_text", "oldText", "old_string", "oldString", "before",
            "from", "find", "search", "target",
        ),
        "new_text": (
            "new_text", "newText", "new_string", "newString", "after",
            "new_value", "newValue", "to", "replace", "replacement",
        ),
        "content": (
            "content", "text", "input", "data", "body", "value",
            "new_string", "newString", "new_text", "newText",
        ),
        "message": ("message", "content", "text", "input", "prompt"),
        "recipient": ("recipient", "to", "target", "agent", "name"),
        "chars": ("chars", "text", "input", "stdin", "content", "data"),
        "todos": ("todos", "todo", "items", "tasks", "list"),
        "plan": ("plan", "steps", "items", "tasks", "todos"),
        "status": ("status", "state", "result"),
        "skill": ("skill", "skill_name", "skillName", "name", "tool", "input"),
        "name": ("name", "tool", "skill", "subagent_type", "agent_type", "type"),
        "session_id": ("session_id", "sessionId", "session", "id", "process_id", "processId"),
        "justification": ("justification", "reason", "description", "why"),
    }

    result = {}
    leftovers = {}
    for raw_key, value in arguments.items():
        key = str(raw_key)
        lowered = key.lower()
        compact = re.sub(r"[^a-z0-9]", "", lowered)
        target = (
            prop_by_lower.get(lowered)
            or prop_by_compact.get(compact)
        )
        if target:
            result[target] = value
        else:
            leftovers[key] = value

    for prop in props:
        if prop in result and result[prop] not in (None, ""):
            continue
        lowered_prop = prop.lower()
        compact_prop = re.sub(r"[^a-z0-9]", "", lowered_prop)
        candidates = aliases.get(lowered_prop, ()) + aliases.get(compact_prop, ())
        for candidate in candidates:
            # Never alias-fill from a key that is itself an advertised
            # property: the model targeted that prop deliberately. This used
            # to copy new_string into content on edit tools that declare
            # both, inventing an argument (2026-07-06 audit).
            lowered_candidate = candidate.lower()
            compact_candidate = re.sub(r"[^a-z0-9]", "", lowered_candidate)
            if (
                lowered_candidate in prop_by_lower
                or compact_candidate in prop_by_compact
            ):
                continue
            if candidate in arguments and arguments[candidate] not in (None, ""):
                result[prop] = arguments[candidate]
                break
            for raw_key, value in arguments.items():
                raw_compact = re.sub(r"[^a-z0-9]", "", str(raw_key).lower())
                if raw_compact == compact_candidate and value not in (None, ""):
                    result[prop] = value
                    break
            if prop in result:
                break

    if (
        len(required) == 1
        and required[0] not in result
        and len(arguments) == 1
    ):
        only_key, only_value = next(iter(arguments.items()))
        only_lower = str(only_key).lower()
        only_compact = re.sub(r"[^a-z0-9]", "", only_lower)
        # Do not reinterpret a different advertised property as the required
        # field. Example: create_goal(token_budget=1000) is missing objective,
        # not objective="1000".
        if (
            only_lower not in prop_by_lower
            and only_compact not in prop_by_compact
            and only_value not in (None, "")
        ):
            result[required[0]] = only_value

    command_prop = next(
        (
            prop_by_lower[prop]
            for prop in ("cmd", "command", "input")
            if prop in prop_by_lower and prop_by_lower[prop] in result
        ),
        None,
    )
    justification_prop = prop_by_lower.get("justification")
    if (
        TOOL_SYNTH_JUSTIFICATION
        and command_prop
        and justification_prop
        and justification_prop not in result
        and result.get(command_prop)
    ):
        result[justification_prop] = _summarize_shell_command_for_tool(
            str(result[command_prop])
        )

    # Keep optional known fields only. Unknown keys are often what strict
    # agent shims reject even though the OpenAI API itself is permissive.
    # Schemas that explicitly allow additionalProperties keep their extra
    # keys — dropping them loses real data (2026-07-06 audit).
    # Upstream XML parsers sometimes materialize an omitted optional field as
    # null (observed with Edit.replace_all). Treat that exactly like absence;
    # retaining it converts an otherwise complete call into a schema mismatch.
    # Required nulls remain for the normal missing-required diagnostic.
    shaped = {
        prop: result[prop]
        for prop in props
        if prop in result and (prop in required or result[prop] is not None)
    }
    if leftovers and _tool_schema_allows_additional_properties(tools, name):
        for key, value in leftovers.items():
            shaped.setdefault(key, value)
    return shaped


def _coerce_codex_control_tool_arguments(arguments, tools, name):
    if not isinstance(arguments, dict):
        return {}
    result = dict(arguments)
    normalized_name = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    if normalized_name == "todowrite" and isinstance(result.get("todos"), list):
        spec = _tool_schema_property_specs(tools, name).get("todos") or {}
        item_spec = spec.get("items") if isinstance(spec, dict) else {}
        item_spec = item_spec if isinstance(item_spec, dict) else {}
        child_props = item_spec.get("properties")
        child_props = child_props if isinstance(child_props, dict) else {}
        child_required = item_spec.get("required")
        child_required = set(child_required) if isinstance(child_required, list) else set()
        normalized_todos = []
        for item in result["todos"]:
            if not isinstance(item, dict):
                normalized_todos.append(item)
                continue
            normalized = dict(item)
            if (
                "activeForm" in child_required
                and not normalized.get("activeForm")
                and normalized.get("content")
            ):
                normalized["activeForm"] = str(normalized["content"])
            if "status" in child_required and not normalized.get("status"):
                status_spec = child_props.get("status") or {}
                allowed = status_spec.get("enum") if isinstance(status_spec, dict) else None
                if not allowed or "pending" in allowed:
                    normalized["status"] = "pending"
            if "priority" in child_required and not normalized.get("priority"):
                priority_spec = child_props.get("priority") or {}
                allowed = priority_spec.get("enum") if isinstance(priority_spec, dict) else None
                if not allowed or "medium" in allowed:
                    normalized["priority"] = "medium"
            normalized_todos.append(normalized)
        result["todos"] = normalized_todos
        return result
    if name != "update_plan" or "plan" not in _tool_schema_property_names(tools, name):
        return result
    plan = result.get("plan")
    default_status = result.get("status")
    if default_status not in {"pending", "in_progress", "completed"}:
        default_status = "in_progress"

    def normalize_item(item):
        if isinstance(item, dict):
            normalized = dict(item)
            step = str(normalized.get("step") or normalized.get("text") or normalized.get("task") or "").strip()
            status = normalized.get("status")
            if status not in {"pending", "in_progress", "completed"}:
                status = default_status
            if not step:
                return None
            return {"step": step, "status": status}
        text = str(item or "").strip()
        if not text:
            return None
        return {"step": text, "status": default_status}

    if isinstance(plan, list):
        coerced = [normalize_item(item) for item in plan]
        result["plan"] = [item for item in coerced if item]
    elif isinstance(plan, dict):
        item = normalize_item(plan)
        result["plan"] = [item] if item else []
    elif plan not in (None, ""):
        item = normalize_item(plan)
        result["plan"] = [item] if item else []
    return result


def _coerce_json_encoded_schema_values(arguments, tools, name):
    """Coerce unambiguous XML strings to their advertised JSON types.

    MiniMax occasionally emits a valid JSON array as the string value of an
    array-typed argument (observed with OpenCode ``todowrite.todos``). Strict
    clients reject that call even though all information is present. XML tool
    output also represents booleans and numbers as text. Keep native values
    unchanged unless the schema asks for a non-string type and conversion is
    exact and lossless.
    """
    if not isinstance(arguments, dict):
        return {}
    specs = _tool_schema_property_specs(tools, name)
    if not specs:
        return dict(arguments)
    result = dict(arguments)
    container_types = {
        "array": list,
        "object": dict,
    }
    for key, value in list(result.items()):
        spec = specs.get(key)
        expected = spec.get("type") if isinstance(spec, dict) else None
        expected_set = set(expected) if isinstance(expected, list) else {expected}
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if "boolean" in expected_set and lowered in {"true", "false"}:
            result[key] = lowered == "true"
            continue
        if "integer" in expected_set and re.fullmatch(r"[-+]?\d+", stripped):
            try:
                result[key] = int(stripped)
            except ValueError:
                pass
            continue
        if "number" in expected_set and re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            stripped,
        ):
            try:
                result[key] = float(stripped)
            except ValueError:
                pass
            continue
        py_type = next(
            (container_types[item] for item in expected_set if item in container_types),
            None,
        )
        if py_type is None:
            continue
        marker = "]<]minimax[>["
        while stripped.startswith(marker):
            stripped = stripped[len(marker):].lstrip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError):
            try:
                decoded = json.loads(stripped, strict=False)
            except (TypeError, ValueError):
                continue
        if isinstance(decoded, py_type):
            result[key] = decoded
    return result


def _summarize_shell_command_for_tool(cmd):
    text = (cmd or "").strip()
    if not text:
        return "Run command"
    first = text.splitlines()[0].strip()
    lowered = first.lower()
    if re.search(r"\b(ls|find|tree)\b", lowered):
        return "List files"
    if re.search(r"\b(rg|grep|ag)\b", lowered):
        return "Search files"
    if re.search(r"\b(cat|sed|head|tail|nl)\b", lowered):
        return "Read file"
    if re.search(r"\b(git status|git diff|git log|git show)\b", lowered):
        return "Inspect git state"
    if re.search(r"\b(python|python3|node|npm|pytest|uv|curl)\b", lowered):
        return "Run project check"
    return "Run command"


def _apply_patch_payload(arguments):
    if not isinstance(arguments, dict):
        return ""
    for key in ("patch", "input", "content", "diff"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _apply_patch_payload_is_valid(arguments):
    payload = _apply_patch_payload(arguments)
    if not payload:
        return False
    if not payload.startswith("*** Begin Patch"):
        return False
    if "*** End Patch" not in payload:
        return False
    return any(
        marker in payload
        for marker in (
            "\n*** Add File:",
            "\n*** Update File:",
            "\n*** Delete File:",
        )
    )


def _is_apply_patch_tool_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower()) == "applypatch"


def _exec_write_from_malformed_patch(arguments, tools):
    """Recover a file write from a malformed apply_patch payload.

    MiniMax states the Add File intent reliably even when it breaks the patch
    envelope (missing Begin/End markers or dropped + prefixes), and resampling
    rarely fixes the format. Rebuild the write as an exec_command from the
    model's own patch text. Never touch Update/Delete ops; a partial diff
    cannot be reconstructed safely.
    """
    payload = _apply_patch_payload(arguments)
    if not payload:
        return None, None
    if re.search(r"\*{0,3}\s*(?:Update|Delete)\s+File\s*:", payload, re.IGNORECASE):
        return None, None
    command_name = _command_tool_name_from_schema(tools)
    if not command_name:
        return None, None
    add_re = re.compile(
        r"^\s*(?:\*{1,3}\s*)?Add\s+File\s*:\s*(?P<path>\S[^\n]*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(add_re.finditer(payload))
    if not matches:
        # MiniMax also emits an explicit file+content shape:
        # <invoke name="apply_patch"><file>path</file><patch>full source</patch>
        # No envelope markers at all means the payload IS the file body.
        path_arg = next(
            (
                str(arguments.get(key)).strip()
                for key in ("file", "path", "file_path", "filename", "target")
                if isinstance(arguments.get(key), str) and arguments.get(key).strip()
            ),
            None,
        )
        if (
            path_arg
            and "*** " not in payload
            and "<old_text>" not in payload
            and not path_arg.startswith("-")
            and "*" not in path_arg
        ):
            content = payload
            cmd = (
                f"mkdir -p {shlex.quote(os.path.dirname(path_arg) or '.')} && "
                f"printf %s {shlex.quote(content)} > {shlex.quote(path_arg)} && "
                f"ls -l {shlex.quote(path_arg)}"
            )
            args = _canonicalize_tool_argument_keys(
                {
                    "cmd": cmd,
                    "command": cmd,
                    "input": cmd,
                    "justification": f"Create {path_arg}",
                },
                tools,
                command_name,
            )
            args = _coerce_codex_control_tool_arguments(args, tools, command_name)
            if args:
                return command_name, args
        return None, None
    writes = []
    for index, match in enumerate(matches):
        path = match.group("path").strip().strip("'\"`")
        if not path or path.startswith("-") or "*" in path:
            return None, None
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(payload)
        content_lines = []
        for line in payload[body_start:body_end].splitlines():
            if re.match(r"^\s*\*{1,3}\s*(?:Begin|End)\s+Patch", line, re.IGNORECASE):
                break
            content_lines.append(line)
        while content_lines and not content_lines[0].strip():
            content_lines.pop(0)
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()
        if not content_lines:
            return None, None
        plus_lines = sum(1 for line in content_lines if line.startswith("+"))
        if plus_lines >= max(1, len(content_lines) // 2):
            content_lines = [
                line[1:] if line.startswith("+") else line
                for line in content_lines
            ]
        writes.append((path, "\n".join(content_lines)))
    commands = []
    for path, content in writes:
        commands.append(f"mkdir -p {shlex.quote(os.path.dirname(path) or '.')}")
        commands.append(f"printf %s {shlex.quote(content)} > {shlex.quote(path)}")
    commands.append("ls -l " + " ".join(shlex.quote(path) for path, _ in writes))
    cmd = " && ".join(commands)
    args = _canonicalize_tool_argument_keys(
        {
            "cmd": cmd,
            "command": cmd,
            "input": cmd,
            "justification": "Create " + ", ".join(path for path, _ in writes),
        },
        tools,
        command_name,
    )
    args = _coerce_codex_control_tool_arguments(args, tools, command_name)
    if not args:
        return None, None
    return command_name, args


def _coerce_file_tool_to_command(raw_name, arguments, tools):
    """Map common file-read hallucinations to the advertised shell tool.

    Codex-shaped clients often expose only exec_command for filesystem reads.
    MiniMax sometimes emits a convenience read_file/open_file/view_file tool
    anyway. Returning that unknown name breaks strict clients, so convert it to
    a small shell read when an exec-style tool is available.
    """
    if not isinstance(raw_name, str) or not isinstance(arguments, dict):
        return None, None
    compact_name = re.sub(r"[^a-z0-9]", "", raw_name.lower())
    if compact_name not in {
        "read",
        "readfile",
        "openfile",
        "viewfile",
        "catfile",
        "showfile",
        "view",
    }:
        return None, None
    allowed = _tool_names_from_schema(tools)
    name_map = _tool_name_map_from_schema(tools)
    command_name = None
    for candidate in (
        "exec_command",
        "invoke_command",
        "run_command",
        "execute_command",
        "bash",
        "shell",
        "terminal",
        "Bash",
        "Shell",
    ):
        canonical = _canonical_tool_name(candidate, name_map)
        if canonical in allowed:
            command_name = canonical
            break
    if not command_name:
        return None, None

    path = None
    for key in (
        "path",
        "file_path",
        "filePath",
        "filename",
        "file",
        "target",
        "input",
    ):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            path = value.strip()
            break
    if not path:
        return None, None

    start = arguments.get("start") or arguments.get("line") or arguments.get("from")
    end = arguments.get("end") or arguments.get("limit") or arguments.get("to")
    try:
        start_i = max(1, int(start)) if start not in (None, "") else 1
    except (TypeError, ValueError):
        start_i = 1
    try:
        end_i = max(start_i, int(end)) if end not in (None, "") else start_i + 219
    except (TypeError, ValueError):
        end_i = start_i + 219
    end_i = min(end_i, start_i + 499)

    quoted = shlex.quote(path)
    cmd = f"sed -n '{start_i},{end_i}p' {quoted}"
    coerced = _canonicalize_tool_argument_keys(
        {
            "cmd": cmd,
            "command": cmd,
            "input": cmd,
            "justification": f"Read {path}",
        },
        tools,
        command_name,
    )
    if not coerced:
        return None, None
    return command_name, coerced


def _command_tool_name_from_schema(tools):
    allowed = _tool_names_from_schema(tools)
    name_map = _tool_name_map_from_schema(tools)
    for candidate in (
        "exec_command",
        "invoke_command",
        "run_command",
        "execute_command",
        "bash",
        "shell",
        "terminal",
        "Bash",
        "Shell",
    ):
        canonical = _canonical_tool_name(candidate, name_map)
        if canonical in allowed:
            return canonical
    return None


def _command_tool_payload_violation(tool_name, arguments, tools):
    """Reject fragments that cannot be an executable shell command.

    This protects a native-parser edge where a malformed Edit retry was
    assigned to Bash with an entire Python function as ``command``.  Normal
    shell commands, Python ``-c`` calls, and explicit heredocs remain valid.
    """
    if not isinstance(arguments, dict):
        return ""
    command_name = _command_tool_name_from_schema(tools)
    if not command_name or tool_name != command_name:
        return ""
    command = next(
        (
            arguments.get(key)
            for key in ("command", "cmd", "input")
            if isinstance(arguments.get(key), str)
        ),
        "",
    )
    text = command.strip()
    if not text:
        return ""
    if re.fullmatch(r"[0-9]+", text):
        return "command is a bare numeric fragment"
    # MiniMax can nest a malformed Edit invocation inside Bash and leave the
    # parser with a command beginning ``Edit</name><parameter ...>``. That is
    # tool protocol, not shell input. Reject only protocol-shaped prefixes so
    # legitimate commands that grep logs for marker text remain available.
    if (
        text.startswith("]<]minimax[>[")
        or re.match(r"(?is)^\s*<(?:tool_call|invoke|parameter)\b", text)
        or re.match(
            r"(?is)^\s*[A-Za-z_][A-Za-z0-9_.:-]*\s*</name>\s*"
            r"<parameter\b",
            text,
        )
    ):
        return "command begins with embedded tool-call markup"
    # A complete XML tool block can still carry an incomplete shell heredoc.
    # Executing it lets the client runner's own status trailer become file
    # content (observed in ZCode as ``__zcode_status=$?`` inside a .py file).
    # Require every advertised delimiter to appear on a line by itself before
    # the command is allowed to leave the server.
    heredoc_re = re.compile(
        r"(?<!<)<<(?P<strip>-)?\s*"
        r"(?:(?P<quote>['\"])(?P<quoted>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?P=quote)|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
    )
    for heredoc in heredoc_re.finditer(text):
        delimiter = heredoc.group("quoted") or heredoc.group("bare")
        body = text[heredoc.end():]
        indent = r"\t*" if heredoc.group("strip") else ""
        if not re.search(
            rf"(?:^|\r?\n){indent}{re.escape(delimiter)}(?:\r?\n|$)",
            body,
        ):
            return "command contains an unterminated heredoc"
    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        "",
    )
    if re.match(
        r"^(?:async\s+def|def|class|from\s+\S+\s+import|import\s+\S+|"
        r"@[A-Za-z_]|#!\s*/.*python)\b",
        first_line,
    ):
        return "command is bare Python source without an interpreter or heredoc"
    if first_line.startswith(('"""', "'''")):
        return "command is bare source code beginning with a module docstring"
    # A quoted multiline program passed to an explicit interpreter is still
    # one executable shell command.  The source-line heuristic below used to
    # count the Python statements inside ``python3 -c "..."`` and reject a
    # perfectly valid ZCode Bash call after long tool runs.  Check only the
    # shell-bearing first line so prose that merely mentions ``python -c`` in
    # a later source line cannot bypass the malformed-command guard.
    if re.search(
        r"(?:^|(?:&&|\|\||;|\|)\s*)"
        r"(?:env\s+)?(?:[^\s;&|]*/)?python(?:3(?:\.\d+)*)?\s+"
        r"(?:-[A-Za-z]+\s+)*-c(?:\s|$)",
        first_line,
        flags=re.IGNORECASE,
    ):
        return ""
    if re.search(r"<<-?\s*(?:['\"])?[A-Za-z_][A-Za-z0-9_]*(?:['\"])?", first_line):
        return ""
    source_lines = sum(
        bool(re.match(
            r"\s*(?:async\s+def|def|class|from\s+\S+\s+import|import\s+\S+|"
            r"if\s+__name__\s*==|@[A-Za-z_])\b",
            line,
        ))
        for line in text.splitlines()
    )
    if source_lines >= 2:
        return "command contains bare Python source without an interpreter or heredoc"
    return ""


def _repair_non_executable_python_command(
    tool_name,
    arguments,
    tools,
    processed_messages,
):
    """Run a proven non-executable ``*.py`` command through Python.

    This is only applied after the client returned ``permission denied`` for
    that exact one-token command, so first attempts and executable scripts are
    untouched.
    """
    if not isinstance(arguments, dict):
        return arguments, None
    command_name = _command_tool_name_from_schema(tools)
    if not command_name or tool_name != command_name:
        return arguments, None
    command_key = next(
        (
            key for key in ("command", "cmd", "input")
            if isinstance(arguments.get(key), str)
        ),
        "",
    )
    command = arguments.get(command_key, "").strip() if command_key else ""
    try:
        parts = shlex.split(command)
    except ValueError:
        return arguments, None
    if len(parts) != 1 or not parts[0].lower().endswith(".py"):
        return arguments, None
    path = parts[0]

    calls_by_id = {}
    for message in processed_messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                calls_by_id[str(call["id"])] = call
    proven_failure = False
    for message in reversed(processed_messages or []):
        if not isinstance(message, dict) or message.get("role") not in {
            "tool", "function"
        }:
            continue
        call = calls_by_id.get(str(message.get("tool_call_id") or ""))
        if not call or _tool_call_name_for_loop(call) != command_name:
            continue
        prior_args = _tool_call_arguments_dict(call)
        prior_command = next(
            (
                prior_args.get(key)
                for key in ("command", "cmd", "input")
                if isinstance(prior_args.get(key), str)
            ),
            "",
        ).strip()
        if prior_command != command:
            continue
        proven_failure = "permission denied" in _tool_message_text(message).lower()
        break
    if not proven_failure:
        return arguments, None

    repaired = dict(arguments)
    repaired[command_key] = f"python3 {shlex.quote(path)}"
    return repaired, {
        "key": command_key,
        "source": command,
        "target": repaired[command_key],
    }


def _last_user_text(processed_messages):
    for message in reversed(processed_messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


def _last_user_instruction_text(processed_messages):
    """Return the latest real user instruction, skipping gateway tool results."""
    for message in reversed(processed_messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content or re.match(r"^Tool result(?:\s+\S+)?:", content):
            continue
        return content
    return ""


_TOOL_ACTION_VERBS = (
    "add|audit|browse|build|check|copy|create|delete|download|edit|email|execute|explore|fetch|"
    "find|fix|implement|inspect|install|list|make|modify|move|open|patch|"
    "navigate|read|remove|rename|review|rewrite|run|save|search|send|strip|test|update|upload|validate|"
    "verify|write|set"
)


def _tool_text_requests_action(user_text):
    """Return True for a clear action request, tolerant of CLI wrappers."""
    if not isinstance(user_text, str) or not user_text.strip():
        return False
    normalized = re.sub(r"\s+", " ", user_text.strip()).lower()
    if re.search(
        r"\b(?:use|call|invoke)\s+(?:the\s+)?(?:available\s+)?"
        r"(?:[a-z0-9_.:-]+\s+)?(?:function|tool)\b",
        normalized,
    ):
        return True
    if re.search(
        rf"^[^a-z0-9]{{0,32}}(?:please\s+)?(?:{_TOOL_ACTION_VERBS})\b",
        normalized,
    ):
        return True
    if re.search(
        rf"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?"
        rf"(?:{_TOOL_ACTION_VERBS})\b",
        normalized,
    ):
        return True
    return bool(re.search(
        rf"\b(?:i\s+(?:want|need)\s+you\s+to|go\s+ahead\s+and)\s+"
        rf"(?:{_TOOL_ACTION_VERBS})\b",
        normalized,
    ))


def _tool_request_requires_call(processed_messages, request=None):
    """Whether this turn needs an actual call rather than a prose answer.

    OpenAI ``tool_choice=required`` remains authoritative. For ``auto`` we
    infer only clear action requests on the first agent turn. Once the
    transcript contains a tool result after the latest user request, prose is
    valid again so an agent can finish instead of being forced into a loop.
    """
    if request and _tool_choice_required_name(request)[0]:
        return True

    messages = list(processed_messages or [])
    last_user_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = index
    if last_user_index < 0:
        return False

    for message in messages[last_user_index + 1:]:
        if not isinstance(message, dict):
            continue
        if message.get("role") in {"tool", "function"}:
            return False
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return False

    return _tool_text_requests_action(_last_user_text(messages))


def _tool_call_started(text, tool_module):
    """Cheaply recognize that generation entered an actual call payload."""
    if not isinstance(text, str) or not text:
        return False
    start_marker, _ = _tool_call_markers(tool_module)
    markers = tuple(filter(None, (
        start_marker,
        "<tool_call",
        "<invoke",
        "[Tool call:",
        '"tool_calls"',
    )))
    return any(marker in text for marker in markers)


def _tool_invocation_match(text, tool_name):
    """Match a valid or narrowly malformed invocation for one known tool.

    MiniMax occasionally moves the tool name outside the ``name=`` attribute,
    for example ``<invoke>edit\">{...}``.  Recognizing that opener lets the
    bounded-write guard stop a giant malformed payload early.  The match is
    deliberately tied to an advertised tool name and an invocation marker;
    it does not make the malformed call executable.
    """
    if not isinstance(text, str) or not text or not tool_name:
        return None
    escaped = re.escape(str(tool_name))
    patterns = (
        rf"(?is)<invoke\s+name\s*=\s*[\"']{escaped}[\"']",
        rf"(?is)<invoke\s+name\s+[\"']{escaped}[\"']",
        rf"(?is)<invoke\s*=\s*[\"']{escaped}[\"']",
        rf"(?is)<invoke\s+{escaped}\s*>",
        rf"(?is)<invoke\s*>\s*[\"']?{escaped}[\"']?\s*(?:[>:]|\{{)",
        rf"(?is)\[tool\s+call:\s*{escaped}\b",
        rf"(?is)<tool_name>{escaped}</tool_name>",
        # MiniMax's native parser also emits the advertised tool name as the
        # invocation element itself: <tool_call><write>...</write>. This form
        # must participate in the early payload guard just like <invoke>.
        rf"(?is)<{escaped}\s*>",
    )
    matches = [re.search(pattern, text) for pattern in patterns]
    matches = [match for match in matches if match]
    return min(matches, key=lambda match: match.start()) if matches else None


def _file_mutation_payload_info(text, tools):
    """Describe the active advertised file mutation, if one has started."""
    if not isinstance(text, str) or not text:
        return None
    for tool in tools or []:
        name = _tool_function_name(tool)
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        if normalized not in _MUTATING_FILE_TOOL_NAMES:
            continue
        match = _tool_invocation_match(text, name)
        if match:
            return {
                "name": name,
                "normalized_name": normalized,
                "payload_chars": len(text) - match.end(),
                "scaffoldable": normalized in _WRITE_FILE_TOOL_NAMES,
            }
    return None


def _file_write_payload_chars(text, tools):
    """Characters emitted after a file mutation invocation, or zero."""
    info = _file_mutation_payload_info(text, tools)
    return int((info or {}).get("payload_chars") or 0)


def _shell_create_file_payload_info(text, tools):
    """Return a bounded-write target embedded in a command tool, if present.

    This is intentionally narrow: only a command tool that starts a new file
    through cat/tee/printf/echo/base64 redirection is recognized. Ordinary
    Bash, test commands, and append operations remain untouched.
    """
    if not isinstance(text, str) or not text:
        return None
    command_name = _command_tool_name_from_schema(tools)
    if not command_name:
        return None
    escaped = re.escape(command_name)
    invocation = re.search(
        rf"(?is)(?:<invoke\s+name=[\"']{escaped}[\"']|"
        rf"\[tool\s+call:\s*{escaped}\b|<tool_name>{escaped}</tool_name>)",
        text,
    )
    if not invocation:
        return None
    payload = text[invocation.end():]
    head = payload[:1600]
    path_atom = (
        r'(?:"(?P<dq>[^"\r\n]+)"|\'(?P<sq>[^\'\r\n]+)\'|'
        r'(?P<bare>[^\s;|<>&\r\n]+))'
    )
    patterns = (
        rf"(?is)\bcat\b[^\r\n]{{0,320}}?(?<!>)>(?!>)\s*{path_atom}",
        rf"(?is)\btee\b\s+(?:-[A-Za-z]+\s+)*{path_atom}",
        rf"(?is)\b(?:printf|echo|base64)\b[^\r\n]{{0,640}}?"
        rf"(?<!>)>(?!>)\s*{path_atom}",
    )
    for pattern in patterns:
        match = re.search(pattern, head)
        if not match:
            continue
        path = next(
            (
                match.group(key).strip()
                for key in ("dq", "sq", "bare")
                if match.groupdict().get(key)
            ),
            "",
        )
        if (
            not path
            or path.startswith("-")
            or "$" in path
            or len(path) > 1024
            or not os.path.splitext(path)[1]
        ):
            continue
        return {
            "command_name": command_name,
            "path": path,
            "payload_chars": len(payload),
        }
    return None


def _file_mutation_stop_info(text, tools):
    """Return the applicable early-stop policy for an open file mutation.

    Atomic Write and shell-create calls can be replaced by a safe scaffold,
    so they use the lower scaffold threshold. Edit payloads cannot be
    truncated without changing replacement semantics; let those calls close
    naturally up to the existing hard mutation ceiling instead of clipping
    them at the Write-only threshold and entering an expensive retry ladder.
    """
    direct = _file_mutation_payload_info(text, tools)
    shell = _shell_create_file_payload_info(text, tools)
    direct_chars = int((direct or {}).get("payload_chars") or 0)
    shell_chars = int((shell or {}).get("payload_chars") or 0)
    if direct_chars <= 0 and shell_chars <= 0:
        return None
    if shell_chars >= direct_chars:
        return {
            "kind": "file-producing shell",
            "payload_chars": shell_chars,
            "threshold_chars": _tool_write_early_stop_chars(),
            "scaffoldable": True,
        }
    scaffoldable = bool((direct or {}).get("scaffoldable"))
    normalized = str((direct or {}).get("normalized_name") or "")
    apply_patch = normalized == "applypatch"
    return {
        "kind": (
            "file-write"
            if scaffoldable
            else "apply-patch"
            if apply_patch
            else "file-edit"
        ),
        "normalized_name": normalized,
        "payload_chars": direct_chars,
        "threshold_chars": (
            _tool_write_early_stop_chars()
            if scaffoldable
            or apply_patch
            else TOOL_WRITE_CHUNK_MAX_CHARS
        ),
        "scaffoldable": scaffoldable,
    }


def _tool_intent_without_call(text):
    """Detect visible promises/drafts that should have been a tool call."""
    if not isinstance(text, str) or not text:
        return False
    visible = text
    for marker in ("</mm:think>", "</think>"):
        if marker in visible:
            visible = visible.rsplit(marker, 1)[1]
    if (
        ("<mm:think>" in visible or "<think>" in visible)
        and "</mm:think>" not in text
        and "</think>" not in text
    ):
        return False
    normalized = re.sub(r"\s+", " ", visible.strip()).lower()
    if not normalized:
        return False
    if normalized.startswith("```"):
        return True
    return bool(re.search(
        rf"(?:^|[.!?]\s+)(?:let me|now\s+let me|now i(?:'ll| will| need to| should)?|"
        rf"i(?:'ll| will| need to| should)|next i(?:'ll| will)?)\s+"
        rf"(?:carefully\s+|quickly\s+)?(?:{_TOOL_ACTION_VERBS})\b",
        normalized,
    ))


_MUTATING_FILE_TOOL_NAMES = {
    "applypatch",
    "edit",
    "editfile",
    "makefile",
    "multiedit",
    "write",
    "writefile",
}
_WORKSPACE_FILE_TOOL_NAMES = {
    "openfile",
    "read",
    "readfile",
    "viewfile",
}
_WRITE_FILE_TOOL_NAMES = {
    "makefile",
    "write",
    "writefile",
}
_FILE_PATH_ARGUMENT_KEYS = (
    "filePath",
    "file_path",
    "path",
    "filename",
    "file",
    "target",
)


def _strip_minimax_closing_tags_from_paths(arguments):
    """Remove a parser-leaked MiniMax closing tag from path-only arguments."""
    if not isinstance(arguments, dict):
        return arguments, []
    cleaned = dict(arguments)
    changes = []
    suffix = re.compile(
        r"(?is)(?:\]<\]minimax\[>\[)?"
        r"</(?:parameter|file_?path|filepath|filename|path|file)>\s*$"
    )
    for key in _FILE_PATH_ARGUMENT_KEYS:
        value = cleaned.get(key)
        if not isinstance(value, str) or not value:
            continue
        repaired = value
        while True:
            next_value = suffix.sub("", repaired).rstrip()
            if next_value == repaired:
                break
            repaired = next_value
        if repaired != value and repaired:
            cleaned[key] = repaired
            changes.append((key, value, repaired))
    return cleaned, changes


def _strip_minimax_closing_tags_from_payloads(tool_name, arguments):
    """Remove only parser-leaked closing fragments at mutation payload ends."""
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if (
        normalized_name not in _MUTATING_FILE_TOOL_NAMES
        or not isinstance(arguments, dict)
    ):
        return arguments, []
    cleaned = dict(arguments)
    changes = []
    suffix = re.compile(
        r"(?is)(?:\s*\]<\]minimax\[>\[\s*"
        r"<?/?(?:parameter|content|contents|text|data|new_?string|"
        r"newtext|patch|command|cmd|input|invoke|tool_?call)>?)+\s*$"
    )
    for key in (
        "content", "contents", "text", "data", "newString", "new_string",
        "newText", "new_text", "replacement", "patch", "patch_text",
        "patchText",
    ):
        value = cleaned.get(key)
        if not isinstance(value, str) or not value:
            continue
        if not suffix.search(value):
            continue
        repaired = suffix.sub("", value).rstrip()
        if repaired != value:
            cleaned[key] = repaired
            changes.append((key, len(value), len(repaired)))
    return cleaned, changes


def _file_write_path(arguments):
    if not isinstance(arguments, dict):
        return ""
    for key in _FILE_PATH_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _repair_reversed_write_path_and_content(tool_name, arguments):
    """Swap a Write path/payload pair only when their shapes prove inversion.

    A malformed MiniMax XML closer can make the parser put a multiline source
    payload in ``file_path`` and the absolute destination in ``content``.
    Strictly require those opposite shapes; a normal path or ordinary prose is
    never rewritten.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in _WRITE_FILE_TOOL_NAMES:
        return arguments, None
    if not isinstance(arguments, dict):
        return arguments, None
    path_key = next(
        (
            key for key in _FILE_PATH_ARGUMENT_KEYS
            if isinstance(arguments.get(key), str) and arguments.get(key).strip()
        ),
        "",
    )
    content_key = next(
        (
            key for key in ("content", "contents", "text", "data", "body")
            if isinstance(arguments.get(key), str) and arguments.get(key).strip()
        ),
        "",
    )
    if not path_key or not content_key:
        return arguments, None
    path_value = arguments[path_key].strip()
    content_value = arguments[content_key].strip()
    content_is_path = bool(
        (content_value.startswith("/") or content_value.startswith("~/"))
        and "\n" not in content_value
        and len(content_value) < 2048
    )
    path_is_payload = bool(
        "\n" in path_value
        or (
            len(path_value) >= 256
            and re.search(
                r"(?:\b(?:def|class|import|from|return|const|function)\b|"
                r"[{};]|```|^#\s)",
                path_value,
                flags=re.MULTILINE,
            )
        )
    )
    if not (content_is_path and path_is_payload):
        return arguments, None
    repaired = dict(arguments)
    repaired[path_key] = content_value
    repaired[content_key] = path_value
    return repaired, {
        "path_key": path_key,
        "content_key": content_key,
        "path": content_value,
        "payload_chars": len(path_value),
    }


def _tool_message_text(message):
    """Return text from an OpenAI-style message without assuming one shape."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text") or part.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _exact_mutation_fingerprint(tool_name, arguments):
    """Hash one complete native file mutation without truncating its payload."""
    normalized_name = re.sub(
        r"[^a-z0-9]", "", str(tool_name or "").lower()
    )
    if (
        normalized_name not in _MUTATING_FILE_TOOL_NAMES
        or not isinstance(arguments, dict)
    ):
        return ""
    try:
        canonical = json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(
        f"{normalized_name}\0{canonical}".encode("utf-8")
    ).hexdigest()


def _tool_result_is_failure(message):
    """Recognize explicit client-side failures without guessing from prose."""
    if not isinstance(message, dict):
        return True
    if message.get("is_error") is True or message.get("error"):
        return True
    status = str(message.get("status") or "").strip().lower()
    if status in {"error", "failed", "failure", "cancelled", "canceled"}:
        return True
    text = _tool_message_text(message).strip()
    return bool(re.search(
        r"(?is)(?:^|\n)\s*(?:error|failed|failure|traceback|exception)\b|"
        r"\b(?:permission denied|no such file|was not applied|did not match|"
        r"could not find|tool call was not executed)\b",
        text,
    ))


def _successful_exact_mutation_fingerprints(processed_messages):
    """Return exact file mutations completed after the latest user request.

    This is deliberately narrower than the general loop detector. It requires
    an assistant call id, a matching non-error tool result, and byte-identical
    canonical arguments. Distinct edits, retries after failures, and a later
    user turn are untouched.
    """
    messages = processed_messages or []
    latest_user_index = max(
        (
            index for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        default=-1,
    )
    pending = {}
    successful = set()
    for message in messages[latest_user_index + 1:]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or "").strip()
                fingerprint = _exact_mutation_fingerprint(
                    _tool_call_name_for_loop(call),
                    _tool_call_arguments_dict(call),
                )
                if call_id and fingerprint:
                    pending[call_id] = fingerprint
            continue
        if message.get("role") not in {"tool", "function"}:
            continue
        call_id = str(message.get("tool_call_id") or "").strip()
        fingerprint = pending.pop(call_id, "") if call_id else ""
        if fingerprint and not _tool_result_is_failure(message):
            successful.add(fingerprint)
    return successful


def _normalized_read_snapshot(text):
    """Remove common client line-number wrappers from a Read tool result."""
    if not isinstance(text, str) or not text:
        return ""
    content_match = re.search(
        r"(?is)<content>\s*\n(?P<content>.*?)\n\s*</content>",
        text,
    )
    body = content_match.group("content") if content_match else text
    lines = []
    for line in body.splitlines():
        if re.match(r"^\s*\(End of file\b", line):
            continue
        line = re.sub(r"^\s*\d+(?::|\t)\s?", "", line)
        lines.append(line)
    return "\n".join(lines).strip()


def _repair_reversed_edit_arguments_after_failure(
    tool_name,
    arguments,
    processed_messages,
):
    """Swap an inverted Edit old/new pair only when the transcript proves it.

    OpenCode occasionally receives a semantically reversed MiniMax call: the
    desired replacement is placed in ``oldString`` and the exact current file
    block in ``newString``. Wait for one real client-side edit failure, then
    use an earlier Read result for the same path as evidence. This keeps valid
    native calls untouched and works even when the client filesystem is remote
    from the inference server.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in {"edit", "editfile"}:
        return arguments, None
    if not isinstance(arguments, dict):
        return arguments, None

    old_key = next(
        (key for key in ("oldString", "old_string", "oldText", "old_text")
         if isinstance(arguments.get(key), str) and arguments.get(key)),
        "",
    )
    new_key = next(
        (key for key in ("newString", "new_string", "newText", "new_text")
         if isinstance(arguments.get(key), str) and arguments.get(key)),
        "",
    )
    path = _file_write_path(arguments)
    if not old_key or not new_key or not path:
        return arguments, None
    old_value = arguments[old_key].strip()
    new_value = arguments[new_key].strip()
    if not old_value or not new_value or old_value == new_value:
        return arguments, None

    calls_by_id = {}
    for message in processed_messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "").strip()
            if call_id:
                calls_by_id[call_id] = call

    latest_edit_failed = False
    for message in reversed(processed_messages or []):
        if not isinstance(message, dict) or message.get("role") not in {
            "tool", "function"
        }:
            continue
        call_id = str(message.get("tool_call_id") or "").strip()
        call = calls_by_id.get(call_id)
        if not call:
            continue
        call_name = re.sub(
            r"[^a-z0-9]", "", _tool_call_name_for_loop(call).lower()
        )
        call_args = _tool_call_arguments_dict(call)
        if call_name not in {"edit", "editfile"}:
            continue
        if os.path.normpath(_file_write_path(call_args)) != os.path.normpath(path):
            continue
        result_text = _tool_message_text(message).lower()
        latest_edit_failed = (
            "could not find oldstring" in result_text
            or "could not find old_string" in result_text
            or (
                "must match exactly" in result_text
                and ("oldstring" in result_text or "old_string" in result_text)
            )
        )
        break
    if not latest_edit_failed:
        return arguments, None

    snapshots = []
    for message in processed_messages or []:
        if not isinstance(message, dict) or message.get("role") not in {
            "tool", "function"
        }:
            continue
        call_id = str(message.get("tool_call_id") or "").strip()
        call = calls_by_id.get(call_id)
        if not call:
            continue
        call_name = re.sub(
            r"[^a-z0-9]", "", _tool_call_name_for_loop(call).lower()
        )
        if call_name not in {"read", "readfile"}:
            continue
        call_args = _tool_call_arguments_dict(call)
        if os.path.normpath(_file_write_path(call_args)) != os.path.normpath(path):
            continue
        snapshot = _normalized_read_snapshot(_tool_message_text(message))
        if snapshot:
            snapshots.append(snapshot)
    if not snapshots:
        return arguments, None
    snapshot = snapshots[-1]
    if new_value not in snapshot or old_value in snapshot:
        return arguments, None

    repaired = dict(arguments)
    repaired[old_key] = arguments[new_key]
    repaired[new_key] = arguments[old_key]
    return repaired, {
        "path": path,
        "old_key": old_key,
        "new_key": new_key,
    }


def _file_write_content_key(arguments):
    if not isinstance(arguments, dict):
        return ""
    for key in ("content", "contents", "text", "data", "new_string", "newString"):
        if key in arguments:
            return key
    return ""


def _oversized_mutating_file_payload(tool_name, arguments):
    """Return the largest oversized edit/patch string, if one exists.

    Complete calls need the same bounded-write policy as unterminated calls.
    Write calls are handled by ``_bound_large_file_write_arguments``; edits
    cannot be truncated or scaffolded safely because doing so changes their
    replacement semantics, so they must be retried instead.
    """
    if TOOL_WRITE_CHUNK_MAX_CHARS <= 0 or not isinstance(arguments, dict):
        return None
    normalized = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized not in _MUTATING_FILE_TOOL_NAMES:
        return None
    candidates = (
        "newString", "new_string", "newText", "new_text", "replacement",
        "patch", "patch_text", "patchText",
    )
    if normalized == "applypatch":
        candidates += ("input", "content", "diff")
    oversized = [
        (key, len(arguments[key]))
        for key in candidates
        if isinstance(arguments.get(key), str)
        and len(arguments[key]) > TOOL_WRITE_CHUNK_MAX_CHARS
    ]
    return max(oversized, key=lambda item: item[1]) if oversized else None


def _small_file_scaffold(path):
    """Return a valid, intentionally tiny first stage for a large file."""
    extension = os.path.splitext(str(path or ""))[1].lower()
    if extension in {".html", ".htm"}:
        return (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "  <title>Work in progress</title>\n"
            "</head>\n"
            "<body>\n"
            "  <main id=\"app\"></main>\n"
            "  <!-- THUNDERMLX_CONTINUE: add the next bounded section here -->\n"
            "</body>\n"
            "</html>\n"
        )
    if extension == ".json":
        return "{}\n"
    if extension in {".md", ".mdx"}:
        return "<!-- THUNDERMLX_CONTINUE: add the next bounded section here -->\n"
    if extension in {".py", ".pyi", ".sh", ".rb", ".pl", ".r"}:
        return "# THUNDERMLX_CONTINUE: implement in bounded sections.\n"
    if extension in {
        ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".java",
        ".js", ".jsx", ".kt", ".m", ".mm", ".rs", ".scss", ".swift",
        ".ts", ".tsx",
    }:
        return "// THUNDERMLX_CONTINUE: implement in bounded sections.\n"
    return "THUNDERMLX_CONTINUE: add the next bounded section here.\n"


def _bound_large_file_write_arguments(tool_name, arguments):
    """Replace a giant atomic Write payload with a safe first-stage scaffold."""
    normalized = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized not in _WRITE_FILE_TOOL_NAMES or not isinstance(arguments, dict):
        return arguments, 0
    content_key = _file_write_content_key(arguments)
    content = arguments.get(content_key) if content_key else None
    scaffold_threshold = _tool_write_early_stop_chars()
    if (
        scaffold_threshold <= 0
        or not isinstance(content, str)
        or len(content) <= scaffold_threshold
    ):
        return arguments, 0
    bounded = dict(arguments)
    bounded[content_key] = _small_file_scaffold(_file_write_path(arguments))
    return bounded, len(content)


def _extract_incomplete_file_write_path(text, tool_name):
    if not isinstance(text, str) or not text:
        return ""
    for key in _FILE_PATH_ARGUMENT_KEYS:
        escaped = re.escape(key)
        match = re.search(
            rf"(?is)<(?:parameter\s+name=[\"']{escaped}[\"']|{escaped})>"
            rf"(?P<value>.*?)(?:</(?:parameter|{escaped})>|\n|$)",
            text,
        )
        if match:
            value = match.group("value").strip().strip("'\"`")
            value = re.sub(r"\]<\]minimax\[>\[\s*$", "", value).strip()
            if value:
                return value
        match = re.search(
            rf"(?is)[\"']{escaped}[\"']\s*:\s*[\"'](?P<value>.*?)[\"']",
            text,
        )
        if match:
            value = match.group("value").strip()
            if value:
                return value
    return ""


def _synthesize_bounded_write_scaffold_text(full_output, tools):
    """Recover an oversized unterminated Write as one parseable small call."""
    direct_payload_chars = _file_write_payload_chars(full_output, tools)
    shell_write = _shell_create_file_payload_info(full_output, tools)
    shell_payload_chars = int((shell_write or {}).get("payload_chars") or 0)
    scaffold_threshold = _tool_write_early_stop_chars()
    if (
        scaffold_threshold <= 0
        or max(direct_payload_chars, shell_payload_chars)
        <= scaffold_threshold
    ):
        return ""
    name_map = _tool_name_map_from_schema(tools)
    selected_name = ""
    path = ""
    if direct_payload_chars > scaffold_threshold:
        for tool in tools or []:
            name = _tool_function_name(tool)
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            if normalized not in _WRITE_FILE_TOOL_NAMES:
                continue
            if _tool_invocation_match(full_output or "", name):
                selected_name = _canonical_tool_name(name, name_map) or name
                path = _extract_incomplete_file_write_path(
                    full_output,
                    selected_name,
                )
                break
    if not selected_name and shell_payload_chars > scaffold_threshold:
        for tool in tools or []:
            name = _tool_function_name(tool)
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            if normalized in _WRITE_FILE_TOOL_NAMES:
                selected_name = _canonical_tool_name(name, name_map) or name
                path = str((shell_write or {}).get("path") or "")
                break
    if not selected_name:
        return ""
    if not path:
        return ""
    arguments = _canonicalize_tool_argument_keys(
        {
            "file_path": path,
            "filePath": path,
            "path": path,
            "filename": path,
            "content": _small_file_scaffold(path),
        },
        tools,
        selected_name,
    )
    required = _tool_schema_required_names(tools, selected_name)
    if any(arguments.get(key) in (None, "") for key in required):
        return ""
    return f"[Tool call: {selected_name}]\n{json.dumps(arguments, ensure_ascii=False)}"


def _complete_html_document_prefix(payload):
    """Return one structurally complete HTML document from a noisy tail."""
    if not isinstance(payload, str) or not payload:
        return ""
    # MiniMax can place its namespace separator immediately before ordinary
    # HTML closing tags. It is transport markup, not part of the requested
    # file. Removing it also exposes the first real document boundary before
    # a repeated closing-tag spiral.
    cleaned = payload.replace("]<]minimax[>[", "").lstrip()
    if not re.match(r"(?is)(?:<!doctype\s+html\b|<html\b)", cleaned):
        return ""
    lower = cleaned.lower()
    body_open = lower.find("<body")
    if body_open < 0:
        return ""
    for match in re.finditer(r"(?is)</html\s*>", cleaned):
        body_close = lower.rfind("</body>", body_open, match.start())
        if body_close < body_open:
            continue
        candidate = cleaned[:match.end()].rstrip()
        if "<script" in candidate.lower():
            script_close = candidate.lower().rfind("</script>")
            if script_close < body_open or script_close > body_close:
                continue
        return candidate
    return ""


def _safe_recovered_file_path(value):
    path = str(value or "").strip().strip("'\"`")
    if (
        not path
        or len(path) > 1024
        or path.startswith("-")
        or "*" in path
        or "]<]minimax[>[" in path
        or any(ord(char) < 32 for char in path)
        or any(char in path for char in "<>")
    ):
        return ""
    return path


def _synthesize_complete_add_file_artifact_text(full_output, tools):
    """Close a malformed Add File envelope around a complete HTML artifact.

    This is intentionally narrower than general partial-patch recovery. It
    accepts only an advertised apply_patch invocation, one safe HTML target,
    and a document that already contains its own closing body/html boundary.
    Update/Delete patches and incomplete source remain non-executable.
    """
    if not isinstance(full_output, str) or not full_output or not tools:
        return ""
    apply_names = [
        name
        for name in _tool_names_from_schema(tools)
        if _is_apply_patch_tool_name(name)
    ]
    if len(apply_names) != 1:
        return ""
    apply_name = apply_names[0]
    invocation = _tool_invocation_match(full_output, apply_name)
    if not invocation:
        return ""
    region = full_output[invocation.start():]
    if re.search(
        r"\*{3}\s*(?:Update|Delete)\s+File\s*:",
        region,
        flags=re.IGNORECASE,
    ):
        return ""

    path = ""
    content = ""
    # Native malformed flavor observed in Codex-shaped traffic:
    # <invoke name="apply_patch"><file_path>...</file_path><patch>...</patch>
    path_match = re.search(
        r"(?is)<(?P<tag>file_?path|filepath|target|file)>"
        r"(?P<value>.*?)</(?P=tag)>",
        region,
    )
    payload_match = re.search(
        r"(?is)<(?:patch|input|content|diff)>",
        region,
    )
    if path_match and payload_match:
        path = _safe_recovered_file_path(
            path_match.group("value").replace("]<]minimax[>[", "")
        )
        content = _complete_html_document_prefix(
            region[payload_match.end():]
        )

    # Standard free-form apply_patch flavor with a missing End Patch marker.
    if not path or not content:
        plain = region.replace("]<]minimax[>[", "")
        if "*** Begin Patch" not in plain:
            return ""
        add_match = re.search(
            r"(?im)^\s*\*{3}\s*Add\s+File\s*:\s*(?P<path>[^\r\n]+)$",
            plain,
        )
        if not add_match:
            return ""
        path = _safe_recovered_file_path(add_match.group("path"))
        body_lines = plain[add_match.end():].lstrip("\r\n").splitlines()
        added_lines = []
        for line in body_lines:
            if re.match(r"^\s*\*{3}\s*End\s+Patch", line, re.IGNORECASE):
                break
            if not line.startswith("+"):
                return ""
            added_lines.append(line[1:])
        content = _complete_html_document_prefix("\n".join(added_lines))

    if not path or not content:
        return ""
    if os.path.splitext(path)[1].lower() not in {".html", ".htm"}:
        return ""
    patch_lines = [
        "*** Begin Patch",
        f"*** Add File: {path}",
        *(f"+{line}" for line in content.splitlines()),
        "*** End Patch",
    ]
    patch = "\n".join(patch_lines)
    if TOOL_WRITE_CHUNK_MAX_CHARS > 0 and len(patch) > TOOL_WRITE_CHUNK_MAX_CHARS:
        return ""
    required = _tool_schema_required_names(tools, apply_name)
    properties = _tool_schema_property_names(tools, apply_name)
    payload_key = next(
        (
            key
            for key in required
            if re.sub(r"[^a-z0-9]", "", key.lower())
            in {"input", "patch", "patchtext", "content", "diff"}
        ),
        None,
    ) or next(
        (
            key
            for key in properties
            if re.sub(r"[^a-z0-9]", "", key.lower())
            in {"input", "patch", "patchtext", "content", "diff"}
        ),
        None,
    )
    if not payload_key:
        return ""
    arguments = _canonicalize_tool_argument_keys(
        {payload_key: patch},
        tools,
        apply_name,
    )
    if any(arguments.get(key) in (None, "") for key in required):
        return ""
    candidate = _openai_tool_call(apply_name, arguments, 0)
    validated = _validate_outgoing_tool_calls([candidate], tools)
    if len(validated) != 1:
        return ""
    return _tool_call_as_display_text(validated[0])


_RELATIVE_MUTATION_TARGET_RE = re.compile(
    r"\b(?:create|write|save|edit|update|modify|patch|rename|move|copy)\s+"
    r"(?:a\s+|an\s+|the\s+)?(?:new\s+)?(?:text\s+)?(?:file\s+)?"
    r"(?:named\s+|called\s+|at\s+|to\s+)?"
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)*"
    r"[A-Za-z0-9_.-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)",
    re.IGNORECASE,
)


def _named_proven_file_target(raw_output, processed_messages, working_directory):
    """Return one same-turn, tool-proven file named in pre-tool intent.

    A successful Read is the strongest evidence. A successful Write/Edit from
    the same user turn is also authoritative for a bounded retry: MiniMax can
    preserve the basename while dropping an intermediate directory on the
    next call. We still require exactly one mentioned basename and keep the
    target inside the client working directory.
    """
    if not isinstance(raw_output, str) or not raw_output or not working_directory:
        return ""
    intent = re.split(
        r"(?is)<tool_call|\[tool\s+call\s*:",
        raw_output,
        maxsplit=1,
    )[0]
    mentioned = []
    proven_paths = [
        *_successful_read_paths_after_last_user(processed_messages),
        *_successful_mutating_paths_after_last_user(processed_messages),
    ]
    for proven_path in proven_paths:
        proven = os.path.normpath(proven_path)
        if not os.path.isabs(proven):
            proven = os.path.normpath(os.path.join(working_directory, proven))
        basename = os.path.basename(proven)
        if not basename or not re.search(
            rf"(?<![A-Za-z0-9_.-]){re.escape(basename)}"
            rf"(?![A-Za-z0-9_.-])",
            intent,
        ):
            continue
        try:
            inside = (
                os.path.commonpath([working_directory, proven])
                == working_directory
            )
        except ValueError:
            inside = False
        if inside and proven not in mentioned:
            mentioned.append(proven)
    return mentioned[0] if len(mentioned) == 1 else ""


def _fill_missing_mutating_tool_path(
    tool_name,
    arguments,
    tools,
    processed_messages,
    raw_output=None,
):
    """Fill one missing required path from one explicit relative user target.

    This only repairs mutating file tools when the schema requires a path, the
    client supplied an exact working directory, and the real user instruction
    names exactly one safe relative file. It never guesses among multiple
    targets and never rewrites an explicit absolute user path.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in _MUTATING_FILE_TOOL_NAMES:
        return arguments, []
    if not isinstance(arguments, dict):
        return arguments, []
    if any(
        isinstance(arguments.get(key), str) and arguments.get(key).strip()
        for key in _FILE_PATH_ARGUMENT_KEYS
    ):
        return arguments, []
    required = set(_tool_schema_required_names(tools, tool_name))
    required_path_keys = [
        key for key in _FILE_PATH_ARGUMENT_KEYS if key in required
    ]
    if len(required_path_keys) != 1:
        return arguments, []
    working_directory = _tool_working_directory_from_messages(
        processed_messages
    )
    instruction = _last_user_instruction_text(processed_messages)
    if not working_directory:
        return arguments, []
    targets = []
    for match in _RELATIVE_MUTATION_TARGET_RE.finditer(instruction or ""):
        target = os.path.normpath(match.group("path").strip())
        if (
            target
            and not os.path.isabs(target)
            and ".." not in target.split(os.sep)
            and target not in targets
        ):
            targets.append(target)
    candidate = ""
    if len(targets) == 1:
        candidate = os.path.normpath(
            os.path.join(working_directory, targets[0])
        )
    elif isinstance(raw_output, str) and raw_output:
        # A bounded retry can correctly emit Write content while dropping only
        # its path. Recover that path solely from an exact filename named in
        # the model's pre-tool intent and proven by a successful Read in this
        # same user turn. This avoids guessing among multi-file agent tasks.
        candidate = _named_proven_file_target(
            raw_output,
            processed_messages,
            working_directory,
        )
    if not candidate:
        return arguments, []
    try:
        inside = os.path.commonpath([working_directory, candidate]) == working_directory
    except ValueError:
        inside = False
    if not inside:
        return arguments, []
    key = required_path_keys[0]
    repaired = dict(arguments)
    repaired[key] = candidate
    return repaired, [(key, candidate)]


def _anchor_mutating_tool_path_from_named_read(
    tool_name,
    arguments,
    processed_messages,
    raw_output,
):
    """Repair a drifted mutation path from one named, tool-proven file."""
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in _MUTATING_FILE_TOOL_NAMES:
        return arguments, []
    if not isinstance(arguments, dict):
        return arguments, []
    working_directory = _tool_working_directory_from_messages(processed_messages)
    target = _named_proven_file_target(
        raw_output,
        processed_messages,
        working_directory,
    )
    if not target:
        return arguments, []
    user_text = _last_user_instruction_text(processed_messages)
    anchored = dict(arguments)
    changes = []
    for key in _FILE_PATH_ARGUMENT_KEYS:
        source = anchored.get(key)
        if not isinstance(source, str) or not source.strip():
            continue
        source = os.path.normpath(source.strip())
        if source == target or source in user_text:
            continue
        anchored[key] = target
        changes.append((key, source, target))
    return anchored, changes


def _anchor_mutating_tool_path_from_read_basename(
    tool_name,
    arguments,
    processed_messages,
):
    """Anchor a drifted mutation to one same-basename successful Read.

    Some MiniMax retries emit no pre-tool prose, so the intent-name anchor has
    nothing to inspect. A unique successful Read of the same basename in the
    current user turn is still exact evidence; ambiguous reads and explicit
    absolute user targets remain untouched.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in _MUTATING_FILE_TOOL_NAMES:
        return arguments, []
    if not isinstance(arguments, dict):
        return arguments, []
    working_directory = _tool_working_directory_from_messages(processed_messages)
    user_text = _last_user_instruction_text(processed_messages)
    read_paths = _successful_read_paths_after_last_user(processed_messages)
    anchored = dict(arguments)
    changes = []
    for key in _FILE_PATH_ARGUMENT_KEYS:
        source = anchored.get(key)
        if not isinstance(source, str) or not source.strip():
            continue
        source = os.path.normpath(source.strip())
        if os.path.isabs(source) and source in user_text:
            continue
        basename = os.path.basename(source)
        if not basename:
            continue
        candidates = []
        for read_path in read_paths:
            target = os.path.normpath(read_path)
            if not os.path.isabs(target):
                # Preserve the exact relative path that the client already
                # executed successfully. ZCode can advertise the desktop
                # process home as its cwd while resolving file tools against
                # the selected workspace. Expanding ``word_stats.py`` against
                # that misleading home turned a proven relative target into a
                # wrong absolute path and fed the model a failure loop.
                if ".." in target.split(os.sep):
                    continue
            if os.path.basename(target) != basename:
                continue
            if working_directory and os.path.isabs(target):
                try:
                    if os.path.commonpath([working_directory, target]) != working_directory:
                        continue
                except ValueError:
                    continue
            if target not in candidates:
                candidates.append(target)
        if len(candidates) != 1 or candidates[0] == source:
            continue
        anchored[key] = candidates[0]
        changes.append((key, source, candidates[0]))
    return anchored, changes


def _anchor_file_tool_path_to_user_relative_target(
    tool_name,
    arguments,
    processed_messages,
):
    """Keep an explicitly user-named workspace file relative for the client.

    ZCode's provider metadata can advertise the desktop process home while its
    file tools resolve relative paths inside the selected workspace. MiniMax
    then expands ``word_stats.py`` to that home (or a memorized example home),
    even though the user's relative target is unambiguous. Preserve that exact
    relative spelling for Read-family calls; later mutations can then reuse
    that exact client-proven path through the successful-Read anchor. The
    client that owns the workspace remains the authority that resolves it.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in _WORKSPACE_FILE_TOOL_NAMES:
        return arguments, []
    if not isinstance(arguments, dict):
        return arguments, []
    user_text = _last_user_instruction_text(processed_messages)
    if not user_text:
        return arguments, []

    anchored = dict(arguments)
    changes = []
    for key in _FILE_PATH_ARGUMENT_KEYS:
        source = anchored.get(key)
        if not isinstance(source, str) or not os.path.isabs(source):
            continue
        source = os.path.normpath(source.strip())
        if source in user_text:
            continue
        basename = os.path.basename(source)
        if not basename or basename in {".", ".."}:
            continue
        relative_pattern = re.compile(
            rf"(?<![A-Za-z0-9_./-])"
            rf"(?P<path>(?:[A-Za-z0-9_.-]+/)*{re.escape(basename)})"
            r"(?=$|[\s,;:!?)}\]'\"`]|[.](?=\s|$))"
        )
        candidates = []
        for match in relative_pattern.finditer(user_text):
            candidate = os.path.normpath(match.group("path"))
            if (
                candidate
                and not os.path.isabs(candidate)
                and ".." not in candidate.split(os.sep)
                and candidate not in candidates
            ):
                candidates.append(candidate)
        if len(candidates) != 1:
            continue
        anchored[key] = candidates[0]
        changes.append((key, source, candidates[0]))
    return anchored, changes


def _anchor_mutating_tool_paths(tool_name, arguments, processed_messages):
    """Rewrite a hallucinated absolute target to a user-named relative path.

    This is intentionally narrow: the tool must mutate a file, the client must
    provide an exact working directory, and the latest user request must name
    the same basename as a relative path. Explicit absolute user targets are
    never rewritten. Returns ``(arguments, changes)``.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in _MUTATING_FILE_TOOL_NAMES:
        return arguments, []
    if not isinstance(arguments, dict):
        return arguments, []
    working_directory = _tool_working_directory_from_messages(
        processed_messages
    )
    if not working_directory:
        return arguments, []
    user_text = _last_user_instruction_text(processed_messages)
    if not user_text:
        return arguments, []

    anchored = dict(arguments)
    changes = []
    for key in _FILE_PATH_ARGUMENT_KEYS:
        raw_path = anchored.get(key)
        if not isinstance(raw_path, str) or not os.path.isabs(raw_path):
            continue
        raw_path = os.path.normpath(raw_path.strip())
        try:
            if os.path.commonpath([working_directory, raw_path]) == working_directory:
                continue
        except ValueError:
            pass
        basename = os.path.basename(raw_path)
        if not basename or basename in {".", ".."}:
            continue
        # An explicit absolute path is user authority. Leave it untouched so
        # the normal boundary either accepts the exact path or rejects a drift.
        absolute_pattern = re.compile(
            rf"(?<![A-Za-z0-9_.-])/"
            rf"(?:[^/\s'\"`<>]+/)*{re.escape(basename)}"
        )
        if absolute_pattern.search(user_text):
            continue
        relative_pattern = re.compile(
            rf"(?<![A-Za-z0-9_.-])"
            rf"((?:[A-Za-z0-9_.-]+/)*{re.escape(basename)})"
            rf"(?![A-Za-z0-9_.-])"
        )
        matches = [match.group(1) for match in relative_pattern.finditer(user_text)]
        if not matches:
            continue
        relative_target = max(matches, key=len)
        candidate = os.path.normpath(
            os.path.join(working_directory, relative_target)
        )
        try:
            inside = (
                os.path.commonpath([working_directory, candidate])
                == working_directory
            )
        except ValueError:
            inside = False
        if not inside:
            continue
        anchored[key] = candidate
        changes.append((key, raw_path, candidate))
    return anchored, changes


def _successful_read_paths_after_last_user(processed_messages):
    latest_user_index = max(
        (
            index for index, message in enumerate(processed_messages or [])
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        default=-1,
    )
    read_calls = {}
    successful_paths = []
    for message in (processed_messages or [])[latest_user_index + 1:]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_name = re.sub(
                    r"[^a-z0-9]",
                    "",
                    _tool_call_name_for_loop(call).lower(),
                )
                if call_name not in {"read", "readfile", "openfile", "viewfile"}:
                    continue
                call_id = str(call.get("id") or "").strip()
                call_path = _file_write_path(_tool_call_arguments_dict(call))
                if call_id and call_path:
                    read_calls[call_id] = call_path
        elif message.get("role") in {"tool", "function"}:
            call_id = str(message.get("tool_call_id") or "").strip()
            call_path = read_calls.get(call_id)
            if not call_path:
                continue
            result_text = _tool_message_text(message)
            if re.search(
                r"(?i)(?:file not found|no such file|not_found|error:|failed)",
                result_text,
            ):
                continue
            successful_paths.append(call_path)
    return successful_paths


def _successful_mutating_paths_after_last_user(processed_messages):
    """Return file targets successfully mutated after the latest user turn."""
    latest_user_index = max(
        (
            index for index, message in enumerate(processed_messages or [])
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        default=-1,
    )
    mutating_calls = {}
    successful_paths = []
    for message in (processed_messages or [])[latest_user_index + 1:]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_name = re.sub(
                    r"[^a-z0-9]",
                    "",
                    _tool_call_name_for_loop(call).lower(),
                )
                if call_name not in _MUTATING_FILE_TOOL_NAMES:
                    continue
                call_id = str(call.get("id") or "").strip()
                call_path = _file_write_path(_tool_call_arguments_dict(call))
                if call_id and call_path:
                    mutating_calls[call_id] = call_path
        elif message.get("role") in {"tool", "function"}:
            call_id = str(message.get("tool_call_id") or "").strip()
            call_path = mutating_calls.get(call_id)
            if not call_path:
                continue
            result_text = _tool_message_text(message)
            if re.search(
                r"(?i)(?:file not found|no such file|not_found|error:|failed|"
                r"permission denied|was not applied|did not match)",
                result_text,
            ):
                continue
            successful_paths.append(call_path)
    return successful_paths


def _successful_command_workdirs_after_last_user(processed_messages, command_name):
    """Return explicit workdirs from successful same-turn command calls."""
    latest_user_index = max(
        (
            index for index, message in enumerate(processed_messages or [])
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        default=-1,
    )
    calls = {}
    successful = []
    for message in (processed_messages or [])[latest_user_index + 1:]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                if _tool_call_name_for_loop(call) != command_name:
                    continue
                args = _tool_call_arguments_dict(call)
                workdir = next(
                    (
                        args.get(key).strip()
                        for key in (
                            "workdir", "cwd", "working_directory",
                            "workingDirectory", "directory",
                        )
                        if isinstance(args.get(key), str) and args.get(key).strip()
                    ),
                    "",
                )
                call_id = str(call.get("id") or "").strip()
                if call_id and os.path.isabs(workdir):
                    calls[call_id] = os.path.normpath(workdir)
        elif message.get("role") in {"tool", "function"}:
            workdir = calls.get(str(message.get("tool_call_id") or "").strip())
            if not workdir:
                continue
            result_text = _tool_message_text(message)
            if re.search(
                r"(?i)(?:notfound|no such file|permission denied|error:|failed|"
                r"traceback)",
                result_text,
            ):
                continue
            if workdir not in successful:
                successful.append(workdir)
    return successful


def _anchor_command_working_directory(
    tool_name,
    arguments,
    tools,
    processed_messages,
):
    """Repair a command cwd that drifted from the client's advertised root.

    Agent prompts provide an exact working directory in trusted system/developer
    metadata. MiniMax occasionally preserves the final directory name while
    corrupting an earlier path component, which makes an otherwise valid shell
    call fail and repeat forever. Only repair an external absolute cwd whose
    basename exactly matches the advertised root; legitimate subdirectories and
    user-requested absolute directories remain untouched.
    """
    if not isinstance(arguments, dict):
        return arguments, []
    command_name = _command_tool_name_from_schema(tools)
    if not command_name or tool_name != command_name:
        return arguments, []
    proven_workdirs = _successful_command_workdirs_after_last_user(
        processed_messages,
        command_name,
    )
    working_directory = _tool_working_directory_from_messages(
        processed_messages
    )
    history_only_workdir = False
    if not working_directory and len(proven_workdirs) == 1:
        working_directory = proven_workdirs[0]
        history_only_workdir = True
    if not working_directory:
        return arguments, []
    user_text = _last_user_instruction_text(processed_messages)
    anchored = dict(arguments)
    changes = []
    for key in (
        "workdir",
        "cwd",
        "working_directory",
        "workingDirectory",
        "directory",
    ):
        raw_value = anchored.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        raw_path = os.path.normpath(raw_value.strip())
        if not os.path.isabs(raw_path):
            continue
        within_working_directory = False
        try:
            within_working_directory = (
                os.path.commonpath([working_directory, raw_path])
                == working_directory
            )
        except ValueError:
            pass
        if within_working_directory and not (
            history_only_workdir
            and len(proven_workdirs) == 1
            and not os.path.exists(raw_path)
        ):
            continue
        if raw_path in user_text:
            continue
        raw_base = os.path.basename(raw_path)
        working_base = os.path.basename(working_directory)
        same_parent = (
            os.path.dirname(raw_path) == os.path.dirname(working_directory)
        )
        near_match = bool(
            same_parent
            and abs(len(raw_base) - len(working_base)) <= 3
            and difflib.SequenceMatcher(
                None,
                raw_base,
                working_base,
            ).ratio() >= 0.90
        )
        anchor_target = working_directory
        if raw_base != working_base and not near_match:
            if (
                len(proven_workdirs) != 1
                or os.path.exists(raw_path)
                or not os.path.isdir(proven_workdirs[0])
            ):
                continue
            anchor_target = proven_workdirs[0]
        anchored[key] = anchor_target
        changes.append((key, raw_path, anchor_target))
    # Some schemas (notably ZCode's Bash tool) carry no separate workdir and
    # put it in a leading ``cd ... &&``.  Repair only a direct child named in
    # the latest user instruction, anchored below the exact advertised root.
    # This caught ``...-20260712/build`` after the model dropped ``-2335``.
    for key in ("command", "cmd", "input"):
        command = anchored.get(key)
        if not isinstance(command, str) or not command.strip():
            continue
        cd_match = re.match(
            r"^(?P<prefix>\s*cd\s+)(?P<quote>['\"]?)"
            r"(?P<path>/.*?)(?P=quote)(?P<separator>\s*(?:&&|;))",
            command,
            flags=re.DOTALL,
        )
        if not cd_match:
            continue
        raw_path = os.path.normpath(cd_match.group("path").strip())
        try:
            if os.path.commonpath([working_directory, raw_path]) == working_directory:
                continue
        except ValueError:
            pass
        if raw_path in user_text:
            continue
        child = os.path.basename(raw_path)
        if (
            not child
            or child in {".", ".."}
            or not re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(child)}(?:/|\b)",
                user_text,
            )
        ):
            continue
        candidate = os.path.normpath(os.path.join(working_directory, child))
        try:
            if os.path.commonpath([working_directory, candidate]) != working_directory:
                continue
        except ValueError:
            continue
        quote = cd_match.group("quote")
        replacement = (
            cd_match.group("prefix")
            + quote
            + candidate
            + quote
            + cd_match.group("separator")
        )
        repaired_command = replacement + command[cd_match.end():]
        anchored[key] = repaired_command
        changes.append((key, raw_path, candidate))
    # Repair a near-matching sibling root embedded in a read-only command such
    # as ``ls /tmp/task-20260712/build``. This remains scoped to an exact
    # client working-directory directive and only rewrites a sibling whose
    # basename is a close/prefix drift of that root.
    working_parent = os.path.dirname(working_directory)
    working_base = os.path.basename(working_directory)
    for key in ("command", "cmd", "input"):
        command = anchored.get(key)
        if not isinstance(command, str) or not command:
            continue
        rewritten = command
        raw_tokens = sorted(
            set(re.findall(r"/[A-Za-z0-9_./-]+", rewritten)),
            key=len,
            reverse=True,
        )
        for raw_token in raw_tokens:
            token = raw_token.rstrip(".,:)")
            normalized = os.path.normpath(token)
            try:
                if os.path.commonpath([working_directory, normalized]) == working_directory:
                    continue
            except ValueError:
                pass
            if token in user_text:
                continue
            cursor = normalized
            sibling_root = ""
            while cursor and cursor != os.path.dirname(cursor):
                if os.path.dirname(cursor) == working_parent:
                    candidate_base = os.path.basename(cursor)
                    length_delta = abs(len(candidate_base) - len(working_base))
                    near_match = bool(
                        candidate_base != working_base
                        and length_delta <= 16
                        and (
                            working_base.startswith(candidate_base)
                            or difflib.SequenceMatcher(
                                None,
                                candidate_base,
                                working_base,
                            ).ratio() >= 0.88
                        )
                    )
                    if near_match:
                        sibling_root = cursor
                    break
                cursor = os.path.dirname(cursor)
            if not sibling_root:
                continue
            suffix = os.path.relpath(normalized, sibling_root)
            replacement = working_directory
            if suffix != ".":
                replacement = os.path.join(working_directory, suffix)
            if raw_token.endswith("/"):
                replacement += "/"
            rewritten = rewritten.replace(raw_token, replacement)
            changes.append((key, sibling_root, replacement.rstrip("/")))
        anchored[key] = rewritten
    return anchored, changes


def _anchor_command_paths_from_read_history(
    tool_name,
    arguments,
    tools,
    processed_messages,
):
    """Repair a shell path only when it matches a proven read target suffix."""
    if not isinstance(arguments, dict):
        return arguments, []
    command_name = _command_tool_name_from_schema(tools)
    if not command_name or tool_name != command_name:
        return arguments, []
    working_directory = _tool_working_directory_from_messages(processed_messages)
    if not working_directory:
        return arguments, []
    successful_paths = _successful_read_paths_after_last_user(processed_messages)
    if not successful_paths:
        return arguments, []
    anchored = dict(arguments)
    changes = []
    user_text = _last_user_instruction_text(processed_messages)
    for key in ("command", "cmd", "input"):
        command = anchored.get(key)
        if not isinstance(command, str) or not command:
            continue
        rewritten = command
        for successful_path in successful_paths:
            known = os.path.normpath(successful_path)
            if not os.path.isabs(known):
                known = os.path.normpath(os.path.join(working_directory, known))
            try:
                relative = os.path.relpath(known, working_directory)
            except ValueError:
                continue
            if relative.startswith(".."):
                continue
            for candidate in set(re.findall(r"/[A-Za-z0-9_./-]+", rewritten)):
                candidate = candidate.rstrip(".,:)")
                if candidate == known or candidate in user_text:
                    continue
                if not candidate.endswith(os.sep + relative):
                    continue
                rewritten = rewritten.replace(candidate, known)
                changes.append((key, candidate, known))
        anchored[key] = rewritten
    return anchored, changes


def _tool_request_path_violation(tool_name, arguments, processed_messages):
    """Return a reason when a mutating call invents an external path.

    Relative paths remain client-resolved. Explicit external paths in the
    latest user request remain valid. This only catches absolute mutation
    targets that conflict with an anchored client working directory.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(tool_name or "").lower())
    if normalized_name not in _MUTATING_FILE_TOOL_NAMES:
        return ""
    if not isinstance(arguments, dict):
        return ""
    working_directory = _tool_working_directory_from_messages(
        processed_messages
    )
    if not working_directory:
        return ""
    user_text = _last_user_instruction_text(processed_messages)
    for key in _FILE_PATH_ARGUMENT_KEYS:
        raw_path = arguments.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        raw_path = raw_path.strip()
        if len(raw_path) > 1024 or any(ord(ch) < 32 for ch in raw_path):
            return (
                f"{tool_name}.{key} is not a sane file path "
                f"(length={len(raw_path)}, contains_control="
                f"{any(ord(ch) < 32 for ch in raw_path)})"
            )
        if not os.path.isabs(raw_path):
            continue
        candidate = os.path.normpath(raw_path)
        try:
            inside_working_directory = (
                os.path.commonpath([working_directory, candidate])
                == working_directory
            )
        except ValueError:
            inside_working_directory = False
        if inside_working_directory or candidate in user_text:
            continue
        return (
            f"{tool_name}.{key} absolute path {candidate!r} is outside "
            f"client working directory {working_directory!r}"
        )
    return ""


def _extract_simple_write_request(text):
    if not isinstance(text, str) or not text.strip():
        return None
    lowered = text.lower()
    if not any(word in lowered for word in ("write", "create", "save")):
        return None
    if "containing exactly" not in lowered:
        return None
    content_match = re.search(
        r"containing exactly\s+(.+?)(?:\s*,?\s+then\b|[.。]\s*$|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not content_match:
        return None
    window_start = max(0, content_match.start() - 500)
    prefix_window = text[window_start:content_match.start()]
    relative_matches = list(re.finditer(
        r"\b(?:create|write|save)\s+(?:a\s+)?(?:simple\s+)?(?:text\s+)?"
        r"(?:file\s+)?(?:named\s+|at\s+)?([A-Za-z0-9._~/-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)",
        prefix_window,
        re.IGNORECASE,
    ))
    absolute_matches = list(re.finditer(r"(/[^\s'\"`;,<>]+)", prefix_window))
    path_match = absolute_matches[-1] if absolute_matches else (relative_matches[-1] if relative_matches else None)
    if not path_match:
        return None
    file_path = path_match.group(1).strip().strip("'\"`")
    content = content_match.group(1).strip().strip("'\"`")
    if content.endswith(".") and "\n" not in content:
        content = content[:-1]
    if not file_path or not content:
        return None
    return {"filename": file_path, "content": content}


def _synthesize_write_command_tool_call(processed_messages, tools, dropped_tool_names=None):
    dropped = {
        re.sub(r"[^a-z0-9]", "", str(name or "").lower())
        for name in (dropped_tool_names or [])
    }
    if dropped and "applypatch" not in dropped:
        return None
    request = _extract_simple_write_request(
        _last_user_instruction_text(processed_messages)
    )
    if not request:
        return None
    command_name = _command_tool_name_from_schema(tools)
    if not command_name:
        return None
    filename = request["filename"]
    content = request["content"]
    cmd = (
        f"mkdir -p {shlex.quote(os.path.dirname(filename) or '.')} && "
        f"printf %s {shlex.quote(content)} > {shlex.quote(filename)} && "
        f"cat {shlex.quote(filename)}"
    )
    args = _canonicalize_tool_argument_keys(
        {
            "cmd": cmd,
            "command": cmd,
            "input": cmd,
            "justification": f"Create {filename}",
        },
        tools,
        command_name,
    )
    args = _coerce_codex_control_tool_arguments(args, tools, command_name)
    if not args:
        return None
    return _openai_tool_call(command_name, args, 0)


def _synthesize_explicit_read_tool_call(processed_messages, tools):
    """Build one read call only from an explicit path in the user request."""
    user_text = _last_user_instruction_text(processed_messages)
    if not user_text:
        return None
    matches = list(re.finditer(
        r"(?i)\bread\s+(?:the\s+)?(?:existing\s+)?(?:file\s+)?"
        r"(?P<quote>[`\"']?)"
        r"(?P<path>(?:/|\.?\.?/)?[A-Za-z0-9_.-]+"
        r"(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9][A-Za-z0-9._-]*)"
        r"(?P=quote)",
        user_text,
    ))
    if not matches:
        return None
    path = matches[-1].group("path").strip()
    if not path or len(path) > 1024 or any(ord(ch) < 32 for ch in path):
        return None
    successful_paths = _successful_read_paths_after_last_user(
        processed_messages
    )
    desired_suffix = os.path.normpath(path).lstrip("./")
    for successful_path in successful_paths:
        normalized = os.path.normpath(successful_path)
        if (
            normalized == os.path.normpath(path)
            or normalized.endswith(os.sep + desired_suffix)
        ):
            # The client has already completed this explicit read in the
            # current user turn. Do not synthesize it again when the next
            # write/edit call is malformed.
            return None
    read_name = ""
    for name in _tool_names_from_schema(tools):
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        if normalized not in {"read", "readfile", "openfile", "viewfile"}:
            continue
        props = {
            re.sub(r"[^a-z0-9]", "", prop.lower())
            for prop in _tool_schema_property_names(tools, name)
        }
        if props & {"path", "filepath", "file", "filename"}:
            read_name = name
            break
    if not read_name:
        return None
    args = _canonicalize_tool_argument_keys(
        {
            "path": path,
            "file_path": path,
            "filePath": path,
            "filename": path,
        },
        tools,
        read_name,
    )
    if _tool_schema_type_mismatches(args, tools, read_name):
        return None
    required = _tool_schema_required_names(tools, read_name)
    if any(args.get(key) in (None, "") for key in required):
        return None
    return _openai_tool_call(read_name, args, 0)


def _recover_named_empty_read_tool_call(text, tools):
    """Recover one non-mutating read named in a complete empty invocation.

    MiniMax occasionally reasons ``read `file.py` `` and then emits a named
    but argument-less ``<invoke name="Read"></invoke>``. A retry can repeat
    that deterministic omission. Recover only a declared read-like tool and
    only when the immediately preceding reasoning contains one unambiguous
    backtick-quoted file path. Mutating tools are never synthesized here.
    """
    if not isinstance(text, str) or "invoke" not in text.lower():
        return []
    compact = text.replace("]<]minimax[>[", "")
    matches = list(re.finditer(
        r"(?is)<invoke\s+name\s*=\s*[\"']"
        r"(?P<name>[A-Za-z_$][\w:.$-]*)[\"']\s*>\s*</invoke>",
        compact,
    ))
    if not matches:
        return []
    name_map = _tool_name_map_from_schema(tools)
    match = matches[-1]
    name = _canonical_tool_name(match.group("name"), name_map)
    normalized = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    if normalized not in {"read", "readfile", "openfile", "viewfile"}:
        return []
    required = set(_tool_schema_required_names(tools, name))
    path_props = [
        prop for prop in _tool_schema_property_names(tools, name)
        if re.sub(r"[^a-z0-9]", "", prop.lower())
        in {"path", "filepath", "file", "filename"}
    ]
    required_paths = [prop for prop in path_props if prop in required]
    if len(required_paths) != 1:
        return []
    planning_tail = compact[max(0, match.start() - 1200):match.start()]
    if not re.search(r"\bread\b", planning_tail, flags=re.IGNORECASE):
        return []
    candidates = []
    for raw_path in re.findall(r"`([^`\r\n]{1,1024})`", planning_tail):
        path = raw_path.strip()
        basename = os.path.basename(path)
        if (
            not path
            or "." not in basename
            or any(ord(ch) < 32 for ch in path)
            or any(marker in path for marker in ("<", ">", "\x00"))
        ):
            continue
        candidates.append(path)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        return []
    args = _canonicalize_tool_argument_keys(
        {
            "path": candidates[0],
            "file_path": candidates[0],
            "filePath": candidates[0],
            "filename": candidates[0],
        },
        tools,
        name,
    )
    if _tool_schema_type_mismatches(args, tools, name):
        return []
    if any(args.get(key) in (None, "") for key in required):
        return []
    logger.warning(
        "recovered named empty %s invocation from explicit reasoning path %r",
        name,
        candidates[0],
    )
    return [_openai_tool_call(name, args, 0)]


def _tool_call_as_display_text(tool_call):
    if not isinstance(tool_call, dict):
        return ""
    fn = tool_call.get("function")
    if not isinstance(fn, dict) or not fn.get("name"):
        return ""
    arguments = fn.get("arguments") or "{}"
    try:
        decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return ""
    if not isinstance(decoded, dict):
        return ""
    return (
        f"[Tool call: {fn['name']}]\n"
        + json.dumps(decoded, ensure_ascii=False)
    )


def _tool_retry_gen_params(gen_params, attempt):
    """Sampling nudge for a tool retry; a temp-0 replay would be deterministic."""
    retry = dict(gen_params or {})
    temps = TOOL_UNUSABLE_RETRY_TEMPERATURES
    temp = temps[min(max(attempt, 1), len(temps)) - 1]
    try:
        current = float(retry.get("temperature") or 0.0)
    except (TypeError, ValueError):
        current = 0.0
    retry["temperature"] = max(temp, current)
    retry.pop("seed", None)
    return retry


def _tool_retry_recovery_hint(
    full_output,
    tool_module,
    tools,
    processed_messages=None,
):
    """Return targeted feedback for a structurally invalid tool call."""
    def _mutation_hint(name, required, reason):
        normalized = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
        required_text = ", ".join(f"`{key}`" for key in sorted(required))
        base = (
            f"Tool-call recovery: the previous `{name}` call {reason}. "
            f"Emit exactly one valid `{name}` call"
            + (f" containing every required argument: {required_text}. "
               if required_text else ". ")
        )
        if normalized == "applypatch":
            patch_arg = next(
                (
                    key
                    for key in ("input", "patch", "patch_text", "patchText")
                    if key in required
                ),
                "input",
            )
            return base + (
                f"Put one complete focused patch in `{patch_arg}`. It must "
                "start with `*** Begin Patch`, contain one Add/Update/Delete "
                "File operation, and end with `*** End Patch`. Keep the patch "
                f"near {TOOL_WRITE_CHUNK_TARGET_CHARS} characters and below "
                f"{TOOL_WRITE_CHUNK_MAX_CHARS} characters. Do not regenerate "
                "the entire artifact in one patch; continue with another "
                "focused patch after the client executes this one."
            )
        if normalized in {"edit", "editfile", "multiedit"}:
            size_limit = ""
            if TOOL_WRITE_CHUNK_MAX_CHARS > 0:
                size_limit = (
                    f" targeting at most {TOOL_WRITE_CHUNK_TARGET_CHARS} "
                    "characters and never exceeding the hard ceiling of "
                    f"{TOOL_WRITE_CHUNK_MAX_CHARS} characters"
                )
            return base + (
                "Make one focused replacement only. Use the exact existing "
                "text as the old/target argument and a bounded replacement "
                f"{size_limit} as the new argument. Do not regenerate the "
                "entire file in one Edit."
            )
        if TOOL_WRITE_CHUNK_MAX_CHARS > 0:
            return base + (
                "Place the destination path argument before the content. "
                "Create only a small working scaffold and keep the file "
                f"content near {TOOL_WRITE_CHUNK_TARGET_CHARS} characters "
                "and below the hard ceiling of "
                f"{TOOL_WRITE_CHUNK_MAX_CHARS} characters. Do not attempt "
                "the complete large file in this "
                "retry; continue with focused Edit or bounded append calls "
                "after the client executes the scaffold."
            )
        return base + (
            "Emit a complete atomic call. For a large file, create a small "
            "working scaffold first, then continue with focused Edit or "
            "bounded append calls after the client executes it."
        )

    def _unknown_invoke_hint(raw_output):
        advertised = [
            _tool_function_name(tool)
            for tool in (tools or [])
            if _tool_function_name(tool)
        ]
        advertised_by_normalized = {
            re.sub(r"[^a-z0-9]", "", name.lower()): name
            for name in advertised
        }
        candidates = []
        for match in re.finditer(
            r"(?is)<invoke(?:\s+name\s*=\s*[\"']([^\"']+)[\"']|"
            r"\s+([A-Za-z_][A-Za-z0-9_.:-]*))",
            raw_output or "",
        ):
            candidate = (match.group(1) or match.group(2) or "").strip()
            if candidate:
                candidates.append(candidate)
        if not candidates:
            return ""
        normalized_candidates = [
            re.sub(r"[^a-z0-9]", "", candidate.lower())
            for candidate in candidates
        ]
        # If the malformed wrapper still names a real advertised tool, the
        # schema-specific recovery paths should guide that call instead of
        # blaming an outer pseudo-wrapper such as `function_calls`.
        if any(name in advertised_by_normalized for name in normalized_candidates):
            return ""
        unknown = candidates[0]
        available = ", ".join(f"`{name}`" for name in advertised[:24])
        command_name = _command_tool_name_from_schema(tools)
        command_hint = (
            f" For filesystem search or inspection, use `{command_name}` with "
            "a concrete shell command."
            if command_name else ""
        )
        return (
            f"Tool-call recovery: `{unknown}` is not an advertised tool for "
            f"this request. Emit exactly one complete call using one of: "
            f"{available}.{command_hint} Do not emit `{unknown}` again."
        )

    tool_calls, _ = _parse_tool_calls(full_output or "", tool_module, tools)
    for tool_call in tool_calls:
        name = _tool_call_name_for_loop(tool_call)
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        if normalized not in _MUTATING_FILE_TOOL_NAMES:
            continue
        arguments = _tool_call_arguments_dict(tool_call)
        required = set(_tool_schema_required_names(tools, name))
        mutation_fingerprint = _exact_mutation_fingerprint(name, arguments)
        if mutation_fingerprint in _successful_exact_mutation_fingerprints(
            processed_messages
        ):
            return (
                f"Tool-call recovery: this exact `{name}` action already "
                "completed successfully in the current user turn. Do not "
                "repeat it. Continue with a different advertised tool or "
                "different arguments if more work is required; otherwise "
                "provide the final answer from the completed result."
            )
        oversized = _oversized_mutating_file_payload(name, arguments)
        if oversized:
            return _mutation_hint(
                name,
                required,
                f"exceeded the bounded payload limit in `{oversized[0]}` "
                f"({oversized[1]} characters)",
            )
        command_violation = _command_tool_payload_violation(
            name,
            arguments,
            tools,
        )
        if command_violation:
            write_tools = [
                _tool_function_name(tool)
                for tool in (tools or [])
                if re.sub(
                    r"[^a-z0-9]",
                    "",
                    _tool_function_name(tool).lower(),
                ) in _WRITE_FILE_TOOL_NAMES
            ]
            write_instruction = (
                f" Use `{write_tools[0]}` with an explicit path and bounded "
                "content, or use one focused Edit call."
                if write_tools else
                " Use an explicit interpreter command or a proper file tool."
            )
            return (
                f"Tool-call recovery: the previous `{name}` call was invalid "
                f"because {command_violation}.{write_instruction} Emit exactly "
                "one complete advertised tool call and no prose."
            )
        missing = [
            key for key in sorted(required)
            if arguments.get(key) in (None, "")
        ]
        if missing:
            return _mutation_hint(
                name,
                required,
                "was invalid because required argument(s) "
                + ", ".join(f"`{key}`" for key in missing)
                + " were missing",
            )
        type_mismatches = _tool_schema_type_mismatches(
            arguments,
            tools,
            name,
        )
        if type_mismatches:
            specs = _tool_schema_property_specs(tools, name)
            expected = []
            for key in type_mismatches:
                spec = specs.get(key) if isinstance(specs, dict) else None
                schema_type = spec.get("type") if isinstance(spec, dict) else None
                expected.append(
                    f"`{key}` as JSON {schema_type or 'the advertised type'}"
                )
            return _mutation_hint(
                name,
                required,
                "was invalid because it must provide "
                + ", ".join(expected)
                + " rather than arrays, objects, or nested tags",
            )
    raw_output = full_output or ""
    if _has_empty_native_invoke(raw_output):
        # Empty native invocations often follow perfectly explicit planning,
        # for example "continue with focused Edit calls" followed by
        # `<invoke:></invoke>`. Ground the retry in that advertised schema
        # instead of giving MiniMax a broad list of every available tool.
        for tool in tools or []:
            name = _tool_function_name(tool)
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            if normalized not in _MUTATING_FILE_TOOL_NAMES:
                continue
            if not re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}"
                r"(?:\s+(?:tool|call))?s?(?![A-Za-z0-9_])",
                raw_output,
                flags=re.IGNORECASE,
            ):
                continue
            return _mutation_hint(
                name,
                set(_tool_schema_required_names(tools, name)),
                "was named in the plan but omitted from the empty invocation",
            )
    explicit_read = _synthesize_explicit_read_tool_call(
        processed_messages,
        tools,
    )
    if explicit_read and (
        re.search(r"\bread\b", raw_output, flags=re.IGNORECASE)
        or "<invoke" in raw_output
    ):
        fn = explicit_read["function"]
        return (
            "Tool-call recovery: the previous response did not execute the "
            "explicit read. Emit exactly one complete "
            f"`{fn['name']}` call with arguments {fn['arguments']}. "
            "Do not narrate or change the path."
        )
    unknown_hint = _unknown_invoke_hint(raw_output)
    if unknown_hint:
        return unknown_hint
    if _looks_like_raw_tool_fragment(raw_output, tool_module):
        for tool in tools or []:
            name = _tool_function_name(tool)
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            if normalized not in _MUTATING_FILE_TOOL_NAMES:
                continue
            if _tool_invocation_match(raw_output, name):
                return _mutation_hint(
                    name,
                    set(_tool_schema_required_names(tools, name)),
                    "did not close before the generation budget ended",
                )
    single_apply_patch = _single_apply_patch_tool_name(tools)
    if single_apply_patch:
        return _mutation_hint(
            single_apply_patch,
            set(_tool_schema_required_names(tools, single_apply_patch)),
            "did not produce an executable call",
        )
    if _tool_intent_without_call(raw_output):
        if explicit_read:
            fn = explicit_read["function"]
            return (
                "Tool-call recovery: the previous response promised the "
                "explicit read but emitted an empty invocation. Emit exactly "
                f"one complete `{fn['name']}` call with arguments "
                f"{fn['arguments']}. Do not narrate or change the path."
            )
        advertised = [
            _tool_function_name(tool)
            for tool in (tools or [])
            if _tool_function_name(tool)
        ]
        available = ", ".join(f"`{name}`" for name in advertised[:24])
        return (
            "Tool-call recovery: the previous response promised an action "
            "but emitted no executable call. Emit exactly one complete tool "
            f"call now using one of: {available}. Do not narrate, plan, ask "
            "the user to run it, or repeat the promise; execute the next "
            "concrete step with an advertised tool."
        )
    return ""


def _tool_retry_messages(processed_messages, recovery_hint):
    """Append retry guidance without invalidating the cached transcript.

    Prepending a new system message makes the rendered retry diverge near
    token zero. At long context that can leave the completed KV resident while
    MLX allocates a second near-full cache for the retry, exhausting the
    smaller rank. A final user turn preserves the complete conversation prefix
    and turns recovery into the small incremental prefill it should be.
    """
    return [
        *[dict(message) for message in processed_messages],
        {"role": "user", "content": recovery_hint},
    ]


def _tool_retry_prefix_safety(original_token_ids, retry_token_ids, *,
                              min_context_tokens=8192,
                              min_reuse_ratio=0.50):
    """Describe whether a long retry must release its incompatible RAM KV.

    The normal appended retry should retain almost the full token prefix. If a
    tokenizer/template change unexpectedly collapses that overlap, explicitly
    release RAM caches on both ranks before retrying. This is a rare stability
    fallback; SSD metadata remains available for later session restoration.
    """
    original = list(original_token_ids or [])
    retry = list(retry_token_ids or [])
    comparable = min(len(original), len(retry))
    common = _common_prefix_len(original, retry)
    ratio = (common / comparable) if comparable else 0.0
    reset = bool(
        comparable >= max(1, int(min_context_tokens or 0))
        and ratio < float(min_reuse_ratio)
    )
    return {
        "reset": reset,
        "original_tokens": len(original),
        "retry_tokens": len(retry),
        "common_prefix_tokens": common,
        "reuse_ratio": ratio,
    }


def _tool_retry_thinking_mode(thinking_mode, prefer_no_think=None):
    """Select the MiniMax template for a bounded malformed-tool retry.

    Normal retries retain the request's mode so the long KV prefix remains
    reusable. A caller may explicitly request the native no-thinking template
    for a final recovery attempt; prefix-safety then cold-rebuilds if that
    template switch invalidates the resident KV.
    """
    prefer_no_think = (
        TOOL_RETRY_NO_THINK if prefer_no_think is None else bool(prefer_no_think)
    )
    if prefer_no_think and _enable_thinking_for_generation(thinking_mode):
        return "disabled"
    return thinking_mode


def _tool_retry_prefers_no_think(
    thinking_mode,
    attempt,
    total_attempts,
    prompt_tokens=0,
):
    """Use the incompatible no-thinking template only for short last resorts."""
    prompt_tokens = max(0, int(prompt_tokens or 0))
    return bool(
        TOOL_RETRY_NO_THINK
        and _enable_thinking_for_generation(thinking_mode)
        and int(attempt or 0) >= max(1, int(total_attempts or 0))
        and (
            TOOL_RETRY_NO_THINK_MAX_PROMPT_TOKENS <= 0
            or prompt_tokens <= TOOL_RETRY_NO_THINK_MAX_PROMPT_TOKENS
        )
    )


def _single_apply_patch_tool_name(tools):
    """Return the advertised name when apply_patch is the only tool."""
    names = _tool_names_from_schema(tools)
    if len(names) != 1:
        return ""
    name = next(iter(names))
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return name if normalized == "applypatch" else ""


def _single_apply_patch_fast_recovery(tools, prompt_tokens=0):
    """Allow an immediate no-thinking retry for a short apply_patch turn."""
    prompt_tokens = max(0, int(prompt_tokens or 0))
    return bool(
        TOOL_RETRY_NO_THINK
        and _single_apply_patch_tool_name(tools)
        and (
            TOOL_RETRY_NO_THINK_MAX_PROMPT_TOKENS <= 0
            or prompt_tokens <= TOOL_RETRY_NO_THINK_MAX_PROMPT_TOKENS
        )
    )


def _tool_retry_no_call_budget(
    thinking_mode,
    *,
    action_tool_task=False,
    require_call=False,
):
    """Leave enough retry room for legitimate thinking before a tool call."""
    budget = max(0, int(TOOL_RETRY_NO_CALL_TOKEN_BUDGET or 0))
    if (
        budget > 0
        and _enable_thinking_for_generation(thinking_mode)
        and (action_tool_task or require_call)
    ):
        budget = max(
            budget,
            max(0, int(TOOL_ACTION_NO_CALL_TOKEN_BUDGET or 0)),
        )
        if TOOL_THINKING_RUNAWAY_TOKEN_BUDGET > 0:
            budget = min(budget, TOOL_THINKING_RUNAWAY_TOKEN_BUDGET)
    return budget


def _render_tool_retry_prompt(model, processor, processed_messages, tools,
                              retry_thinking_mode, recovery_hint):
    if not recovery_hint:
        return None
    from mlx_vlm.prompt_utils import apply_chat_template

    retry_messages = _tool_retry_messages(processed_messages, recovery_hint)
    template_kwargs = _thinking_template_kwargs(
        model.config,
        enable_thinking=(retry_thinking_mode == "enabled"),
        thinking_mode=retry_thinking_mode,
    )
    if tools:
        template_kwargs["tools"] = _model_facing_tool_schemas(tools)
    with _tokenizer_runtime_lock:
        return apply_chat_template(
            processor,
            model.config,
            retry_messages,
            add_generation_prompt=True,
            num_images=0,
            **template_kwargs,
        )


def _usable_tool_turn(full_output, tool_module, tools, processed_messages,
                      thinking_mode, require_call=False):
    """True when a buffered tool-mode generation can be returned as-is.

    Mirrors the response-path checks: a validated tool call, a synthesizable
    simple write, or real visible content all count; empty/malformed tool
    markup and leaked reasoning do not. With require_call (OpenAI
    tool_choice "required"/named), ONLY a validated call counts — prose is
    unusable and the retry ladder regenerates.
    """
    tool_calls, remaining_text = _parse_tool_calls(
        full_output or "", tool_module, tools
    )
    if require_call and not tool_calls:
        return False
    if tool_calls:
        validated, dropped, dropped_names = _validate_outgoing_tool_calls(
            tool_calls,
            tools,
            return_dropped=True,
            return_dropped_names=True,
            processed_messages=processed_messages,
            raw_output=full_output,
        )
        if validated:
            return True
        if dropped and _synthesize_write_command_tool_call(
            processed_messages,
            tools,
            dropped_names,
        ):
            return True
        return False
    # An UNTERMINATED tool marker — the model closes thinking, emits the
    # opening `]<]minimax[>[<tool_call>` (or `<invoke`), then stops (EOS) with
    # no body and no close — is NOT removed by _strip_raw_tool_blocks (it
    # needs a start..end pair), so the leftover marker leaks as "content" and
    # slips past the strip-diff check below. Any raw tool fragment with zero
    # parsed calls is a botched call, never usable content — force the retry
    # ladder. (2026-07-10 hermes: "Let me make targeted edits" then a bare
    # <tool_call>; the turn shipped "could not produce a valid tool call" and
    # the agent stalled.)
    if _looks_like_raw_tool_fragment(full_output or "", tool_module):
        return False
    start_marker, _ = _tool_call_markers(tool_module)
    if start_marker and start_marker in (full_output or ""):
        return False
    raw_source = remaining_text or full_output or ""
    safe_text = _strip_raw_tool_blocks(raw_source, tool_module)
    if (safe_text or "").strip() != raw_source.strip():
        # Tool markup was present but produced ZERO parseable calls: the
        # model narrated an intent ("let me check the schedule...") and
        # botched the call. The narration used to count as usable content,
        # so the turn shipped as innocent text and agent loops silently
        # stalled (2026-07-09 hermes World Cup drop). Intent + failed call
        # is NOT usable — force the retry ladder.
        return False
    _, content = split_thinking_text(
        safe_text,
        assume_in_thinking=_enable_thinking_for_generation(thinking_mode),
    )
    content = _scrub_goal_state_echo((content or "").strip())
    if _tool_intent_without_call(content):
        # A post-tool turn can still end with "Now let me write/run..." and
        # no call. Treat that as an unfinished action, not a final answer, so
        # the bounded retry ladder emits the tool the model promised.
        return False
    # A turn whose visible content is itself a copy-spiral (2026-07-10: a
    # No-Think retry looped a ~300-char analysis paragraph to its 4096 cap
    # and shipped as the answer) is never usable — force the ladder onward.
    if _looks_like_degenerate_repetition(content):
        return False
    return bool(content) and not _looks_like_leaked_reasoning_content(content)


def _buffered_tool_reasoning(full_output, tool_module, thinking_mode):
    """Return reasoning that is safe to emit after a tool turn validates."""
    reasoning, _ = split_thinking_text(
        full_output or "",
        assume_in_thinking=_enable_thinking_for_generation(thinking_mode),
    )
    reasoning = _strip_raw_tool_blocks(reasoning or "", tool_module)
    reasoning = _strip_thinking_control_markers(reasoning).strip()
    if not reasoning or _looks_like_raw_tool_fragment(reasoning, tool_module):
        return ""
    return reasoning


def _remaining_tool_reasoning(buffered_reasoning, live_reasoning):
    """Return only reasoning not already emitted by the live stream.

    A malformed first attempt can be replaced by an in-place retry. In that
    case its reasoning will differ from the already-visible prefix; suppress
    the retry's private reasoning instead of duplicating or interleaving two
    plans before the recovered tool call.
    """
    buffered = buffered_reasoning or ""
    live = live_reasoning or ""
    if not buffered:
        return ""
    if not live:
        return buffered
    if buffered.startswith(live):
        return buffered[len(live):]
    return ""


def _native_tool_retry_ram_reset_reason(full_output, prompt_tokens):
    """Return why a native retry must start from a clean RAM cache."""
    output = full_output or ""
    control_bytes = sum(
        1
        for char in output
        if ord(char) < 32 and char not in "\n\r\t"
    )
    if "\x00" in output or control_bytes >= 8:
        return "corrupt_control_bytes"
    prompt_tokens = max(0, int(prompt_tokens or 0))
    if (
        NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS > 0
        and prompt_tokens >= NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS
    ):
        return (
            "long_context:"
            f"{prompt_tokens}>={NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS}"
        )
    return None


def _ensure_usable_tool_turn(model, processor, rank, *, full_output,
                             rank_request, prompt, max_tokens, thinking_mode,
                             gen_params, image_path, token_ids, session_id,
                             session_source, tool_module, tools,
                             processed_messages, req_id, stream,
                             should_abort=None, progress_cb=None,
                             action_tool_task=False):
    """Regenerate an unusable tool turn in place instead of falling back.

    MiniMax occasionally emits empty or malformed tool markup, especially in
    thinking mode at long context. Tool-mode output is buffered, so a bounded
    resample at a nudged temperature is invisible to the client and keeps
    Codex/Claude Code agent loops alive; the prose fallback would be treated
    as a final answer and end a long-running goal. Each retry re-broadcasts a
    normal generation request, so rank 1 mirrors it like any client retry.
    """
    # Explicit OpenAI forcing and clear first-turn action requests both need
    # a real call. A prose draft is not successful execution. The inference
    # helper deliberately stops inferring once a tool result exists, so a
    # normal final answer remains possible after the work is complete.
    require_call = _tool_request_requires_call(
        processed_messages,
        rank_request or {},
    )
    native_marker_retry = bool(
        not TOOL_COMPAT_OVERLAY
        and NATIVE_TOOL_ACTION_RETRY_ATTEMPTS > 0
        and _looks_like_raw_tool_fragment(full_output or "", tool_module)
    )
    native_action_retry = bool(
        not TOOL_COMPAT_OVERLAY
        and NATIVE_TOOL_ACTION_RETRY_ATTEMPTS > 0
        and (action_tool_task or require_call or native_marker_retry)
    )
    if not (
        tools
        and tool_module is not None
        and (TOOL_COMPAT_OVERLAY or native_action_retry)
    ):
        return full_output
    if TOOL_UNUSABLE_RETRY_ATTEMPTS <= 0:
        return full_output
    if _usable_tool_turn(full_output, tool_module, tools, processed_messages,
                         thinking_mode, require_call=require_call):
        return full_output
    label = "stream" if stream else "non-stream"
    completed_artifact = _synthesize_complete_add_file_artifact_text(
        full_output,
        tools,
    )
    if completed_artifact and _usable_tool_turn(
        completed_artifact,
        tool_module,
        tools,
        processed_messages,
        thinking_mode,
        require_call=require_call,
    ):
        logger.warning(
            "[rank 0] %s %s recovered a complete Add File artifact from "
            "an unterminated apply_patch envelope",
            label,
            req_id,
        )
        return completed_artifact
    bounded_scaffold = _synthesize_bounded_write_scaffold_text(
        full_output,
        tools,
    )
    if bounded_scaffold and _usable_tool_turn(
        bounded_scaffold,
        tool_module,
        tools,
        processed_messages,
        thinking_mode,
        require_call=require_call,
    ):
        logger.warning(
            "[rank 0] %s %s returning a bounded scaffold immediately after "
            "an oversized incomplete Write",
            label,
            req_id,
        )
        return bounded_scaffold
    ceiling = MAX_TOKENS_CEILING if MAX_TOKENS_CEILING > 0 else 16384
    retry_max_tokens = min(ceiling, max(int(max_tokens or 0), 2048) * 2)
    if TOOL_UNUSABLE_RETRY_MAX_TOKENS > 0:
        retry_max_tokens = min(retry_max_tokens, TOOL_UNUSABLE_RETRY_MAX_TOKENS)
    mutation_stop = _file_mutation_stop_info(full_output, tools) or {}
    oversized_mutation_recovery = bool(
        int(mutation_stop.get("payload_chars") or 0)
        > int(mutation_stop.get("threshold_chars") or 0)
        > 0
    )
    oversized_apply_patch = bool(
        oversized_mutation_recovery
        and mutation_stop.get("normalized_name") == "applypatch"
    )
    single_apply_patch = bool(_single_apply_patch_tool_name(tools))
    fast_apply_patch_recovery = _single_apply_patch_fast_recovery(
        tools,
        len(token_ids or []),
    )
    retry_attempts = TOOL_UNUSABLE_RETRY_ATTEMPTS
    if native_action_retry:
        retry_attempts = min(
            retry_attempts,
            NATIVE_TOOL_ACTION_RETRY_ATTEMPTS,
        )
    if oversized_mutation_recovery and not oversized_apply_patch:
        # A malformed giant mutation must not receive another full-size
        # runway to repeat the same payload. Atomic Write can become a small
        # scaffold. apply_patch must retain enough runway to close a complete
        # focused patch because partial patches are never safe to execute.
        retry_max_tokens = min(retry_max_tokens, 4096)
    if oversized_apply_patch or single_apply_patch:
        retry_attempts = min(retry_attempts, 2)
    logged_no_think_recovery = False
    retry_ram_reset_done = False
    for attempt in range(1, retry_attempts + 1):
        if should_abort and should_abort():
            return full_output
        retry_params = _tool_retry_gen_params(gen_params, attempt)
        snippet = re.sub(r"\s+", " ", full_output or "")
        logger.warning(
            "[rank 0] %s %s produced an unusable tool turn; retry %d/%d "
            "(temperature=%s, max_tokens=%s, raw_head=%r, raw_tail=%r)",
            label, req_id, attempt, retry_attempts,
            retry_params.get("temperature"), retry_max_tokens,
            snippet[:200], snippet[-160:],
        )
        # Preserve the original thinking template for the cheap recovery
        # attempts so the long prompt prefix remains reusable. Their separate
        # no-call guard keeps duplicate private reasoning bounded. Switch to
        # native no-thinking only for the final fallback, where prefix safety
        # deliberately releases incompatible RAM KV before a cold rebuild.
        hidden_no_think_recovery = bool(
            TOOL_COMPAT_OVERLAY
            and _enable_thinking_for_generation(thinking_mode)
            and (
                fast_apply_patch_recovery
                or _tool_retry_prefers_no_think(
                    thinking_mode,
                    attempt,
                    retry_attempts,
                    len(token_ids or []),
                )
            )
        )
        retry_thinking_mode = _tool_retry_thinking_mode(
            thinking_mode,
            prefer_no_think=hidden_no_think_recovery,
        )
        if hidden_no_think_recovery and not logged_no_think_recovery:
            logger.warning(
                "[rank 0] %s %s bounded hidden tool retries are using the "
                "no-thinking template after the visible thinking attempt",
                label,
                req_id,
            )
            logged_no_think_recovery = True
        retry_prompt = prompt
        retry_token_ids = token_ids
        recovery_hint = _tool_retry_recovery_hint(
            full_output,
            tool_module,
            tools,
            processed_messages,
        )
        if recovery_hint and not image_path:
            try:
                retry_prompt = _render_tool_retry_prompt(
                    model,
                    processor,
                    processed_messages,
                    tools,
                    retry_thinking_mode,
                    recovery_hint,
                ) or prompt
                rendered_token_ids = _tokenize_prompt(
                    processor,
                    retry_prompt,
                )
                if not rendered_token_ids:
                    raise RuntimeError(
                        "targeted retry prompt tokenization returned no ids"
                    )
                retry_token_ids = rendered_token_ids
                retry_prefix = _tool_retry_prefix_safety(
                    token_ids,
                    retry_token_ids,
                )
                logger.warning(
                    "[rank 0] %s %s tool retry %d using targeted "
                    "large-write recovery guidance (%d prompt tokens, "
                    "prefix=%d, reuse=%.4f)",
                    label,
                    req_id,
                    attempt,
                    len(retry_token_ids),
                    retry_prefix["common_prefix_tokens"],
                    retry_prefix["reuse_ratio"],
                )
                if retry_prefix["reset"]:
                    logger.error(
                        "[rank 0] %s %s tool retry %d unexpectedly collapsed "
                        "a long prompt prefix (%d/%d, ratio=%.4f); releasing "
                        "distributed RAM KV before cold retry",
                        label,
                        req_id,
                        attempt,
                        retry_prefix["common_prefix_tokens"],
                        min(
                            retry_prefix["original_tokens"],
                            retry_prefix["retry_tokens"],
                        ),
                        retry_prefix["reuse_ratio"],
                    )
                    _reset_prompt_cache_on_all_ranks(
                        rank,
                        reason="tool retry prefix collapse",
                        clear_memory=True,
                        clear_manifest=False,
                        clear_resident=True,
                    )
                    retry_ram_reset_done = True
            except Exception as e:
                logger.warning(
                    "[rank 0] %s %s could not render targeted tool retry "
                    "prompt; using original prompt: %s",
                    label,
                    req_id,
                    e,
                )
        retry_request = dict(rank_request)
        retry_request["gen_params"] = retry_params
        retry_request["max_tokens"] = retry_max_tokens
        retry_request["thinking_mode"] = retry_thinking_mode
        retry_request["prompt"] = retry_prompt
        retry_request["token_ids"] = retry_token_ids
        retry_no_call_budget = _tool_retry_no_call_budget(
            retry_thinking_mode,
            action_tool_task=action_tool_task,
            require_call=require_call,
        )
        retry_request["no_call_token_budget"] = retry_no_call_budget
        logger.info(
            "[rank 0] %s %s tool retry %d no-call budget=%d "
            "(mode=%s, action_task=%s, required=%s)",
            label,
            req_id,
            attempt,
            retry_no_call_budget,
            retry_thinking_mode,
            bool(action_tool_task),
            bool(require_call),
        )
        retry_ram_reset_reason = (
            _native_tool_retry_ram_reset_reason(
                full_output,
                len(retry_token_ids or []),
            )
            if native_action_retry and not retry_ram_reset_done
            else None
        )
        if retry_ram_reset_reason:
            logger.warning(
                "[rank 0] %s %s tool retry %d releasing active distributed "
                "RAM KV before recovery (%s); preserving SSD checkpoints",
                label,
                req_id,
                attempt,
                retry_ram_reset_reason,
            )
            _reset_prompt_cache_on_all_ranks(
                rank,
                reason=(
                    "native tool retry RAM reset:"
                    f"{retry_ram_reset_reason}"
                ),
                clear_memory=True,
                clear_manifest=False,
                clear_resident=False,
            )
            retry_ram_reset_done = True
        _clear_stop_request()
        _clear_prefill_stop_file("rank 0 tool retry")
        _bcast(retry_request, rank)
        try:
            full_output = run_generation(
                model, processor, retry_prompt, retry_max_tokens, rank,
                image=image_path, thinking_mode=retry_thinking_mode,
                gen_params=retry_params, progress_cb=progress_cb,
                token_ids=retry_token_ids,
                session_id=session_id, session_source=session_source,
                reset_incomplete_thinking_on_limit=False,
                tool_module=tool_module, tools=tools,
                require_tool_call=require_call,
                action_tool_task=action_tool_task,
                no_call_token_budget=retry_no_call_budget,
            )
        except Exception as e:
            logger.error(
                "[rank 0] %s %s tool retry %d failed: %s",
                label, req_id, attempt, e,
            )
            return full_output
        retry_artifact = _synthesize_complete_add_file_artifact_text(
            full_output,
            tools,
        )
        if retry_artifact and _usable_tool_turn(
            retry_artifact,
            tool_module,
            tools,
            processed_messages,
            retry_thinking_mode,
            require_call=require_call,
        ):
            logger.warning(
                "[rank 0] %s %s tool retry %d/%d recovered a complete "
                "Add File artifact from an unterminated apply_patch envelope",
                label,
                req_id,
                attempt,
                retry_attempts,
            )
            return retry_artifact
        retry_scaffold = _synthesize_bounded_write_scaffold_text(
            full_output,
            tools,
        )
        if retry_scaffold and _usable_tool_turn(
            retry_scaffold,
            tool_module,
            tools,
            processed_messages,
            retry_thinking_mode,
            require_call=require_call,
        ):
            logger.warning(
                "[rank 0] %s %s tool retry %d/%d was still an oversized "
                "Write; returning its bounded scaffold immediately",
                label,
                req_id,
                attempt,
                retry_attempts,
            )
            return retry_scaffold
        if _usable_tool_turn(full_output, tool_module, tools,
                             processed_messages, retry_thinking_mode,
                             require_call=require_call):
            logger.warning(
                "[rank 0] %s %s tool retry %d/%d recovered a usable tool turn "
                "(mode=%s)",
                label, req_id, attempt, retry_attempts,
                retry_thinking_mode,
            )
            return full_output
        if attempt < retry_attempts:
            # rank 1 returns to its request loop between mirrored retries and
            # clears transient Metal allocations there. Rank 0 stays inside
            # this helper, so give it the same cleanup cadence without touching
            # the live prompt cache backing arrays.
            try:
                mx.clear_cache()
            except Exception as e:
                logger.debug(
                    "[rank 0] %s %s tool retry transient cleanup failed: %s",
                    label,
                    req_id,
                    e,
                )
            gc.collect()
    explicit_read = _synthesize_explicit_read_tool_call(
        processed_messages,
        tools,
    )
    explicit_read_text = _tool_call_as_display_text(explicit_read)
    if explicit_read_text and _usable_tool_turn(
        explicit_read_text,
        tool_module,
        tools,
        processed_messages,
        thinking_mode,
        require_call=require_call,
    ):
        logger.warning(
            "[rank 0] %s %s synthesized the explicit user-requested read "
            "after %d malformed retries",
            label,
            req_id,
            retry_attempts,
        )
        return explicit_read_text
    snippet = re.sub(r"\s+", " ", full_output or "")
    logger.warning(
        "[rank 0] %s %s tool turn still unusable after %d retries; "
        "sending compatibility fallback (raw_head=%r, raw_tail=%r)",
        label, req_id, retry_attempts,
        snippet[:200], snippet[-160:],
    )
    # Forensics: every exhausted ladder means a markup flavor all parse
    # rungs miss. Persist the raw output so the next flavor is diagnosed
    # from evidence (the 2026-07-09 pandoc turn burned 3x8k tokens and left
    # only a 200-char head in the log).
    try:
        _fail_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "ops", "logs", "tool_parse_failures")
        os.makedirs(_fail_dir, exist_ok=True)
        with open(os.path.join(_fail_dir, f"{req_id}.txt"), "w") as _fh:
            _fh.write(full_output or "")
    except Exception:
        pass
    return full_output


def _validate_outgoing_tool_calls(
    tool_calls,
    tools,
    *,
    return_dropped=False,
    return_dropped_names=False,
    processed_messages=None,
    raw_output=None,
):
    """Return OpenAI tool_calls that match the submitted schema exactly.

    The MiniMax parser and recovery paths can infer aliases such as
    `invoke_command`, `cmd`, or `shell`. Some agent shims are stricter than
    OpenAI here and stop if the returned function name is not byte-for-byte one
    of the names they advertised. Canonicalize once at the boundary.
    """
    if not tool_calls:
        if return_dropped and return_dropped_names:
            return [], 0, []
        return ([], 0) if return_dropped else []
    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    successful_mutations = _successful_exact_mutation_fingerprints(
        processed_messages
    )
    validated = []
    dropped = 0
    dropped_names = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            dropped += 1
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw_name = fn.get("name") or call.get("name") or call.get("tool")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                decoded = {"input": arguments}
        elif isinstance(arguments, dict):
            decoded = arguments
        else:
            decoded = {}

        if not TOOL_COMPAT_OVERLAY:
            name = raw_name if isinstance(raw_name, str) else ""
            if not name or (allowed and name not in allowed):
                logger.warning("dropping outgoing tool call with unknown name=%r", raw_name)
                dropped += 1
                if raw_name:
                    dropped_names.append(str(raw_name))
                continue
            # Match oMLX / mlx-vlm native behavior: preserve a complete call
            # and let the client execute it or return an ordinary tool error.
            # Keep only lossless schema shaping. Native mode must not rewrite
            # paths, invent fields, bound payloads, suppress mutations, or
            # synthesize a different command.
            if isinstance(arguments, str):
                try:
                    native_decoded = (
                        json.loads(arguments) if arguments.strip() else {}
                    )
                except json.JSONDecodeError:
                    native_decoded = None
            elif isinstance(arguments, dict):
                native_decoded = dict(arguments)
            else:
                native_decoded = None
            if isinstance(native_decoded, dict):
                native_decoded = _canonicalize_tool_argument_keys(
                    native_decoded, tools, name
                )
                native_decoded = _coerce_json_encoded_schema_values(
                    native_decoded, tools, name
                )
                if call.get("_m3_schema_recovered"):
                    mismatches = _tool_schema_type_mismatches(
                        native_decoded, tools, name
                    )
                    required = _tool_schema_required_names(tools, name)
                    missing = [
                        key for key in required
                        if key not in native_decoded
                        or native_decoded.get(key) in (None, "")
                    ]
                    if mismatches or missing:
                        logger.warning(
                            "dropping schema-invalid recovered %s call "
                            "(mismatches=%s, missing=%s)",
                            name,
                            mismatches,
                            missing,
                        )
                        dropped += 1
                        dropped_names.append(str(name))
                        continue
                native_arguments = json.dumps(
                    native_decoded, ensure_ascii=False
                )
            elif isinstance(arguments, str):
                native_arguments = arguments
            else:
                native_arguments = json.dumps(arguments, ensure_ascii=False)
            validated.append({
                "type": "function",
                "index": index,
                "id": call.get("id") or str(uuid.uuid4()),
                "function": {
                    "name": name,
                    "arguments": native_arguments,
                },
            })
            continue

        name = _canonical_tool_name(raw_name, name_map)
        if not name or (allowed and name not in allowed):
            coerced_name, coerced_args = _coerce_file_tool_to_command(
                raw_name,
                decoded,
                tools,
            )
            if not (coerced_name and coerced_args) and _is_apply_patch_tool_name(raw_name):
                # apply_patch may be hidden from the schema on purpose; keep
                # honoring add-file intents by rewriting them as shell writes.
                coerced_name, coerced_args = _exec_write_from_malformed_patch(
                    decoded,
                    tools,
                )
            if coerced_name and coerced_args:
                logger.warning(
                    "coerced outgoing %s tool call to %s",
                    raw_name,
                    coerced_name,
                )
                name = coerced_name
                decoded = coerced_args
            else:
                logger.warning("dropping outgoing tool call with unknown name=%r", raw_name)
                dropped += 1
                if raw_name:
                    dropped_names.append(str(raw_name))
                continue
        decoded = _canonicalize_tool_argument_keys(decoded, tools, name)
        decoded = _coerce_json_encoded_schema_values(decoded, tools, name)
        decoded = _coerce_codex_control_tool_arguments(decoded, tools, name)
        decoded, stripped_path_tags = _strip_minimax_closing_tags_from_paths(
            decoded
        )
        for key, source, target in stripped_path_tags:
            logger.warning(
                "stripped MiniMax closing tag from outgoing %s.%s: %r -> %r",
                name,
                key,
                source,
                target,
            )
        decoded, stripped_payload_tags = (
            _strip_minimax_closing_tags_from_payloads(name, decoded)
        )
        for key, source_len, target_len in stripped_payload_tags:
            logger.warning(
                "stripped MiniMax closing fragment from outgoing %s.%s "
                "(%d -> %d characters)",
                name,
                key,
                source_len,
                target_len,
            )
        decoded, reversed_write = _repair_reversed_write_path_and_content(
            name,
            decoded,
        )
        if reversed_write:
            logger.warning(
                "swapped reversed outgoing %s %s/%s arguments for %r "
                "(%d payload characters)",
                name,
                reversed_write["path_key"],
                reversed_write["content_key"],
                reversed_write["path"],
                reversed_write["payload_chars"],
            )
        decoded, relative_user_paths = (
            _anchor_file_tool_path_to_user_relative_target(
                name,
                decoded,
                processed_messages,
            )
        )
        for key, source, target in relative_user_paths:
            logger.warning(
                "anchored outgoing %s.%s from %r to user workspace relative "
                "path %r",
                name,
                key,
                source,
                target,
            )
        decoded, reversed_edit = _repair_reversed_edit_arguments_after_failure(
            name,
            decoded,
            processed_messages,
        )
        if reversed_edit:
            logger.warning(
                "swapped outgoing %s %s/%s arguments after a prior failed "
                "edit matched the client's Read snapshot for %r",
                name,
                reversed_edit["old_key"],
                reversed_edit["new_key"],
                reversed_edit["path"],
            )
        decoded, filled_paths = _fill_missing_mutating_tool_path(
            name,
            decoded,
            tools,
            processed_messages,
            raw_output,
        )
        for key, target in filled_paths:
            logger.warning(
                "filled missing outgoing %s.%s from the explicit client "
                "target %r",
                name,
                key,
                target,
            )
        decoded, intent_anchored_paths = _anchor_mutating_tool_path_from_named_read(
            name,
            decoded,
            processed_messages,
            raw_output,
        )
        for key, source, target in intent_anchored_paths:
            logger.warning(
                "anchored outgoing %s.%s from %r to named read target %r",
                name,
                key,
                source,
                target,
            )
        decoded, basename_anchored_paths = (
            _anchor_mutating_tool_path_from_read_basename(
                name,
                decoded,
                processed_messages,
            )
        )
        for key, source, target in basename_anchored_paths:
            logger.warning(
                "anchored outgoing %s.%s from %r to unique read target %r",
                name,
                key,
                source,
                target,
            )
        decoded, anchored_paths = _anchor_mutating_tool_paths(
            name,
            decoded,
            processed_messages,
        )
        for key, source, target in anchored_paths:
            logger.warning(
                "anchored outgoing %s.%s from %r to client path %r",
                name,
                key,
                source,
                target,
            )
        decoded, anchored_workdirs = _anchor_command_working_directory(
            name,
            decoded,
            tools,
            processed_messages,
        )
        for key, source, target in anchored_workdirs:
            logger.warning(
                "anchored outgoing %s.%s from %r to client working "
                "directory %r",
                name,
                key,
                source,
                target,
            )
        decoded, anchored_command_paths = _anchor_command_paths_from_read_history(
            name,
            decoded,
            tools,
            processed_messages,
        )
        for key, source, target in anchored_command_paths:
            logger.warning(
                "anchored outgoing %s.%s path from %r to proven read "
                "target %r",
                name,
                key,
                source,
                target,
            )
        decoded, repaired_python_command = _repair_non_executable_python_command(
            name,
            decoded,
            tools,
            processed_messages,
        )
        if repaired_python_command:
            logger.warning(
                "repaired outgoing %s.%s from %r to %r after proven "
                "permission-denied result",
                name,
                repaired_python_command["key"],
                repaired_python_command["source"],
                repaired_python_command["target"],
            )
        mutation_fingerprint = _exact_mutation_fingerprint(name, decoded)
        if mutation_fingerprint in successful_mutations:
            logger.warning(
                "dropping exact duplicate %s mutation that already completed "
                "successfully in this user turn (path=%r)",
                name,
                _file_write_path(decoded),
            )
            dropped += 1
            dropped_names.append(str(name))
            continue
        decoded, oversized_write_chars = _bound_large_file_write_arguments(
            name,
            decoded,
        )
        if oversized_write_chars:
            logger.warning(
                "bounded outgoing %s content from %d characters to a small "
                "scaffold; continue through subsequent Edit calls",
                name,
                oversized_write_chars,
            )
        type_mismatches = _tool_schema_type_mismatches(
            decoded,
            tools,
            name,
        )
        if type_mismatches:
            logger.warning(
                "dropping outgoing %s tool call with schema type "
                "mismatches=%s",
                name,
                type_mismatches,
            )
            dropped += 1
            dropped_names.append(str(name))
            continue
        if not decoded and _tool_schema_expects_arguments(tools, name):
            logger.warning("dropping outgoing %s tool call with empty arguments", name)
            dropped += 1
            if name:
                dropped_names.append(str(name))
            continue
        required = _tool_schema_required_names(tools, name)
        missing_required = [
            key for key in required
            if key not in decoded or decoded.get(key) in (None, "")
        ]
        if missing_required:
            logger.warning(
                "dropping outgoing %s tool call missing required arguments=%s",
                name,
                missing_required,
            )
            dropped += 1
            if name:
                dropped_names.append(str(name))
            continue
        oversized_mutation = _oversized_mutating_file_payload(name, decoded)
        if oversized_mutation:
            logger.warning(
                "dropping outgoing %s tool call with oversized %s=%d "
                "characters (budget=%d)",
                name,
                oversized_mutation[0],
                oversized_mutation[1],
                TOOL_WRITE_CHUNK_MAX_CHARS,
            )
            dropped += 1
            dropped_names.append(str(name))
            continue
        command_violation = _command_tool_payload_violation(
            name,
            decoded,
            tools,
        )
        if command_violation:
            logger.warning(
                "dropping outgoing %s tool call: %s",
                name,
                command_violation,
            )
            dropped += 1
            dropped_names.append(str(name))
            continue
        if _is_apply_patch_tool_name(name) and not _apply_patch_payload_is_valid(decoded):
            coerced_name, coerced_args = _exec_write_from_malformed_patch(decoded, tools)
            if coerced_name and coerced_args:
                logger.warning(
                    "coerced malformed apply_patch add-file payload into %s write",
                    coerced_name,
                )
                validated.append({
                    "type": "function",
                    "index": index,
                    "id": call.get("id") or str(uuid.uuid4()),
                    "function": {
                        "name": coerced_name,
                        "arguments": json.dumps(coerced_args, ensure_ascii=False),
                    },
                })
                continue
            logger.warning("dropping outgoing apply_patch tool call with malformed patch payload")
            dropped += 1
            dropped_names.append(str(name))
            continue
        path_violation = _tool_request_path_violation(
            name, decoded, processed_messages
        )
        if path_violation:
            logger.warning("dropping outgoing tool call: %s", path_violation)
            dropped += 1
            dropped_names.append(str(name))
            continue
        validated.append({
            "type": "function",
            "index": index,
            "id": call.get("id") or str(uuid.uuid4()),
            "function": {
                "name": name,
                "arguments": json.dumps(decoded, ensure_ascii=False),
            },
        })
    if return_dropped and return_dropped_names:
        return validated, dropped, dropped_names
    return (validated, dropped) if return_dropped else validated


def _tool_call_names(tool_calls):
    names = []
    for call in tool_calls or []:
        fn = call.get("function") if isinstance(call, dict) else None
        if isinstance(fn, dict):
            names.append(str(fn.get("name") or ""))
    return [name for name in names if name]


def _tool_call_arg_keys(tool_calls):
    keys = []
    for call in tool_calls or []:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict):
            keys.append([])
            continue
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        keys.append(sorted(args.keys()) if isinstance(args, dict) else [])
    return keys


def _sanitize_inbound_tool_call_content(message, content_text):
    if isinstance(message, dict) and message.get("tool_calls"):
        return ""
    return content_text


def _tool_call_blocks(text, tool_module):
    start, end = _tool_call_markers(tool_module)
    if not text or not start:
        return []
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if start not in text:
        if ns_token and _looks_like_raw_tool_fragment(text, tool_module):
            return [text[text.find(ns_token):]]
        return []
    if end:
        pattern = re.compile(
            f"{re.escape(start)}(?P<body>.*?){re.escape(end)}",
            flags=re.DOTALL,
        )
        blocks = [m.group("body") for m in pattern.finditer(text)]
        if blocks:
            return blocks
        # Agent-shaped MiniMax outputs sometimes close the inner <invoke> but
        # omit </tool_call>. Inspect the trailing block only after the explicit
        # start marker so normal prose is not treated as a tool.
        start_pos = text.find(start)
        if start_pos >= 0:
            return [text[start_pos + len(start):]]
        return []
    pattern = re.compile(f"{re.escape(start)}(?P<body>.*?)(?:\n|$)", re.DOTALL)
    return [m.group("body") for m in pattern.finditer(text)]


def _clean_loose_tool_segment(segment):
    segment = (segment or "").strip()
    if not segment:
        return ""
    if re.fullmatch(r"[\[\]>\s]*", segment):
        return ""
    if (
        len(segment) >= 2
        and segment.startswith("[")
        and segment.endswith("]")
        and not segment.startswith("[Tool call:")
    ):
        segment = segment[1:-1].strip()
        if not segment or re.fullmatch(r"[\[\]>\s]*", segment):
            return ""
    if segment.startswith("</"):
        return ""
    segment = re.sub(r"^<(?P<tag>[A-Za-z_$][\w:.$-]*)(?:\s[^>]*)?>", "", segment).strip()
    segment = re.sub(r"</(?P<tag>[A-Za-z_$][\w:.$-]*)>$", "", segment).strip()
    if re.fullmatch(r"[\[\]>\s]*", segment):
        return ""
    return segment


def _loose_tool_segments(body, ns_token):
    if not body or not ns_token:
        return []
    segments = []
    for part in body.split(ns_token):
        cleaned = _clean_loose_tool_segment(part)
        if cleaned and not cleaned.startswith("<"):
            segments.append(cleaned)
    return segments


def _looks_like_shell_command(value):
    text = (value or "").strip()
    if not text:
        return False
    first = text.split(maxsplit=1)[0]
    commands = {
        "ls", "pwd", "cat", "rg", "grep", "find", "git", "python",
        "python3", "node", "npm", "pnpm", "yarn", "uv", "pytest",
        "sed", "awk", "curl", "mkdir", "touch", "cp", "mv", "rm",
    }
    return (
        first in commands
        or text.startswith(("./", "../", "/", "~"))
        or ("/" in text and (" " in text or '"' in text or "'" in text))
    )


def _normalize_loose_shell_command(cmd):
    text = (cmd or "").strip()
    if not text:
        return ""
    # MiniMax occasionally emits typographic dashes in bare command fragments.
    # Keep this narrow: only normalize a dash immediately before an option word.
    text = re.sub(r"(?<=\s)[\u2010-\u2015](?=[A-Za-z])", "--", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def _loose_shell_command_is_complete(cmd):
    text = _normalize_loose_shell_command(cmd)
    if not text:
        return False
    if re.search(r"(?:&&|\|\||\||;|\\)\s*$", text):
        return False
    try:
        parts = shlex.split(text)
    except ValueError:
        return False
    if not parts:
        return False
    first = os.path.basename(parts[0])
    if first in {"python", "python3"} and "-c" in parts:
        idx = parts.index("-c")
        if idx + 1 >= len(parts):
            return False
        code = parts[idx + 1].strip()
        # A truncated Python -c body such as "import" or "from" is almost
        # always a partial MiniMax tool fragment, not an executable intent.
        if code in {"import", "from", "print", "def", "class", "with", "for", "if"}:
            return False
    return True


_NS_ARG_TAG_RE = re.compile(r"<(?P<close>/)?(?P<tag>[A-Za-z_$][\w:.$-]*)\s*>")


def _nested_array_args_from_ns_tags(raw_body, ns_token, tools, name):
    """Recover one schema-declared array of XML ``<item>`` objects.

    MiniMax/OpenCode sometimes emits ``<invoke=\"todowrite\">`` followed by
    ``<todos><item><content>...`` instead of JSON. Keep this schema-driven:
    only explicit array/object properties and their declared child fields are
    accepted, so arbitrary nested markup cannot become a tool invocation.
    """
    if not raw_body or not ns_token:
        return None
    specs = _tool_schema_property_specs(tools, name)
    array_props = [
        (key, spec)
        for key, spec in specs.items()
        if isinstance(spec, dict)
        and spec.get("type") == "array"
        and isinstance(spec.get("items"), dict)
        and spec["items"].get("type") == "object"
    ]
    if len(array_props) != 1:
        return None
    prop, spec = array_props[0]
    open_tag = f"{ns_token}<{prop}>"
    close_tag = f"{ns_token}</{prop}>"
    item_open = f"{ns_token}<item>"
    item_close = f"{ns_token}</item>"
    start = raw_body.find(open_tag)
    opener_len = len(open_tag)
    end = raw_body.find(close_tag, start + opener_len) if start >= 0 else -1
    if start < 0:
        # Real ZCode long-turn drift observed 2026-07-12: the array opener
        # alone was reversed (``</todos>``), while its item objects and the
        # final array close remained complete. Require two explicit container
        # closers with an item boundary between them; a lone close can never
        # be promoted into an opener.
        first_close = raw_body.find(close_tag)
        second_close = (
            raw_body.find(close_tag, first_close + len(close_tag))
            if first_close >= 0 else -1
        )
        between = (
            raw_body[first_close + len(close_tag):second_close]
            if second_close >= 0 else ""
        )
        if second_close >= 0 and (item_open in between or item_close in between):
            start = first_close
            opener_len = len(close_tag)
            end = second_close
    wrapperless_items = False
    if start < 0 or end < 0:
        # Native MiniMax retry drift observed with ZCode TodoWrite: the model
        # emitted several complete schema-shaped item groups and explicit
        # </item> boundaries, but omitted only the outer ``<todos>`` wrapper.
        # Recover this only when the tool has exactly one required array field
        # and at least two item boundaries. A lone loose object remains too
        # ambiguous and is regenerated by the bounded native retry instead.
        top_required = set(_tool_schema_required_names(tools, name))
        if (
            prop not in top_required
            or raw_body.count(item_close) < 2
            or item_open in raw_body
        ):
            return None
        container = raw_body
        wrapperless_items = True
    else:
        container = raw_body[start + opener_len:end]
    item_spec = spec.get("items") or {}
    child_specs = item_spec.get("properties")
    child_specs = child_specs if isinstance(child_specs, dict) else {}
    child_required = item_spec.get("required")
    child_required = (
        [key for key in child_required if isinstance(key, str)]
        if isinstance(child_required, list) else []
    )
    if not child_specs:
        return None
    item_bodies = []
    if item_open in container:
        cursor = 0
        while True:
            item_start = container.find(item_open, cursor)
            if item_start < 0:
                break
            item_end = container.find(item_close, item_start + len(item_open))
            if item_end < 0:
                return None
            item_bodies.append(
                container[item_start + len(item_open):item_end]
            )
            cursor = item_end + len(item_close)
    elif item_close in container:
        # ZCode long-turn wire form observed 2026-07-12: MiniMax reversed
        # every ``<item>`` opener to ``</item>`` while preserving complete,
        # schema-named child fields and the final container close. Splitting
        # between those explicit structural closers recovers the same atomic
        # objects without guessing any content.
        child_names = {str(key).lower() for key in child_specs}
        for candidate in container.split(item_close):
            structural_names = {
                match.group("tag").lower()
                for match in re.finditer(
                    rf"{re.escape(ns_token)}<(?P<tag>[A-Za-z_$][\w:.$-]*)\s*>",
                    candidate,
                )
            }
            if structural_names & child_names:
                item_bodies.append(candidate)
    if not item_bodies:
        return None

    items = []
    for item_body in item_bodies:
        item = {}
        for child, child_spec in child_specs.items():
            child_open = re.search(
                rf"{re.escape(ns_token)}<{re.escape(child)}\s*>",
                item_body,
                flags=re.IGNORECASE,
            )
            if child_open is None:
                continue
            child_close = re.search(
                rf"{re.escape(ns_token)}</{re.escape(child)}\s*>",
                item_body[child_open.end():],
                flags=re.IGNORECASE,
            )
            if child_close is None:
                continue
            value = item_body[
                child_open.end():child_open.end() + child_close.start()
            ].strip()
            expected = child_spec.get("type") if isinstance(child_spec, dict) else None
            if expected == "integer":
                try:
                    value = int(value)
                except ValueError:
                    continue
            elif expected == "number":
                try:
                    value = float(value)
                except ValueError:
                    continue
            elif expected == "boolean":
                lowered = value.lower()
                if lowered not in {"true", "false"}:
                    continue
                value = lowered == "true"
            allowed = (
                child_spec.get("enum")
                if isinstance(child_spec, dict) else None
            )
            if isinstance(allowed, list) and allowed and value not in allowed:
                continue
            item[child] = value
        # ZCode occasionally omits only the display-oriented activeForm on a
        # later todo item while still emitting complete content/status. Its
        # own schema requires activeForm, so derive that label from content
        # rather than dropping the entire otherwise valid todo array.
        for child in child_required:
            if item.get(child) not in (None, ""):
                continue
            compact = re.sub(r"[^a-z0-9]", "", child.lower())
            if compact == "activeform" and item.get("content"):
                item[child] = str(item["content"])
        if any(item.get(key) in (None, "") for key in child_required):
            return None
        if item:
            items.append(item)
    result = {prop: items} if items else None
    if result and _tool_schema_type_mismatches(result, tools, name):
        return None
    if wrapperless_items and len(items) < 2:
        return None
    return result


def _labeled_json_array_args_from_body(raw_body, tools, name):
    """Recover ``property: [...]`` for one schema-declared array field."""
    if not raw_body:
        return None
    specs = _tool_schema_property_specs(tools, name)
    array_props = [
        key for key, spec in specs.items()
        if isinstance(spec, dict) and spec.get("type") == "array"
    ]
    if len(array_props) != 1:
        return None
    prop = array_props[0]
    label = re.search(
        rf"(?i)(?:^|[\s>{{]){re.escape(prop)}\s*:\s*",
        raw_body,
    )
    if label is None:
        return None
    start = raw_body.find("[", label.end())
    if start < 0:
        return None
    end = _json_balanced_end(raw_body, start)
    if end <= start:
        return None
    try:
        value = json.loads(raw_body[start:end])
    except json.JSONDecodeError:
        try:
            value = json.loads(raw_body[start:end], strict=False)
        except json.JSONDecodeError:
            return None
    return {prop: value} if isinstance(value, list) else None


def _legacy_question_args_from_body(raw_body, ns_token, tools, name):
    """Recover an explicit bare question as one schema-valid free-form prompt.

    MiniMax occasionally emits ``<invoke><question>...</question></invoke>``
    for OpenCode's declared question tool, omitting the outer ``questions``
    array. The user-facing question is complete and atomic, but no answer
    choices were emitted. Preserve that intent without inventing choices by
    sending an empty options array; OpenCode then exposes its normal custom
    answer field.
    """
    if not raw_body or not ns_token:
        return None
    specs = _tool_schema_property_specs(tools, name)
    array_fields = [
        (key, spec)
        for key, spec in specs.items()
        if re.sub(r"[^a-z0-9]", "", key.lower()) == "questions"
        and isinstance(spec, dict)
        and spec.get("type") == "array"
        and isinstance(spec.get("items"), dict)
        and spec["items"].get("type") == "object"
    ]
    if len(array_fields) != 1:
        return None
    field, array_spec = array_fields[0]
    item_spec = array_spec.get("items") or {}
    child_specs = item_spec.get("properties")
    child_specs = child_specs if isinstance(child_specs, dict) else {}
    if not child_specs:
        return None
    child_by_compact = {
        re.sub(r"[^a-z0-9]", "", key.lower()): key
        for key in child_specs
    }
    question_key = child_by_compact.get("question")
    header_key = child_by_compact.get("header")
    options_key = child_by_compact.get("options")
    if not (question_key and header_key and options_key):
        return None

    def _tag_value(tag):
        match = re.search(
            rf"(?is)(?:{re.escape(ns_token)})?"
            rf"<{re.escape(tag)}\b[^>]*>"
            rf"(?P<value>.*?)"
            rf"(?:{re.escape(ns_token)})?"
            rf"</{re.escape(tag)}\s*>",
            raw_body,
        )
        if match is None:
            return ""
        value = match.group("value").strip()
        return value if value and "<" not in value and ">" not in value else ""

    question = _tag_value("question")
    if not question or len(question) > 2000:
        return None
    header = _tag_value("header")
    if not header:
        header_words = re.findall(r"[A-Za-z0-9_./+#-]+", question)[:4]
        header = " ".join(header_words) or "Clarification"
    header = header[:30].strip() or "Clarification"
    item = {
        question_key: question,
        header_key: header,
        options_key: [],
    }
    child_required = item_spec.get("required")
    child_required = child_required if isinstance(child_required, list) else []
    for child in child_required:
        if not isinstance(child, str) or item.get(child) not in (None, ""):
            continue
        spec = child_specs.get(child)
        expected = spec.get("type") if isinstance(spec, dict) else None
        if expected == "boolean":
            item[child] = False
        elif expected == "array":
            item[child] = []
        else:
            return None
    args = {field: [item]}
    if _tool_schema_type_mismatches(args, tools, name):
        return None
    return args


def _arguments_from_ns_arg_tags(raw_body, ns_token, tools, name):
    """Drift-tolerant extractor for MiniMax-M3 direct arg tags.

    Well-formed wire format: every STRUCTURAL tag is ns-prefixed
    (ns<path>value ns</path>); anything not behind the ns token — bare HTML,
    JSX, generics — is value content by contract. The stock parser enforces
    this strictly, so a drifted block (bare closer, missing closer) lands in
    the repair chain, whose _normalize_body rewrite ns-prefixes tags INSIDE
    values and shreds markup payloads (2026-07-06 audit, SEV-3). This
    extractor splits on the ns token instead: values round-trip byte-exact,
    the only repairs are stripping a bare same-name closer off a value tail
    and closing a dangling arg at block end.
    """
    if not raw_body or ns_token not in raw_body:
        return None
    props = _tool_schema_property_names(tools, name)
    if not props:
        return None
    args = {}
    current = None
    parts = []

    def _finish():
        val = "".join(parts)
        stripped_tail = val.rstrip()
        bare_close = f"</{current}>"
        if stripped_tail.endswith(bare_close):
            val = stripped_tail[: -len(bare_close)]
        return val.strip()

    for i, segment in enumerate(raw_body.split(ns_token)):
        m = _NS_ARG_TAG_RE.match(segment) if segment else None
        if m and not m.group("close") and m.group("tag") not in ("invoke", "tool_call"):
            if current is not None:
                args[current] = _finish()
            current = m.group("tag")
            parts = [segment[m.end():]]
        elif m and m.group("close"):
            if current is not None and m.group("tag") in (current, "invoke", "tool_call"):
                args[current] = _finish()
                current = None
                parts = []
            elif current is not None:
                # bare closer for something else mid-value: content
                parts.append(ns_token + segment)
        else:
            if current is not None:
                # literal ns token inside a value — restore it verbatim
                parts.append(ns_token + segment)
            # else: preamble/whitespace between args — drift noise
    if current is not None:
        args[current] = _finish()
    if not args:
        return None
    # Only accept when the extracted keys look like this tool's schema —
    # this is a repair path, not a place to invent arguments.
    lower_props = {p.lower() for p in props}
    matched = sum(1 for k in args if k.lower() in lower_props)
    if not matched or matched < len(args) / 2:
        return None
    return args


def _arguments_from_named_parameter_tags(raw_body, ns_token, tools, name):
    """Recover MiniMax's hybrid ``<parameter name=...>`` argument format.

    Long ZCode Edit turns sometimes emit a bare parameter opener, a namespaced
    closer, and a closer whose tag name does not match the opener. Structural
    closers remain namespaced, so split only on those boundaries and preserve
    HTML/CSS/JS inside argument values byte-for-byte.
    """
    if not raw_body or not ns_token or "<parameter" not in raw_body:
        return None
    opener = re.compile(
        rf"(?is)(?:{re.escape(ns_token)})?<parameter\b[^>]*"
        r"\bname\s*=\s*(?:([\"'])(?P<quoted>.*?)\1|(?P<bare>[^\s>]+))"
        r"[^>]*>",
    )
    matches = list(opener.finditer(raw_body))
    if not matches:
        return None
    args = {}
    for index, match in enumerate(matches):
        raw_name = (match.group("quoted") or match.group("bare") or "").strip()
        if not raw_name:
            continue
        next_opener = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(raw_body)
        )
        close_match = re.search(
            rf"(?is){re.escape(ns_token)}</[A-Za-z_$][\w:.$-]*\s*>",
            raw_body[match.end():next_opener],
        )
        if close_match is None:
            # A bare </parameter> is still unambiguous; bare payload markup is
            # not, so no other bare closer is accepted here.
            close_match = re.search(
                r"(?is)</parameter\s*>",
                raw_body[match.end():next_opener],
            )
        if close_match is None:
            continue
        value_start = match.end()
        value_end = value_start + close_match.start()
        args[raw_name] = raw_body[value_start:value_end]
    if not args:
        return None
    args = _canonicalize_tool_argument_keys(args, tools, name)
    args = _coerce_json_encoded_schema_values(args, tools, name)
    required = _tool_schema_required_names(tools, name)
    if any(args.get(key) in (None, "") for key in required):
        return None
    if _tool_schema_type_mismatches(args, tools, name):
        return None
    return args


def _positional_recovery_is_underspecified(raw_body, ns_token, tools, name):
    """Reject positional XML recovery that cannot cover required fields.

    The MLX-VLM parser can map a lone ``param-1`` payload onto more than one
    required Edit argument. That creates a superficially valid call whose
    old/new strings were never both emitted, so strict clients execute a bad
    edit and enter a repair loop. Explicitly named required fields plus the
    number of positional values must be sufficient before this permissive
    recovery path is allowed to run.
    """
    if not raw_body or not ns_token:
        return False
    required = _tool_schema_required_names(tools, name)
    if len(required) <= 1:
        return False
    positional_tags = re.findall(
        rf"{re.escape(ns_token)}<param(?:eter)?[-_]?\d+\b[^>]*>",
        raw_body,
        flags=re.IGNORECASE,
    )
    if not positional_tags:
        return False
    required_by_compact = {
        re.sub(r"[^a-z0-9]", "", field.lower()): field
        for field in required
    }
    explicit_required = set()
    for match in re.finditer(
        rf"{re.escape(ns_token)}<(?P<tag>[A-Za-z_$][\w:.$-]*)\b[^>]*>",
        raw_body,
    ):
        compact = re.sub(r"[^a-z0-9]", "", match.group("tag").lower())
        field = required_by_compact.get(compact)
        if field:
            explicit_required.add(field)
    return len(explicit_required) + len(positional_tags) < len(required)


def _arguments_from_loose_segments(body, ns_token, tools, name):
    segments = _loose_tool_segments(body, ns_token)
    if not segments:
        return None
    props = _tool_schema_property_names(tools, name)
    required = _tool_schema_required_names(tools, name)
    lower_props = {p.lower(): p for p in props}

    command_prop = next(
        (
            lower_props[prop]
            for prop in ("command", "cmd", "input")
            if prop in lower_props
        ),
        None,
    )
    if name and command_prop and (
        name.lower() in {"bash", "shell"}
        or _canonical_tool_name(name, _tool_name_map_from_schema(tools)) == name
    ):
        command_idx = None
        for i, segment in enumerate(segments):
            if _looks_like_shell_command(segment):
                command_idx = i
                break
        if command_idx is None:
            command_idx = len(segments) - 1
        command = _normalize_loose_shell_command(segments[command_idx])
        if not _loose_shell_command_is_complete(command):
            return None
        args = {command_prop: command}
        if "description" in lower_props:
            desc = next((s for i, s in enumerate(segments) if i != command_idx), "")
            if desc:
                args[lower_props["description"]] = desc
        if "justification" in lower_props:
            desc = next((s for i, s in enumerate(segments) if i != command_idx), "")
            if desc:
                args[lower_props["justification"]] = desc
        if not args.get(command_prop):
            return None
        return args

    ordered = []
    for field in required + props:
        if field not in ordered:
            ordered.append(field)
    if not ordered:
        return {"input": segments[0]} if len(segments) == 1 else {"items": segments}
    args = {}
    for field, value in zip(ordered, segments):
        args[field] = value
    if any(not args.get(field) for field in required):
        return None
    return args


def _json_balanced_end(text, start):
    if start < 0 or start >= len(text) or text[start] not in "{[":
        return -1
    stack = [text[start]]
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    for i in range(start + 1, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if not stack or pairs.get(stack[-1]) != ch:
                return -1
            stack.pop()
            if not stack:
                return i + 1
    return -1


def _loads_json_string_fragment(fragment):
    try:
        return json.loads(f'"{fragment}"')
    except Exception:
        return fragment.replace('\\"', '"').replace("\\\\", "\\")


def _coerce_tool_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            return {"input": arguments}
    return {}


def _tool_call_from_json_object(obj, tools, name_map, index):
    if not isinstance(obj, dict):
        return None
    raw_name = obj.get("name") or obj.get("tool") or obj.get("function")
    if isinstance(raw_name, dict):
        raw_name = raw_name.get("name")
    name = _canonical_tool_name(raw_name, name_map)
    allowed = set(name_map.values())
    if not name or (allowed and name not in allowed):
        return None
    arguments = (
        obj.get("arguments")
        if "arguments" in obj
        else obj.get("args", obj.get("parameters", obj.get("input", {})))
    )
    arguments = _coerce_tool_arguments(arguments)
    arguments = _canonicalize_tool_argument_keys(arguments, tools, name)
    if not arguments and _tool_schema_expects_arguments(tools, name):
        return None
    return _openai_tool_call(name, arguments, index)


def _recover_complete_json_unclosed_tool_call(text, tool_module, tools):
    """Recover an atomic JSON call whose outer MiniMax tags were truncated.

    A complete self-describing JSON object is an atomic tool payload even when
    MiniMax omits the surrounding ``</invoke></tool_call>`` markers. Accept
    only a declared tool plus a balanced object at the end of the open block;
    partial JSON and any substantive trailing bytes remain retry-only.
    """
    if not text or tool_module is None:
        return []
    start, _ = _tool_call_markers(tool_module)
    marker_at = text.rfind(start) if start else -1
    if marker_at < 0:
        return []
    tail = text[marker_at:]
    ns_token = start.removesuffix("<tool_call>")
    name_map = _tool_name_map_from_schema(tools)
    pos = 0
    while True:
        obj_start = tail.find("{", pos)
        if obj_start < 0:
            return []
        obj_end = _json_balanced_end(tail, obj_start)
        if obj_end <= obj_start:
            return []
        payload = tail[obj_start:obj_end]
        obj = None
        for strict in (True, False):
            try:
                obj = json.loads(payload, strict=strict)
                break
            except json.JSONDecodeError:
                continue
        if not isinstance(obj, dict):
            pos = obj_end
            continue
        trailing = tail[obj_end:]
        if ns_token:
            trailing = trailing.replace(ns_token, "")
        trailing = trailing.replace("```", "")
        if re.sub(r"[\s\[\]<>/]+", "", trailing):
            pos = obj_end
            continue
        # Prefer a self-describing object. MiniMax also emits a fully balanced
        # flat argument object after placing the tool name in the immediately
        # preceding invoke tag. Accept that atomic variant only when the name
        # is advertised and every object key maps to an advertised property.
        self_describing = (
            isinstance(obj.get("name"), str)
            and any(
                key in obj
                for key in ("arguments", "args", "parameters", "input")
            )
        )
        if not self_describing:
            before = text[max(0, marker_at - 2048):marker_at]
            invoke_names = re.findall(
                r"(?is)<invoke\b[^>]*?\bname\s*=\s*[\"']([^\"']+)[\"']",
                before,
            )
            inferred = (
                _canonical_tool_name(invoke_names[-1], name_map)
                if invoke_names else ""
            )
            props = _tool_schema_property_names(tools, inferred) if inferred else []
            prop_keys = {
                re.sub(r"[^a-z0-9]", "", key.lower()) for key in props
            }
            object_keys = {
                re.sub(r"[^a-z0-9]", "", str(key).lower()) for key in obj
            }
            if not inferred or not object_keys or not object_keys.issubset(prop_keys):
                pos = obj_end
                continue
            obj = {"name": inferred, "arguments": obj}
        call = _tool_call_from_json_object(obj, tools, name_map, 0)
        return [call] if call is not None else []


def _split_pseudo_tool_args(arg_text):
    """Parse Codex-style pseudo tool args: key: "value", count=3.

    This intentionally accepts only simple literal values inside the explicit
    <<< ... >>> wrapper. It is a compatibility bridge for agent shims that put
    a pseudo call in the assistant text before MiniMax's XML block.
    """
    args = {}
    if not isinstance(arg_text, str) or not arg_text.strip():
        return args
    pair_re = re.compile(
        r"""
        (?P<key>[A-Za-z_][\w-]*)\s*[:=]\s*
        (?:
            (?P<quote>["'])(?P<quoted>(?:\\.|(?!\2).)*)(?P=quote)
            |
            (?P<bare>[^,\s)]+)
        )
        """,
        flags=re.DOTALL | re.VERBOSE,
    )
    for match in pair_re.finditer(arg_text):
        key = match.group("key")
        if match.group("quote"):
            value = _loads_json_string_fragment(match.group("quoted") or "")
        else:
            raw = (match.group("bare") or "").strip()
            if re.fullmatch(r"-?\d+", raw):
                try:
                    value = int(raw)
                except ValueError:
                    value = raw
            elif re.fullmatch(r"-?\d+\.\d+", raw):
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            elif raw.lower() in {"true", "false"}:
                value = raw.lower() == "true"
            else:
                value = raw
        args[key] = value
    return args


def _recover_codex_pseudo_tool_calls(text, tools):
    """Recover explicit Codex pseudo calls such as <<< $g=create_goal(...) >>>."""
    if not text:
        return []
    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    if not allowed:
        return []
    pseudo_re = re.compile(
        r"""
        <<<\s*
        (?:\$?[A-Za-z_][\w-]*\s*=\s*)?
        (?P<name>[A-Za-z_][\w:.-]*)\s*
        \((?P<args>.*?)\)
        \s*;?\s*>>>
        """,
        flags=re.DOTALL | re.VERBOSE,
    )
    calls = []
    for match in pseudo_re.finditer(text):
        name = _canonical_tool_name(match.group("name"), name_map)
        if not name or (allowed and name not in allowed):
            continue
        args = _split_pseudo_tool_args(match.group("args"))
        args = _canonicalize_tool_argument_keys(args, tools, name)
        if not args and _tool_schema_expects_arguments(tools, name):
            continue
        calls.append(_openai_tool_call(name, args, len(calls)))
    if calls:
        logger.warning("recovered %d Codex pseudo tool call(s)", len(calls))
    return calls


def _recover_display_tool_calls(text, tools):
    """Recover display-style tool calls: [Tool call: exec]\n{"cmd": "..."}."""
    if not isinstance(text, str) or "[Tool call:" not in text:
        return []
    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    if not allowed:
        return []
    call_re = re.compile(
        r"""\[Tool call:\s*(?P<name>[A-Za-z_][\w:.-]*)\s*\]""",
        flags=re.IGNORECASE,
    )
    calls = []
    pos = 0
    while True:
        match = call_re.search(text, pos)
        if not match:
            break
        raw_name = match.group("name")
        if raw_name.lower() == "exec":
            raw_name = "exec_command"
        name = _canonical_tool_name(raw_name, name_map)
        if not name or (allowed and name not in allowed):
            pos = match.end()
            continue
        arg_start = match.end()
        while arg_start < len(text) and text[arg_start].isspace():
            arg_start += 1
        # MiniMax often emits the namespace marker before the display line.
        if text.startswith("]<]minimax[>[", arg_start):
            arg_start += len("]<]minimax[>[")
            while arg_start < len(text) and text[arg_start].isspace():
                arg_start += 1
        args = {}
        has_json_args = False
        if arg_start < len(text) and text[arg_start] == "{":
            arg_end = _json_balanced_end(text, arg_start)
            if arg_end > arg_start:
                try:
                    decoded = json.loads(text[arg_start:arg_end])
                    if isinstance(decoded, dict):
                        args = decoded
                        has_json_args = True
                except json.JSONDecodeError:
                    args = {}
                pos = arg_end
            else:
                pos = match.end()
        else:
            pos = match.end()
        # 2026-07-06 audit: without the args JSON, "[Tool call: name]" is
        # indistinguishable from prose mentioning a call (docs, log quotes).
        # Recovering it manufactured executions for zero-arg tools.
        if not has_json_args:
            continue
        args = _canonicalize_tool_argument_keys(args, tools, name)
        if not args and _tool_schema_expects_arguments(tools, name):
            continue
        calls.append(_openai_tool_call(name, args, len(calls)))
    if calls:
        logger.warning("recovered %d display-style tool call(s)", len(calls))
    return calls


def _recover_bare_xml_argument_tool_calls(text, tool_module, tools):
    """Recover a complete tool block with a bare name and XML arguments.

    ZCode/MiniMax can emit ``<tool_call> Edit <file_path>...`` with a trailing
    ``</Edit></invoke>`` instead of an opening ``<invoke name="Edit">``. This
    rung remains schema-grounded: the name must be advertised and every
    required argument must be explicitly present in a matching XML tag.
    """
    if not text or tool_module is None:
        return []
    blocks = _tool_call_blocks(text, tool_module)
    if not blocks:
        return []
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    calls = []
    for block in blocks:
        flat = block.replace(ns_token, "") if ns_token else block
        attr_match = re.search(
            r"(?is)^\s*<invoke\b[^>]*\bname\s*=\s*[\"']"
            r"(?P<name>[A-Za-z_$][\w:.$-]*)[\"'][^>]*>",
            flat,
        )
        bare_match = re.search(
            r"(?is)^\s*(?P<name>[A-Za-z_$][\w:.$-]*)\s*(?:\n|<)",
            flat,
        )
        raw_name = (
            attr_match.group("name") if attr_match
            else bare_match.group("name") if bare_match
            else ""
        )
        name = _canonical_tool_name(raw_name, name_map)
        if not name or (allowed and name not in allowed):
            continue
        props = _tool_schema_property_names(tools, name)
        specs = _tool_schema_property_specs(tools, name)
        aliases = {
            "file_path": ("file_path", "filePath", "path", "filename"),
            "old_string": ("old_string", "oldString", "old_text", "oldText"),
            "new_string": ("new_string", "newString", "new_text", "newText"),
            "old_text": ("old_text", "oldText", "old_string", "oldString"),
            "new_text": ("new_text", "newText", "new_string", "newString"),
            "replace_all": ("replace_all", "replaceAll"),
            "content": ("content", "text", "body"),
            "command": ("command", "cmd"),
        }
        raw_args = {}
        for prop in props:
            variants = aliases.get(prop, (prop,))
            value_match = None
            for variant in variants:
                value_match = re.search(
                    rf"(?is)<{re.escape(variant)}\b[^>]*>"
                    rf"(?P<value>.*?)</{re.escape(variant)}\s*>",
                    flat,
                )
                if value_match:
                    break
            if not value_match:
                continue
            value = value_match.group("value")
            spec = specs.get(prop) if isinstance(specs, dict) else None
            schema_type = spec.get("type") if isinstance(spec, dict) else None
            if schema_type == "boolean":
                lowered = value.strip().lower()
                if lowered not in {"true", "false"}:
                    continue
                value = lowered == "true"
            elif schema_type == "integer":
                try:
                    value = int(value.strip())
                except ValueError:
                    continue
            elif schema_type == "number":
                try:
                    value = float(value.strip())
                except ValueError:
                    continue
            elif schema_type in {"array", "object"}:
                try:
                    value = json.loads(value.strip())
                except json.JSONDecodeError:
                    continue
            elif re.sub(r"[^a-z0-9]", "", prop.lower()) in {
                "path", "filepath", "filename"
            }:
                value = value.strip()
            raw_args[prop] = value
        args = _canonicalize_tool_argument_keys(raw_args, tools, name)
        required = _tool_schema_required_names(tools, name)
        if any(args.get(key) in (None, "") for key in required):
            continue
        if _tool_schema_type_mismatches(args, tools, name):
            continue
        calls.append(_openai_tool_call(name, args, len(calls)))
    if calls:
        logger.warning(
            "recovered %d bare-XML-argument tool call(s)",
            len(calls),
        )
    return calls


def _looks_like_codex_pseudo_tool_call(text):
    return bool(
        isinstance(text, str)
        and "<<<" in text
        and ">>>" in text
        and re.search(r"<<<\s*(?:\$?\w+\s*=\s*)?\w+\s*\(", text)
    )


def _recover_attr_invoke_tool_calls(text, tool_module, tools):
    """Recover the attribute-invoke flavor (2026-07-10 zcode todo_write):

        <invoke id="todo_write" todos="[{...}]" />

    The tool name rides an `id=`/`name=` attribute and the arguments are raw
    XML attributes (not a JSON body / not <parameter> tags), often
    self-closing. Every rung keyed on name= + a JSON/tag body missed it and
    the call was dropped. Parse attributes generically; the arg-carrying
    attribute value is JSON (todos="[...]") or a scalar, matched to the
    declared schema.
    """
    if not text or tool_module is None:
        return []
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if not ns_token or "<invoke" not in text:
        return []
    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    # Strip the namespace tokens so a plain XML-ish attribute scan works.
    flat = text.replace(ns_token, " ")
    calls = []
    # Match the invoke OPENER only; attribute values may contain '>' and
    # unescaped quotes (the model emits JSON inside XML attrs), so scan
    # attributes manually rather than bounding on the first '>'.
    attr_key_re = re.compile(r'([A-Za-z_][\w:.-]*)\s*=\s*"')
    for om in re.finditer(r"<invoke\b", flat):
        seg = flat[om.end():om.end() + 20000]
        attrs = {}
        pos = 0
        while True:
            km = attr_key_re.search(seg, pos)
            if not km:
                break
            tag_close = seg.find(">", 0)
            if 0 <= tag_close < km.start():
                break  # past this invoke's opening tag
            key = km.group(1)
            vstart = km.end()  # just past the opening quote
            if seg[vstart:vstart + 1] in "[{":
                # JSON-valued attribute: balanced end ignores the XML quote
                # AND the unescaped inner quotes that make it invalid XML.
                vend = _json_balanced_end(seg, vstart)
                if vend > vstart:
                    attrs[key] = seg[vstart:vend]
                    close = seg.find('"', vend)
                    pos = (close + 1) if close >= 0 else vend
                    continue
            qm = re.compile(r'(?:\\.|[^"])*"').match(seg, vstart)
            if not qm:
                break
            attrs[key] = seg[vstart:qm.end() - 1]
            pos = qm.end()
        if not attrs:
            continue
        raw_name = attrs.pop("name", None) or attrs.pop("id", None)
        name = _canonical_tool_name(raw_name, name_map)
        if not name or (allowed and name not in allowed):
            continue
        args = {}
        for k, v in attrs.items():
            vs = (v or "").strip()
            parsed = None
            if vs[:1] in "[{":
                for strict in (True, False):
                    try:
                        parsed = json.loads(vs, strict=strict)
                        break
                    except json.JSONDecodeError:
                        parsed = None
            args[k] = parsed if parsed is not None else v
        args = _canonicalize_tool_argument_keys(args, tools, name)
        if args or not _tool_schema_expects_arguments(tools, name):
            calls.append(_openai_tool_call(
                name, json.dumps(args, ensure_ascii=False), len(calls)))
    if calls:
        logger.warning(
            "recovered %d attribute-invoke tool call(s) (id=/name= + attr args)",
            len(calls),
        )
    return calls


def _recover_function_syntax_invoke_tool_calls(text, tool_module, tools):
    """Recover ``<invoke name=\"read_file(path=\"...\")]>`` drift.

    OpenCode/MiniMax occasionally places a whole one-argument function call
    inside the XML ``name`` attribute. Require a closed native tool block, one
    declared function alias, and one schema-mappable string argument.
    """
    if not text or tool_module is None:
        return []
    start, _ = _tool_call_markers(tool_module)
    if not start or start not in text or not _tool_block_emission_finished(
        text,
        tool_module,
    ):
        return []
    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    patterns = [
        re.compile(
            r'''<invoke\s+name="(?P<name>[A-Za-z_$][\w:.$-]*)'''
            r'''\((?P<key>[A-Za-z_$][\w:.$-]*)="(?P<value>[^"]+)"'''
            r'''\)\]?''',
            flags=re.DOTALL,
        ),
        re.compile(
            r"""<invoke\s+name='(?P<name>[A-Za-z_$][\w:.$-]*)"""
            r"""\((?P<key>[A-Za-z_$][\w:.$-]*)='(?P<value>[^']+)'"""
            r"""\)\]?""",
            flags=re.DOTALL,
        ),
        re.compile(
            r'''<invoke>\s*(?P<name>[A-Za-z_$][\w:.$-]*)'''
            r'''\((?P<key>[A-Za-z_$][\w:.$-]*)="(?P<value>[^"]+)"'''
            r'''\)\]?''',
            flags=re.DOTALL,
        ),
        re.compile(
            r"""<invoke>\s*(?P<name>[A-Za-z_$][\w:.$-]*)"""
            r"""\((?P<key>[A-Za-z_$][\w:.$-]*)='(?P<value>[^']+)'"""
            r"""\)\]?""",
            flags=re.DOTALL,
        ),
    ]
    calls = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = _canonical_tool_name(match.group("name"), name_map)
            if not name or (allowed and name not in allowed):
                continue
            args = _canonicalize_tool_argument_keys(
                {match.group("key"): match.group("value")},
                tools,
                name,
            )
            args = _coerce_json_encoded_schema_values(args, tools, name)
            if _tool_schema_type_mismatches(args, tools, name):
                continue
            required = _tool_schema_required_names(tools, name)
            if any(args.get(key) in (None, "") for key in required):
                continue
            calls.append(_openai_tool_call(name, args, len(calls)))
    if calls:
        logger.warning(
            "recovered %d function-syntax invoke tool call(s)",
            len(calls),
        )
    return calls


def _recover_bare_name_tool_calls(text, tool_module, tools):
    """Recover the bare-name flavor: ns_marker + ToolName + {json args}.

    2026-07-10 zcode: `]<]minimax[>[ Bash {"command": ...}` — no
    <tool_call>/<invoke> tags at all, so every block-based rung misses it and
    it leaked verbatim into visible content. The intended call is
    unambiguous: identifier right after the namespace marker + one balanced
    JSON object. Only declared tools are accepted; tolerant JSON (raw
    newlines) handled like the other rungs.
    """
    if not text or tool_module is None:
        return []
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if not ns_token or ns_token not in text:
        return []
    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    calls = []
    pos = 0
    head_re = re.compile(r"\s*<?(?P<name>[A-Za-z_][\w.-]{0,40})>?\s*\{")
    while True:
        idx = text.find(ns_token, pos)
        if idx < 0:
            break
        rest = text[idx + len(ns_token):]
        m = head_re.match(rest)
        name = _canonical_tool_name(m.group("name"), name_map) if m else None
        if m:
            obj_start = idx + len(ns_token) + m.end() - 1  # the '{'
        else:
            brace = re.match(r"\s*\{", rest)
            if not brace:
                pos = idx + len(ns_token)
                continue
            obj_start = idx + len(ns_token) + brace.end() - 1
        obj_end = _json_balanced_end(text, obj_start)
        if obj_end <= obj_start:
            pos = idx + len(ns_token)
            continue
        payload = text[obj_start:obj_end]
        obj = None
        for strict in (True, False):
            try:
                obj = json.loads(payload, strict=strict)
                break
            except json.JSONDecodeError:
                continue
        # Self-describing flavor: {"name": "Write", "arguments": {...}} (also
        # accepts input/parameters as the args key). Overrides a missing or
        # unknown leading identifier.
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            inner_name = _canonical_tool_name(obj["name"], name_map)
            if inner_name:
                name = inner_name
                inner_args = None
                for key in ("arguments", "input", "parameters", "args"):
                    if isinstance(obj.get(key), dict):
                        inner_args = obj[key]
                        break
                obj = inner_args if inner_args is not None else {
                    k: v for k, v in obj.items() if k != "name"}
        if not name or (allowed and name not in allowed):
            pos = obj_end
            continue
        if isinstance(obj, dict) and obj:
            obj = _canonicalize_tool_argument_keys(obj, tools, name)
            if obj or not _tool_schema_expects_arguments(tools, name):
                calls.append(_openai_tool_call(
                    name, json.dumps(obj, ensure_ascii=False), len(calls)))
        pos = obj_end
    if calls:
        logger.warning(
            "recovered %d bare-name tool call(s) (marker + name + JSON, no tags)",
            len(calls),
        )
    return calls


def _recover_jsonish_tool_calls(text, tool_module, tools):
    """Recover common MiniMax/ZCode JSON-ish tool calls after official parse fails.

    MiniMax occasionally emits blocks like:
      {"name": "Bash", {"description": "...", "input": "..."}}
    inside its tool marker. That is not valid JSON/XML, but the intended OpenAI
    function call is clear. Repair only this narrow shape and only for declared
    tools, leaving normal MiniMax XML invocations to the upstream parser.
    """
    if not text or tool_module is None:
        return []
    blocks = _tool_call_blocks(text, tool_module)
    if not blocks:
        return []

    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    calls = []
    name_re = re.compile(r'\{\s*"name"\s*:\s*"(?P<name>(?:\\.|[^"])*)"\s*,', re.DOTALL)
    for block in blocks:
        pos = 0
        while True:
            obj_start = block.find("{", pos)
            if obj_start < 0:
                break
            obj_end = _json_balanced_end(block, obj_start)
            if obj_end <= obj_start:
                pos = obj_start + 1
                continue
            try:
                obj = json.loads(block[obj_start:obj_end])
            except json.JSONDecodeError:
                pos = obj_start + 1
                continue
            call = _tool_call_from_json_object(obj, tools, name_map, len(calls))
            if call is not None:
                calls.append(call)
            pos = obj_end
        pos = 0
        while True:
            match = name_re.search(block, pos)
            if not match:
                break
            raw_name = _loads_json_string_fragment(match.group("name")).strip()
            name = _canonical_tool_name(raw_name, name_map)
            if allowed and name not in allowed:
                pos = match.end()
                continue
            arg_start = match.end()
            while arg_start < len(block) and block[arg_start].isspace():
                arg_start += 1
            # Most observed bad calls put the argument object directly after
            # the comma. Also accept {"arguments": {...}} if the model drifts.
            arguments = None
            if arg_start < len(block) and block[arg_start] == "{":
                arg_end = _json_balanced_end(block, arg_start)
                if arg_end > arg_start:
                    try:
                        arguments = json.loads(block[arg_start:arg_end])
                    except json.JSONDecodeError:
                        arguments = None
                    pos = arg_end
                else:
                    pos = match.end()
            else:
                pos = match.end()
            if isinstance(arguments, dict) and set(arguments) == {"arguments"}:
                inner = arguments.get("arguments")
                arguments = inner if isinstance(inner, dict) else arguments
            if not isinstance(arguments, dict):
                continue
            if not arguments and _tool_schema_expects_arguments(tools, name):
                continue
            calls.append(_openai_tool_call(name, arguments, len(calls)))
    if calls:
        logger.warning("recovered %d malformed JSON-ish tool call(s)", len(calls))
    return calls


def _recover_loose_segment_tool_calls(text, tool_module, tools):
    """Recover nameless MiniMax tool blocks made only of bracket fragments.

    Observed shape from Codex/ZCode-style requests:
      ]<]minimax[>[<tool_call>
      ]<]minimax[>[]<]minimax[>[python3 -version]
      ]<]minimax[>[]<]minimax[>[Run project check]
      ]<]minimax[>[</tool_call>

    There is no invoke tag or JSON object to parse, but if the fragments contain
    a shell command and the client advertised exactly a command-like tool, we can
    safely reconstruct the OpenAI tool call instead of leaking raw markup.
    """
    if not text or tool_module is None:
        return []
    blocks = _tool_call_blocks(text, tool_module)
    if not blocks:
        return []
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if not ns_token:
        return []

    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    command_name = None
    for candidate in (
        "exec_command",
        "invoke_command",
        "run_command",
        "execute_command",
        "Bash",
        "bash",
        "Shell",
        "shell",
        "terminal",
    ):
        canonical = _canonical_tool_name(candidate, name_map)
        if canonical in allowed:
            command_name = canonical
            break
    if not command_name:
        return []

    calls = []
    for block in blocks:
        if re.search(r"<\s*(?:invoke|[A-Za-z_$][\w:.$-]+)\b", block) or "{" in block:
            continue
        segments = _loose_tool_segments(block, ns_token)
        if not segments:
            continue
        command_idx = next(
            (i for i, segment in enumerate(segments) if _looks_like_shell_command(segment)),
            None,
        )
        if command_idx is None:
            continue
        args = _arguments_from_loose_segments(block, ns_token, tools, command_name)
        if args is None:
            continue
        args = _canonicalize_tool_argument_keys(args, tools, command_name)
        if not args and _tool_schema_expects_arguments(tools, command_name):
            continue
        calls.append(_openai_tool_call(command_name, args, len(calls)))
    if calls:
        logger.warning("recovered %d loose-segment MiniMax tool call(s)", len(calls))
    return calls


def _recover_malformed_xml_tool_calls(text, tool_module, tools):
    """Repair narrow MiniMax XML attr drift, e.g. <invoke name Bash">.

    This intentionally refuses empty argument calls for tools whose schema
    declares parameters; sending an empty Bash/Read call is worse than dropping
    a malformed block and asking the model/client to continue.
    """
    if not text or tool_module is None:
        return []
    blocks = _tool_call_blocks(text, tool_module)
    if not blocks:
        return []
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if not ns_token:
        return []

    def _normalize_body(body):
        # MiniMax's parser expects the namespace marker before both opening and
        # closing tags. Some agent runs have produced bare closing tags inside
        # an otherwise MiniMax-shaped block, e.g. <command>ls</command>.
        if not body:
            return body
        tag = r"[A-Za-z_$][\w:.$-]*"
        body = re.sub(
            rf"(?<!{re.escape(ns_token)})</(?P<tag>{tag})>",
            rf"{ns_token}</\g<tag>>",
            body,
        )
        body = re.sub(
            rf"(?<!{re.escape(ns_token)})<(?P<tag>{tag})(?P<attrs>[^>/]*)>",
            rf"{ns_token}<\g<tag>\g<attrs>>",
            body,
        )
        return body

    name_map = _tool_name_map_from_schema(tools)
    allowed = set(name_map.values())
    # The outer <tool_call> marker already proves this block is tool-shaped.
    # MiniMax can place the nested <invoke> opener directly after it, without
    # repeating the namespace token, while still namespacing every argument
    # and closing tag. Accept that one missing boundary marker here.
    invoke_opener = rf"(?:{re.escape(ns_token)}\s*<?invoke\b|<invoke\b)"
    invoke_re = re.compile(
        rf"{invoke_opener}(?P<attrs>[^>]*)>"
        rf"(?P<body>.*?){re.escape(ns_token)}</invoke>",
        flags=re.DOTALL,
    )
    loose_invoke_re = re.compile(
        rf"{invoke_opener}(?P<attrs>[^>]*)>"
        rf"(?P<body>.*?)(?:{re.escape(ns_token)}</tool_call>|$)",
        flags=re.DOTALL,
    )
    named_tag_invoke_re = re.compile(
        rf"{re.escape(ns_token)}\s*<(?P<tag>[A-Za-z_$][\w:.$-]*)(?P<attrs>[^>]*)>"
        rf"(?P<body>.*?){re.escape(ns_token)}</invoke>",
        flags=re.DOTALL,
    )
    name_attr_re = re.compile(
        # MiniMax occasionally borrows the Responses-style `to=` spelling
        # while keeping the rest of its native XML envelope intact. Accept it
        # only inside a completed native block and still require the value to
        # map to an advertised schema name before a call can be emitted.
        r"""\b(?:name|to)\s*=\s*(?:(["'])(?P<quoted>.*?)\1|(?P<bare>[^\s>]+))""",
        flags=re.DOTALL,
    )
    broken_name_re = re.compile(
        r"""\bname\s+(?P<name>[A-Za-z_$][\w:.$-]*)(?:"|')?""",
        flags=re.DOTALL,
    )
    positional_name_re = re.compile(
        r"""^\s*(?P<name>[A-Za-z_$][\w:.$-]*)\s*[\"'}\]),;:]*\s*$""",
        flags=re.DOTALL,
    )
    equals_name_re = re.compile(
        r"""^\s*=\s*(?:(['\"])(?P<quoted>.*?)\1|(?P<bare>[^\s>]+))\s*$""",
        flags=re.DOTALL,
    )
    calls = []
    for block in blocks:
        matches = list(invoke_re.finditer(block))
        if not matches:
            matches = list(loose_invoke_re.finditer(block))
        if not matches:
            matches = list(named_tag_invoke_re.finditer(block))
        for match in matches:
            attrs = match.group("attrs") or ""
            tag_name = match.groupdict().get("tag")
            name_match = name_attr_re.search(attrs)
            raw_name = (
                (name_match.group("quoted") or name_match.group("bare"))
                if name_match else None
            )
            if not raw_name and tag_name and tag_name not in {"invoke", "tool_call"}:
                raw_name = tag_name.strip().strip("\"'")
            if not raw_name:
                broken = broken_name_re.search(attrs)
                raw_name = broken.group("name") if broken else None
            if not raw_name:
                # Some MiniMax/Hermes tool turns drift from
                # <invoke name="execute_search"> to <invoke execute_search>.
                # Treat a lone positional attr as the function name, but still
                # validate it against the declared OpenAI tool schema below.
                positional = positional_name_re.match(attrs or "")
                raw_name = positional.group("name") if positional else None
            if not raw_name:
                # OpenCode thinking turn observed 2026-07-12:
                # ``<invoke="todowrite">``. The upstream parser treated the
                # todo array as Bash.command. Recover the explicit function
                # name before any body-based inference.
                equals_name = equals_name_re.match(attrs or "")
                raw_name = (
                    equals_name.group("quoted") or equals_name.group("bare")
                    if equals_name else None
                )
            name = _canonical_tool_name(raw_name, name_map)
            if not name or (allowed and name not in allowed):
                name = _infer_tool_name_from_body(attrs, match.group("body"), tools, name_map)
            if allowed and name not in allowed:
                continue
            raw_body = match.group("body") or ""
            path_props = [
                prop for prop in _tool_schema_property_names(tools, name)
                if re.sub(r"[^a-z0-9]", "", prop.lower())
                in {"path", "filepath", "filename"}
            ]
            required_paths = [
                prop for prop in path_props
                if prop in _tool_schema_required_names(tools, name)
            ]
            if (
                len(required_paths) == 1
                and len(_tool_schema_required_names(tools, name)) == 1
            ):
                # The same legacy ``<invoke>read_file>`` drift often leaves
                # ordinary (non-namespaced) path tags. Extract that complete,
                # explicit value before positional parsing can absorb the
                # alias and XML closer into the filename.
                path_aliases = list(dict.fromkeys([
                    *path_props,
                    "path",
                    "file_path",
                    "filePath",
                    "filename",
                ]))
                bare_path = None
                for path_alias in path_aliases:
                    tagged_path = re.search(
                        rf"(?is)(?:{re.escape(ns_token)})?"
                        rf"<{re.escape(path_alias)}\b[^>]*>"
                        rf"(?P<value>.*?)"
                        rf"(?:{re.escape(ns_token)})?"
                        rf"</{re.escape(path_alias)}\s*>",
                        raw_body,
                    )
                    if tagged_path:
                        bare_path = tagged_path.group("value").strip()
                        break
                if bare_path and "<" not in bare_path and ">" not in bare_path:
                    calls.append(_openai_tool_call(
                        name,
                        json.dumps(
                            {required_paths[0]: bare_path},
                            ensure_ascii=False,
                        ),
                        len(calls),
                    ))
                    continue
            if len(required_paths) == 1 and not any(
                re.search(
                    rf"{re.escape(ns_token)}<{re.escape(prop)}\s*>",
                    raw_body,
                    flags=re.IGNORECASE,
                )
                for prop in path_props
            ):
                attr_paths = list(dict.fromkeys(re.findall(
                    r"(?<![A-Za-z0-9_])(/[^\s<>\"']+)",
                    attrs,
                )))
                if len(attr_paths) == 1:
                    calls.append(_openai_tool_call(
                        name,
                        json.dumps(
                            {required_paths[0]: attr_paths[0]},
                            ensure_ascii=False,
                        ),
                        len(calls),
                    ))
                    continue
            # Bound for the empty-args salvage at the bottom of the loop: the
            # fast paths skip the repair branch that otherwise assigns it
            # (UnboundLocalError wedge, 2026-07-09 — rank1 died, cache
            # diverged, next request deadlocked).
            body = raw_body
            legacy_question_args = _legacy_question_args_from_body(
                raw_body,
                ns_token,
                tools,
                name,
            )
            if legacy_question_args:
                calls.append(_openai_tool_call(
                    name,
                    json.dumps(legacy_question_args, ensure_ascii=False),
                    len(calls),
                ))
                continue
            labeled_array_args = _labeled_json_array_args_from_body(
                raw_body,
                tools,
                name,
            )
            if labeled_array_args:
                calls.append(_openai_tool_call(
                    name,
                    json.dumps(labeled_array_args, ensure_ascii=False),
                    len(calls),
                ))
                continue
            nested_array_args = _nested_array_args_from_ns_tags(
                raw_body,
                ns_token,
                tools,
                name,
            )
            if nested_array_args:
                calls.append(_openai_tool_call(
                    name,
                    json.dumps(nested_array_args, ensure_ascii=False),
                    len(calls),
                ))
                continue
            named_parameter_args = _arguments_from_named_parameter_tags(
                raw_body,
                ns_token,
                tools,
                name,
            )
            if named_parameter_args:
                calls.append(_openai_tool_call(
                    name,
                    json.dumps(named_parameter_args, ensure_ascii=False),
                    len(calls),
                ))
                continue
            # Payload-safe fast path: when the body carries one balanced JSON
            # object, take it verbatim. The repair chain below rewrites tags
            # (_normalize_body) and slices on tag-like boundaries
            # (loose segments) — both corrupt markup-bearing code payloads
            # (HTML/JSX inside JSON string values): the 2026-07-06
            # shredded-single-file-game bug. JSON is opaque to markup; if it
            # parses whole, no repair is needed or wanted.
            _js = raw_body.find("{")
            if _js >= 0:
                _je = _json_balanced_end(raw_body, _js)
                if _je > _js:
                    try:
                        _obj = json.loads(raw_body[_js:_je])
                    except json.JSONDecodeError:
                        try:
                            # Big multiline payloads (markdown/code inside a
                            # JSON string) often carry RAW newlines/tabs —
                            # illegal in strict JSON but unambiguous. The
                            # 2026-07-09 zcode pandoc turn failed all rungs
                            # 3x on exactly this; strict=False accepts the
                            # control chars and json.dumps re-escapes on the
                            # way out.
                            _obj = json.loads(raw_body[_js:_je], strict=False)
                        except json.JSONDecodeError:
                            _obj = None
                    if isinstance(_obj, dict) and _obj:
                        _obj = _canonicalize_tool_argument_keys(_obj, tools, name)
                        if _obj or not _tool_schema_expects_arguments(tools, name):
                            calls.append(_openai_tool_call(
                                name,
                                json.dumps(_obj, ensure_ascii=False),
                                len(calls),
                            ))
                            continue
            if _positional_recovery_is_underspecified(
                raw_body,
                ns_token,
                tools,
                name,
            ):
                logger.warning(
                    "refusing under-specified positional recovery for %s; "
                    "required schema fields were not all emitted",
                    name,
                )
                continue
            # XML-args fast paths (2026-07-06 audit): _normalize_body
            # prefixes EVERY tag with the namespace token — including markup
            # INSIDE argument values — the same corruption class as the
            # shredded-JSON bug, for XML-style arg encodings. Repair ladder:
            #   1. parse the un-rewritten body (well-formed loose blocks);
            #   2. ns-segment extractor (drifted blocks, values verbatim);
            #   3. only then the rewriting chain below.
            parsed_items = None
            raw_fixed = (
                f'{ns_token}<invoke name="{name}">'
                f'{raw_body}{ns_token}</invoke>'
            )
            try:
                parsed = tool_module.parse_tool_call(raw_fixed, tools)
                parsed_items = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                parsed_items = None
            if not parsed_items:
                ns_args = _arguments_from_ns_arg_tags(
                    raw_body, ns_token, tools, name
                )
                if ns_args:
                    ns_args = _canonicalize_tool_argument_keys(ns_args, tools, name)
                    if ns_args or not _tool_schema_expects_arguments(tools, name):
                        parsed_items = [{
                            "name": name,
                            "arguments": json.dumps(ns_args, ensure_ascii=False),
                        }]
            if not parsed_items:
                body = _normalize_body(match.group("body"))
                fixed = (
                    f'{ns_token}<invoke name="{name}">'
                    f'{body}{ns_token}</invoke>'
                )
                try:
                    parsed = tool_module.parse_tool_call(fixed, tools)
                    parsed_items = parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    args = _arguments_from_loose_segments(body, ns_token, tools, name)
                    if args is None:
                        continue
                    parsed_items = [{"name": name, "arguments": json.dumps(args, ensure_ascii=False)}]
            for item in parsed_items:
                if not isinstance(item, dict):
                    continue
                item_name = _canonical_tool_name(item.get("name"), name_map) or name
                if allowed and item_name not in allowed:
                    continue
                args = item.get("arguments", {})
                arg_obj = None
                if isinstance(args, str):
                    try:
                        arg_obj = json.loads(args)
                    except json.JSONDecodeError:
                        try:
                            arg_obj = json.loads(args, strict=False)
                            # Re-serialize so downstream consumers get
                            # properly escaped strict JSON.
                            args = json.dumps(arg_obj, ensure_ascii=False)
                        except json.JSONDecodeError:
                            arg_obj = None
                elif isinstance(args, dict):
                    arg_obj = args
                if isinstance(arg_obj, dict) and not arg_obj and _tool_schema_expects_arguments(tools, item_name):
                    if body is None:
                        # Fast-path parse with empty args: the old code raised
                        # UnboundLocalError here and the outer except dropped
                        # the parse; drop just this call instead so rank1
                        # survives. The usable-turn ladder regenerates it.
                        continue
                    loose_args = _arguments_from_loose_segments(body, ns_token, tools, item_name)
                    # Empty/None salvage -> skip the call entirely so the
                    # usable-turn retry ladder regenerates it, matching the
                    # pre-2026-07-09 behavior (this branch used to crash and
                    # the whole parse was retried; emitting a degraded call
                    # here would skip that ladder).
                    if not loose_args:
                        continue
                    args = json.dumps(loose_args, ensure_ascii=False)
                calls.append(_openai_tool_call(item_name, args, len(calls)))
    if calls:
        logger.warning("recovered %d malformed MiniMax XML tool call(s)", len(calls))
    return calls


def _recover_complete_calls_before_unclosed_tail(text, tool_module, tools):
    """Keep closed native calls when a later call is left unfinished.

    MiniMax can finish a valid Write/Edit and immediately start a verification
    call before sampling EOS.  Rejecting the final open block must not discard
    the already atomic action.  Parse only the prefix before the last opener,
    require its last native block to be closed, and return schema-valid calls.
    The unfinished tail is never parsed or exposed.
    """
    if not text or tool_module is None or not tools:
        return [], ""
    start, _ = _tool_call_markers(tool_module)
    marker_at = text.rfind(start) if start else -1
    if marker_at <= 0:
        return [], ""
    prefix = text[:marker_at]
    if start not in prefix or not _tool_block_emission_finished(
        prefix,
        tool_module,
    ):
        return [], ""
    try:
        from mlx_vlm.server.responses_state import process_tool_calls

        parsed = process_tool_calls(prefix, tool_module, tools)
        calls = parsed.get("calls") or []
        validated = _validate_outgoing_tool_calls(calls, tools)
        if not validated:
            return [], ""
        # Narration after a completed action belongs to the abandoned second
        # action. Preserve only text that preceded the first native call.
        remaining = prefix[:prefix.find(start)]
        return validated, remaining.rstrip()
    except Exception:
        return [], ""


def _capture_suspicious_native_tool_parse(text, calls, tools):
    """Persist an opt-in raw fixture when a large Write collapses to a stub.

    MiniMax tool output can contain source markup that resembles its own XML
    argument envelope. Keep diagnostics disabled by default because the raw
    response can contain user data; this hook is only for local parser audits.
    """
    if not TOOL_PARSE_DIAGNOSTICS or len(text or "") < 4096:
        return

    suspicious = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name") if isinstance(fn, dict) else ""
        normalized_name = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
        if normalized_name not in _WRITE_FILE_TOOL_NAMES:
            continue
        arguments = fn.get("arguments", {}) if isinstance(fn, dict) else {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"$raw": arguments}
        if not isinstance(arguments, dict):
            continue
        payload_key = next(
            (
                key for key in ("content", "contents", "text", "data", "body")
                if isinstance(arguments.get(key), str)
            ),
            "",
        )
        payload = arguments.get(payload_key, "") if payload_key else ""
        # Diagnostics are explicitly opt-in, so prefer a harmless extra capture
        # over missing the failure when the native parser has already discarded
        # the unrecognized material.
        schema_keys = set(_tool_schema_property_names(tools, name))
        extra_keys = [str(key) for key in arguments if key not in schema_keys]
        extra_chars = sum(
            len(value) if isinstance(value, str) else len(json.dumps(value, default=str))
            for key, value in arguments.items()
            if key not in schema_keys
        )
        if len(payload) > 512:
            continue
        suspicious.append({
            "name": name,
            "payload_key": payload_key,
            "payload_chars": len(payload),
            "argument_keys": [str(key) for key in arguments],
            "extra_keys": extra_keys,
            "extra_chars": extra_chars,
        })

    if not suspicious:
        return
    try:
        diagnostic_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ops",
            "logs",
            "tool_parse_diagnostics",
        )
        os.makedirs(diagnostic_dir, exist_ok=True)
        path = os.path.join(
            diagnostic_dir,
            f"native-write-{time.time_ns()}-{uuid.uuid4().hex[:8]}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "captured_at": time.time(),
                    "suspicious": suspicious,
                    "raw_output": text,
                    "parser_calls": calls,
                    "tools": tools,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        logger.warning("captured suspicious native Write parse at %s", path)
    except Exception as exc:
        logger.warning("failed to capture suspicious native Write parse: %s", exc)


def _parse_tool_calls(text, tool_module, tools):
    if not text or tool_module is None:
        return [], text
    start, _ = _tool_call_markers(tool_module)
    # Never execute a parser-recovered call from an unclosed native block.
    # At max_tokens the permissive MLX parser can salvage `file_path` plus a
    # partial `content` string and downstream clients will faithfully write a
    # truncated file. A tool response is atomic: if its last native block did
    # not close, discard every call and let the bounded retry regenerate it.
    marker_at = text.rfind(start) if start else -1
    if marker_at >= 0 and not _tool_block_emission_finished(text, tool_module):
        # A closed inner </invoke> does not make an unclosed outer
        # <tool_call> atomic. MiniMax can abandon a large Write body, start a
        # second complete Bash invoke, and then emit EOS without closing
        # either outer block. The permissive recovery parser used to salvage
        # the first Write and collapse markup-bearing content to its <title>.
        # Only a complete earlier outer block (handled below) or a complete
        # JSON payload is safe to expose to a client.
        completed, completed_remaining = (
            _recover_complete_calls_before_unclosed_tail(
                text,
                tool_module,
                tools,
            )
        )
        if completed:
            logger.warning(
                "preserved %d complete native tool call(s) before an "
                "unfinished trailing call",
                len(completed),
            )
            return completed, completed_remaining
        recovered = (
            _recover_complete_json_unclosed_tool_call(text, tool_module, tools)
            if TOOL_COMPAT_OVERLAY
            else []
        )
        if recovered:
            logger.warning(
                "recovered complete JSON tool payload from unclosed native block"
            )
            visible_prefix = text[:text.find(start)]
            visible_prefix = re.sub(
                r"(?is)(?:\]<\]minimax\[>\[)?<invoke\b[^>]*>?\s*$",
                "",
                visible_prefix,
            )
            return recovered, visible_prefix.rstrip()
        logger.warning(
            "native tool invocation ended before its closing tag; refusing "
            "partial tool execution and requesting regeneration"
        )
        return [], text[:text.find(start)].rstrip()
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if (
        TOOL_COMPAT_OVERLAY
        and ns_token
        and re.search(
            rf"{re.escape(ns_token)}\s*<invoke\s*=",
            text,
            flags=re.IGNORECASE,
        )
    ):
        recovered = _recover_malformed_xml_tool_calls(
            text,
            tool_module,
            tools,
        )
        if recovered:
            logger.warning(
                "recovered equals-name MiniMax invoke before native parsing"
            )
            return recovered, _strip_raw_tool_blocks(text, tool_module)
    if TOOL_COMPAT_OVERLAY:
        recovered = _recover_function_syntax_invoke_tool_calls(
            text,
            tool_module,
            tools,
        )
        if recovered:
            return recovered, _strip_raw_tool_blocks(text, tool_module)
        recovered = _recover_named_empty_read_tool_call(text, tools)
        if recovered:
            return recovered, _strip_raw_tool_blocks(text, tool_module)
    try:
        from mlx_vlm.server.responses_state import process_tool_calls

        parsed = process_tool_calls(text, tool_module, tools)
        calls = parsed.get("calls") or []
        remaining = parsed.get("remaining_text") or ""
        if calls:
            _capture_suspicious_native_tool_parse(text, calls, tools)
        if TOOL_COMPAT_OVERLAY and len(calls) == 1:
            parsed_name = _tool_call_name_for_loop(calls[0])
            if _positional_recovery_is_underspecified(
                text,
                ns_token,
                tools,
                parsed_name,
            ):
                logger.warning(
                    "discarding native %s parse backed by too few positional "
                    "arguments for its required schema",
                    parsed_name,
                )
                calls = []
        if calls:
            if TOOL_COMPAT_OVERLAY:
                native_validated, native_dropped = _validate_outgoing_tool_calls(
                    calls,
                    tools,
                    return_dropped=True,
                )
                if native_dropped and not native_validated:
                    recovered = _recover_malformed_xml_tool_calls(
                        text,
                        tool_module,
                        tools,
                    )
                    if recovered:
                        logger.warning(
                            "replaced schema-invalid native tool parse with "
                            "schema-grounded XML recovery"
                        )
                        return recovered, _strip_raw_tool_blocks(
                            text,
                            tool_module,
                        )
            return calls, remaining
        if not TOOL_COMPAT_OVERLAY:
            if start and start in text:
                # Native-first does not mean discarding a complete native
                # block solely because the invoke target drifted from
                # `name="Bash"` to `to="bash"`. This schema-grounded fallback
                # runs only after mlx-vlm returned zero calls.
                recovered = _recover_malformed_xml_tool_calls(
                    text,
                    tool_module,
                    tools,
                )
                if recovered:
                    for call in recovered:
                        if isinstance(call, dict):
                            call["_m3_schema_recovered"] = True
                    logger.warning(
                        "recovered schema-matching native invoke drift after "
                        "mlx-vlm parser miss"
                    )
                    return recovered, _strip_raw_tool_blocks(text, tool_module)
                logger.warning(
                    "native tool marker was generated but no valid OpenAI "
                    "tool_calls were parsed; stripping raw tool markup"
                )
                return [], _strip_raw_tool_blocks(text, tool_module)
            return [], remaining or text
        if (
            (start and start in text)
            or _looks_like_raw_tool_fragment(text, tool_module)
            or _looks_like_codex_pseudo_tool_call(text)
            or "[Tool call:" in text
        ):
            recovered = _recover_jsonish_tool_calls(text, tool_module, tools)
            if not recovered:
                recovered = _recover_bare_name_tool_calls(text, tool_module, tools)
            if not recovered:
                recovered = _recover_attr_invoke_tool_calls(text, tool_module, tools)
            if not recovered:
                recovered = _recover_codex_pseudo_tool_calls(text, tools)
            if not recovered:
                recovered = _recover_display_tool_calls(text, tools)
            if not recovered:
                recovered = _recover_bare_xml_argument_tool_calls(
                    text,
                    tool_module,
                    tools,
                )
            if not recovered:
                recovered = _recover_loose_segment_tool_calls(text, tool_module, tools)
            if not recovered:
                recovered = _recover_malformed_xml_tool_calls(text, tool_module, tools)
            if not recovered:
                recovered = _recover_named_empty_read_tool_call(text, tools)
            if recovered:
                return recovered, _strip_raw_tool_blocks(text, tool_module)
            logger.warning(
                "tool-call marker was generated but no valid OpenAI tool_calls "
                "were parsed; stripping raw tool markup from assistant content"
            )
            # Forensics (2026-07-07): every rung of the repair ladder missed
            # this markup (seen live on zcode turns — the agent's action gets
            # dropped and it looks stalled). Capture the raw bytes so the
            # next occurrence hands us the exact format to add a rung for.
            try:
                import json as _json
                with open(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "ops", "logs", "tool_parse_failures.jsonl"), "a") as _f:
                    # Marker-centered window: thinking turns run thousands
                    # of reasoning chars before the call; a head-truncated
                    # capture misses the markup entirely (learned 12:28).
                    _idx = text.find(start) if start else -1
                    if _idx < 0:
                        _idx = max(len(text) - 3500, 0)
                    _f.write(_json.dumps({
                        "at": round(time.time(), 3),
                        "tools_advertised": [
                            (t.get("function") or {}).get("name")
                            for t in (tools or [])][:16],
                        "head": text[:600],
                        "marker_at": _idx,
                        "raw": text[max(_idx - 500, 0):_idx + 3500],
                    }) + "\n")
            except Exception:
                pass
            return [], _strip_raw_tool_blocks(text, tool_module)
        return [], remaining or text
    except Exception as e:
        logger.warning(f"tool-call parse failed: {e}")
        return [], _strip_raw_tool_blocks(text, tool_module)


def _tool_block_emission_finished(text, tool_module):
    """True when the last opened tool block has a closing tag.

    Judging stop/invalid on an unclosed block truncates the model mid-call
    (a patch cut off at 31 tokens, `cat >` losing its filename) and then
    manufactures the very malformed calls the recovery paths exist to repair.
    Accept the namespaced or bare closing tag; MiniMax emits both.
    """
    start, _ = _tool_call_markers(tool_module)
    if not start or not text:
        return False
    last_start = text.rfind(start)
    if last_start < 0:
        return False
    return "</tool_call" in text[last_start + len(start):]


def _tool_call_complete_for_stop(text, tool_module, tools):
    """True once a buffered tool response has enough structure to stop decode."""
    if not text or tool_module is None or not tools:
        return False
    start, end = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if not ns_token or ns_token not in text:
        return False
    if not _tool_block_emission_finished(text, tool_module):
        return False
    try:
        from mlx_vlm.server.responses_state import process_tool_calls

        parsed = process_tool_calls(text, tool_module, tools)
        calls = parsed.get("calls") or []
        validated, dropped = _validate_outgoing_tool_calls(
            calls,
            tools,
            return_dropped=True,
        )
        if validated or dropped:
            return True
    except Exception:
        pass
    if not TOOL_COMPAT_OVERLAY:
        return False
    calls = _recover_jsonish_tool_calls(text, tool_module, tools)
    if not calls:
        calls = _recover_bare_xml_argument_tool_calls(
            text,
            tool_module,
            tools,
        )
    if not calls:
        calls = _recover_loose_segment_tool_calls(text, tool_module, tools)
    if not calls:
        calls = _recover_malformed_xml_tool_calls(text, tool_module, tools)
    if calls:
        validated, dropped = _validate_outgoing_tool_calls(
            calls,
            tools,
            return_dropped=True,
        )
        if validated or dropped:
            return True
    return False


def _incomplete_tool_call_budget_reached(
    token_count,
    tool_call_started,
    text,
    tool_module,
):
    """True only for an opened tool block that exceeded its atomic budget."""
    if (
        TOOL_INCOMPLETE_CALL_TOKEN_BUDGET <= 0
        or not tool_call_started
        or int(token_count or 0) < TOOL_INCOMPLETE_CALL_TOKEN_BUDGET
    ):
        return False
    return not _tool_block_emission_finished(text, tool_module)


def _completed_tool_detokenizer_tail_reached(
    silent_tokens,
    tool_call_started,
    text,
    tool_module,
    tools,
):
    """Stop a silent detokenizer tail only after a tool call is complete.

    Empty ``response.text`` values are not evidence of a stalled decode while
    MiniMax is still emitting a native tool block. Structural/control tokens
    can be buffered for many steps, including long array-valued arguments.
    """
    if (
        TOOL_DETOKENIZER_SILENT_TOKEN_BUDGET <= 0
        or not tool_call_started
        or int(silent_tokens or 0) < TOOL_DETOKENIZER_SILENT_TOKEN_BUDGET
    ):
        return False
    return _tool_call_complete_for_stop(text, tool_module, tools)


def _tool_call_contains_complete_but_invalid(text, tool_module, tools):
    if not text or tool_module is None or not tools:
        return False
    start, _ = _tool_call_markers(tool_module)
    ns_token = start.removesuffix("<tool_call>") if start else ""
    if not ns_token or ns_token not in text:
        return False
    if not _tool_block_emission_finished(text, tool_module):
        return False
    candidates = []
    try:
        from mlx_vlm.server.responses_state import process_tool_calls

        parsed = process_tool_calls(text, tool_module, tools)
        candidates.extend(parsed.get("calls") or [])
    except Exception:
        pass
    if not TOOL_COMPAT_OVERLAY:
        return False
    candidates.extend(_recover_jsonish_tool_calls(text, tool_module, tools))
    candidates.extend(
        _recover_bare_xml_argument_tool_calls(text, tool_module, tools)
    )
    candidates.extend(_recover_loose_segment_tool_calls(text, tool_module, tools))
    candidates.extend(_recover_malformed_xml_tool_calls(text, tool_module, tools))
    if not candidates:
        return False
    validated, dropped = _validate_outgoing_tool_calls(
        candidates,
        tools,
        return_dropped=True,
    )
    if dropped and not validated:
        return True
    return False


def _sanitize_inbound_message_content(role, content):
    """Remove prior hidden MiniMax reasoning markup before re-templating chat history."""
    if not isinstance(content, str):
        return content
    if role != "assistant":
        return _canonicalize_volatile_context_lines(content)
    content = _strip_thinking_control_markers(content).strip()
    if _looks_like_tool_compat_fallback_content(content):
        return ""
    if _looks_like_leaked_reasoning_content(content):
        return ""
    return content


def _looks_like_tool_compat_fallback_content(content):
    """Detect server-generated compatibility text that should not steer the model."""
    if not isinstance(content, str):
        return False
    text = re.sub(r"\s+", " ", content.strip().lower())
    if not text or len(text) > 500:
        return False
    markers = (
        "empty tool-call markers",
        "tool-call markers instead of a valid function name",
        "tool action was incomplete",
        "tool call was malformed",
        "call was malformed and was not executed",
        "could not form a complete tool call",
        "could not produce a valid tool call",
        "previous tool action was incomplete",
        "previous apply_patch call was malformed",
        "previous exec_command call was malformed",
        "previous tool call was malformed",
        "malformed apply_patch",
        "malformed exec_command",
        "could not complete that tool step",
        "previous `",
    )
    return any(marker in text for marker in markers)


def _scrub_goal_state_echo(content):
    """Cut a parroted goal-runner JSON state block from visible content.

    Codex-style goal harnesses inject a `{"prompt": ..., "goal": ...,
    "tools": [...]}` state payload into context each cycle; the model
    sometimes copies it verbatim into its visible answer.

    2026-07-06 audit: only treat the blob as an echo when it IS the answer
    (content starts with it). Prose that merely embeds such JSON — "show me
    an example request with tools" — is the user's requested output and was
    being silently deleted by the substring version of this guard.
    """
    if not isinstance(content, str) or '{"prompt"' not in content:
        return content
    if not content.strip().startswith('{"prompt"'):
        return content
    head, sep, tail = content.partition('{"prompt"')
    blob = sep + tail
    if '"tools"' in blob and '"function"' in blob:
        return head.strip()
    return content


def _looks_like_leaked_reasoning_content(content):
    """Detect reasoning text that a client echoed back as assistant content.

    2026-07-06 audit: this predicate used to flag ANY short text opening with
    "let me "/"i need to "/"i should " — killing legitimate replies like
    "Let me know if you want me to continue." and replacing them with the
    canned could-not-produce-a-tool-call fallback (then erasing them from
    re-templated history). A model never opens a reply to its user with
    third-person narration ABOUT the user, so that is the only reliable
    signature. First-person openers are common in real answers; a missed
    leak is cosmetic, a false positive destroys the answer — under-block.
    """
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text or len(text) > 1200:
        return False
    if text.endswith("?"):
        return False
    lowered = text.lower()
    starts_like_reasoning = lowered.startswith((
        "the user ",
        "user asked",
        "user wants",
    ))
    reasoning_phrases = (
        "no tools needed",
        "i should respond",
        "i should use",
        "i got a result",
        "the user asked",
        "the user wants",
        "the user just",
    )
    return bool(
        starts_like_reasoning and any(p in lowered for p in reasoning_phrases)
    )


def _should_emit_reasoning_fields(tools):
    return bool(not tools or EMIT_TOOL_REASONING)


def _tool_request_fallback_content(
    processed_messages,
    *,
    dropped_tool_names=None,
    available_tool_names=None,
    empty_tool_markers=False,
    thinking_mode=None,
):
    dropped_tool_names = [
        str(name).strip()
        for name in (dropped_tool_names or [])
        if str(name or "").strip()
    ]
    available_tool_names = sorted(
        str(name).strip()
        for name in (available_tool_names or [])
        if str(name or "").strip()
    )
    last_user = ""
    for message in reversed(processed_messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                last_user = content.strip()
            break
    if re.fullmatch(r"(?is)\s*(hi|hello|hey|hey!|hi!|hello!|yo|sup)[\s!.]*", last_user):
        return "Hey! How can I help?"
    # One stable sentence for every unusable shape (dropped call, empty
    # markers, incomplete action). It is only reached after the in-place
    # retries in _ensure_usable_tool_turn are exhausted. The phrase
    # "could not produce a valid tool call" is the marker keyed on by
    # _looks_like_tool_compat_fallback_content and the gateway's
    # unusable-content detector, so keep it verbatim.
    return (
        "I could not produce a valid tool call for this step, so it was not "
        "executed. Continue from the results already gathered."
    )


def _canonicalize_volatile_context_lines(text):
    """Stabilize UI-injected clock lines so they do not bust KV reuse every turn."""
    if not isinstance(text, str) or "Current" not in text:
        return text
    text = re.sub(
        r"(?im)^(\s*[-*]?\s*(?:Full\s+)?Current\s+datetime\s*:\s*"
        r"\d{4}-\d{2}-\d{2})(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?.*$",
        r"\1 [stable-time]",
        text,
    )
    text = re.sub(
        r"(?im)^(\s*[-*]?\s*Current\s+time\s*:\s*).*$",
        r"\1[stable]",
        text,
    )
    return text


def _assistant_content_for_template(message, content, *, session_id=None,
                                    session_source=None):
    """Return model-facing assistant content, restoring preserved reasoning.

    OpenAI-compatible UIs often keep visible assistant content separate from
    `reasoning` / `reasoning_content`. MiniMax generated that reasoning inside
    its native thinking block, so dropping it on the next request makes the new
    chat template diverge from the hot KV cache right where the previous
    assistant response begins. Reinsert preserved reasoning only for the
    model-facing prompt; the API response remains split into reasoning/content.
    """
    visible = _sanitize_inbound_message_content("assistant", content or "")
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
    )
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = _recall_assistant_reasoning(
            session_id,
            visible,
            tool_calls=message.get("tool_calls"),
            session_source=session_source,
        )
    if not isinstance(reasoning, str) or not reasoning.strip():
        return visible
    reasoning = _strip_thinking_control_markers(reasoning).strip()
    if not reasoning:
        return visible
    if visible:
        return f"<mm:think>{reasoning}</mm:think>\n{visible}"
    return f"<mm:think>{reasoning}</mm:think>"


def _client_preserves_assistant_reasoning(messages):
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
        )
        if isinstance(reasoning, str) and reasoning.strip():
            return True
    return False


def _has_model_facing_reasoning(content):
    return (
        isinstance(content, str)
        and "<mm:think>" in content
        and "</mm:think>" in content
    )


# Which generator the ACTIVE request runs. EOS-swap arming is safe only on
# the batch path: the stream generator pre-builds step N+1 before yielding N,
# so an injected EOS breaks its loop while asymmetric collectives are queued
# (the photographed mutual-send deadlock). Single generation slot => a plain
# module flag is race-free. Both ranks compute the same value.
_BATCH_PATH_ACTIVE = {"value": False}


def _capture_prompt_ids(gen_kwargs):
    """The ids fed during prefill: the suffix ids under prompt caching, else the
    tokenized prompt. Length must equal the captured prompt positions for the
    prompt_*.npz file to be written (m3_capture guards a mismatch)."""
    iid = gen_kwargs.get("input_ids")
    if iid is not None:
        try:
            return [int(x) for x in iid.reshape(-1).tolist()]
        except Exception:
            return None
    try:
        processor = gen_kwargs.get("processor")
        prompt = gen_kwargs.get("prompt")
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return list(tok.encode(prompt))
    except Exception:
        return None


def _capture_only_iter(m3_capture, gen_kwargs, it):
    """rank0 capture-only tee: pass every GenerationResult through untouched
    while recording the sampled token stream, then write the request's corpus
    files on completion. Token counting mirrors run_generation (dedup by
    generation_tokens) so K == the number of decode captures; m3_capture
    recovers the prompt/decode split as n_prompt = N_captured - K.

    Exit discipline: a clean finish or an early consumer stop (GeneratorExit,
    e.g. client disconnect/stop — the last forward already completed) both
    finalize the on-distribution sequence; a genuine generation error abandons
    the partial (possibly misaligned) accumulator."""
    prompt_ids = _capture_prompt_ids(gen_kwargs)
    m3_capture.begin_request()
    tokens = []
    seen = 0
    try:
        for response in it:
            gtoks = int(getattr(response, "generation_tokens", 0) or 0)
            tok = getattr(response, "token", None)
            if tok is not None and gtoks > seen:
                tokens.append(int(tok))
                seen = gtoks
            yield response
    except GeneratorExit:
        m3_capture.finalize_request(prompt_ids, tokens)
        raise
    except BaseException:
        m3_capture.abort_request()
        raise
    else:
        m3_capture.finalize_request(prompt_ids, tokens)


def _maybe_capture_wrap(rank, gen_kwargs, it):
    """Wrap the non-eagle generator with the capture-only tee when armed. Only
    rank0 accumulates/writes; rank1 rides the pipeline piggyback (env-gated in
    m3_pipeline_patch) but returns the raw iterator. Default OFF => returns `it`
    unchanged, zero behavioral change."""
    if rank != 0:
        return it
    try:
        import m3_capture
    except Exception:
        return it
    if not m3_capture.armed():
        return it
    return _capture_only_iter(m3_capture, gen_kwargs, it)


def _generation_iter(rank, gen_kwargs):
    """The request's token generator: upstream stream_generate, or the
    step-synchronous batch-cancel mirror loop when MLX_M3_BATCH_CANCEL=1.

    Unsupported shapes (multimodal, kv-quant) fall back to the stream path —
    that decision is a pure function of the broadcast request + shared env,
    so both ranks always pick the same generator. Any OTHER construction
    error raises into the existing symmetric generation-error protocol.
    """
    from mlx_vlm.generate import stream_generate
    try:
        from mlx_vlm.models.minimax_m3_vl import language as minimax_language

        begin_generation = getattr(
            minimax_language, "begin_decode_topk_generation", None
        )
        if callable(begin_generation):
            begin_generation()
    except Exception as e:
        logger.debug("decode top-k generation epoch unavailable: %s", e)
    import m3_batch_cancel
    import m3_eagle3

    _BATCH_PATH_ACTIVE["value"] = False
    # EAGLE3 speculative path (2026-07-09): per-request decision travels in
    # the broadcast request op (REQUEST_ACTIVE is set from it on BOTH ranks
    # before run_generation), so the generator choice is always symmetric.
    if m3_eagle3.enabled() and m3_eagle3.REQUEST_ACTIVE.get("value"):
        try:
            it = m3_eagle3.eagle3_stream_generate(rank=rank, **gen_kwargs)
            logger.info("rank %s: eagle3 speculative path active", rank)
            return it
        except m3_eagle3.Unsupported as e:
            logger.info("rank %s: eagle3 unsupported for request (%s); "
                        "falling through", rank, e)
    if m3_batch_cancel.enabled():
        try:
            it = m3_batch_cancel.batch_cancel_stream_generate(
                rank=rank, **gen_kwargs
            )
            _BATCH_PATH_ACTIVE["value"] = True
            return _maybe_capture_wrap(rank, gen_kwargs, it)
        except m3_batch_cancel.Unsupported as e:
            logger.info("rank %s: batch-cancel unsupported for request (%s); "
                        "using stream_generate", rank, e)
    return _maybe_capture_wrap(rank, gen_kwargs, stream_generate(**gen_kwargs))


# ---------------------------------------------------------------------------
# Constrained tool decoding (MLX_M3_CONSTRAINED_TOOLS, default OFF).
# The rank0 sampler hook (_synced_sample_with_positions) consults the module
# global constrained_tools.active(); these helpers arm/disarm it per request
# around the decode loop. Single-owner and generation_lock-serialized, mirroring
# the _FORCE_EOS pattern. Any failure falls back to unconstrained decode.
# ---------------------------------------------------------------------------
def _arm_constrained_tools(processor, tools, rank):
    try:
        import constrained_tools as _ctools
    except Exception:
        return
    _ctools.clear_active()          # never inherit a prior request's grammar
    if rank != 0 or not tools or not _ctools.env_enabled():
        return
    try:
        tk = getattr(processor, "tokenizer", processor)
        con = _ctools.build_from_request(tk, tools)
        if con is not None:
            _ctools.set_active(con)
            logger.info("constrained-tools armed (%d advertised tool(s))",
                        len(getattr(con, "tool_names", []) or []))
    except Exception as e:
        logger.warning("constrained-tools arm failed; decoding unconstrained: %s", e)


def _disarm_constrained_tools():
    try:
        import constrained_tools as _ctools
        _ctools.clear_active()
    except Exception:
        pass


def run_generation(model, processor, prompt, max_tokens, rank, image=None,
                   thinking_mode="adaptive", gen_params=None, progress_cb=None,
                   token_ids=None, session_id=None, session_source=None,
                   prefill_progress_cb=None,
                   reset_incomplete_thinking_on_limit=True,
                   tool_module=None, tools=None,
                   require_tool_call=False,
                   action_tool_task=False,
                   no_call_token_budget=None):
    """Both ranks run stream_generate in lockstep. On ANY error, raise so the
    caller can signal the partner and release memory.

    Robustness for long agentic sessions:
    - prefill_step_size: chunked prefill so long inputs don't cause a memory
      spike AND so each Metal command buffer completes within the GPU driver
      timeout (~10s). The value is configurable via MLX_M3_PREFILL_STEP_SIZE.
    - max_kv_size: passed through to upstream stream_generate. MiniMax-M3 has
      a custom KV cache, so high-context trimming must be proven by soak tests
      rather than assumed from generic MLX rotating-cache behavior.
    - token_ids: when prompt caching is enabled, both ranks receive the same
      token_ids (broadcast by rank 0) and compute the same prefix reuse, so
      only new context is processed on subsequent turns.
    - Stop flag: checked between tokens so POST /v1/stop can end generation
      at the next token boundary (both ranks check at the same boundary).
    """
    from mlx_vlm.generate import stream_generate

    # Disarm any stale stop from a PREVIOUS request. The stream path resets
    # this at the stop-nonce site, but the non-stream path never did — so one
    # /v1/stop poisoned every later non-stream request into an instant-EOS
    # empty reply ("generation complete: 0 chars", found 2026-07-07).
    _FORCE_EOS["active"] = False
    _refresh_generation_stream()

    # Cross-request prompt cache: compute suffix + reused cache (both ranks).
    prompt_to_send = prompt
    cached_prompt_cache = None
    cached_suffix_ids = None
    cache_marked_in_use = False
    cache_mode = _prompt_cache_mode_for_request(thinking_mode, token_ids)
    cache_allowed = _prompt_cache_allowed_for_generation(
        thinking_mode,
        token_ids,
        image,
    )
    if cache_allowed:
        _expire_idle_prompt_cache()
        _mark_prompt_cache_in_use(True)
        cache_marked_in_use = True
        try:
            prompt_to_send, cached_prompt_cache = _prepare_cached_prompt(
                model, processor, prompt, token_ids,
                session_id=session_id,
                session_source=session_source,
                thinking_mode=thinking_mode,
                append_reserve_tokens=max_tokens,
            )
            if (
                PROMPT_CACHE_DIRECT_SUFFIX_IDS
                and cached_prompt_cache is not None
                and image is None
            ):
                cached_suffix_ids = _prompt_cache_last_suffix_ids()
        except Exception:
            _mark_prompt_cache_in_use(False)
            cache_marked_in_use = False
            raise
        prompt_to_send, cached_prompt_cache, cached_suffix_ids = (
            _prefix_plan_consensus(rank, prompt, prompt_to_send,
                                   cached_prompt_cache, cached_suffix_ids)
        )

    gen_kwargs = dict(
        model=model, processor=processor, prompt=prompt_to_send,
        max_tokens=max_tokens,
        enable_thinking=_enable_thinking_for_generation(thinking_mode),
        prefill_step_size=_runtime_prefill_step_size(len(token_ids)),
        max_kv_size=MAX_KV_SIZE,
    )
    gen_kwargs.update(_kv_quant_kwargs())
    if cached_prompt_cache is not None:
        gen_kwargs["prompt_cache"] = cached_prompt_cache
    if cached_prompt_cache is not None and cached_suffix_ids:
        gen_kwargs["input_ids"] = mx.array([cached_suffix_ids], dtype=mx.int32)
        gen_kwargs["mask"] = None
    if image is not None:
        gen_kwargs["image"] = image
    if gen_params:
        gen_kwargs.update(gen_params)
    if prefill_progress_cb is not None or SAFE_DECODE_STOP:
        def _prefill_progress(processed_tokens, total_tokens):
            if prefill_progress_cb is not None:
                prefill_progress_cb(processed_tokens, total_tokens)
            if SAFE_DECODE_STOP:
                _check_prefill_stop(rank, processed_tokens, total_tokens)

        gen_kwargs["prefill_progress_callback"] = _prefill_progress

    # Size the watchdog's prefill stall window to this prompt (fix A): a large
    # prefill legitimately blocks in the jaccl recv longer than the fixed 240s.
    _watchdog_note_prefill(len(token_ids))

    text = ""
    n = 0
    generated_token_ids = []
    tool_accumulated = ""
    tool_call_started = False
    tool_complete_seen = False
    raw_tool_fragment_tokens = 0
    tool_detokenizer_silent_tokens = 0
    stopped = False
    runaway_budget = (
        TOOL_THINKING_RUNAWAY_TOKEN_BUDGET
        if tools else THINKING_RUNAWAY_TOKEN_BUDGET
    )
    if no_call_token_budget is not None:
        no_call_budget = max(0, int(no_call_token_budget))
    else:
        no_call_budget = (
            TOOL_NO_CALL_TOKEN_BUDGET
            if require_tool_call
            else TOOL_ACTION_NO_CALL_TOKEN_BUDGET
            if action_tool_task
            else TOOL_NO_CALL_TOKEN_BUDGET
        )
    write_scaffold_threshold = _tool_write_early_stop_chars()
    request_decode_reuse_state = _begin_request_decode_topk_reuse(tools, rank)
    _arm_constrained_tools(processor, tools, rank)
    try:
        force_eval = _decode_eval_force_for_request(thinking_mode, token_ids)
        with _tokenizer_runtime_lock, _decode_eval_context(force_eval):
            for response in _generation_iter(rank, gen_kwargs):
                generation_tokens = int(getattr(response, "generation_tokens", 0) or 0)
                token = getattr(response, "token", None)
                if token is not None and generation_tokens > len(generated_token_ids):
                    generated_token_ids.append(token)
                # 2026-07-06 TAIL-LOSS FIX: no manual EOS break. The native
                # 0.6.4 generator checks EOS itself on EVERY rank (the synced
                # token is identical), so both generators end naturally and
                # lockstep is preserved without our break. The old manual
                # break fired on the FINAL flush yield — which in 0.6.4
                # carries detokenizer.finalize()'s buffered tail text — and
                # silently dropped the last words of every response.
                if (
                    _FORCE_EOS.get("active")
                    and _FORCE_EOS.get("eos_id") is not None
                    and token is not None
                    and int(token) == int(_FORCE_EOS["eos_id"])
                ):
                    stopped = True  # bookkeeping only; generator ends itself
                token_text = getattr(response, "text", None) or ""
                if token_text:
                    tool_accumulated += token_text
                    if tools and not tool_call_started:
                        tool_call_started = _tool_call_started(
                            tool_accumulated,
                            tool_module,
                        )
                    if rank == 0:
                        text += token_text
                if tools and tool_call_started:
                    if token_text:
                        tool_detokenizer_silent_tokens = 0
                    else:
                        tool_detokenizer_silent_tokens += 1
                else:
                    tool_detokenizer_silent_tokens = 0
                n += 1
                if rank == 0 and progress_cb is not None:
                    metrics = {}
                    if n == 1:
                        metrics = {
                            "prompt_tps": float(getattr(response, "prompt_tps", 0.0) or 0.0),
                            "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                            "cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
                            "prompt_cache_prepare": _prompt_cache_status().get("last_prepare_event"),
                        }
                    try:
                        progress_cb(n, len(text), metrics)
                    except Exception:
                        try:
                            progress_cb(n, len(text))
                        except Exception:
                            pass
                _watchdog_tick(progress=True)
                if (
                    rank == 0
                    and _BATCH_PATH_ACTIVE["value"]
                    and not _FORCE_EOS.get("active")
                    and tools
                    and _completed_tool_detokenizer_tail_reached(
                        tool_detokenizer_silent_tokens,
                        tool_call_started,
                        tool_accumulated,
                        tool_module,
                        tools,
                    )
                ):
                    logger.warning(
                        "rank 0: tool detokenizer produced no text for %d "
                        "consecutive tokens at token %d; arming synchronized "
                        "EOS for bounded recovery",
                        tool_detokenizer_silent_tokens,
                        n,
                    )
                    if _arm_rank0_semantic_eos(
                        rank,
                        "tool_detokenizer_silent_tail",
                        n,
                    ):
                        stopped = True
                # Safe decode-phase stop (2026-07-06 EOS redesign): the stop
                # file exists only on rank 0, so rank 0 arms EOS injection in
                # the sampled-token sync instead of breaking; rank 1 simply
                # follows the synced EOS. `stopped` marks the turn cancelled
                # for cache handling once the loop ends naturally.
                # BATCH PATH ONLY: the stream generator pre-builds the next
                # step, so an injected EOS deadlocks it (rig 0/5 history).
                # Thinking-runaway guard: still inside <mm:think> well past
                # the budget => the model is looping in reasoning and will
                # burn to the ceiling. Arm the SAME proven EOS stop, gated on
                # visible content still being empty (thinking not yet closed)
                # so a turn writing a long answer is never clipped.
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and runaway_budget > 0
                        and thinking_mode == "enabled"
                        and n >= runaway_budget
                        and "</mm:think>" not in text
                        and "</think>" not in text):
                    logger.warning(
                        "rank 0: thinking-runaway guard at token %d "
                        "(still in <mm:think>, no visible answer%s) — forcing "
                        "EOS to release the slot",
                        n,
                        "; tool turn" if tools else "",
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and tools
                        and no_call_budget > 0
                        and n >= no_call_budget
                        and not tool_call_started
                        and (
                            require_tool_call
                            or action_tool_task
                            or _tool_intent_without_call(tool_accumulated)
                        )):
                    logger.warning(
                        "rank 0: no-call tool guard at token %d "
                        "(required=%s, action_task=%s, budget=%d, no call "
                        "marker started) — forcing EOS "
                        "for bounded retry",
                        n,
                        bool(require_tool_call),
                        bool(action_tool_task),
                        no_call_budget,
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and tools
                        and n % 8 == 0
                        and _incomplete_tool_call_budget_reached(
                            n,
                            tool_call_started,
                            tool_accumulated,
                            tool_module,
                        )):
                    logger.warning(
                        "rank 0: incomplete tool-call guard at token %d "
                        "(budget=%d, opened block never closed) — forcing "
                        "EOS for bounded retry",
                        n,
                        TOOL_INCOMPLETE_CALL_TOKEN_BUDGET,
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and tools
                        and tool_call_started
                        and write_scaffold_threshold > 0
                        and n % 8 == 0):
                    mutation_stop = _file_mutation_stop_info(
                        tool_accumulated,
                        tools,
                    ) or {}
                    oversized_payload_chars = int(
                        mutation_stop.get("payload_chars") or 0
                    )
                    mutation_threshold = int(
                        mutation_stop.get("threshold_chars") or 0
                    )
                    if (
                        mutation_threshold > 0
                        and
                        oversized_payload_chars
                        > mutation_threshold
                        and not _tool_call_complete_for_stop(
                            tool_accumulated,
                            tool_module,
                            tools,
                        )
                    ):
                        logger.warning(
                            "rank 0: oversized %s payload at token %d "
                            "(%d chars after invocation, limit=%d, hard=%d) "
                            "— forcing EOS for %s",
                            mutation_stop.get("kind") or "file mutation",
                            n,
                            oversized_payload_chars,
                            mutation_threshold,
                            TOOL_WRITE_CHUNK_MAX_CHARS,
                            (
                                "immediate scaffold"
                                if mutation_stop.get("scaffoldable")
                                else "bounded retry"
                            ),
                        )
                        _FORCE_EOS["active"] = True
                        stopped = True
                # Degenerate-repetition guard: force EOS on a tight copy-spiral
                # (any repeating unit), checked cheaply every 12 tokens.
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and DECODE_REPETITION_GUARD_TOKENS > 0
                        and n % DECODE_REPETITION_GUARD_TOKENS == 0
                        and n >= 48
                        and _looks_like_degenerate_repetition(text)):
                    logger.warning(
                        "rank 0: degenerate-repetition guard at token %d "
                        "(decode tail is a tight copy-spiral) — forcing EOS",
                        n,
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"] and n % 8 == 0):
                    _sp = _read_prefill_stop_file()
                    if (
                        _sp
                        and _sp.get("phase") in {"decode", "any"}
                        and _sp.get("nonce")
                        and _sp.get("nonce") == _STOP_NONCE.get("value")
                        and int(
                            _sp.get("decode_stop_at_tokens")
                            or _sp.get("stop_at_tokens")
                            or 0
                        ) > 0
                        and n >= int(
                            _sp.get("decode_stop_at_tokens")
                            or _sp.get("stop_at_tokens")
                            or 0
                        )
                    ):
                        logger.info(
                            "rank 0: decode stop honored at token %d — "
                            "forcing EOS via sampled-token sync "
                            "(stop_at=%s, reason=%s)",
                            n,
                            _sp.get("decode_stop_at_tokens")
                            or _sp.get("stop_at_tokens"),
                            _sp.get("reason"),
                        )
                        _FORCE_EOS["active"] = True
                        stopped = True
                # In-flight stop: both ranks check at the same token boundary.
                if UNSAFE_INFLIGHT_STOP:
                    stop_now, stop_reason = _stop_requested_synced_reason(rank, n)
                else:
                    stop_now, stop_reason = False, None
                if stop_now:
                    stopped = stop_reason != "tool"
                    logger.info(
                        "rank %s: %s stop requested at token %d; breaking",
                        rank,
                        stop_reason or "unknown",
                        n,
                    )
                    break
                tool_stop_ready = tool_complete_seen or _tool_call_complete_for_stop(
                    tool_accumulated, tool_module, tools
                )
                if tool_stop_ready:
                    if _BATCH_PATH_ACTIVE["value"] and BATCH_TOOL_NATURAL_DRAIN:
                        if not tool_complete_seen:
                            logger.info(
                                "rank %s: complete tool call detected at token "
                                "%d; consuming to natural synchronized EOS",
                                rank,
                                n,
                            )
                            tool_complete_seen = True
                        continue
                    if _BATCH_PATH_ACTIVE["value"]:
                        if rank == 0:
                            logger.info(
                                "rank 0: complete tool call detected at token "
                                "%d; arming synchronized EOS",
                                n,
                            )
                            _arm_rank0_semantic_eos(
                                rank,
                                "complete_tool_call",
                                n,
                            )
                        continue
                    # Upstream stream_generate may have a prefetched pipeline
                    # step in flight when its consumer receives this yield.
                    # Closing either local generator here can split the ranks;
                    # let the model's natural EOS finish unsupported shapes.
                    if rank == 0 and not tool_complete_seen:
                        logger.info(
                            "rank 0: complete tool call detected at token %d "
                            "on upstream generator; draining to natural EOS",
                            n,
                        )
                        tool_complete_seen = True
                    continue
                # Semantic parser state is process-local. Rank 1 must never
                # close its generator from this guard: one such close at token
                # 683 left rank 0 decoding past 5k until the watchdog killed
                # both processes. Rank 0 instead injects EOS into the existing
                # sampled-token sync, so both ranks finish at one boundary.
                if (
                    rank == 0
                    and not _FORCE_EOS.get("active")
                    and tools
                    and THINKING_RAW_SILENT_LIMIT > 0
                    and _tool_fragment_looks_degenerate(tool_accumulated, tool_module)
                ):
                    raw_tool_fragment_tokens += 1
                    if raw_tool_fragment_tokens >= THINKING_RAW_SILENT_LIMIT:
                        logger.warning(
                            "rank 0: incomplete raw tool fragment guard tripped "
                            "at token %d (raw_tool_tokens=%d, limit=%d); "
                            "arming synchronized EOS",
                            n,
                            raw_tool_fragment_tokens,
                            THINKING_RAW_SILENT_LIMIT,
                        )
                        if _arm_rank0_semantic_eos(
                            rank,
                            "incomplete_raw_tool_fragment",
                            n,
                        ):
                            stopped = True
                else:
                    raw_tool_fragment_tokens = 0
        # On clean completion, record the full token sequence in the cache.
        preserve_existing_cache = (
            cache_allowed and _prompt_cache_prepare_preserves_existing_cache()
        )
        incomplete_thinking = (
            bool(reset_incomplete_thinking_on_limit)
            and _thinking_generation_hit_limit(thinking_mode, n, max_tokens)
        )
        if cache_allowed and incomplete_thinking:
            logger.info(
                "prompt-cache: dropping cache after thinking generation hit "
                "max_tokens (%d/%s); treating turn as incomplete",
                n,
                max_tokens,
            )
            _reset_prompt_cache("reset after incomplete thinking output",
                                clear_resident=False)
        elif cache_allowed and not stopped and not preserve_existing_cache:
            _update_prompt_cache_after_generation(
                token_ids,
                generated_token_ids=generated_token_ids,
                generated_tokens=n,
                include_generated_ids=(cache_mode == "full"),
                session_id=session_id,
                session_source=session_source,
                prompt=prompt,
                model=model,
                processor=processor,
                save_reason="nonstream_generation",
            )
        elif preserve_existing_cache:
            logger.info("prompt-cache: preserved existing cache after bypassed request")
        if stopped:
            if not _keep_prompt_prefix_after_cancel(token_ids, "stop"):
                _reset_prompt_cache("reset after stop", clear_resident=False)
        return text if rank == 0 else None
    except GenerationCancelled as e:
        stopped = True
        logger.info("rank %s: %s; generation cancelled", rank, e)
        if not _keep_prompt_prefix_after_cancel(token_ids, "cancel"):
            _reset_prompt_cache("reset after stop", clear_resident=False)
        return text if rank == 0 else None
    except Exception as e:
        logger.error(f"rank {rank}: generation error at token {n}: {e}")
        _reset_prompt_cache("reset after error", clear_resident=False)
        raise
    finally:
        _disarm_constrained_tools()
        _restore_request_decode_topk_reuse(request_decode_reuse_state, rank)
        if cache_marked_in_use:
            _mark_prompt_cache_in_use(False)


def run_generation_stream(model, processor, prompt, max_tokens, rank, image=None,
                          thinking_mode="adaptive", enable_thinking=True,
                          gen_params=None, token_ids=None,
                          session_id=None, session_source=None,
                          prefill_progress_cb=None,
                          reset_incomplete_thinking_on_limit=True,
                          tool_module=None, tools=None,
                          require_tool_call=False,
                          action_tool_task=False):
    """Streaming generation. Yields delta dicts with keys from
    {"reasoning", "content"} in OpenAI delta format.

    Routing matches the official mlx_vlm server: we accumulate the raw text
    and call split_stream_thinking_delta on every token so reasoning/content
    are separated correctly even when </mm:think> spans token boundaries.

    Runs on BOTH ranks in lockstep (same forward calls). Only rank 0 yields.
    token_ids enables cross-request prompt caching (both ranks get the same
    ids via _bcast and compute identical prefix reuse).
    """
    from mlx_vlm.generate import stream_generate

    # Disarm any stale stop from a PREVIOUS request. The stream path resets
    # this at the stop-nonce site, but the non-stream path never did — so one
    # /v1/stop poisoned every later non-stream request into an instant-EOS
    # empty reply ("generation complete: 0 chars", found 2026-07-07).
    _FORCE_EOS["active"] = False
    _refresh_generation_stream()

    # Cross-request prompt cache: compute suffix + reused cache (both ranks).
    prompt_to_send = prompt
    cached_prompt_cache = None
    cached_suffix_ids = None
    cache_marked_in_use = False
    cache_mode = _prompt_cache_mode_for_request(thinking_mode, token_ids)
    cache_allowed = _prompt_cache_allowed_for_generation(
        thinking_mode,
        token_ids,
        image,
    )
    timing = {
        "cache_prepare_started_at": None,
        "cache_prepare_finished_at": None,
        "stream_generate_started_at": None,
        "runtime_lock_wait_started_at": None,
        "runtime_lock_acquired_at": None,
        "first_generator_yield_at": None,
    }
    if cache_allowed:
        _expire_idle_prompt_cache()
        _mark_prompt_cache_in_use(True)
        cache_marked_in_use = True
        try:
            timing["cache_prepare_started_at"] = time.time()
            prompt_to_send, cached_prompt_cache = _prepare_cached_prompt(
                model, processor, prompt, token_ids,
                session_id=session_id,
                session_source=session_source,
                thinking_mode=thinking_mode,
                append_reserve_tokens=max_tokens,
            )
            if (
                PROMPT_CACHE_DIRECT_SUFFIX_IDS
                and cached_prompt_cache is not None
                and image is None
            ):
                cached_suffix_ids = _prompt_cache_last_suffix_ids()
            prompt_to_send, cached_prompt_cache, cached_suffix_ids = (
                _prefix_plan_consensus(rank, prompt, prompt_to_send,
                                       cached_prompt_cache, cached_suffix_ids)
            )
            timing["cache_prepare_finished_at"] = time.time()
        except Exception:
            _mark_prompt_cache_in_use(False)
            cache_marked_in_use = False
            raise

    gen_kwargs = dict(
        model=model, processor=processor, prompt=prompt_to_send,
        max_tokens=max_tokens,
        enable_thinking=_enable_thinking_for_generation(thinking_mode),
        prefill_step_size=_runtime_prefill_step_size(len(token_ids)),
        max_kv_size=MAX_KV_SIZE,
    )
    gen_kwargs.update(_kv_quant_kwargs())
    if cached_prompt_cache is not None:
        gen_kwargs["prompt_cache"] = cached_prompt_cache
    if cached_prompt_cache is not None and cached_suffix_ids:
        gen_kwargs["input_ids"] = mx.array([cached_suffix_ids], dtype=mx.int32)
        gen_kwargs["mask"] = None
    if image is not None:
        gen_kwargs["image"] = image
    if gen_params:
        gen_kwargs.update(gen_params)
    if prefill_progress_cb is not None or SAFE_DECODE_STOP:
        def _prefill_progress(processed_tokens, total_tokens):
            if prefill_progress_cb is not None:
                prefill_progress_cb(processed_tokens, total_tokens)
            if SAFE_DECODE_STOP:
                _check_prefill_stop(rank, processed_tokens, total_tokens)

        gen_kwargs["prefill_progress_callback"] = _prefill_progress

    # Size the watchdog's prefill stall window to this prompt (fix A): a large
    # prefill legitimately blocks in the jaccl recv longer than the fixed 240s.
    _watchdog_note_prefill(len(token_ids))

    # Routing state (mirrors mlx_vlm/server/openai.py:1493-1497)
    in_thinking = bool(enable_thinking)
    accumulated = ""
    raw_tail = ""  # rolling raw-text tail for the repetition guard
    tool_guard_text = ""
    tool_call_started = False
    at_response_start = True
    n = 0
    generated_token_ids = []
    stopped = False
    malformed_thinking = False
    raw_silent_tokens = 0
    tool_detokenizer_silent_tokens = 0
    tool_complete_seen = False
    thinking_active = _enable_thinking_for_generation(thinking_mode)
    runaway_budget = (
        TOOL_THINKING_RUNAWAY_TOKEN_BUDGET
        if tools else THINKING_RUNAWAY_TOKEN_BUDGET
    )
    no_call_budget = (
        TOOL_NO_CALL_TOKEN_BUDGET
        if require_tool_call
        else TOOL_ACTION_NO_CALL_TOKEN_BUDGET
        if action_tool_task
        else TOOL_NO_CALL_TOKEN_BUDGET
    )
    write_scaffold_threshold = _tool_write_early_stop_chars()
    request_decode_reuse_state = _begin_request_decode_topk_reuse(tools, rank)
    _arm_constrained_tools(processor, tools, rank)
    try:
        force_eval = _decode_eval_force_for_request(thinking_mode, token_ids)
        timing["stream_generate_started_at"] = time.time()
        timing["runtime_lock_wait_started_at"] = time.time()
        with _tokenizer_runtime_lock, _decode_eval_context(force_eval):
            timing["runtime_lock_acquired_at"] = time.time()
            for response in _generation_iter(rank, gen_kwargs):
                if timing["first_generator_yield_at"] is None:
                    timing["first_generator_yield_at"] = time.time()
                    if rank == 0:
                        _t0 = timing.get("cache_prepare_started_at")
                        _t1 = timing.get("cache_prepare_finished_at")
                        _lw = timing.get("runtime_lock_wait_started_at")
                        _la = timing.get("runtime_lock_acquired_at")
                        _fy = timing["first_generator_yield_at"]
                        logger.info(
                            "[rank 0] ttft breakdown: prepare=%.2fs lock_wait=%.2fs "
                            "generator=%.2fs (prepare start -> first yield %.2fs)",
                            (_t1 - _t0) if _t0 and _t1 else -1,
                            (_la - _lw) if _lw and _la else -1,
                            _fy - (_la or _fy),
                            _fy - (_t0 or _fy),
                        )
                generation_tokens = int(getattr(response, "generation_tokens", 0) or 0)
                token = getattr(response, "token", None)
                if token is not None and generation_tokens > len(generated_token_ids):
                    generated_token_ids.append(token)
                # 2026-07-06 TAIL-LOSS FIX: no manual EOS break (see mirror
                # loop). The 0.6.4 generator ends on EOS itself on every rank;
                # the final flush yield carries the buffered tail text and
                # must be processed, not skipped.
                if (
                    _FORCE_EOS.get("active")
                    and _FORCE_EOS.get("eos_id") is not None
                    and token is not None
                    and int(token) == int(_FORCE_EOS["eos_id"])
                ):
                    stopped = True  # bookkeeping only; generator ends itself
                token_text = getattr(response, "text", None) or ""
                n += 1
                # Rolling raw-text tail: `accumulated` is CONSUMED by
                # split_stream_thinking_delta each token (it is a pending
                # buffer, near-empty most steps), so guards must not read it
                # for history. 2026-07-10 ca0f2748: a retry spiraled 3.8k
                # tokens of bare markers past a guard checking `accumulated`.
                raw_tail = (raw_tail + token_text)[-600:]
                if tools and token_text:
                    tool_guard_text += token_text
                    if not tool_call_started:
                        tool_call_started = _tool_call_started(
                            tool_guard_text,
                            tool_module,
                        )
                if tools and tool_call_started:
                    if token_text:
                        tool_detokenizer_silent_tokens = 0
                    else:
                        tool_detokenizer_silent_tokens += 1
                else:
                    tool_detokenizer_silent_tokens = 0
                _watchdog_tick(progress=True)
                if (
                    rank == 0
                    and _BATCH_PATH_ACTIVE["value"]
                    and not _FORCE_EOS.get("active")
                    and tools
                    and _completed_tool_detokenizer_tail_reached(
                        tool_detokenizer_silent_tokens,
                        tool_call_started,
                        tool_guard_text,
                        tool_module,
                        tools,
                    )
                ):
                    logger.warning(
                        "rank 0: streaming tool detokenizer produced no text "
                        "for %d consecutive tokens at token %d; arming "
                        "synchronized EOS for bounded recovery",
                        tool_detokenizer_silent_tokens,
                        n,
                    )
                    if _arm_rank0_semantic_eos(
                        rank,
                        "stream_tool_detokenizer_silent_tail",
                        n,
                    ):
                        stopped = True
                # Safe decode-phase stop: rank 0 watches the coordinated stop
                # file every 8 tokens and, when the boundary is reached, ARMS
                # EOS injection (it does NOT break here — that would desync
                # rank 1). The EOS then flows to both ranks through the sync
                # and the unconditional EOS break above ends both identically.
                # BATCH PATH ONLY: the stream generator pre-builds the next
                # step, so an injected EOS deadlocks it (rig 0/5 history).
                # Thinking-runaway guard: still inside <mm:think> well past
                # the budget => the model is looping in reasoning and will
                # burn to the ceiling. Arm the SAME proven EOS stop, gated on
                # the splitter's authoritative in_thinking state (substring
                # checks on the consumed `accumulated` buffer never matched).
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and runaway_budget > 0
                        and thinking_mode == "enabled"
                        and n >= runaway_budget
                        and in_thinking):
                    logger.warning(
                        "rank 0: thinking-runaway guard at token %d "
                        "(still in <mm:think>, no visible answer%s) — forcing "
                        "EOS to release the slot",
                        n,
                        "; tool turn" if tools else "",
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and tools
                        and no_call_budget > 0
                        and n >= no_call_budget
                        and not tool_call_started
                        and (
                            require_tool_call
                            or action_tool_task
                            or _tool_intent_without_call(tool_guard_text)
                        )):
                    logger.warning(
                        "rank 0: no-call tool guard at token %d "
                        "(required=%s, action_task=%s, budget=%d, no call "
                        "marker started) — forcing EOS "
                        "for bounded retry",
                        n,
                        bool(require_tool_call),
                        bool(action_tool_task),
                        no_call_budget,
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and tools
                        and n % 8 == 0
                        and _incomplete_tool_call_budget_reached(
                            n,
                            tool_call_started,
                            tool_guard_text,
                            tool_module,
                        )):
                    logger.warning(
                        "rank 0: incomplete streaming tool-call guard at "
                        "token %d (budget=%d, opened block never closed) — "
                        "forcing EOS for bounded retry",
                        n,
                        TOOL_INCOMPLETE_CALL_TOKEN_BUDGET,
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and tools
                        and tool_call_started
                        and write_scaffold_threshold > 0
                        and n % 8 == 0):
                    mutation_stop = _file_mutation_stop_info(
                        tool_guard_text,
                        tools,
                    ) or {}
                    oversized_payload_chars = int(
                        mutation_stop.get("payload_chars") or 0
                    )
                    mutation_threshold = int(
                        mutation_stop.get("threshold_chars") or 0
                    )
                    if (
                        mutation_threshold > 0
                        and
                        oversized_payload_chars
                        > mutation_threshold
                        and not _tool_call_complete_for_stop(
                            tool_guard_text,
                            tool_module,
                            tools,
                        )
                    ):
                        logger.warning(
                            "rank 0: oversized streaming %s payload at token "
                            "%d (%d chars after invocation, limit=%d, "
                            "hard=%d) — forcing EOS for %s",
                            mutation_stop.get("kind") or "file mutation",
                            n,
                            oversized_payload_chars,
                            mutation_threshold,
                            TOOL_WRITE_CHUNK_MAX_CHARS,
                            (
                                "immediate scaffold"
                                if mutation_stop.get("scaffoldable")
                                else "bounded retry"
                            ),
                        )
                        _FORCE_EOS["active"] = True
                        stopped = True
                # Degenerate-repetition guard on the raw tail.
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"]
                        and DECODE_REPETITION_GUARD_TOKENS > 0
                        and n % DECODE_REPETITION_GUARD_TOKENS == 0
                        and n >= 48
                        and _looks_like_degenerate_repetition(raw_tail)):
                    logger.warning(
                        "rank 0: degenerate-repetition guard at token %d "
                        "(decode tail is a tight copy-spiral) — forcing EOS",
                        n,
                    )
                    _FORCE_EOS["active"] = True
                    stopped = True
                if (rank == 0 and _BATCH_PATH_ACTIVE["value"]
                        and not _FORCE_EOS["active"] and n % 8 == 0):
                    _sp = _read_prefill_stop_file()
                    if (
                        _sp
                        and _sp.get("phase") in {"decode", "any"}
                        # nonce match = this stop belongs to THIS generation
                        and _sp.get("nonce")
                        and _sp.get("nonce") == _STOP_NONCE.get("value")
                        and int(
                            _sp.get("decode_stop_at_tokens")
                            or _sp.get("stop_at_tokens")
                            or 0
                        ) > 0
                        and n >= int(
                            _sp.get("decode_stop_at_tokens")
                            or _sp.get("stop_at_tokens")
                            or 0
                        )
                    ):
                        logger.info(
                            "rank 0: decode stop honored at token %d — "
                            "forcing EOS via sampled-token sync "
                            "(stop_at=%s, reason=%s)",
                            n,
                            _sp.get("decode_stop_at_tokens")
                            or _sp.get("stop_at_tokens"),
                            _sp.get("reason"),
                        )
                        _FORCE_EOS["active"] = True
                        stopped = True
                # Check distributed stop before local stream guards. A prior
                # tool-complete stop may already be armed; if rank 0 trips a
                # local guard first, the next request can enter JACCL with the
                # mirror rank still in the previous decode loop.
                if UNSAFE_INFLIGHT_STOP:
                    stop_now, stop_reason = _stop_requested_synced_reason(rank, n)
                else:
                    stop_now, stop_reason = False, None
                if stop_now:
                    stopped = stop_reason != "tool"
                    logger.info(
                        "rank %s: %s stop requested at token %d; breaking",
                        rank,
                        stop_reason or "unknown",
                        n,
                    )
                    break
                accumulated += token_text
                (in_thinking, accumulated, at_response_start,
                 delta_reasoning, delta_content) = split_stream_thinking_delta(
                    accumulated, token_text, in_thinking,
                    at_response_start=at_response_start,
                )
                routed_delta_text = (delta_reasoning or "") + (delta_content or "")
                visible_delta_text = _strip_thinking_control_markers(
                    routed_delta_text
                ).strip()
                # Count decode steps that do not produce visible routed
                # reasoning/content. MiniMax can emit long runs of control
                # markup that split_stream_thinking_delta technically routes
                # but the OpenAI stream will not expose as useful progress.
                # Treat those as silent in both thinking and no-thinking modes
                # so we break inside the generation loop before the distributed
                # process watchdog has to kill both ranks.
                if visible_delta_text:
                    raw_silent_tokens = 0
                else:
                    # A HEALTHY tool call is invisible by design: after the
                    # ns marker the holdback silences the client while the
                    # block buffers, so a big Write/Edit call is hundreds of
                    # "silent" tokens. Killing at 32 truncated every large
                    # call two tokens into <invoke (12:45 specimen) — the
                    # dropped-action bug. Buffering an OPEN, non-degenerate
                    # block is progress; only true silence and marker-spam
                    # (the degenerate classifier) count toward the limit.
                    _tool_buffering = bool(
                        tools and tool_module is not None
                        and _looks_like_raw_tool_fragment(accumulated, tool_module)
                        and not _tool_block_emission_finished(accumulated, tool_module)
                        and not _tool_fragment_looks_degenerate(accumulated, tool_module)
                    )
                    if _tool_buffering:
                        raw_silent_tokens = 0
                    else:
                        raw_silent_tokens += 1

                if (
                    rank == 0
                    and _BATCH_PATH_ACTIVE["value"]
                    and not _FORCE_EOS.get("active")
                    and THINKING_RAW_SILENT_LIMIT > 0
                    and not tools  # oMLX PARITY (2026-07-07): oMLX has NO
                    # no-visible guard. On a TOOL turn the holdback silences
                    # the client for the whole (legitimately long) tool block,
                    # and in thinking mode the model reasons for thousands of
                    # tokens before the call — the guard fired at raw_silent=32
                    # and killed generation right at <invoke (token 2841,
                    # every Flappy-Bird/file-write turn). Marker-spam runaway
                    # is still caught by the degenerate-fragment guard; total
                    # runaway by max_tokens + the decode watchdog. The
                    # no-visible guard now protects ONLY toolless turns.
                    and n >= THINKING_RAW_SILENT_LIMIT
                    and raw_silent_tokens >= THINKING_RAW_SILENT_LIMIT
                ):
                    malformed_thinking = True
                    logger.warning(
                        "rank 0: no-visible stream guard tripped at token %d "
                        "(raw_silent=%d, limit=%d, thinking=%s); arming "
                        "synchronized EOS",
                        n,
                        raw_silent_tokens,
                        THINKING_RAW_SILENT_LIMIT,
                        thinking_active,
                    )
                    if _arm_rank0_semantic_eos(
                        rank,
                        "no_visible_stream",
                        n,
                    ):
                        stopped = True
                tool_stop_ready = tool_complete_seen or _tool_call_complete_for_stop(
                    tool_guard_text if tools else accumulated,
                    tool_module,
                    tools,
                )
                if rank != 0:
                    if tool_stop_ready:
                        if (
                            _BATCH_PATH_ACTIVE["value"]
                            and BATCH_TOOL_NATURAL_DRAIN
                        ):
                            if not tool_complete_seen:
                                logger.info(
                                    "rank %s: complete tool call detected at "
                                    "token %d; consuming mirror to natural "
                                    "synchronized EOS",
                                    rank,
                                    n,
                                )
                                tool_complete_seen = True
                            continue
                        # Rank 1 never turns semantic parser state into local
                        # control flow. Rank 0 either injects synchronized EOS
                        # on the batch path or both ranks drain natural EOS on
                        # the upstream fallback path.
                        continue
                    continue  # mirror rank: run forward calls, but emit nothing

                metrics = {}
                if n == 1:
                    metrics = {
                        "_prompt_tps": float(getattr(response, "prompt_tps", 0.0) or 0.0),
                        "_prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                        "_cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
                        "_prompt_cache_prepare": _prompt_cache_status().get("last_prepare_event"),
                        "_cache_prepare_started_at": timing["cache_prepare_started_at"],
                        "_cache_prepare_finished_at": timing["cache_prepare_finished_at"],
                        "_stream_generate_started_at": timing["stream_generate_started_at"],
                        "_runtime_lock_wait_started_at": timing["runtime_lock_wait_started_at"],
                        "_runtime_lock_acquired_at": timing["runtime_lock_acquired_at"],
                        "_first_generator_yield_at": timing["first_generator_yield_at"],
                    }
                metrics["_generation_tokens"] = generation_tokens or n
                raw_attached = False
                if delta_reasoning:
                    yield {"reasoning": delta_reasoning, "_raw": token_text} | metrics
                    raw_attached = True
                if delta_content:
                    chunk = {"content": delta_content}
                    if not raw_attached:
                        chunk["_raw"] = token_text
                        raw_attached = True
                    yield chunk | metrics
                if token_text and not delta_reasoning and not delta_content:
                    yield {"_raw": token_text} | metrics
                elif metrics and not delta_reasoning and not delta_content and not token_text:
                    yield metrics

                if tool_stop_ready:
                    if _BATCH_PATH_ACTIVE["value"] and BATCH_TOOL_NATURAL_DRAIN:
                        if not tool_complete_seen:
                            logger.info(
                                "rank %s: complete tool call detected at token "
                                "%d; consuming stream to natural synchronized "
                                "EOS",
                                rank,
                                n,
                            )
                            tool_complete_seen = True
                        continue
                    if _BATCH_PATH_ACTIVE["value"]:
                        logger.info(
                            "rank 0: complete tool call detected at token %d; "
                            "arming synchronized stream EOS",
                            n,
                        )
                        _arm_rank0_semantic_eos(
                            rank,
                            "complete_stream_tool_call",
                            n,
                        )
                    elif not tool_complete_seen:
                        logger.info(
                            "rank 0: complete tool call detected at token %d "
                            "on upstream stream; draining to natural EOS",
                            n,
                        )
                        tool_complete_seen = True
                    continue

        # Flush any trailing buffered text
        if rank == 0 and accumulated and not stopped:
            if malformed_thinking:
                fallback = _strip_thinking_control_markers(accumulated).strip()
                if fallback:
                    yield {
                        "content": fallback,
                        "_malformed_thinking_truncated": True,
                    }
            else:
                reasoning, content = split_thinking_text(
                    accumulated, assume_in_thinking=in_thinking)
                if reasoning:
                    yield {"reasoning": reasoning}
                if content:
                    yield {"content": content}
        # On clean completion, record the full token sequence in the cache.
        if cache_allowed:
            preserve_existing_cache = _prompt_cache_prepare_preserves_existing_cache()
            incomplete_thinking = (
                bool(reset_incomplete_thinking_on_limit)
                and _thinking_generation_hit_limit(thinking_mode, n, max_tokens)
            )
            if stopped:
                if not _keep_prompt_prefix_after_cancel(token_ids, "stream stop"):
                    _reset_prompt_cache("reset after stop", clear_resident=False)
            elif malformed_thinking:
                _reset_prompt_cache("reset after no-visible generation stream")
            elif incomplete_thinking:
                logger.info(
                    "prompt-cache: dropping cache after thinking stream hit "
                    "max_tokens (%d/%s); treating turn as incomplete",
                    n,
                    max_tokens,
                )
                _reset_prompt_cache("reset after incomplete thinking output",
                                clear_resident=False)
            elif preserve_existing_cache:
                logger.info("prompt-cache: preserved existing cache after bypassed stream request")
            else:
                _update_prompt_cache_after_generation(
                    token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_tokens=n,
                    include_generated_ids=(cache_mode == "full"),
                    session_id=session_id,
                    session_source=session_source,
                    prompt=prompt,
                    model=model,
                    processor=processor,
                    save_reason="stream_generation",
                )
    except GenerationCancelled as e:
        stopped = True
        logger.info("rank %s: %s; stream generation cancelled", rank, e)
        if not _keep_prompt_prefix_after_cancel(token_ids, "stream cancel"):
            _reset_prompt_cache("reset after stop", clear_resident=False)
        if rank == 0:
            yield {
                "_cancelled": True,
                "_raw": accumulated,
                "_generation_tokens": n,
            }
        return
    except Exception as e:
        logger.error(f"rank {rank}: stream generation error at token {n}: {e}")
        _reset_prompt_cache("reset after error", clear_resident=False)
        raise
    finally:
        _disarm_constrained_tools()
        _restore_request_decode_topk_reuse(request_decode_reuse_state, rank)
        if cache_marked_in_use:
            _mark_prompt_cache_in_use(False)


def _materialize_image(image_source):
    """Return a local temp-file path for an OpenAI image_url/input_image value."""
    if not image_source:
        return None

    if isinstance(image_source, dict):
        image_source = image_source.get("url") or image_source.get("image_url")
    if not isinstance(image_source, str) or not image_source:
        return None

    import base64
    import tempfile
    import urllib.request

    if image_source.startswith("data:"):
        header, b64 = image_source.split(",", 1)
        suffix = ".png"
        if "jpeg" in header or "jpg" in header:
            suffix = ".jpg"
        elif "webp" in header:
            suffix = ".webp"
        data = base64.b64decode(b64)
        tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tf.write(data)
        tf.close()
        return tf.name

    tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    urllib.request.urlretrieve(image_source, tf.name)
    tf.close()
    return tf.name


def _extract_image_source(part):
    """Extract an image URL/path/data URI from OpenAI chat/response content."""
    if not isinstance(part, dict):
        return None
    for key in ("image_url", "input_image", "image", "url", "path"):
        value = part.get(key)
        if isinstance(value, dict):
            value = value.get("url") or value.get("image_url") or value.get("path")
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _install_rank0_token_sync(group):
    """Force every distributed rank to feed rank 0's sampled token.

    MLX generation runs the sampler independently in each process. That is OK
    only while every rank produces exactly the same token. MiniMax-M3 tensor and
    pipeline modes both depend on the next decode step seeing the same token ids
    on every rank, so gather the sampled token and keep rank 0 as the source of
    truth.
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
        logger.warning("rank0 sampled-token sync unavailable: %s", e)
        return

    orig = ar_mod._sample_with_positions
    if getattr(orig, "_m3_rank0_token_sync", False):
        return

    sync_rank = group.rank()
    try:
        import constrained_tools as _ctools
    except Exception:
        _ctools = None
    _ct_on = _ctools is not None and _ctools.env_enabled()

    def _synced_sample_with_positions(*args, **kwargs):
        # Constrained tool decoding (rank0 only): mask the logits BEFORE the
        # sampler draws so malformed tool-markup tokens are unsamplable, then
        # fold the sampled token into the automaton. No-op unless the env flag
        # is on AND a per-request grammar is armed. rank>0 never masks — it
        # consumes rank0's token via the all_gather below.
        con = _ctools.active() if (_ct_on and sync_rank == 0) else None
        if con is not None and len(args) >= 2:
            try:
                masked = con.mask_logits(args[1])
                if masked is not args[1]:
                    args = (args[0], masked) + tuple(args[2:])
            except Exception:
                con = None
        y = orig(*args, **kwargs)
        # Decode stop = rank 0 forces EOS as its sampled token; every rank
        # receives it through this same all_gather and the generation ends
        # identically everywhere (no per-rank stop files, no extra
        # collectives, no break-point drift).
        if (
            sync_rank == 0
            and _FORCE_EOS["active"]
            and _FORCE_EOS["eos_id"] is not None
        ):
            y = mx.full(y.shape, _FORCE_EOS["eos_id"], dtype=y.dtype)
        if con is not None:
            # fold the actually-sampled token (post force-eos) into the automaton
            try:
                con.observe(int(y.reshape(-1)[0]))
            except Exception:
                pass
        gathered = mx.distributed.all_gather(y, group=group)
        return gathered[: y.shape[0]]

    _synced_sample_with_positions._m3_rank0_token_sync = True
    _synced_sample_with_positions._m3_original = orig
    ar_mod._sample_with_positions = _synced_sample_with_positions
    logger.info("rank0 sampled-token sync enabled")


def sharded_load_tensor(repo):
    """Load MiniMax-M3 using its built-in tensor-parallel language shard.

    mlx_vlm.utils.sharded_load() only checks the top-level VLM wrapper for
    .shard(), but MiniMax-M3 exposes tensor parallelism on
    model.language_model.shard(group). This path keeps the official MiniMax
    forward/cache implementation intact and avoids the custom layer-pipeline
    forward that currently stalls at the sparse-attention block boundary.
    """
    from mlx_vlm.utils import (
        get_model_path,
        load_model,
        load_processor,
        load_image_processor,
    )

    group = mx.distributed.init()
    model_path = get_model_path(repo)
    model = load_model(model_path, lazy=True, strict=False)
    if not hasattr(model, "language_model") or not hasattr(model.language_model, "shard"):
        raise ValueError("MiniMax tensor loader requires model.language_model.shard()")
    model.language_model.shard(group)
    logger.info("rank %s: applied MiniMax language tensor shard", group.rank())

    logger.info("rank %s: materializing tensor-sharded model", group.rank())
    mx.eval(model.language_model.parameters())
    model.eval()
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))

    eos_token_id = getattr(model.config, "eos_token_id", None)
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _install_shutdown_handlers()

    # NOTE: we do NOT override mx.async_eval. An earlier version forced it
    # synchronous (mx.async_eval = lambda *a: mx.eval(list(a))) based on a
    # theory that async command buffers cause GPU timeouts. That theory was
    # WRONG — the real crash causes were (a) the mx.eval(h) after all_gather
    # in the pipeline patch and (b) the watchdog false-positive during long
    # prefills, both now fixed. Forcing async_eval synchronous destroys
    # prefill performance (no compute/transfer overlap -> 64-token prompts
    # went from ~6s to >90s) AND starves the watchdog's heartbeat ticker
    # thread of the GIL. The canonical mlx_lm server and glm4_moe pipeline
    # never override async_eval. Leave it native.
    logger.info("mx.async_eval left native (canonical behavior)")
    logger.info(
        "prompt cache: enabled=%s min_reuse=%s ttl=%ss max_tokens=%s ceiling=%s",
        PROMPT_CACHE_ENABLED,
        PROMPT_CACHE_MIN_REUSE if PROMPT_CACHE_ENABLED else "n/a",
        PROMPT_CACHE_TTL_SECONDS if PROMPT_CACHE_ENABLED else "n/a",
        PROMPT_CACHE_MAX_TOKENS if PROMPT_CACHE_MAX_TOKENS > 0 else "off",
        MAX_TOKENS_CEILING,
    )
    _install_omlx_minimax_overlay()
    _configure_metal_memory_limits()
    _install_decode_eval_patch()

    group = mx.distributed.init()
    world, rank = group.size(), group.rank()
    logging.getLogger().handlers[0].setFormatter(
        logging.Formatter(f"%(asctime)s [rank {rank}] %(levelname)s %(message)s"))
    logger.info(f"distributed init OK: rank {rank} of {world}")
    if SHARDING_MODE == "tensor":
        _install_rank0_token_sync(group)

    if SHARDING_MODE == "pipeline":
        from m3_pipeline_patch import sharded_load_pipeline
        logger.info(f"sharded_load_pipeline on rank {rank} ...")
        model, processor = sharded_load_pipeline(MODEL)
    elif SHARDING_MODE == "tensor":
        logger.info(f"sharded_load_tensor on rank {rank} ...")
        model, processor = sharded_load_tensor(MODEL)
    else:
        raise ValueError(f"unsupported M3_SHARDING_MODE={SHARDING_MODE!r}")
    logger.info(f"rank {rank}: model loaded")
    try:
        _tok = getattr(processor, "tokenizer", None) or processor
        _eid = getattr(_tok, "eos_token_id", None)
        if isinstance(_eid, (list, tuple, set)):
            _eid = next(iter(_eid), None)
        _FORCE_EOS["eos_id"] = int(_eid) if _eid is not None else None
        logger.info("decode-stop EOS id: %s", _FORCE_EOS["eos_id"])
    except Exception as e:
        logger.warning("decode-stop EOS id unavailable: %s", e)
    with _prompt_cache_lock:
        _load_prompt_cache_session_manifest_unlocked()
    _start_prompt_cache_janitor()

    if rank == 0:
        run_http_server(model, processor, rank)
    else:
        run_mirror(model, processor, rank)


def run_http_server(model, processor, rank):
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    generation_lock = threading.Lock()
    generation_lock_owner = {
        "kind": None,
        "started_at": None,
        "reason": None,
        "changed_at": time.time(),
    }

    def set_generation_lock_owner(kind, reason=None):
        with state_lock:
            now = time.time()
            generation_lock_owner["kind"] = kind
            generation_lock_owner["started_at"] = now if kind else None
            generation_lock_owner["reason"] = reason
            generation_lock_owner["changed_at"] = now

    def clear_generation_lock_owner(kind=None, reason=None, changed_at=None):
        with state_lock:
            if kind is not None and generation_lock_owner.get("kind") != kind:
                return False
            if reason is not None and generation_lock_owner.get("reason") != reason:
                return False
            if (
                changed_at is not None
                and generation_lock_owner.get("changed_at") != changed_at
            ):
                return False
            generation_lock_owner["kind"] = None
            generation_lock_owner["started_at"] = None
            generation_lock_owner["reason"] = None
            generation_lock_owner["changed_at"] = time.time()
            return True
    state_lock = threading.Lock()
    request_state = {
        "queued": 0,
        "active": None,
        "completed": 0,
        "failed": 0,
        "last_error": None,
        "releasing": None,
        "shutdown_requested": False,
        # last_request: summary of the most recently completed request, so the
        # dashboard can show an authoritative tokens/sec per request.
        "last_request": None,
        # Latest request with enough output tokens for decode TPS to be
        # meaningful. Very short replies are dominated by TTFT and prompt work.
        "last_meaningful_request": None,
        "recent_requests": [],
        "lifetime_tokens": {
            "requests": 0,
            "prompt_processed": 0,
            "prompt_logical": 0,
            "prompt_cached": 0,
            "prompt_avoided": 0,
            "decode": 0,
            "processed_total": 0,
            "logical_total": 0,
        },
    }
    nonstream_coalescer = _NonstreamRequestCoalescer(
        enabled=NONSTREAM_COALESCE_ENABLED,
        replay_grace_seconds=NONSTREAM_COALESCE_GRACE_SECONDS,
        disconnect_grace_seconds=NONSTREAM_DISCONNECT_GRACE_SECONDS,
        max_entries=NONSTREAM_COALESCE_MAX_ENTRIES,
    )

    # Lifetime token counters survive restarts: overlay the last persisted
    # snapshot at boot and rewrite it (atomically) after every release.
    lifetime_tokens_path = os.environ.get(
        "MLX_M3_LIFETIME_TOKENS_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "lifetime_tokens.json"),
    )

    def _load_lifetime_tokens():
        try:
            with open(lifetime_tokens_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("lifetime tokens load failed (%s); starting fresh", exc)
            return
        if not isinstance(saved, dict):
            return
        lifetime = request_state["lifetime_tokens"]
        for key in lifetime:
            try:
                lifetime[key] = int((saved.get("lifetime_tokens") or {}).get(key) or 0)
            except (TypeError, ValueError):
                pass
        for key in ("completed", "failed"):
            try:
                request_state[key] = int(saved.get(key) or 0)
            except (TypeError, ValueError):
                pass
        logger.info(
            "lifetime tokens restored: %s requests, %s tokens total",
            lifetime.get("requests"), lifetime.get("processed_total"),
        )

    def _persist_lifetime_tokens():
        if rank != 0:
            return
        with state_lock:
            snapshot = {
                "lifetime_tokens": dict(request_state["lifetime_tokens"]),
                "completed": request_state.get("completed"),
                "failed": request_state.get("failed"),
                "saved_at": time.time(),
            }
        tmp_path = lifetime_tokens_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh)
            os.replace(tmp_path, lifetime_tokens_path)
        except Exception as exc:
            logger.warning("lifetime tokens persist failed: %s", exc)

    if rank == 0:
        # NOTE: a later `import atexit` inside this function makes the name
        # function-local; import here too or this line hits it unbound.
        import atexit
        _load_lifetime_tokens()
        atexit.register(_persist_lifetime_tokens)
    generation_jobs = queue.Queue()

    def schedule_process_exit(delay=0.25):
        def _delayed_exit():
            time.sleep(delay)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_delayed_exit, daemon=True).start()

    def shutdown_idle_cluster(reason):
        logger.info("shutting down idle cluster (%s)", reason)
        try:
            _bcast({"shutdown": True}, rank)
        except Exception as e:
            logger.warning(f"shutdown broadcast failed: {e}")
        schedule_process_exit()

    def admin_localhost_only(req: Request):
        client_host = req.client.host if req.client else ""
        return client_host in ("127.0.0.1", "::1", "localhost")

    def generation_worker():
        logger.info("[rank 0] persistent generation worker started")
        while True:
            job = generation_jobs.get()
            if job is None:
                return
            try:
                job()
            except Exception:
                logger.error(
                    "[rank 0] unhandled generation worker error:\n%s",
                    traceback.format_exc(),
                )

    threading.Thread(target=generation_worker, daemon=True).start()

    def submit_generation_job(job):
        generation_jobs.put(job)

    async def run_generation_control_job(name, job, timeout=1800):
        """Run a localhost admin control op on the generation worker thread.

        Some MLX tensors in the live prompt cache are owned by the generation
        worker's stream. Running serialization from uvicorn's request thread can
        fail with a missing current Stream, so control jobs that touch KV state
        must execute on the same worker lane as generation.
        """
        done = threading.Event()
        result = {}

        def _job():
            acquired = False
            try:
                recover_stale_generation_lock(f"before control {name}")
                generation_lock.acquire()
                acquired = True
                set_generation_lock_owner("control", name)
                # generation_lock first, then the op-channel mutex: control
                # jobs may broadcast ops (e.g. SSD save) and must own the
                # channel for the whole job.
                _RANK0_OP_MUTEX.acquire()
                try:
                    result["value"] = job()
                finally:
                    _RANK0_OP_MUTEX.release()
            except BaseException as e:
                result["error"] = e
                logger.error(
                    "[rank 0] generation control job %s failed:\n%s",
                    name,
                    traceback.format_exc(),
                )
            finally:
                if acquired:
                    clear_generation_lock_owner("control")
                    try:
                        generation_lock.release()
                    except RuntimeError:
                        pass
                done.set()

        submit_generation_job(_job)
        completed = await asyncio.to_thread(done.wait, timeout)
        if not completed:
            raise TimeoutError(f"generation control job timed out: {name}")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def coordinated_keepwarm(size=32, repeats=1, reason="prompt-cache keepwarm"):
        with state_lock:
            if (
                request_state.get("active") is not None
                or int(request_state.get("queued") or 0) > 0
                or request_state.get("shutdown_requested")
            ):
                return {"skipped": True, "reason": "generation busy"}
        acquired = generation_lock.acquire(blocking=False)
        if not acquired:
            return {"skipped": True, "reason": "generation lock busy"}
        set_generation_lock_owner("keepwarm", reason)
        op_channel_acquired = False
        try:
            # generation_lock first, then the op-channel mutex (lock order).
            # Bounded wait: skip rather than queue behind a long request
            # transaction holding the broadcast channel.
            op_channel_acquired = _RANK0_OP_MUTEX.acquire(timeout=2)
            if not op_channel_acquired:
                return {"skipped": True, "reason": "rank0 op channel busy"}
            with state_lock:
                if (
                    request_state.get("active") is not None
                    or int(request_state.get("queued") or 0) > 0
                    or request_state.get("shutdown_requested")
                ):
                    return {"skipped": True, "reason": "generation busy"}
            if PROMPT_CACHE_KEEPWARM_MODE == "prewarm":
                snapshot = _prompt_cache_current_prompt_snapshot()
                if not snapshot:
                    return {"skipped": True, "reason": "no exact prompt snapshot"}
                try:
                    _bcast({
                        "op": "prewarm_prompt_cache",
                        "prompt": snapshot["prompt"],
                        "token_ids": snapshot["token_ids"],
                        "reason": reason,
                        "visible_source": "idle_keepwarm",
                        "thinking_mode": DEFAULT_THINKING_MODE,
                        "session_id": snapshot.get("session_id"),
                        "session_source": snapshot.get("session_source"),
                    }, rank)
                except Exception as e:
                    logger.debug("coordinated keepwarm prewarm broadcast failed: %s", e)
                    return {"ok": False, "error": str(e), "reason": reason}
                started = time.time()
                ok = _prewarm_prompt_cache(
                    model,
                    processor,
                    snapshot["prompt"],
                    snapshot["token_ids"],
                    reason=reason,
                    session_id=snapshot.get("session_id"),
                    session_source=snapshot.get("session_source"),
                    reset_on_failure=False,
                )
                return {
                    "ok": bool(ok),
                    "action": "prompt_cache_prewarm",
                    "reason": reason,
                    "at": round(time.time(), 3),
                    "prompt_tokens": len(snapshot["token_ids"]),
                    "cache_len": snapshot.get("cache_len"),
                    "elapsed_ms": round((time.time() - started) * 1000, 3),
                }
            try:
                _bcast({
                    "op": "metal_warmup",
                    "matrix_size": size,
                    "repeats": repeats,
                    "reason": reason,
                }, rank)
            except Exception as e:
                logger.debug("coordinated keepwarm broadcast failed: %s", e)
                return {"ok": False, "error": str(e), "reason": reason}
            return _metal_warmup_touch(size=size, repeats=repeats, reason=reason)
        finally:
            if op_channel_acquired:
                _RANK0_OP_MUTEX.release()
            clear_generation_lock_owner("keepwarm")
            generation_lock.release()

    _start_prompt_cache_keepwarm(warmup_cb=coordinated_keepwarm)

    post_response_keepwarm_state = {"scheduled": False}
    post_response_keepwarm_state_lock = threading.Lock()

    def schedule_post_response_keepwarm(req_id):
        if not PROMPT_CACHE_POST_RESPONSE_KEEPWARM_ENABLED:
            return False
        if not _prompt_cache_request_start_keepwarm_candidate(min_idle_seconds=0):
            return False
        with post_response_keepwarm_state_lock:
            if post_response_keepwarm_state["scheduled"]:
                return False
            post_response_keepwarm_state["scheduled"] = True

        def _run():
            delay = max(0.0, PROMPT_CACHE_POST_RESPONSE_KEEPWARM_DELAY_SECONDS)
            try:
                if delay > 0:
                    time.sleep(delay)
                with state_lock:
                    busy = (
                        request_state.get("active") is not None
                        or int(request_state.get("queued") or 0) > 0
                        or request_state.get("shutdown_requested")
                    )
                if busy:
                    return
                acquired = generation_lock.acquire(blocking=False)
                if not acquired:
                    return
                set_generation_lock_owner("keepwarm", f"post-response:{req_id}")
                op_channel_acquired = False
                try:
                    # generation_lock first, then the op-channel mutex (lock
                    # order). Bounded wait: skip rather than queue behind a
                    # long request transaction holding the broadcast channel.
                    op_channel_acquired = _RANK0_OP_MUTEX.acquire(timeout=2)
                    if not op_channel_acquired:
                        return
                    with state_lock:
                        busy = (
                            request_state.get("active") is not None
                            or int(request_state.get("queued") or 0) > 0
                            or request_state.get("shutdown_requested")
                        )
                    if busy:
                        return
                    candidate = _prompt_cache_request_start_keepwarm_candidate(
                        min_idle_seconds=0
                    )
                    if not candidate:
                        return
                    size = PROMPT_CACHE_POST_RESPONSE_KEEPWARM_MATRIX_SIZE
                    repeats = PROMPT_CACHE_POST_RESPONSE_KEEPWARM_REPEATS
                    reason = f"post-response keepwarm:{req_id}"
                    started = time.time()
                    _bcast({
                        "op": "metal_warmup",
                        "matrix_size": size,
                        "repeats": repeats,
                        "reason": reason,
                    }, rank)
                    local = _metal_warmup_touch(
                        size=size,
                        repeats=repeats,
                        reason=reason,
                    )
                    event = {
                        "ok": bool(local.get("ok")),
                        "action": "post_response_metal_touch",
                        "at": round(time.time(), 3),
                        "elapsed_ms": round((time.time() - started) * 1000, 3),
                        "matrix_size": int(local.get("matrix_size") or size),
                        "repeats": int(local.get("repeats") or repeats),
                        **candidate,
                    }
                    if not local.get("ok"):
                        event["error"] = local.get("error")
                    with _prompt_cache_lock:
                        holder = _prompt_cache_holder
                        holder["last_keepwarm_event"] = event
                        holder["last_keepwarm_at"] = event["at"]
                        holder["keepwarm_count"] = int(holder.get("keepwarm_count") or 0) + 1
                    logger.info(
                        "prompt-cache post-response keepwarm after %s: %s",
                        req_id,
                        event,
                    )
                finally:
                    if op_channel_acquired:
                        _RANK0_OP_MUTEX.release()
                    clear_generation_lock_owner("keepwarm")
                    try:
                        generation_lock.release()
                    except RuntimeError:
                        pass
            except Exception as e:
                with _prompt_cache_lock:
                    _prompt_cache_holder["last_keepwarm_event"] = {
                        "ok": False,
                        "action": "post_response_metal_touch_error",
                        "at": round(time.time(), 3),
                        "error": str(e),
                    }
                logger.debug("prompt-cache post-response keepwarm failed: %s", e)
            finally:
                with post_response_keepwarm_state_lock:
                    post_response_keepwarm_state["scheduled"] = False

        threading.Thread(
            target=_run,
            name="prompt-cache-post-response-keepwarm",
            daemon=True,
        ).start()
        return True

    def request_start_keepwarm(req_id):
        if not PROMPT_CACHE_REQUEST_START_KEEPWARM_ENABLED:
            return None
        candidate = _prompt_cache_request_start_keepwarm_candidate()
        if not candidate:
            return None
        size = PROMPT_CACHE_REQUEST_START_KEEPWARM_MATRIX_SIZE
        repeats = PROMPT_CACHE_REQUEST_START_KEEPWARM_REPEATS
        reason = f"request-start keepwarm:{req_id}"
        started = time.time()
        # Warm bcast + local touch is one op-channel transaction. Reentrant
        # under the request transaction (the producer already owns the
        # channel); bounded so any other caller skips instead of queueing.
        if not _RANK0_OP_MUTEX.acquire(timeout=2):
            return None
        try:
            _bcast({
                "op": "metal_warmup",
                "matrix_size": size,
                "repeats": repeats,
                "reason": reason,
            }, rank)
            local = _metal_warmup_touch(
                size=size,
                repeats=repeats,
                reason=reason,
            )
            event = {
                "ok": bool(local.get("ok")),
                "action": "request_start_metal_touch",
                "at": round(time.time(), 3),
                "elapsed_ms": round((time.time() - started) * 1000, 3),
                "matrix_size": int(local.get("matrix_size") or size),
                "repeats": int(local.get("repeats") or repeats),
                **candidate,
            }
            if not local.get("ok"):
                event["error"] = local.get("error")
            with _prompt_cache_lock:
                holder = _prompt_cache_holder
                holder["last_keepwarm_event"] = event
                holder["last_keepwarm_at"] = event["at"]
                holder["keepwarm_count"] = int(holder.get("keepwarm_count") or 0) + 1
            logger.info(
                "prompt-cache request-start keepwarm for %s: %s",
                req_id,
                event,
            )
            return event
        except Exception as e:
            event = {
                "ok": False,
                "action": "request_start_metal_touch_error",
                "at": round(time.time(), 3),
                "elapsed_ms": round((time.time() - started) * 1000, 3),
                "error": str(e),
                **candidate,
            }
            with _prompt_cache_lock:
                _prompt_cache_holder["last_keepwarm_event"] = event
            logger.debug("prompt-cache request-start keepwarm failed: %s", e)
            return event
        finally:
            _RANK0_OP_MUTEX.release()

    def _cache_effective_metrics(summary):
        shape = summary.get("request_shape") or {}
        prepare = summary.get("prompt_cache_prepare") or {}
        processed_prompt_tokens = int(summary.get("prompt_tokens") or 0)
        full_prompt_tokens = int(
            shape.get("full_prompt_tokens")
            or prepare.get("prompt_tokens")
            or processed_prompt_tokens
        )
        reuse_tokens = int(prepare.get("reuse_tokens") or 0)
        avoided_tokens = max(0, full_prompt_tokens - processed_prompt_tokens, reuse_tokens)
        first_token_s = float(summary.get("first_token_s") or 0.0)
        cache_efficiency = (
            round(avoided_tokens / full_prompt_tokens, 4)
            if full_prompt_tokens > 0 else 0.0
        )
        effective_prompt_tps = (
            round(full_prompt_tokens / first_token_s, 2)
            if first_token_s > 0 and full_prompt_tokens > 0 else 0.0
        )
        return {
            "full_prompt_tokens": full_prompt_tokens,
            "processed_prompt_tokens": processed_prompt_tokens,
            "cache_avoided_prompt_tokens": avoided_tokens,
            "cache_efficiency": cache_efficiency,
            "effective_prompt_tps": effective_prompt_tps,
            "prompt_tps_excluding_cache": float(summary.get("prompt_tps") or 0.0),
        }

    def _history_row(summary):
        shape = summary.get("request_shape") or {}
        prepare = summary.get("prompt_cache_prepare") or {}
        cache_metrics = _cache_effective_metrics(summary)
        return {
            "id": summary.get("id"),
            "ok": bool(summary.get("ok")),
            "finished_at": summary.get("finished_at"),
            "model": shape.get("response_model") or shape.get("requested_model"),
            "thinking_mode": shape.get("thinking_mode"),
            "stream": bool(summary.get("stream")),
            "tools_count": int(shape.get("tools_count") or 0),
            "image_count": int(shape.get("image_count") or 0),
            "tokens": int(summary.get("tokens") or 0),
            "reasoning_chars": int(summary.get("reasoning_chars") or 0),
            "content_chars": int(summary.get("content_chars") or 0),
            "reasoning_only": (
                int(summary.get("reasoning_chars") or 0) > 0
                and int(summary.get("content_chars") or 0) == 0
            ),
            "prompt_tokens": int(summary.get("prompt_tokens") or 0),
            "full_prompt_tokens": cache_metrics["full_prompt_tokens"],
            "processed_prompt_tokens": cache_metrics["processed_prompt_tokens"],
            "prompt_tps": float(summary.get("prompt_tps") or 0.0),
            "prompt_tps_excluding_cache": cache_metrics["prompt_tps_excluding_cache"],
            "effective_prompt_tps": cache_metrics["effective_prompt_tps"],
            "decode_tps": float(summary.get("decode_tps") or 0.0),
            "request_tps": float(summary.get("tps") or 0.0),
            "ttft_s": float(summary.get("first_token_s") or 0.0),
            "first_generator_token_s": float(
                summary.get("first_generator_token_s") or 0.0
            ),
            "first_visible_delta_s": float(
                summary.get("first_visible_delta_s") or 0.0
            ),
            "cache_prepare_s": float(summary.get("cache_prepare_s") or 0.0),
            "first_yield_after_generate_s": float(
                summary.get("first_yield_after_generate_s") or 0.0
            ),
            "runtime_lock_wait_s": float(
                summary.get("runtime_lock_wait_s") or 0.0
            ),
            "first_yield_after_lock_s": float(
                summary.get("first_yield_after_lock_s") or 0.0
            ),
            "cache_action": prepare.get("action"),
            "cache_miss_reason": prepare.get("miss_reason"),
            # 2026-07-06 audit: prepare-time reuse_ratio lies on rebuild rows
            # (hard 0 even when a prefix matched) and on bypass rows (positive
            # reuse that was never applied). The generator's cached_tokens is
            # ground truth for what this request actually skipped.
            "cache_reuse_ratio": (
                round(
                    cache_metrics["cache_avoided_prompt_tokens"]
                    / cache_metrics["full_prompt_tokens"],
                    4,
                )
                if cache_metrics.get("full_prompt_tokens")
                else prepare.get("reuse_ratio")
            ),
            "cache_prepare_reuse_ratio": prepare.get("reuse_ratio"),
            "cache_reuse_tokens": prepare.get("reuse_tokens"),
            "cache_suffix_tokens": prepare.get("suffix_tokens"),
            "cache_min_suffix_backtrack_tokens": prepare.get("min_suffix_backtrack_tokens"),
            "cache_missed_tokens": prepare.get("missed_tokens"),
            "cache_avoided_prompt_tokens": cache_metrics["cache_avoided_prompt_tokens"],
            "cache_efficiency": cache_metrics["cache_efficiency"],
            "cache_would_reprocess_tokens": prepare.get("would_reprocess_tokens"),
            "protected_cache_tokens": prepare.get("protected_cache_tokens"),
            "protected_session_id": prepare.get("protected_session_id"),
        }

    def _history_stats(rows):
        ok_rows = [
            row for row in rows
            if row.get("ok") and int(row.get("tokens") or 0) > 0
        ]
        meaningful = [
            row for row in ok_rows
            if int(row.get("tokens") or 0) >= 16
        ]
        prompt_meaningful = [
            row for row in ok_rows
            if int(row.get("processed_prompt_tokens") or row.get("prompt_tokens") or 0) >= 512
        ]
        hot = [
            row for row in ok_rows
            if float(row.get("cache_reuse_ratio") or 0.0) >= 0.95
        ]
        effective_hot = [
            row for row in ok_rows
            if float(row.get("cache_efficiency") or 0.0) >= 0.95
        ]
        bypassed = [
            row for row in ok_rows
            if str(row.get("cache_action") or "").startswith("bypass_preserve")
        ]
        if not ok_rows:
            return {
                "count": len(rows),
                "ok_count": 0,
                "meaningful_count": 0,
                "hot_count": 0,
                "effective_hot_count": 0,
                "protected_bypass_count": 0,
                "total_cache_reuse_tokens": 0,
                "total_cache_avoided_prompt_tokens": 0,
            }

        def avg(key, items):
            # Ratios/efficiencies must average over ALL rows: excluding zeros
            # turned "avg reuse" into "avg reuse among requests that reused
            # anything", hiding total cache loss (2026-07-06 audit).
            if key.endswith("_ratio") or key.endswith("efficiency"):
                vals = [float(row.get(key) or 0.0) for row in items]
                return round(sum(vals) / len(vals), 4) if vals else 0.0
            vals = [float(row.get(key) or 0.0) for row in items if float(row.get(key) or 0.0) > 0]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        def total(key, items):
            return int(sum(int(row.get(key) or 0) for row in items))

        return {
            "count": len(rows),
            "ok_count": len(ok_rows),
            "meaningful_count": len(meaningful),
            "hot_count": len(hot),
            "effective_hot_count": len(effective_hot),
            "protected_bypass_count": len(bypassed),
            "avg_decode_tps": avg("decode_tps", meaningful),
            "avg_prompt_tps": avg("prompt_tps", ok_rows),
            "avg_prompt_tps_meaningful": avg("prompt_tps", prompt_meaningful),
            "avg_effective_prompt_tps": avg("effective_prompt_tps", ok_rows),
            "avg_effective_prompt_tps_meaningful": avg("effective_prompt_tps", prompt_meaningful),
            "avg_ttft_s": avg("ttft_s", ok_rows),
            "avg_cache_reuse_ratio": avg("cache_reuse_ratio", ok_rows),
            "avg_cache_efficiency": avg("cache_efficiency", ok_rows),
            "effective_hot_avg_ttft_s": avg("ttft_s", effective_hot),
            "effective_hot_avg_decode_tps": avg("decode_tps", effective_hot),
            "effective_hot_avg_effective_prompt_tps": avg("effective_prompt_tps", effective_hot),
            "avg_cache_suffix_tokens": avg("cache_suffix_tokens", ok_rows),
            "hot_avg_suffix_tokens": avg("cache_suffix_tokens", hot),
            "hot_avg_ttft_s": avg("ttft_s", hot),
            "hot_avg_decode_tps": avg("decode_tps", hot),
            "hot_avg_effective_prompt_tps": avg("effective_prompt_tps", hot),
            "total_cache_reuse_tokens": total("cache_reuse_tokens", ok_rows),
            "total_cache_avoided_prompt_tokens": total("cache_avoided_prompt_tokens", ok_rows),
            "total_cache_suffix_tokens": total("cache_suffix_tokens", ok_rows),
        }

    def recover_stale_generation_lock(reason):
        with state_lock:
            owner = dict(generation_lock_owner)
            owner_kind = owner.get("kind")
            now = time.time()
            owner_age = (
                now - float(owner.get("started_at") or 0.0)
                if owner.get("started_at") else None
            )
            transition_age = (
                now - float(owner.get("changed_at") or 0.0)
                if owner.get("changed_at") else None
            )
            stale = _should_recover_generation_lock(
                lock_locked=generation_lock.locked(),
                active_present=request_state.get("active") is not None,
                releasing_present=request_state.get("releasing") is not None,
                owner_kind=owner_kind,
                owner_age=owner_age,
                transition_age=transition_age,
            )
        if not stale:
            return False
        # Clear only the owner snapshot we inspected. A waiter may acquire the
        # lock between recovery checks; never erase or release that newer turn.
        if not clear_generation_lock_owner(
            owner_kind,
            owner.get("reason"),
            owner.get("changed_at"),
        ):
            return False
        try:
            generation_lock.release()
            with state_lock:
                request_state["last_error"] = (
                    f"recovered stale generation lock: {reason}"
                )
            logger.warning("recovered stale generation lock (%s)", reason)
            return True
        except RuntimeError:
            return False

    async def acquire_generation_slot(
        req_id,
        *,
        stream,
        max_tokens,
        image_count,
        request_shape=None,
        stop_nonce=None,
    ):
        recover_stale_generation_lock(f"before acquire {req_id}")
        acquired = False
        queued_counted = False
        with state_lock:
            request_state["queued"] += 1
            queued_counted = True
        wait_started = time.time()
        try:
            # Bounded waits + recovery retries: a leaked lock with a single
            # queued waiter used to block until ANOTHER request arrived to
            # trigger the before-acquire recovery (P5b sat 6.5 min behind the
            # one observed leak, 2026-07-07 09:52). Now every waiter re-runs
            # the stale check itself every 30s — any leak costs <=30s.
            while True:
                got = await asyncio.to_thread(generation_lock.acquire, True, 30.0)
                if got:
                    break
                recover_stale_generation_lock(f"acquire-wait retry {req_id}")
            acquired = True
            set_generation_lock_owner("request", req_id)
            active = {
                "id": req_id,
                "stream": bool(stream),
                "max_tokens": int(max_tokens),
                "image_count": int(image_count),
                "started": time.time(),
                "wait_s": round(time.time() - wait_started, 3),
                "client_connected": True,
                "tokens_emitted": 0,
                "chunks_emitted": 0,
                "chars_raw": 0,
                "reasoning_chars": 0,
                "content_chars": 0,
                "first_token_s": 0.0,
                "last_token_s": 0.0,
                "first_generator_token_s": 0.0,
                "first_visible_delta_s": 0.0,
                "cache_prepare_s": 0.0,
                "first_yield_after_generate_s": 0.0,
                "runtime_lock_wait_s": 0.0,
                "first_yield_after_lock_s": 0.0,
                "prefill_processed_tokens": 0,
                "prefill_total_tokens": 0,
                "prefill_progress": 0.0,
                "prefill_last_progress_s": 0.0,
                "prompt_tps": 0.0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "prompt_cache_prepare": None,
                "generation_elapsed_s": 0.0,
                "last_progress_s": 0.0,
                "request_shape": request_shape or {},
            }
            with state_lock:
                request_state["queued"] -= 1
                queued_counted = False
                request_state["active"] = active
                _ACTIVE_STOP_TARGET["request_id"] = req_id
                _ACTIVE_STOP_TARGET["nonce"] = stop_nonce
        except BaseException:
            with state_lock:
                if queued_counted:
                    request_state["queued"] -= 1
            if acquired:
                try:
                    generation_lock.release()
                except RuntimeError:
                    pass
            raise
        logger.info(
            "[rank 0] request %s acquired distributed generation slot "
            "(stream=%s, max_tokens=%s, images=%s, waited=%.3fs)",
            req_id, stream, max_tokens, image_count, active["wait_s"],
        )
        return active

    def update_generation_slot(active, **updates):
        with state_lock:
            current = request_state.get("active")
            if current is active:
                if "tokens_emitted" in updates:
                    emitted = int(updates.get("tokens_emitted") or 0)
                    previous = int(current.get("tokens_emitted") or 0)
                    if emitted > previous:
                        token_s = round(time.time() - current["started"], 3)
                        if not current.get("first_token_s"):
                            updates["first_token_s"] = token_s
                        updates["last_token_s"] = token_s
                current.update(updates)

    def _lifetime_with_active(snapshot, active):
        lifetime = dict(snapshot.get("lifetime_tokens") or {})
        active_prompt = int(active.get("prompt_tokens") or 0) if active else 0
        active_decode = int(active.get("tokens_emitted") or 0) if active else 0
        lifetime["active_prompt_processed"] = active_prompt
        lifetime["active_decode"] = active_decode
        lifetime["active_processed_total"] = active_prompt + active_decode
        lifetime["processed_total_live"] = (
            int(lifetime.get("processed_total") or 0)
            + lifetime["active_processed_total"]
        )
        lifetime["decode_live"] = int(lifetime.get("decode") or 0) + active_decode
        lifetime["prompt_processed_live"] = (
            int(lifetime.get("prompt_processed") or 0) + active_prompt
        )
        return lifetime

    def release_generation_slot(req_id, active, error=None):
        if active is None:
            logger.warning(
                "[rank 0] release requested for %s with no active slot object",
                req_id,
            )
            return
        with state_lock:
            if active.get("_released"):
                logger.warning(
                    "[rank 0] duplicate release ignored for request %s",
                    req_id,
                )
                return
            active["_released"] = True
            owns_active_slot = request_state.get("active") is active
            if owns_active_slot:
                request_state["releasing"] = req_id
        if not owns_active_slot:
            logger.warning(
                "[rank 0] stale release for %s did not own the published "
                "active slot; preserving the newer request",
                req_id,
            )
        shutdown_after_release = False
        total_elapsed = time.time() - active["started"]
        generation_elapsed = float(
            active.get("generation_elapsed_s")
            or active.get("last_token_s")
            or 0.0
        )
        elapsed = generation_elapsed if generation_elapsed > 0 else total_elapsed
        tokens = int(active.get("tokens_emitted") or 0)
        first_token_s = float(active.get("first_token_s") or 0.0)
        post_first_s = max(0.0, elapsed - first_token_s) if first_token_s > 0 else 0.0
        decode_tokens = max(0, tokens - 1)
        decode_tps = (
            round(decode_tokens / post_first_s, 2)
            if post_first_s > 0 and decode_tokens > 0 else 0.0
        )
        # Authoritative total TPS for the dashboard. This includes prefill /
        # time-to-first-token, while decode_tps isolates token generation after
        # the first token has landed.
        tps = round(tokens / elapsed, 2) if elapsed > 0 and tokens > 0 else 0.0
        last_summary = {
            "id": req_id,
            "tokens": tokens,
            "elapsed_s": round(elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "post_generation_s": round(max(0.0, total_elapsed - elapsed), 2),
            "tps": tps,
            "first_token_s": round(first_token_s, 2),
            "last_token_s": round(
                float(active.get("last_token_s") or 0.0), 3
            ),
            "first_generator_token_s": round(
                float(active.get("first_generator_token_s") or 0.0), 2
            ),
            "first_visible_delta_s": round(
                float(active.get("first_visible_delta_s") or 0.0), 2
            ),
            "cache_prepare_s": round(
                float(active.get("cache_prepare_s") or 0.0), 3
            ),
            "first_yield_after_generate_s": round(
                float(active.get("first_yield_after_generate_s") or 0.0), 3
            ),
            "runtime_lock_wait_s": round(
                float(active.get("runtime_lock_wait_s") or 0.0), 3
            ),
            "first_yield_after_lock_s": round(
                float(active.get("first_yield_after_lock_s") or 0.0), 3
            ),
            "prompt_tps": round(float(active.get("prompt_tps") or 0.0), 2),
            "prompt_tokens": int(active.get("prompt_tokens") or 0),
            "cached_tokens": int(active.get("cached_tokens") or 0),
            "prompt_cache_prepare": active.get("prompt_cache_prepare"),
            "decode_tps": decode_tps,
            "short_output": tokens > 0 and tokens < 16,
            "chars": int(active.get("chars_raw") or 0),
            "reasoning_chars": int(active.get("reasoning_chars") or 0),
            "content_chars": int(active.get("content_chars") or 0),
            "stream": bool(active.get("stream")),
            "ok": error is None,
            "finished_at": time.time(),
            "request_shape": active.get("request_shape") or {},
        }
        last_summary.update(_cache_effective_metrics(last_summary))
        processed_prompt_tokens = int(last_summary.get("processed_prompt_tokens") or 0)
        full_prompt_tokens = int(last_summary.get("full_prompt_tokens") or 0)
        cached_tokens = int(last_summary.get("cached_tokens") or 0)
        avoided_prompt_tokens = int(last_summary.get("cache_avoided_prompt_tokens") or 0)
        decoded_tokens = int(last_summary.get("tokens") or 0)
        lifetime_delta = {
            "requests": 1,
            "prompt_processed": processed_prompt_tokens,
            "prompt_logical": full_prompt_tokens,
            "prompt_cached": cached_tokens,
            "prompt_avoided": avoided_prompt_tokens,
            "decode": decoded_tokens,
            "processed_total": processed_prompt_tokens + decoded_tokens,
            "logical_total": full_prompt_tokens + decoded_tokens,
        }
        if error is None:
            with state_lock:
                lifetime = request_state["lifetime_tokens"]
                for key, value in lifetime_delta.items():
                    lifetime[key] = int(lifetime.get(key) or 0) + int(value or 0)
                request_state["completed"] += 1
                request_state["last_error"] = None
                if request_state.get("active") is active:
                    request_state["active"] = None
                request_state["last_request"] = last_summary
                if tokens >= 32:
                    request_state["last_meaningful_request"] = last_summary
                history = request_state["recent_requests"]
                history.append(_history_row(last_summary))
                if REQUEST_HISTORY_MAX > 0:
                    del history[:-REQUEST_HISTORY_MAX]
                else:
                    history.clear()
                shutdown_after_release = bool(request_state["shutdown_requested"])
            _persist_lifetime_tokens()
        else:
            with state_lock:
                lifetime = request_state["lifetime_tokens"]
                for key, value in lifetime_delta.items():
                    lifetime[key] = int(lifetime.get(key) or 0) + int(value or 0)
                request_state["failed"] += 1
                request_state["last_error"] = f"{type(error).__name__}: {error}"
                if request_state.get("active") is active:
                    request_state["active"] = None
                request_state["last_request"] = last_summary
                if tokens >= 32:
                    request_state["last_meaningful_request"] = last_summary
                history = request_state["recent_requests"]
                history.append(_history_row(last_summary))
                if REQUEST_HISTORY_MAX > 0:
                    del history[:-REQUEST_HISTORY_MAX]
                else:
                    history.clear()
                shutdown_after_release = bool(request_state["shutdown_requested"])
            _persist_lifetime_tokens()
        with state_lock:
            if _ACTIVE_STOP_TARGET.get("request_id") == req_id:
                _ACTIVE_STOP_TARGET["request_id"] = None
                _ACTIVE_STOP_TARGET["nonce"] = None
        logger.info(
            "[rank 0] request %s released distributed generation slot "
            "(elapsed=%.2fs, first_token=%.2fs, prompt_tps=%.2f, tokens=%s, "
            "tps=%.2f, decode_tps=%.2f)",
            req_id, elapsed, first_token_s, last_summary["prompt_tps"], tokens,
            tps, decode_tps,
        )
        owner_cleared = clear_generation_lock_owner("request", req_id)
        if owner_cleared:
            try:
                generation_lock.release()
            except RuntimeError:
                logger.warning(
                    "[rank 0] generation lock was already released for request %s",
                    req_id,
                )
        else:
            logger.warning(
                "[rank 0] request %s did not release a generation lock owned "
                "by another request",
                req_id,
            )
        with state_lock:
            if request_state.get("releasing") == req_id:
                request_state["releasing"] = None
        if error is None and tokens > 0:
            schedule_post_response_keepwarm(req_id)
        if shutdown_after_release:
            shutdown_idle_cluster(f"deferred after request {req_id}")

    @asynccontextmanager
    async def generation_slot(req_id, *, stream, max_tokens, image_count,
                              request_shape=None):
        active = await acquire_generation_slot(
            req_id, stream=stream, max_tokens=max_tokens,
            image_count=image_count, request_shape=request_shape
        )
        error = None
        try:
            yield
        except BaseException as e:
            error = e
            raise
        finally:
            release_generation_slot(req_id, active, error)

    @app.get("/health")
    async def health():
        with state_lock:
            snapshot = dict(request_state)
            recent_requests = list(request_state.get("recent_requests") or [])
        active = snapshot["active"]
        if active is None and int(snapshot.get("queued") or 0) == 0:
            _recover_stale_prompt_cache_in_use("health idle recovery")
        if active is not None:
            active = dict(active)
            elapsed = time.time() - active["started"]
            active["elapsed_s"] = round(elapsed, 3)
            tokens = int(active.get("tokens_emitted") or 0)
            first_token_s = float(active.get("first_token_s") or 0.0)
            decode_clock_s = float(active.get("last_token_s") or elapsed)
            post_first_s = (
                max(0.0, decode_clock_s - first_token_s)
                if first_token_s > 0 else 0.0
            )
            decode_tokens = max(0, tokens - 1)
            active["decode_tps"] = round(decode_tokens / post_first_s, 2) if post_first_s > 0 and decode_tokens > 0 else 0.0
            active["request_tps"] = round(tokens / elapsed, 2) if elapsed > 0 and tokens > 0 else 0.0
            last_progress = float(active.get("last_progress_s") or 0.0)
            active["seconds_since_progress"] = round(
                elapsed - last_progress if last_progress > 0 else elapsed,
                3,
            )
            processed = int(active.get("prefill_processed_tokens") or 0)
            total = int(active.get("prefill_total_tokens") or 0)
            if tokens > 0:
                active["phase"] = "decode"
            elif total > 0 and processed < total:
                active["phase"] = "prefill"
            elif total > 0 and processed >= total:
                active["phase"] = "decode_starting"
            else:
                active["phase"] = "preparing"
        return {"status": "healthy", "loaded_model": MODEL_ID,
                "loaded_model_path": MODEL,
                "distributed": SHARDING_MODE, "ranks": int(mx.distributed.init().size()),
                "generation_lock_owner": dict(generation_lock_owner),
                "metal_limits": _METAL_LIMITS,
                "omlx_minimax_overlay": OMLX_MINIMAX_OVERLAY,
                "single_flight": EFFECTIVE_MAX_CONCURRENT_REQUESTS == 1,
                "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "effective_max_concurrent_requests": EFFECTIVE_MAX_CONCURRENT_REQUESTS,
                "stream_mode": STREAM_MODE,
                "nonstream_coalescing": nonstream_coalescer.status(),
                "generation_defaults": _generation_defaults_status(),
                "kernel_stats": _kernel_stats_status(),
                "metal_warmup": _metal_warmup_status(),
                "request_queue_depth": snapshot["queued"],
                "active_request": active,
                "requests_completed": snapshot["completed"],
                "requests_failed": snapshot["failed"],
                "prompt_cache": _prompt_cache_status(),
                "capture": _capture_corpus_status(),
                "lifetime_tokens": _lifetime_with_active(snapshot, active),
                "last_request": snapshot.get("last_request"),
                "last_meaningful_request": snapshot.get("last_meaningful_request"),
                "recent_requests": recent_requests,
                "recent_request_stats": _history_stats(recent_requests),
                "last_error": snapshot["last_error"]}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1,
                "owned_by": "mlx-vlm",
                "max_model_len": ADVERTISED_MAX_MODEL_LEN,
                "native_context_window": MAX_KV_SIZE,
            }
            for model_id in VISIBLE_MODEL_IDS
        ]}

    @app.post("/admin/request-history/reset")
    async def admin_request_history_reset(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "request history reset is localhost-only"},
            )
        with state_lock:
            cleared = len(request_state.get("recent_requests") or [])
            request_state["recent_requests"] = []
        return {"ok": True, "cleared": cleared, "recent_request_stats": _history_stats([])}

    @app.post("/admin/eagle3")
    async def admin_eagle3_toggle(req: Request):
        """Dashboard toggle for the EAGLE3 speculative path. The flag is
        read by rank 0 per request and travels in the broadcast request op,
        so flipping it mid-session can never desync the ranks."""
        import m3_eagle3 as _m3e3
        try:
            body = await req.json()
        except Exception:
            body = {}
        if "enabled" in (body or {}):
            _m3e3.RUNTIME_ENABLED["value"] = bool(body["enabled"])
        return {
            "armed": _m3e3.enabled(),
            "enabled": bool(_m3e3.RUNTIME_ENABLED.get("value")),
            "stats": _m3e3.acceptance_stats(),
        }

    @app.get("/admin/eagle3")
    async def admin_eagle3_status():
        import m3_eagle3 as _m3e3
        return {
            "armed": _m3e3.enabled(),
            "enabled": bool(_m3e3.RUNTIME_ENABLED.get("value")),
            "stats": _m3e3.acceptance_stats(),
        }

    @app.post("/admin/shutdown")
    async def admin_shutdown(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "admin shutdown is localhost-only"},
            )

        logger.info("localhost admin shutdown requested")
        with state_lock:
            request_state["shutdown_requested"] = True
            active = request_state["active"]
            if active is not None:
                active = dict(active)
                active["elapsed_s"] = round(time.time() - active["started"], 3)

        if active is not None:
            logger.info("shutdown deferred until active request finishes: %s", active)
            return {"status": "deferred", "active_request": active}

        shutdown_idle_cluster("admin request")
        return {"status": "shutting_down"}

    @app.post("/unload")
    @app.post("/v1/unload")
    async def unload(req: Request):
        # A pipeline-split M3 cannot be hot-unloaded safely while keeping the
        # distributed process alive. Match the user-facing intent of mlx_vlm's
        # unload endpoint by doing the clean distributed shutdown path.
        return await admin_shutdown(req)

    @app.post("/admin/prompt-cache/reset")
    @app.post("/v1/prompt-cache/reset")
    async def admin_prompt_cache_reset(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "prompt cache reset is localhost-only"},
            )
        with state_lock:
            active = request_state.get("active")
            queued = int(request_state.get("queued") or 0)
            active_info = dict(active) if active else None
        if active_info is not None or queued > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "generation in progress; retry when idle",
                    "active_request": active_info,
                    "queue_depth": queued,
                },
            )
        body = {}
        try:
            body = await req.json()
        except Exception:
            body = {}
        clear_memory = bool(body.get("clear_memory", True))
        clear_manifest = bool(body.get("clear_manifest", False))
        reason = str(body.get("reason") or "admin reset")
        logger.info(
            "localhost prompt-cache reset requested (clear_memory=%s, clear_manifest=%s)",
            clear_memory,
            clear_manifest,
        )
        try:
            _bcast({
                "op": "reset_prompt_cache",
                "reason": reason,
                "clear_memory": clear_memory,
                "clear_manifest": clear_manifest,
            }, rank)
        except Exception as e:
            logger.warning("prompt-cache reset broadcast failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": f"prompt-cache reset broadcast failed: {e}"},
            )
        if clear_memory:
            _reset_prompt_cache_and_clear_memory(reason, clear_manifest=clear_manifest)
        else:
            _reset_prompt_cache(reason, clear_manifest=clear_manifest)
        return {"ok": True, "prompt_cache": _prompt_cache_status()}

    @app.post("/admin/prompt-cache/ssd/prune")
    @app.post("/v1/prompt-cache/ssd/prune")
    async def admin_prompt_cache_ssd_prune(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "persistent cache prune is localhost-only"},
            )
        with state_lock:
            active = request_state.get("active")
            queued = int(request_state.get("queued") or 0)
            active_info = dict(active) if active else None
        if active_info is not None or queued > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "generation in progress; retry when idle",
                    "active_request": active_info,
                    "queue_depth": queued,
                },
            )
        try:
            body = await req.json()
        except Exception:
            body = {}
        reason = str(body.get("reason") or "admin prune")
        # bcast + local prune is one op-channel transaction. Bounded wait: a
        # request racing past the idle check above must not park the event
        # loop behind its whole generation.
        if not _RANK0_OP_MUTEX.acquire(timeout=2):
            return JSONResponse(
                status_code=409,
                content={"error": "rank0 op channel busy; retry when idle"},
            )
        try:
            try:
                _bcast({"op": "prompt_cache_ssd_prune", "reason": reason}, rank)
            except Exception as e:
                logger.warning("prompt-cache SSD prune broadcast failed: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"prompt-cache SSD prune broadcast failed: {e}"},
                )
            with _prompt_cache_lock:
                result = _prompt_cache_ssd_prune_unlocked(reason=reason)
                status = _prompt_cache_status()
            return {"ok": bool(result.get("ok")), "result": result, "prompt_cache": status}
        finally:
            _RANK0_OP_MUTEX.release()

    def _admin_prompt_cache_ssd_save_impl(reason):
        # bcast + checkpoint + save is one op-channel transaction; reentrant
        # under run_generation_control_job, which already owns the channel.
        _RANK0_OP_MUTEX.acquire()
        try:
            try:
                _bcast({"op": "prompt_cache_ssd_save", "reason": reason}, rank)
            except Exception as e:
                logger.warning("prompt-cache SSD save broadcast failed: %s", e)
                raise RuntimeError(f"prompt-cache SSD save broadcast failed: {e}") from e
            with _prompt_cache_lock:
                checkpointed = _prompt_cache_make_ssd_checkpoint_unlocked(
                    reason=reason,
                )
                saved = _prompt_cache_ssd_save_current_unlocked(
                    model,
                    processor,
                    reason=reason,
                )
                status = _prompt_cache_status()
            return {
                "ok": bool(saved),
                "saved": bool(saved),
                "checkpointed": bool(checkpointed),
                "prompt_cache": status,
            }
        finally:
            _RANK0_OP_MUTEX.release()

    @app.post("/admin/prompt-cache/ssd/save")
    @app.post("/v1/prompt-cache/ssd/save")
    async def admin_prompt_cache_ssd_save(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "persistent cache save is localhost-only"},
            )
        with state_lock:
            active = request_state.get("active")
            queued = int(request_state.get("queued") or 0)
            active_info = dict(active) if active else None
        if active_info is not None or queued > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "generation in progress; retry when idle",
                    "active_request": active_info,
                    "queue_depth": queued,
                },
            )
        try:
            body = await req.json()
        except Exception:
            body = {}
        reason = str(body.get("reason") or "admin save")
        try:
            return await run_generation_control_job(
                "prompt_cache_ssd_save",
                lambda: _admin_prompt_cache_ssd_save_impl(reason),
            )
        except Exception as e:
            logger.warning("prompt-cache SSD save failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": f"prompt-cache SSD save failed: {e}"},
            )

    @app.post("/admin/prompt-cache/ssd/clear")
    @app.post("/v1/prompt-cache/ssd/clear")
    async def admin_prompt_cache_ssd_clear(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "persistent cache clear is localhost-only"},
            )
        with state_lock:
            active = request_state.get("active")
            queued = int(request_state.get("queued") or 0)
            active_info = dict(active) if active else None
        if active_info is not None or queued > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "generation in progress; retry when idle",
                    "active_request": active_info,
                    "queue_depth": queued,
                },
            )
        try:
            body = await req.json()
        except Exception:
            body = {}
        reason = str(body.get("reason") or "admin clear")
        # bcast + local clear is one op-channel transaction. Bounded wait: a
        # request racing past the idle check above must not park the event
        # loop behind its whole generation.
        if not _RANK0_OP_MUTEX.acquire(timeout=2):
            return JSONResponse(
                status_code=409,
                content={"error": "rank0 op channel busy; retry when idle"},
            )
        try:
            try:
                _bcast({"op": "prompt_cache_ssd_clear", "reason": reason}, rank)
            except Exception as e:
                logger.warning("prompt-cache SSD clear broadcast failed: %s", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"prompt-cache SSD clear broadcast failed: {e}"},
                )
            with _prompt_cache_lock:
                result = _prompt_cache_ssd_clear_unlocked(reason=reason)
                status = _prompt_cache_status()
            return {"ok": bool(result.get("ok")), "result": result, "prompt_cache": status}
        finally:
            _RANK0_OP_MUTEX.release()

    @app.post("/admin/metal-warmup")
    async def admin_metal_warmup(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "metal warmup is localhost-only"},
            )
        with state_lock:
            active = request_state.get("active")
            queued = int(request_state.get("queued") or 0)
            active_info = dict(active) if active else None
        if active_info is not None or queued > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "generation in progress; retry when idle",
                    "active_request": active_info,
                    "queue_depth": queued,
                },
            )
        try:
            body = await req.json()
        except Exception:
            body = {}
        size = int(body.get("matrix_size") or 128)
        repeats = int(body.get("repeats") or 2)
        reason = str(body.get("reason") or "admin request")
        try:
            _bcast({
                "op": "metal_warmup",
                "matrix_size": size,
                "repeats": repeats,
                "reason": reason,
            }, rank)
        except Exception as e:
            logger.warning("metal warmup broadcast failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": f"metal warmup broadcast failed: {e}"},
            )
        local = _metal_warmup_touch(size=size, repeats=repeats, reason=reason)
        return {"ok": bool(local.get("ok")), "local": local}

    @app.post("/admin/runtime-tuning")
    async def admin_runtime_tuning(req: Request):
        if not admin_localhost_only(req):
            return JSONResponse(
                status_code=403,
                content={"error": "runtime tuning is localhost-only"},
            )
        with state_lock:
            active = request_state.get("active")
            queued = int(request_state.get("queued") or 0)
            active_info = dict(active) if active else None
        if active_info is not None or queued > 0:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "generation in progress; retry when idle",
                    "active_request": active_info,
                    "queue_depth": queued,
                },
            )
        try:
            body = await req.json()
        except Exception:
            body = {}
        values = body.get("values") if isinstance(body.get("values"), dict) else body
        if not isinstance(values, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "runtime tuning body must be an object"},
            )
        previous = _runtime_tuning_status()

        # Changing decode top-k state (reuse window, sparse block count, sort)
        # clears per-layer selection caches inside the model on both ranks. On
        # 2026-07-01 doing that under a hot RAM prompt cache desynchronized
        # distributed decode (wedged at 132 tokens until the watchdog killed
        # both ranks and orphaned wired memory). Drop the RAM cache on both
        # ranks before applying such a change; the next request rebuilds it
        # from prefill or SSD. Normal requests never hit this path.
        def _tuning_cmp(value):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return str(value)

        decode_state_keys = (
            "decode_topk_reuse_tokens",
            "sparse_topk_blocks",
            "compact_decode_sort_topk",
        )
        decode_state_changing = any(
            key in values and _tuning_cmp(values[key]) != _tuning_cmp(previous.get(key))
            for key in decode_state_keys
        )
        cache_reset_before_tuning = False
        if decode_state_changing:
            pc_status = _prompt_cache_status()
            resident_slots = (pc_status.get("session_map") or {}).get("resident_slots") or []
            if pc_status.get("loaded") or resident_slots:
                reset_reason = "runtime tuning decode top-k state change"
                try:
                    _bcast({
                        "op": "reset_prompt_cache",
                        "reason": reset_reason,
                        "clear_memory": False,
                        "clear_manifest": False,
                    }, rank)
                except Exception as e:
                    logger.warning("pre-tuning prompt-cache reset broadcast failed: %s", e)
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"pre-tuning prompt-cache reset broadcast failed: {e}"},
                    )
                _reset_prompt_cache(reset_reason, clear_manifest=False)
                cache_reset_before_tuning = True
                logger.info(
                    "dropped RAM prompt cache before decode top-k tuning change: %s",
                    {k: values[k] for k in decode_state_keys if k in values},
                )

        clamped = {}
        try:
            changed = _set_runtime_tuning(values or {}, clamped_out=clamped)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            # Storage caps are rank0-local by design: rank1's SSD prune keeps
            # its env-seeded cap (different disk budget) and capture-corpus
            # writes only happen on rank0, so those keys never ride the
            # tuning broadcast.
            _bcast({
                "op": "runtime_tuning",
                "values": {
                    key: value
                    for key, value in _runtime_tuning_status().items()
                    if key not in _STORAGE_TUNING_KEYS
                },
            }, rank)
        except Exception as e:
            _set_runtime_tuning(previous)
            _apply_runtime_model_tuning(model)
            logger.warning("runtime tuning broadcast failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": f"runtime tuning broadcast failed: {e}"},
            )
        model_tuning = _apply_runtime_model_tuning(model)
        logger.info("localhost runtime tuning updated: %s", changed)
        response = {
            "ok": True,
            "changed": changed,
            "clamped": clamped,
            "cache_reset_before_tuning": cache_reset_before_tuning,
            "model_tuning": model_tuning,
            "runtime_tuning": _runtime_tuning_status(),
            "generation_defaults": _generation_defaults_status(),
        }
        if any(key in _STORAGE_TUNING_KEYS for key in (values or {})):
            response["storage_note"] = (
                "storage caps are rank0-local: rank1's SSD prune keeps its "
                "env-seeded cap until the next restart re-reads .env.local"
            )
        return response

    @app.post("/v1/stop")
    @app.post("/stop")
    async def stop_generation(req: Request):
        """Request an in-flight generation to stop at the next token boundary.

        Sets a flag that run_generation / run_generation_stream check between
        tokens. Both ranks check at the same boundary (they process identical
        tokens per step), so the pipeline stays in lockstep when it breaks.
        The flag is cleared at the start of the next request. This does NOT
        interrupt a collective mid-flight; it takes effect at the next decode
        step. If no generation is active, this is a no-op.
        """
        try:
            body = await req.json()
        except Exception:
            body = {}
        expected_request_id = _stop_request_target(body)
        with state_lock:
            active = request_state.get("active")
            active_info = dict(active) if active else None
            stop_target = dict(_ACTIVE_STOP_TARGET)
        if active_info is None:
            return {
                "stopped": False,
                "mode": "drain_only",
                "reason": "no active generation",
                "expected_request_id": expected_request_id,
                "active_request": None,
            }
        if not _stop_request_matches_active(body, active_info):
            return JSONResponse(
                status_code=409,
                content={
                    "stopped": False,
                    "mode": "request_id_guard",
                    "reason": "active request changed before cancellation",
                    "expected_request_id": expected_request_id,
                    "active_request": active_info,
                },
            )
        if not SAFE_DECODE_STOP:
            return {
                "stopped": False,
                "mode": "drain_only",
                "reason": ("decode-phase stop is gated off (MLX_M3_SAFE_DECODE_STOP=0) "
                           "pending offline acceptance; the request drains to its budget"),
                "active_request": active_info,
            }
        active_request_id = str(active_info.get("id") or "") or None
        stop_nonce = (
            stop_target.get("nonce")
            if stop_target.get("request_id") == active_request_id
            else None
        )
        stop_state = _request_inflight_stop(
            "api stop",
            active_info,
            request_id=active_request_id,
            stop_nonce=stop_nonce,
        )
        if active_info is not None:
            logger.info("[rank 0] in-flight stop requested for %s", active_info.get("id"))
        return {
            "stopped": True,
            "mode": "distributed_token_boundary",
            "stop_check_every": STOP_CHECK_EVERY,
            "prefill_stop_check_every": PREFILL_STOP_CHECK_EVERY,
            **stop_state,
            "active_request": active_info,
        }

    @app.post("/v1/chat/completions")
    async def chat(request: dict, http_request: Request = None):
        with state_lock:
            if request_state["shutdown_requested"]:
                return JSONResponse(
                    status_code=503,
                    content={"error": {
                        "message": "Server is shutting down",
                        "type": "server_shutdown",
                    }},
                )

        msgs = request["messages"]
        if int(request.get("n", 1) or 1) != 1:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Only n=1 is supported by this distributed endpoint",
                                   "type": "invalid_request_error"}},
            )
        requested_max_tokens = request.get(
            "max_tokens",
            request.get("max_completion_tokens"),
        )
        max_tokens_source = "request" if requested_max_tokens is not None else "default"
        max_tokens = requested_max_tokens if requested_max_tokens is not None else DEFAULT_MAX_TOKENS
        max_tokens = int(max_tokens or DEFAULT_MAX_TOKENS)
        thinking_mode = _resolve_thinking_mode(request)
        stream = bool(request.get("stream", False))
        nonstream_request_key = (
            _nonstream_request_fingerprint(request)
            if NONSTREAM_COALESCE_ENABLED and not stream
            else None
        )
        response_model = _response_model_id(request.get("model"))
        cache_session_id, cache_session_source = _request_cache_session(request)
        tools = _tools_from_request(request)
        title_metadata_request = _request_looks_like_title_metadata(
            request,
            msgs,
            tools,
        )
        if title_metadata_request:
            # Chat frontends often inherit the selected thinking model for
            # title generation. Letting that tiny no-tool sidecar use the
            # agent-sized default can occupy the single distributed slot for
            # thousands of hidden reasoning tokens and trigger duplicate
            # client retries. Explicit max-token requests, streaming chat,
            # and every tool request bypass this narrowly scoped policy.
            request["_metadata_request"] = "title"
            thinking_mode = "disabled"
            if max_tokens > TITLE_DEFAULT_MAX_TOKENS:
                logger.info(
                    "using title metadata max_tokens=%s instead of text default %s",
                    TITLE_DEFAULT_MAX_TOKENS,
                    max_tokens,
                )
                max_tokens = TITLE_DEFAULT_MAX_TOKENS
                max_tokens_source = "title_metadata_default"
        tool_choice_error = _tool_choice_validation_error(request, tools)
        if tool_choice_error:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": tool_choice_error,
                    "type": "invalid_request_error",
                    "param": "tool_choice",
                }},
            )
        # Required/named choices need an instruction in a template-visible
        # role. A trailing system role is ignored by MiniMax's native template.
        if tools and _tool_choice_required_name(request)[0]:
            msgs = _apply_tool_choice_instruction(msgs, request)
        if (
            tools
            and TOOL_THINKING_MODE != "request"
            and TOOL_THINKING_MODE in VALID_THINKING_MODES
            and thinking_mode != TOOL_THINKING_MODE
        ):
            logger.info(
                "tool request using internal thinking_mode=%s (client requested %s)",
                TOOL_THINKING_MODE,
                thinking_mode,
            )
            thinking_mode = TOOL_THINKING_MODE
        tool_loop_diag = _tool_loop_steering_diag(msgs, tools) if TOOL_COMPAT_OVERLAY else None
        if tool_loop_diag:
            reasons = set(tool_loop_diag.get("reasons") or [])
            if (
                "repeated_user_tool_prompt" in reasons
                and int(tool_loop_diag.get("repeated_user_prompt_count") or 0) >= 3
            ):
                logger.warning(
                    "[rank 0] rejecting repeated user tool prompt loop "
                    "(count=%s, stream=%s, model=%s)",
                    tool_loop_diag.get("repeated_user_prompt_count"),
                    stream,
                    response_model,
                )
                return JSONResponse(
                    status_code=409,
                    headers={"Retry-After": "30"},
                    content={"error": {
                        "message": (
                            "Repeated long tool prompt loop detected before "
                            "inference. Stop this agent run and start a fresh "
                            "session, or ask for a final answer from gathered "
                            "context."
                        ),
                        "type": "tool_loop_detected",
                        "code": "repeated_user_tool_prompt",
                    }},
                )
            if "force_final" in reasons:
                tools = None
                request["_tool_source"] = "tools_loop_force_final"
            else:
                tools, filtered_control_tools = _filter_looping_control_tools(
                    tools,
                    tool_loop_diag,
                )
                if filtered_control_tools:
                    tool_loop_diag["filtered_tools"] = filtered_control_tools
                    request["_tool_source"] = "tools_control_loop_filtered"
                    if _require_alternate_work_tool(
                        request,
                        tools,
                        filtered_control_tools,
                    ):
                        request["_tool_source"] = (
                            "tools_repeated_work_filtered_required"
                        )
            request["_tool_loop_steering"] = tool_loop_diag
            logger.warning(
                "tool-loop steering hint active "
                "(reasons=%s, tool_turns=%s, tool_results=%s, repeated_tool=%s/%s, filtered=%s, source=%s)",
                ",".join(tool_loop_diag.get("reasons") or []),
                tool_loop_diag.get("assistant_tool_turns"),
                tool_loop_diag.get("tool_results"),
                tool_loop_diag.get("repeated_tool"),
                tool_loop_diag.get("repeated_tool_count"),
                ",".join(tool_loop_diag.get("filtered_tools") or []),
                request.get("_tool_source"),
            )
        loop_force_final = (
            request.get("_tool_source") == "tools_loop_force_final"
        )
        parser_needed = bool(tools or loop_force_final)
        tool_module = _load_tool_parser(processor) if parser_needed else None
        gen_params = _request_generation_params(request, tools=tools)

        # Preserve OpenAI message roles/content for the model's native chat
        # template. Flattening this to "user: ..." text drops assistant/system
        # structure and, worse, used to disable add_generation_prompt via a
        # positional-argument mixup below.
        processed_messages, images = [], []
        client_preserves_reasoning = _client_preserves_assistant_reasoning(msgs)
        for m in msgs:
            role = m.get("role", "user")
            msg = {"role": role}
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in ("text", "input_text"):
                            text_parts.append(part.get("text", "") or part.get("content", ""))
                        else:
                            image_source = _extract_image_source(part)
                            if image_source:
                                images.append(image_source)
                content_text = "\n".join(p for p in text_parts if p)
            else:
                content_text = content
            if role == "assistant":
                content_text = _sanitize_inbound_tool_call_content(m, content_text)
                msg["content"] = _assistant_content_for_template(
                    m,
                    content_text,
                    session_id=cache_session_id,
                    session_source=cache_session_source,
                )
                if (
                    _enable_thinking_for_generation(thinking_mode)
                    and PROMPT_CACHE_THINKING_MODE == "visible"
                    and not _has_model_facing_reasoning(msg.get("content"))
                ):
                    msg["reasoning_content"] = " "
            else:
                msg["content"] = _sanitize_inbound_message_content(role, content_text)
            if "tool_calls" in m:
                msg["tool_calls"] = _normalize_tool_calls(m["tool_calls"])
            for key in ("tool_call_id", "name"):
                if key in m:
                    msg[key] = m[key]
            processed_messages.append(msg)
        processed_messages, date_context_injected = _add_date_system_context(
            processed_messages,
            session_id=cache_session_id,
        )
        request["_date_context_injected"] = date_context_injected
        processed_messages = _add_tool_system_hint_if_needed(
            processed_messages,
            request,
            tools,
            tool_loop_diag=tool_loop_diag,
        )
        require_tool_call = bool(
            tools
            and _tool_request_requires_call(processed_messages, request)
        )
        action_tool_task = bool(
            tools
            and (
                _tool_choice_required_name(request)[0]
                or _tool_text_requests_action(
                    _last_user_instruction_text(processed_messages)
                )
            )
        )

        if (
            requested_max_tokens is None
            and OPENWEBUI_DEFAULT_MAX_TOKENS > 0
            # Agent/tool requests need room to finish a structured call. The
            # old stream_options heuristic also matched ZCode and truncated a
            # large write at 2,048 tokens before the write/scaffold guards
            # could produce an executable call.
            and not tools
            and _request_looks_like_openwebui(request, processed_messages)
        ):
            if max_tokens > OPENWEBUI_DEFAULT_MAX_TOKENS:
                logger.info(
                    "using OpenWebUI default max_tokens=%s instead of text default %s",
                    OPENWEBUI_DEFAULT_MAX_TOKENS,
                    max_tokens,
                )
                max_tokens = OPENWEBUI_DEFAULT_MAX_TOKENS
                max_tokens_source = "openwebui_default"

        prompt_char_count = sum(
            len(m.get("content") or "") for m in processed_messages
            if isinstance(m.get("content"), str)
        )
        logger.info(
            "[rank 0] chat request prepared "
            "(messages=%s, prompt_chars=%s, images=%s, stream=%s, max_tokens=%s, "
            "thinking=%s, model=%s)",
            len(processed_messages), prompt_char_count, len(images), stream,
            max_tokens, thinking_mode, response_model,
        )

        nonstream_req_id = None
        coalesce_entry = None
        if not stream:
            nonstream_req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            coalesce_entry, coalesce_owner, coalesce_replay = (
                nonstream_coalescer.claim(
                    nonstream_request_key,
                    nonstream_req_id,
                )
            )
            if not coalesce_owner:
                owner_id = coalesce_entry.get("owner_request_id")
                logger.info(
                    "[rank 0] non-stream %s joined exact retry owner=%s key=%s "
                    "(completed_replay=%s)",
                    nonstream_req_id,
                    owner_id,
                    (nonstream_request_key or "")[:12],
                    coalesce_replay,
                )
                while not coalesce_entry["event"].is_set():
                    if (
                        http_request is not None
                        and await http_request.is_disconnected()
                    ):
                        connected = nonstream_coalescer.disconnect(
                            coalesce_entry, nonstream_req_id
                        )
                        logger.info(
                            "[rank 0] coalesced non-stream retry %s disconnected "
                            "while owner=%s continued (connected_clients=%s)",
                            nonstream_req_id,
                            owner_id,
                            connected,
                        )
                        return JSONResponse(
                            status_code=499,
                            content={"error": {
                                "message": (
                                    "Client disconnected while waiting for the "
                                    "original request"
                                ),
                                "type": "client_disconnected",
                            }},
                        )
                    await asyncio.sleep(0.25)
                coalesced_response = nonstream_coalescer.response(coalesce_entry)
                if coalesced_response is None:
                    return JSONResponse(
                        status_code=503,
                        content={"error": {
                            "message": (
                                "Original request ended without a reusable response"
                            ),
                            "type": "coalesced_request_unavailable",
                        }},
                    )
                response_status, response_payload = coalesced_response
                logger.info(
                    "[rank 0] non-stream %s replaying owner=%s response status=%s",
                    nonstream_req_id,
                    owner_id,
                    response_status,
                )
                if 200 <= response_status < 300:
                    return response_payload
                return JSONResponse(
                    status_code=response_status,
                    content=response_payload,
                )

        # Build prompt with M3's chat template + requested thinking_mode
        from mlx_vlm.prompt_utils import apply_chat_template
        try:
            tk = _thinking_template_kwargs(
                model.config,
                enable_thinking=(thinking_mode == "enabled"),
                thinking_mode=thinking_mode,
            )
            if tools:
                tk["tools"] = _model_facing_tool_schemas(tools)
            def _render_prompt_with_lock():
                with _tokenizer_runtime_lock:
                    return apply_chat_template(
                        processor,
                        model.config,
                        processed_messages,
                        add_generation_prompt=True,
                        num_images=len(images),
                        **tk,
                    )

            # Do not block uvicorn's event loop while another request is
            # actively decoding and holding the tokenizer/runtime lock. Without
            # this, a follow-up OpenWebUI request can make /health time out and
            # look like an orphan even while the active generation is draining.
            prompt = await asyncio.to_thread(_render_prompt_with_lock)
        except Exception as e:
            logger.warning(f"chat template fallback: {e}")
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in processed_messages
            )

        # Broadcast image sources so rank 1 materializes the same visual input
        # instead of mirroring a text-only request.
        image_sources = images or None
        image_path = None
        if image_sources:
            if (
                requested_max_tokens is None
                and IMAGE_DEFAULT_MAX_TOKENS > 0
                # A compact image-chat answer can use the image default, but
                # tool clients may need thousands of tokens to emit an atomic
                # Write/Edit after inspecting an image. The 768-token chat cap
                # truncated those calls on every first attempt; structured
                # write guards already bound their payload safely.
                and not tools
            ):
                logger.info(
                    "using image default max_tokens=%s instead of text default %s",
                    IMAGE_DEFAULT_MAX_TOKENS,
                    max_tokens,
                )
                max_tokens = IMAGE_DEFAULT_MAX_TOKENS
                max_tokens_source = "image_default"
            if IMAGE_MAX_TOKENS > 0 and max_tokens > IMAGE_MAX_TOKENS:
                logger.info(
                    "clamping image request max_tokens from %s to %s",
                    max_tokens,
                    IMAGE_MAX_TOKENS,
                )
                max_tokens = IMAGE_MAX_TOKENS
                max_tokens_source = "image_clamp"
            try:
                image_path = [_materialize_image(src) for src in image_sources]
                image_path = [p for p in image_path if p]
                if len(image_path) == 1:
                    image_path = image_path[0]
            except Exception as e:
                logger.warning(f"image load failed: {e}")

        if (
            requested_max_tokens is None
            and not stream
            and not image_sources
            # Tool turns need room for narration + the call itself; clamping
            # them to the short chat default guarantees truncated, unusable
            # tool emissions (observed: 512-token cap eaten by narration).
            and not tools
            # Thinking turns burn the whole short budget on reasoning and
            # come back incomplete with empty content (512/512 mid-think,
            # 2026-07-09 zcode sidecars) — only clamp when thinking is off.
            and thinking_mode != "enabled"
            and NONSTREAM_DEFAULT_MAX_TOKENS > 0
            and max_tokens > NONSTREAM_DEFAULT_MAX_TOKENS
        ):
            logger.info(
                "using non-stream default max_tokens=%s instead of text default %s",
                NONSTREAM_DEFAULT_MAX_TOKENS,
                max_tokens,
            )
            max_tokens = NONSTREAM_DEFAULT_MAX_TOKENS
            max_tokens_source = "nonstream_default"

        _apply_default_thinking_budget(gen_params, thinking_mode, max_tokens)

        # Tokenize the rendered prompt so both ranks share the same cache key
        # (rank 0 broadcasts token_ids to rank 1 via _bcast). Used for cross-
        # request prompt caching when MLX_M3_PROMPT_CACHE=1.
        try:
            request_token_ids = (
                await asyncio.to_thread(_tokenize_prompt, processor, prompt)
                if _should_tokenize_prompt_for_cache(thinking_mode)
                else None
            )
        except Exception as e:
            logger.error("prompt tokenization failed: %s", e)
            error_payload = {
                "error": {
                    "message": f"Prompt tokenization failed: {e}",
                    "type": "generation_error",
                }
            }
            nonstream_coalescer.complete(
                coalesce_entry,
                error_payload,
                status_code=500,
            )
            return JSONResponse(status_code=500, content=error_payload)
        if request_token_ids is not None:
            logger.info(
                "prompt-cache: tokenized prompt -> %d tokens", len(request_token_ids)
            )
            if (
                HARD_MAX_INPUT_TOKENS > 0
                and len(request_token_ids) > HARD_MAX_INPUT_TOKENS
            ):
                logger.warning(
                    "rejecting %d-token input above safe cluster ceiling %d",
                    len(request_token_ids),
                    HARD_MAX_INPUT_TOKENS,
                )
                error_payload = {
                    "error": {
                        "message": (
                            "This request has "
                            f"{len(request_token_ids)} input tokens, above "
                            f"the cluster's safe limit of {HARD_MAX_INPUT_TOKENS}. "
                            "Compact the conversation and retry."
                        ),
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": "context_length_exceeded",
                    }
                }
                nonstream_coalescer.complete(
                    coalesce_entry,
                    error_payload,
                    status_code=400,
                )
                return JSONResponse(status_code=400, content=error_payload)
        elif PROMPT_CACHE_ENABLED and _enable_thinking_for_generation(thinking_mode):
            with _prompt_cache_lock:
                _set_prompt_cache_event(
                    "thinking_cache_bypass",
                    prompt_tokens=0,
                    reuse_tokens=0,
                    reason=f"MLX_M3_PROMPT_CACHE_THINKING_MODE={PROMPT_CACHE_THINKING_MODE}",
                    tokenization_skipped=True,
                )

        # Enforce a sane max_tokens ceiling so a runaway request (e.g. an agent
        # sending max_tokens=65536) can't grind the single-flight slot for ages.
        if MAX_TOKENS_CEILING > 0 and max_tokens > MAX_TOKENS_CEILING:
            logger.info(
                "clamping max_tokens %s -> %s (MLX_M3_MAX_TOKENS_CEILING)",
                max_tokens, MAX_TOKENS_CEILING,
            )
            max_tokens = MAX_TOKENS_CEILING
            max_tokens_source = "ceiling"

        request_shape = _request_shape_summary(
            request,
            processed_messages,
            prompt,
            request_token_ids,
            thinking_mode=thinking_mode,
            response_model=response_model,
            max_tokens=max_tokens,
            max_tokens_source=max_tokens_source,
            stream=stream,
            image_count=len(images),
            tools=tools,
            gen_params=gen_params,
        )
        if request_token_ids is not None:
            request_shape["prefill_step_size"] = _runtime_prefill_step_size(
                len(request_token_ids)
            )
        request_shape["require_tool_call"] = require_tool_call
        request_shape["action_tool_task"] = action_tool_task

        # Do not clear stop state while preparing. An exact retry can reach
        # this point while its owner is still generating; clearing here would
        # erase the owner's cancellation. Each generation worker clears stop
        # state only after it owns the distributed slot.
        import m3_eagle3 as _m3e3
        _eagle3_on = bool(
            _m3e3.enabled()
            and _m3e3.RUNTIME_ENABLED.get("value")
            and not image_sources
        )
        _m3e3.REQUEST_ACTIVE["value"] = _eagle3_on
        rank_request = {"stop_nonce": uuid.uuid4().hex,
                        "prompt": prompt, "max_tokens": max_tokens,
                        "thinking_mode": thinking_mode,
                        "gen_params": gen_params,
                        "image_sources": image_sources,
                        "token_ids": request_token_ids,
                        "session_id": cache_session_id,
                        "session_source": cache_session_source,
                        "eagle3": _eagle3_on,
                        "tools": tools,
                        # OpenAI tool_choice forcing: the usable-turn ladder
                        # reads this to treat call-less turns as unusable
                        # when 'required'/named (2026-07-09).
                        "tool_choice": request.get(
                            "tool_choice", request.get("function_call")),
                        "require_tool_call": require_tool_call,
                        "action_tool_task": action_tool_task,
                        # Tool-bearing requests are buffered and parsed after
                        # decode, so reaching max_tokens can leave no visible
                        # content even when the stable tool/template prefix is
                        # still safe to reuse. Keep the stricter incomplete
                        # thinking reset for normal chat/agent text, but do
                        # not let short tool probes wipe the shared tool prefix.
                        "reset_incomplete_thinking_on_limit": not bool(tools)}

        # Generate WITH robust error handling — never hang, never orphan memory
        if stream:
            from fastapi.responses import StreamingResponse
            import json

            async def sse_stream():
                req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())
                out_q = queue.Queue()
                done_event = threading.Event()
                client_connected = threading.Event()
                client_connected.set()

                def _put_sse(payload):
                    if client_connected.is_set():
                        out_q.put(f"data: {json.dumps(payload)}\n\n")

                def _producer():
                    """Live token streaming with synchronized cancel on disconnect.

                    Uses run_generation_stream() (the per-token generator) so each
                    reasoning/content delta is pushed to the SSE queue IMMEDIATELY
                    as tokens are decoded -> OpenWebUI shows live token generation.

                    On client disconnect, the SSE wrapper sets the shared stop flag.
                    Both ranks then observe that flag at the same token-boundary
                    all-sum check inside run_generation_stream(), release the slot,
                    and avoid the old drain-to-max_tokens behavior.
                    """
                    producer_error = None
                    full_output = ""
                    visible_output = ""
                    reasoning_recall_stored = False
                    generation_tokens = 0
                    chunks_emitted = 0
                    chars_raw = 0
                    reasoning_chars = 0
                    content_chars = 0
                    generation_cancelled = False
                    rank_request_broadcast = False
                    op_channel_held = False
                    tool_stream_pending = ""      # live tool-turn content awaiting holdback release
                    tool_stream_silenced = False  # once markup risk appears, buffer the rest
                    tool_streamed_len = 0         # chars already streamed live this tool turn
                    tool_reasoning_live = ""      # reasoning already streamed live this tool turn
                    try:
                        # Own the rank-0 op channel for the WHOLE request
                        # transaction: keepwarm bcast, request bcast, every
                        # decode collective, and the mirror barrier in the
                        # finally below. generation_lock is already held by
                        # this request (acquired in acquire_generation_slot),
                        # so lock order holds. RLock: same-thread re-entry by
                        # request_start_keepwarm is fine.
                        _RANK0_OP_MUTEX.acquire()
                        op_channel_held = True
                        _clear_stop_request()
                        _clear_prefill_stop_file("rank 0 stream generation start")
                        _STOP_NONCE["value"] = rank_request.get("stop_nonce")
                        _FORCE_EOS["active"] = False
                        request_start_keepwarm(req_id)
                        _bcast(rank_request, rank)
                        rank_request_broadcast = True
                        _put_sse({
                            "id": req_id, "object": "chat.completion.chunk",
                            "created": created, "model": response_model,
                            "choices": [{"index": 0, "delta": {"role": "assistant"},
                                         "finish_reason": None}],
                        })

                        def _prefill_progress(processed_tokens, total_tokens):
                            total_tokens = int(total_tokens or 0)
                            processed_tokens = int(processed_tokens or 0)
                            progress = (
                                processed_tokens / total_tokens
                                if total_tokens > 0 else 0.0
                            )
                            _watchdog_tick(progress=True)
                            update_generation_slot(
                                active,
                                prefill_processed_tokens=processed_tokens,
                                prefill_total_tokens=total_tokens,
                                prefill_progress=round(progress, 4),
                                prefill_last_progress_s=round(
                                    time.time() - active["started"], 3
                                ),
                                last_progress_s=round(
                                    time.time() - active["started"], 3
                                ),
                            )

                        for chunk in run_generation_stream(
                            model, processor, prompt, max_tokens, rank,
                            image=image_path, thinking_mode=thinking_mode,
                            enable_thinking=_enable_thinking_for_generation(thinking_mode),
                            gen_params=gen_params,
                            token_ids=request_token_ids,
                            session_id=cache_session_id,
                            session_source=cache_session_source,
                            reset_incomplete_thinking_on_limit=not bool(tools),
                            prefill_progress_cb=_prefill_progress,
                            tool_module=tool_module,
                            tools=tools,
                            require_tool_call=require_tool_call,
                            action_tool_task=action_tool_task,
                        ):
                            chunks_emitted += 1
                            prompt_tps = chunk.pop("_prompt_tps", None) if chunk else None
                            prompt_tokens = chunk.pop("_prompt_tokens", None) if chunk else None
                            cached_tokens = chunk.pop("_cached_tokens", None) if chunk else None
                            prompt_cache_prepare = chunk.pop("_prompt_cache_prepare", None) if chunk else None
                            cache_prepare_started_at = chunk.pop("_cache_prepare_started_at", None) if chunk else None
                            cache_prepare_finished_at = chunk.pop("_cache_prepare_finished_at", None) if chunk else None
                            stream_generate_started_at = chunk.pop("_stream_generate_started_at", None) if chunk else None
                            runtime_lock_wait_started_at = chunk.pop("_runtime_lock_wait_started_at", None) if chunk else None
                            runtime_lock_acquired_at = chunk.pop("_runtime_lock_acquired_at", None) if chunk else None
                            first_generator_yield_at = chunk.pop("_first_generator_yield_at", None) if chunk else None
                            if chunk and chunk.pop("_cancelled", False):
                                generation_cancelled = True
                            if chunk:
                                generation_tokens = int(
                                    chunk.pop("_generation_tokens", None)
                                    or generation_tokens
                                    or chunks_emitted
                                )
                            metric_updates = {}
                            if prompt_tps:
                                metric_updates["prompt_tps"] = float(prompt_tps)
                            if prompt_tokens:
                                metric_updates["prompt_tokens"] = int(prompt_tokens)
                            if cached_tokens:
                                metric_updates["cached_tokens"] = int(cached_tokens)
                            if prompt_cache_prepare:
                                metric_updates["prompt_cache_prepare"] = prompt_cache_prepare
                            if (
                                cache_prepare_started_at
                                and cache_prepare_finished_at
                            ):
                                metric_updates["cache_prepare_s"] = round(
                                    cache_prepare_finished_at
                                    - cache_prepare_started_at,
                                    3,
                                )
                            if first_generator_yield_at:
                                metric_updates["first_generator_token_s"] = round(
                                    first_generator_yield_at - active["started"],
                                    3,
                                )
                            if (
                                first_generator_yield_at
                                and stream_generate_started_at
                            ):
                                metric_updates["first_yield_after_generate_s"] = round(
                                    first_generator_yield_at
                                    - stream_generate_started_at,
                                    3,
                                )
                            if (
                                runtime_lock_wait_started_at
                                and runtime_lock_acquired_at
                            ):
                                metric_updates["runtime_lock_wait_s"] = round(
                                    runtime_lock_acquired_at
                                    - runtime_lock_wait_started_at,
                                    3,
                                )
                            if (
                                first_generator_yield_at
                                and runtime_lock_acquired_at
                            ):
                                metric_updates["first_yield_after_lock_s"] = round(
                                    first_generator_yield_at
                                    - runtime_lock_acquired_at,
                                    3,
                                )
                            if chunk is None:
                                update_generation_slot(
                                    active,
                                    tokens_emitted=generation_tokens,
                                    chunks_emitted=chunks_emitted,
                                    last_progress_s=round(
                                        time.time() - active["started"], 3
                                    ),
                                    **metric_updates,
                                )
                                continue
                            # _raw carries the full decoded text for tool-call parsing later
                            raw = chunk.pop("_raw", None)
                            if raw:
                                chars_raw += len(raw)
                                full_output += raw
                            update_generation_slot(
                                active,
                                chars_raw=chars_raw,
                                reasoning_chars=reasoning_chars,
                                content_chars=content_chars,
                                tokens_emitted=generation_tokens,
                                chunks_emitted=chunks_emitted,
                                last_progress_s=round(time.time() - active["started"], 3),
                                **metric_updates,
                            )
                            # If the client disconnected, the SSE wrapper has
                            # already requested distributed cancellation. Keep
                            # consuming until the next synchronized stop boundary,
                            # but do not queue more SSE chunks.
                            if not client_connected.is_set():
                                continue
                            # Tool-bearing requests are transaction-buffered by
                            # default. Native XML may be repairable only after
                            # EOS. Reasoning is a separate channel and ends
                            # before native tool XML begins, so it can stream
                            # live while visible content and tool markup remain
                            # buffered for validation.
                            if parser_needed and TOOL_STREAM_BUFFER_ALL:
                                r_piece = chunk.get("reasoning") or ""
                                if r_piece and EMIT_TOOL_REASONING:
                                    reasoning_chars += len(r_piece)
                                    tool_reasoning_live += r_piece
                                    if not active.get("first_visible_delta_s"):
                                        metric_updates["first_visible_delta_s"] = round(
                                            time.time() - active["started"], 3
                                        )
                                        update_generation_slot(active, **metric_updates)
                                    _put_sse({
                                        "id": req_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": response_model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"reasoning_content": r_piece},
                                            "finish_reason": None,
                                        }],
                                    })
                                continue
                            # Native live tool-delta path. Ordinary text streams
                            # immediately; native tool XML is held for EOS parse.
                            if tools:
                                # Thinking streams live on tool turns exactly
                                # like oMLX: the splitter routes tool markup to
                                # the content channel (post-</mm:think>), so
                                # reasoning deltas can never leak tool XML and
                                # need no holdback. Degenerate marker spam is
                                # still reaped by the classifier on _raw.
                                r_piece = chunk.get("reasoning") or ""
                                if r_piece and EMIT_TOOL_REASONING:
                                    reasoning_chars += len(r_piece)
                                    tool_reasoning_live += r_piece
                                    if not active.get("first_visible_delta_s"):
                                        metric_updates["first_visible_delta_s"] = round(
                                            time.time() - active["started"], 3
                                        )
                                        update_generation_slot(active, **metric_updates)
                                    _put_sse({
                                        "id": req_id, "object": "chat.completion.chunk",
                                        "created": created, "model": response_model,
                                        "choices": [{"index": 0,
                                                     "delta": {"reasoning_content": r_piece},
                                                     "finish_reason": None}],
                                    })
                                if not TOOL_STREAM_CONTENT or tool_stream_silenced:
                                    continue
                                piece = chunk.get("content") or ""
                                if not piece:
                                    continue
                                tool_stream_pending += piece
                                # Release text up to any '<', then classify it:
                                # a real tool-marker prefix silences the turn;
                                # ordinary angle brackets (code! generics!
                                # HTML!) stream through. v1 silenced on EVERY
                                # '<' and killed streaming ~100 tokens into
                                # code-heavy turns.
                                # "]<]minimax" catches the namespace token from
                                # its first byte; "<]minimax" catches it when
                                # the scan lands on its inner '<'. Without
                                # them the raw marker streamed to clients
                                # before "<tool_call" tripped silencing
                                # (2026-07-06 audit).
                                _markers = ("<tool_call", "</tool_call",
                                            "<invoke", "</invoke", "<minimax",
                                            "<]minimax", "]<]minimax")
                                out_parts = []
                                while True:
                                    lt_a = tool_stream_pending.find("<")
                                    lt_b = tool_stream_pending.find("]")
                                    candidates = [x for x in (lt_a, lt_b) if x >= 0]
                                    lt = min(candidates) if candidates else -1
                                    if lt < 0:
                                        safe_len = len(tool_stream_pending) - TOOL_STREAM_HOLDBACK_CHARS
                                        if safe_len > 0:
                                            out_parts.append(tool_stream_pending[:safe_len])
                                            tool_stream_pending = tool_stream_pending[safe_len:]
                                        break
                                    out_parts.append(tool_stream_pending[:lt])
                                    rest = tool_stream_pending[lt:]
                                    if any(rest.startswith(m) for m in _markers):
                                        tool_stream_pending = ""
                                        tool_stream_silenced = True
                                        break
                                    if any(m.startswith(rest) for m in _markers):
                                        # could still become a marker — wait
                                        tool_stream_pending = rest
                                        break
                                    out_parts.append(rest[0])
                                    tool_stream_pending = rest[1:]
                                emit_piece = "".join(out_parts)
                                if not emit_piece:
                                    continue
                                visible_output += emit_piece
                                content_chars += len(emit_piece)
                                tool_streamed_len += len(emit_piece)
                                if not active.get("first_visible_delta_s"):
                                    metric_updates["first_visible_delta_s"] = round(
                                        time.time() - active["started"], 3
                                    )
                                    update_generation_slot(active, **metric_updates)
                                _put_sse({
                                    "id": req_id, "object": "chat.completion.chunk",
                                    "created": created, "model": response_model,
                                    "choices": [{"index": 0,
                                                 "delta": {"content": emit_piece},
                                                 "finish_reason": None}],
                                })
                                continue
                            # Emit any reasoning/content delta live
                            delta = {}
                            if chunk.get("reasoning"):
                                # reasoning_content ONLY — oMLX parity. An extra
                                # "reasoning" alias makes downstream shims
                                # italicize thinking into visible chat text.
                                delta["reasoning_content"] = chunk["reasoning"]
                                reasoning_chars += len(chunk["reasoning"])
                            if chunk.get("content"):
                                delta["content"] = chunk["content"]
                                visible_output += chunk["content"]
                                content_chars += len(chunk["content"])
                            if not delta:
                                continue
                            if not active.get("first_visible_delta_s"):
                                metric_updates["first_visible_delta_s"] = round(
                                    time.time() - active["started"],
                                    3,
                                )
                                update_generation_slot(active, **metric_updates)
                            _put_sse({
                                "id": req_id, "object": "chat.completion.chunk",
                                "created": created, "model": response_model,
                                "choices": [{"index": 0, "delta": delta,
                                             "finish_reason": None}],
                            })

                        # Generation complete. Parse tool calls from full output.
                        if generation_cancelled or _user_stop_requested(req_id):
                            # A cancelled distributed decode can leave one
                            # rank with a locally retained input prefix while
                            # the peer has already dropped its physical KV.
                            # The next prefix-consensus call is too late: JACCL
                            # may already see mismatched cache shapes. Reset
                            # only the uncertain active RAM cache on both ranks
                            # now; completed resident and SSD sessions remain.
                            _reset_prompt_cache_on_all_ranks(
                                rank,
                                "reset after cancelled stream",
                                clear_memory=False,
                                clear_manifest=False,
                                clear_resident=False,
                            )
                            logger.info(
                                "[rank 0] stream %s cancelled before final "
                                "tool/content parse; finishing without retry text",
                                req_id,
                            )
                            if client_connected.is_set():
                                _put_sse({
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": response_model,
                                    "choices": [{"index": 0, "delta": {},
                                                 "finish_reason": "stop"}],
                                })
                                out_q.put("data: [DONE]\n\n")
                            return
                        if tools and client_connected.is_set():
                            def _retry_progress(tokens, chars, metrics=None):
                                update_generation_slot(
                                    active,
                                    tokens_emitted=int(tokens),
                                    chars_raw=int(chars),
                                    last_progress_s=round(
                                        time.time() - active["started"], 3
                                    ),
                                )
                            full_output = _ensure_usable_tool_turn(
                                model, processor, rank,
                                full_output=full_output,
                                rank_request=rank_request,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                thinking_mode=thinking_mode,
                                gen_params=gen_params,
                                image_path=image_path,
                                token_ids=request_token_ids,
                                session_id=cache_session_id,
                                session_source=cache_session_source,
                                tool_module=tool_module,
                                tools=tools,
                                processed_messages=processed_messages,
                                req_id=req_id,
                                stream=True,
                                should_abort=lambda: (
                                    not client_connected.is_set()
                                    or _user_stop_requested(req_id)
                                ),
                                progress_cb=_retry_progress,
                                action_tool_task=action_tool_task,
                            )
                        tool_calls, remaining_text = _parse_tool_calls(
                            full_output, tool_module, tools
                        )
                        safe_remaining_text = (
                            _strip_raw_tool_blocks(
                                remaining_text or full_output or "",
                                tool_module,
                            )
                            if tool_module else (remaining_text or full_output or "")
                        )
                        if loop_force_final and tool_calls:
                            logger.warning(
                                "[rank 0] stream %s suppressed a repeated tool "
                                "call on a forced-final loop turn",
                                req_id,
                            )
                            tool_calls = []
                            safe_remaining_text = _strip_raw_tool_blocks(
                                remaining_text or full_output or "",
                                tool_module,
                            ).strip()
                            if not safe_remaining_text:
                                safe_remaining_text = (
                                    _tool_loop_forced_final_fallback(tool_loop_diag)
                                )
                        raw_tool_marker_only = bool(
                            tools
                            and not tool_calls
                            and (
                                _looks_like_raw_tool_fragment(full_output, tool_module)
                                or "[Tool call:" in (full_output or "")
                            )
                            and not _strip_thinking_control_markers(
                                safe_remaining_text or ""
                            ).strip()
                        )
                        if not client_connected.is_set():
                            _reset_prompt_cache_on_all_ranks(
                                rank,
                                "reset after disconnected stream",
                                clear_memory=False,
                                clear_manifest=False,
                                clear_resident=False,
                            )
                            logger.warning(
                                "[rank 0] disconnected stream %s finished after "
                                "cancel; skipped tool parsing/prewarm cache update",
                                req_id,
                            )
                            return
                        prewarm_done = False
                        prewarm_attempted = False
                        unsafe_empty_tool_turn = False
                        dropped_invalid_tool_calls = 0
                        dropped_invalid_tool_names = []
                        emit_reasoning_fields = _should_emit_reasoning_fields(tools)
                        if client_connected.is_set():
                            if tool_calls:
                                (
                                    tool_calls,
                                    dropped_invalid_tool_calls,
                                    dropped_invalid_tool_names,
                                ) = (
                                    _validate_outgoing_tool_calls(
                                        tool_calls,
                                        tools,
                                        return_dropped=True,
                                        return_dropped_names=True,
                                        processed_messages=processed_messages,
                                        raw_output=full_output,
                                    )
                                )
                            if (
                                TOOL_COMPAT_OVERLAY
                                and not tool_calls
                                and dropped_invalid_tool_calls
                            ):
                                synthesized = _synthesize_write_command_tool_call(
                                    processed_messages,
                                    tools,
                                    dropped_invalid_tool_names,
                                )
                                if synthesized:
                                    logger.warning(
                                        "[rank 0] stream %s converted malformed "
                                        "tool call %s into %s for simple write request",
                                        req_id,
                                        dropped_invalid_tool_names,
                                        synthesized["function"]["name"],
                                    )
                                    tool_calls = [synthesized]
                                    dropped_invalid_tool_calls = 0
                                    dropped_invalid_tool_names = []
                            if tool_calls:
                                if EMIT_TOOL_REASONING:
                                    buffered_reasoning = _buffered_tool_reasoning(
                                        full_output,
                                        tool_module,
                                        thinking_mode,
                                    )
                                    buffered_reasoning = _remaining_tool_reasoning(
                                        buffered_reasoning,
                                        tool_reasoning_live,
                                    )
                                    if buffered_reasoning:
                                        reasoning_chars += len(buffered_reasoning)
                                        _put_sse({
                                            "id": req_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": response_model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "reasoning_content": buffered_reasoning,
                                                },
                                                "finish_reason": None,
                                            }],
                                        })
                                reasoning_recall_stored = _remember_assistant_reasoning(
                                    cache_session_id,
                                    visible_output,
                                    full_output,
                                    thinking_mode=thinking_mode,
                                    session_source=cache_session_source,
                                    tool_calls=tool_calls,
                                )
                                logger.info(
                                    "[rank 0] stream %s returning tool_calls=%s arg_keys=%s",
                                    req_id,
                                    _tool_call_names(tool_calls),
                                    _tool_call_arg_keys(tool_calls),
                                )
                                _put_sse({
                                    "id": req_id, "object": "chat.completion.chunk",
                                    "created": created, "model": response_model,
                                    "choices": [{"index": 0,
                                                 "delta": {"role": "assistant",
                                                           "tool_calls": tool_calls},
                                                "finish_reason": "tool_calls"}],
                                })
                            elif loop_force_final:
                                content = _strip_thinking_control_markers(
                                    safe_remaining_text or ""
                                ).strip()
                                if (
                                    not content
                                    or _looks_like_leaked_reasoning_content(content)
                                ):
                                    content = _tool_loop_forced_final_fallback(
                                        tool_loop_diag
                                    )
                                visible_output += content
                                content_chars += len(content)
                                _put_sse({
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": response_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }],
                                })
                            elif tools:
                                reasoning, content = split_thinking_text(
                                    safe_remaining_text,
                                    assume_in_thinking=_enable_thinking_for_generation(
                                        thinking_mode
                                    ),
                                )
                                if reasoning and tool_reasoning_live:
                                    reasoning = _remaining_tool_reasoning(
                                        reasoning,
                                        tool_reasoning_live,
                                    ) or None
                                if content:
                                    content = _scrub_goal_state_echo(content)
                                if content and _looks_like_leaked_reasoning_content(content):
                                    content = None
                                if content and tool_streamed_len:
                                    # live tool-content streaming already
                                    # delivered a prefix; send only the
                                    # remainder (or separator + full text when
                                    # an in-place retry rewrote the turn)
                                    streamed_prefix = visible_output[-tool_streamed_len:]
                                    if content.startswith(streamed_prefix):
                                        content = content[len(streamed_prefix):] or None
                                    else:
                                        content = "\n" + content
                                delta = {}
                                if reasoning and emit_reasoning_fields:
                                    delta["reasoning_content"] = reasoning
                                    reasoning_chars += len(reasoning)
                                if content:
                                    delta["content"] = content
                                    visible_output += content
                                    content_chars += len(content)
                                if TOOL_COMPAT_OVERLAY and (
                                    dropped_invalid_tool_calls
                                    or (not content and not tool_streamed_len)
                                ):
                                    unsafe_empty_tool_turn = True
                                    if dropped_invalid_tool_calls:
                                        fallback = _tool_request_fallback_content(
                                            processed_messages,
                                            dropped_tool_names=dropped_invalid_tool_names,
                                            available_tool_names=_tool_names_from_schema(tools),
                                        )
                                    elif _looks_like_raw_tool_fragment(
                                        full_output,
                                        tool_module,
                                    ):
                                        # Never promote a malformed tool
                                        # turn's planning text into assistant
                                        # content. It was already emitted, if
                                        # requested, through reasoning_content.
                                        fallback = _tool_request_fallback_content(
                                            processed_messages,
                                            empty_tool_markers=True,
                                            thinking_mode=thinking_mode,
                                        )
                                    else:
                                        fallback = _strip_raw_tool_blocks(
                                            _strip_thinking_control_markers(
                                                safe_remaining_text
                                            ),
                                            tool_module,
                                        ).strip()
                                        if (
                                            not fallback
                                            or _looks_like_leaked_reasoning_content(fallback)
                                            or len(fallback) < 24
                                        ):
                                            fallback = _tool_request_fallback_content(
                                                processed_messages,
                                                empty_tool_markers=raw_tool_marker_only,
                                                thinking_mode=thinking_mode,
                                            )
                                        elif _looks_like_leaked_reasoning_content(fallback):
                                            fallback = _tool_request_fallback_content(
                                                processed_messages,
                                                empty_tool_markers=raw_tool_marker_only,
                                                thinking_mode=thinking_mode,
                                            )
                                    delta["content"] = fallback
                                    visible_output += fallback
                                    content_chars += len(fallback)
                                    logger.warning(
                                        "[rank 0] tool request %s produced no "
                                        "valid tool_calls or visible content "
                                        "(dropped_invalid_tool_calls=%s); "
                                        "sent compatibility fallback and kept RAM prefix cache",
                                        req_id,
                                        dropped_invalid_tool_calls,
                                    )
                                if delta:
                                    _put_sse({
                                        "id": req_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": response_model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }],
                                    })
                            if not tool_calls and not unsafe_empty_tool_turn:
                                reasoning_recall_stored = _remember_assistant_reasoning(
                                    cache_session_id,
                                    visible_output,
                                    full_output,
                                    thinking_mode=thinking_mode,
                                    session_source=cache_session_source,
                                )
                            if (
                                VISIBLE_TRANSCRIPT_PREWARM_BLOCKING
                                and not unsafe_empty_tool_turn
                            ):
                                prewarm_attempted = True
                                prewarm_done = _maybe_prewarm_visible_transcript(
                                    model,
                                    processor,
                                    rank,
                                    processed_messages,
                                    full_output,
                                    thinking_mode=thinking_mode,
                                    generated_tokens=generation_tokens,
                                    num_images=len(images),
                                    tools=tools,
                                    visible_output=visible_output,
                                    session_id=cache_session_id,
                                    session_source=cache_session_source,
                                    preserve_reasoning=(
                                        client_preserves_reasoning
                                        or reasoning_recall_stored
                                    ),
                                )
                            _put_sse({
                                "id": req_id, "object": "chat.completion.chunk",
                                "created": created, "model": response_model,
                                "choices": [{"index": 0, "delta": {},
                                             "finish_reason": "tool_calls" if tool_calls else "stop"}],
                            })
                            out_q.put("data: [DONE]\n\n")
                            update_generation_slot(
                                active,
                                finalization_elapsed_s=round(
                                    time.time() - active["started"], 3
                                ),
                                reasoning_chars=reasoning_chars,
                                content_chars=content_chars,
                            )
                        if (
                            not prewarm_done
                            and not prewarm_attempted
                            and not unsafe_empty_tool_turn
                        ):
                            if not tool_calls and not reasoning_recall_stored:
                                reasoning_recall_stored = _remember_assistant_reasoning(
                                    cache_session_id,
                                    visible_output,
                                    full_output,
                                    thinking_mode=thinking_mode,
                                    session_source=cache_session_source,
                                )
                            prewarm_attempted = True
                            prewarm_done = _maybe_prewarm_visible_transcript(
                                model,
                                processor,
                                rank,
                                processed_messages,
                                full_output,
                                thinking_mode=thinking_mode,
                                generated_tokens=generation_tokens,
                                num_images=len(images),
                                tools=tools,
                                visible_output=visible_output,
                                session_id=cache_session_id,
                                session_source=cache_session_source,
                                preserve_reasoning=(
                                    client_preserves_reasoning
                                    or reasoning_recall_stored
                                ),
                            )
                        if (
                            prewarm_attempted
                            and not prewarm_done
                            and _enable_thinking_for_generation(thinking_mode)
                            and PROMPT_CACHE_THINKING_MODE == "visible"
                        ):
                            # 2026-07-06 cache audit (LEAK 1): this used to drop
                            # the RAM cache on every gate-skipped prewarm "to
                            # avoid stale generated-tail reuse". In visible mode
                            # the cache key stores INPUT ids only and
                            # _prepare_cached_prompt trims the generated tail,
                            # so stale-tail reuse cannot occur through the
                            # prefix math; genuinely failed prewarms already
                            # self-reset inside _prewarm_prompt_cache. The drop
                            # converted benign skips into full-transcript
                            # re-prefills (and wiped all resident slots).
                            logger.info(
                                "[rank 0] visible thinking prewarm skipped for "
                                "%s; keeping RAM prompt cache (input-ids key; "
                                "next turn trims any generated tail)",
                                req_id,
                            )
                    except Exception as e:
                        producer_error = e
                        logger.error(f"[rank 0] stream FAILED: {e}\n{traceback.format_exc()}")
                        if client_connected.is_set():
                            _put_sse({
                                "id": req_id, "object": "chat.completion.chunk",
                                "created": created, "model": response_model,
                                "choices": [{"index": 0, "delta": {
                                    "content": f"\n\n[Generation error: {e}]"
                                }, "finish_reason": "stop"}],
                            })
                            out_q.put("data: [DONE]\n\n")
                    finally:
                        if rank_request_broadcast:
                            try:
                                update_generation_slot(
                                    active,
                                    mirror_sync_started_s=round(
                                        time.time() - active["started"], 3
                                    ),
                                    last_progress_s=round(
                                        time.time() - active["started"], 3
                                    ),
                                )
                                logger.info(
                                    "[rank 0] stream %s waiting for rank 1 mirror barrier",
                                    req_id,
                                )
                                _bcast(
                                    {
                                        "op": "generation_barrier",
                                        "request_id": req_id,
                                        "stream": True,
                                    },
                                    rank,
                                )
                                update_generation_slot(
                                    active,
                                    mirror_sync_done_s=round(
                                        time.time() - active["started"], 3
                                    ),
                                    last_progress_s=round(
                                        time.time() - active["started"], 3
                                    ),
                                )
                                logger.info(
                                    "[rank 0] stream %s rank 1 mirror barrier complete",
                                    req_id,
                                )
                            except Exception as e:
                                producer_error = producer_error or e
                                logger.error(
                                    "[rank 0] stream %s mirror barrier failed: %s",
                                    req_id,
                                    e,
                                )
                        # Transaction over (the mirror barrier above was its
                        # last collective) — release the op channel before
                        # the slot releases.
                        if op_channel_held:
                            _RANK0_OP_MUTEX.release()
                        if CLEAR_CACHE_AFTER_REQUEST or (
                            producer_error is not None and CLEAR_CACHE_AFTER_ERROR
                        ):
                            _clear_transient_mlx_memory(
                                f"rank 0 stream {req_id} complete"
                            )
                        else:
                            gc.collect()
                        release_generation_slot(req_id, active, producer_error)
                        done_event.set()
                        if client_connected.is_set():
                            out_q.put(None)
                        logger.info(
                            "[rank 0] stream producer done, mlx_cache_cleared=%s "
                            "(tokens_emitted=%s, chunks=%s, chars=%s, client=%s)",
                            bool(
                                CLEAR_CACHE_AFTER_REQUEST
                                or (
                                    producer_error is not None
                                    and CLEAR_CACHE_AFTER_ERROR
                                )
                            ),
                            generation_tokens, chunks_emitted, chars_raw,
                            "connected" if client_connected.is_set() else "disconnected-cancelled",
                        )

                active = await acquire_generation_slot(
                    req_id, stream=True, max_tokens=max_tokens,
                    image_count=len(images), request_shape=request_shape,
                    stop_nonce=rank_request.get("stop_nonce"),
                )
                submit_generation_job(_producer)
                last_payload_at = time.monotonic()
                last_progress_pulse_at = last_payload_at
                try:
                    while True:
                        try:
                            if SSE_KEEPALIVE_SECONDS > 0:
                                item = await asyncio.to_thread(
                                    out_q.get, True, SSE_KEEPALIVE_SECONDS
                                )
                            else:
                                item = await asyncio.to_thread(out_q.get)
                        except queue.Empty:
                            now = time.monotonic()
                            progress_delta = {}
                            if (
                                tools
                                and TOOL_STREAM_BUFFER_ALL
                                and TOOL_STREAM_PROGRESS_SECONDS > 0
                                and now - last_payload_at
                                >= TOOL_STREAM_PROGRESS_SECONDS
                                and now - last_progress_pulse_at
                                >= TOOL_STREAM_PROGRESS_SECONDS
                            ):
                                progress_delta["reasoning_content"] = (
                                    "\n[Tool action is still processing.]\n"
                                )
                                last_progress_pulse_at = now
                            # A comment is the SSE-standard transport
                            # heartbeat. Keep the empty OpenAI delta below as
                            # well because a few agent clients only reset
                            # their idle timer after parsing a data event.
                            yield _sse_keepalive_comment()
                            keepalive = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": response_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": progress_delta,
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(keepalive)}\n\n"
                            continue
                        if item is None:
                            break
                        last_payload_at = time.monotonic()
                        last_progress_pulse_at = last_payload_at
                        yield item
                except (asyncio.CancelledError, GeneratorExit):
                    client_connected.clear()
                    updates = {"client_connected": False}
                    # Client disconnects trigger the SAFE coordinated stop (file-based,
                    # both ranks halt at an agreed future token boundary — no
                    # collectives, no race). The old additional UNSAFE_INFLIGHT_STOP
                    # gate was historical over-caution that made every client's stop
                    # button a no-op; the only working stop was killing the cluster.
                    if STOP_ON_CLIENT_DISCONNECT and SAFE_DECODE_STOP:
                        stop_state = _request_inflight_stop(
                            "client_disconnect",
                            dict(active) if active else None,
                            request_id=req_id,
                            stop_nonce=rank_request.get("stop_nonce"),
                        )
                        updates["cancel_requested"] = True
                        updates["cancel_reason"] = "client_disconnect"
                        updates.update(stop_state)
                        logger.warning(
                            "[rank 0] stream client disconnected; distributed "
                            "stop requested for %s at next token boundary",
                            req_id,
                        )
                    update_generation_slot(active, **updates)
                    logger.warning(
                        "[rank 0] stream client disconnected for request %s "
                        "(stop_on_disconnect=%s, inflight_stop=%s)",
                        req_id, STOP_ON_CLIENT_DISCONNECT, UNSAFE_INFLIGHT_STOP,
                    )
                    raise

            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
                headers=SSE_STREAM_HEADERS,
            )

        # Non-streaming path
        req_id = nonstream_req_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"
        try:
            active = await acquire_generation_slot(
                req_id, stream=False, max_tokens=max_tokens,
                image_count=len(images), request_shape=request_shape,
                stop_nonce=rank_request.get("stop_nonce"),
            )
            done_event = threading.Event()
            result = {}

            def _job():
                job_error = None
                rank_request_broadcast = False
                op_channel_held = False
                try:
                    # Own the rank-0 op channel for the WHOLE request
                    # transaction: keepwarm bcast, request bcast, every
                    # decode collective, and the mirror barrier in the
                    # finally below. generation_lock is already held by this
                    # request (acquired in acquire_generation_slot), so lock
                    # order holds. RLock: same-thread re-entry by
                    # request_start_keepwarm is fine.
                    _RANK0_OP_MUTEX.acquire()
                    op_channel_held = True
                    _clear_stop_request()
                    _clear_prefill_stop_file("rank 0 non-stream generation start")
                    _STOP_NONCE["value"] = rank_request.get("stop_nonce")
                    _FORCE_EOS["active"] = False
                    request_start_keepwarm(req_id)
                    _bcast(rank_request, rank)
                    rank_request_broadcast = True
                    def _progress(tokens, chars, metrics=None):
                        metrics = metrics or {}
                        update_generation_slot(
                            active,
                            tokens_emitted=int(tokens),
                            chars_raw=int(chars),
                            last_progress_s=round(time.time() - active["started"], 3),
                            prompt_tps=float(metrics.get("prompt_tps") or active.get("prompt_tps") or 0.0),
                            prompt_tokens=int(metrics.get("prompt_tokens") or active.get("prompt_tokens") or 0),
                            cached_tokens=int(metrics.get("cached_tokens") or active.get("cached_tokens") or 0),
                        )
                    def _prefill_progress(processed_tokens, total_tokens):
                        total_tokens = int(total_tokens or 0)
                        processed_tokens = int(processed_tokens or 0)
                        progress = (
                            processed_tokens / total_tokens
                            if total_tokens > 0 else 0.0
                        )
                        _watchdog_tick(progress=True)
                        update_generation_slot(
                            active,
                            prefill_processed_tokens=processed_tokens,
                            prefill_total_tokens=total_tokens,
                            prefill_progress=round(progress, 4),
                            prefill_last_progress_s=round(
                                time.time() - active["started"], 3
                            ),
                            last_progress_s=round(
                                time.time() - active["started"], 3
                            ),
                        )
                    result["text"] = run_generation(
                        model, processor, prompt, max_tokens, rank,
                        image=image_path, thinking_mode=thinking_mode,
                        gen_params=gen_params, progress_cb=_progress,
                        token_ids=request_token_ids,
                        session_id=cache_session_id,
                        session_source=cache_session_source,
                        reset_incomplete_thinking_on_limit=not bool(tools),
                        prefill_progress_cb=_prefill_progress,
                        tool_module=tool_module,
                        tools=tools,
                        require_tool_call=require_tool_call,
                        action_tool_task=action_tool_task,
                    )
                    if tools and not _user_stop_requested(req_id):
                        result["text"] = _ensure_usable_tool_turn(
                            model, processor, rank,
                            full_output=result["text"],
                            rank_request=rank_request,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            thinking_mode=thinking_mode,
                            gen_params=gen_params,
                            image_path=image_path,
                            token_ids=request_token_ids,
                            session_id=cache_session_id,
                            session_source=cache_session_source,
                            tool_module=tool_module,
                            tools=tools,
                            processed_messages=processed_messages,
                            req_id=req_id,
                            stream=False,
                            should_abort=lambda: _user_stop_requested(req_id),
                            progress_cb=_progress,
                            action_tool_task=action_tool_task,
                        )
                except Exception as e:
                    job_error = e
                    result["error"] = e
                finally:
                    if rank_request_broadcast:
                        try:
                            if _user_stop_requested(req_id):
                                # Match the streaming path: never carry an
                                # unproven rank-local KV prefix across a
                                # cancelled distributed transaction.
                                _reset_prompt_cache_on_all_ranks(
                                    rank,
                                    "reset after cancelled non-stream request",
                                    clear_memory=False,
                                    clear_manifest=False,
                                    clear_resident=False,
                                )
                            update_generation_slot(
                                active,
                                mirror_sync_started_s=round(
                                    time.time() - active["started"], 3
                                ),
                                last_progress_s=round(
                                    time.time() - active["started"], 3
                                ),
                            )
                            logger.info(
                                "[rank 0] non-stream %s waiting for rank 1 mirror barrier",
                                req_id,
                            )
                            _bcast(
                                {
                                    "op": "generation_barrier",
                                    "request_id": req_id,
                                    "stream": False,
                                },
                                rank,
                            )
                            update_generation_slot(
                                active,
                                mirror_sync_done_s=round(
                                    time.time() - active["started"], 3
                                ),
                                last_progress_s=round(
                                    time.time() - active["started"], 3
                                ),
                            )
                            logger.info(
                                "[rank 0] non-stream %s rank 1 mirror barrier complete",
                                req_id,
                            )
                        except Exception as e:
                            job_error = job_error or e
                            result.setdefault("error", e)
                            logger.error(
                                "[rank 0] non-stream %s mirror barrier failed: %s",
                                req_id,
                                e,
                            )
                    # Transaction over (the mirror barrier above was its
                    # last collective) — release the op channel before the
                    # slot releases.
                    if op_channel_held:
                        _RANK0_OP_MUTEX.release()
                    if CLEAR_CACHE_AFTER_REQUEST or (
                        job_error is not None and CLEAR_CACHE_AFTER_ERROR
                    ):
                        _clear_transient_mlx_memory(
                            f"rank 0 non-stream {req_id} complete"
                        )
                    else:
                        gc.collect()
                    release_generation_slot(req_id, active, job_error)
                    done_event.set()

            submit_generation_job(_job)
            # Non-stream orphan guard (2026-07-10): a client that times out
            # or closes mid-generation used to leave the turn decoding into
            # the void for up to max_tokens (an 18-minute orphan held the
            # exclusive slot while the client's retries queued behind it).
            # Poll the connection while the job runs and arm the SAME safe
            # coordinated stop the stream path uses on disconnect.
            disconnect_seen = False
            disconnect_stopped = False
            while not await asyncio.to_thread(done_event.wait, 1.0):
                if (
                    not disconnect_seen
                    and http_request is not None
                    and STOP_ON_CLIENT_DISCONNECT
                    and SAFE_DECODE_STOP
                    and await http_request.is_disconnected()
                ):
                    disconnect_seen = True
                    connected = nonstream_coalescer.disconnect(
                        coalesce_entry, req_id
                    )
                    update_generation_slot(
                        active,
                        client_connected=connected > 0,
                        coalesced_connected_clients=connected,
                    )
                    logger.warning(
                        "[rank 0] non-stream owner client disconnected for %s; "
                        "waiting %.1fs for an exact retry "
                        "(connected_clients=%s)",
                        req_id,
                        NONSTREAM_DISCONNECT_GRACE_SECONDS,
                        connected,
                    )
                if (
                    disconnect_seen
                    and not disconnect_stopped
                    and nonstream_coalescer.should_cancel(coalesce_entry)
                ):
                    disconnect_stopped = True
                    stop_state = _request_inflight_stop(
                        "client_disconnect",
                        dict(active) if active else None,
                        request_id=req_id,
                        stop_nonce=rank_request.get("stop_nonce"),
                    )
                    update_generation_slot(
                        active, client_connected=False,
                        cancel_requested=True,
                        cancel_reason="client_disconnect", **stop_state,
                    )
                    logger.warning(
                        "[rank 0] all clients left non-stream request; distributed "
                        "stop requested for %s at next token boundary",
                        req_id,
                    )
            if "error" in result:
                raise result["error"]
            text = result.get("text")
            logger.info(f"[rank 0] generation complete: {len(text or '')} chars")
            tool_calls, remaining_text = _parse_tool_calls(text or "", tool_module, tools)
            dropped_invalid_tool_calls = 0
            dropped_invalid_tool_names = []
            if loop_force_final and tool_calls:
                logger.warning(
                    "[rank 0] non-stream %s suppressed a repeated tool call "
                    "on a forced-final loop turn",
                    req_id,
                )
                tool_calls = []
                text = _strip_raw_tool_blocks(
                    remaining_text or text or "",
                    tool_module,
                ).strip()
                if not text:
                    text = _tool_loop_forced_final_fallback(tool_loop_diag)
            if tool_calls:
                (
                    tool_calls,
                    dropped_invalid_tool_calls,
                    dropped_invalid_tool_names,
                ) = _validate_outgoing_tool_calls(
                    tool_calls,
                    tools,
                    return_dropped=True,
                    return_dropped_names=True,
                    processed_messages=processed_messages,
                    raw_output=text,
                )
            if (
                TOOL_COMPAT_OVERLAY
                and not tool_calls
                and dropped_invalid_tool_calls
            ):
                synthesized = _synthesize_write_command_tool_call(
                    processed_messages,
                    tools,
                    dropped_invalid_tool_names,
                )
                if synthesized:
                    logger.warning(
                        "[rank 0] non-stream %s converted malformed tool call "
                        "%s into %s for simple write request",
                        req_id,
                        dropped_invalid_tool_names,
                        synthesized["function"]["name"],
                    )
                    tool_calls = [synthesized]
                    dropped_invalid_tool_calls = 0
                    dropped_invalid_tool_names = []
            if tool_calls:
                logger.info(
                    "[rank 0] non-stream %s returning tool_calls=%s arg_keys=%s",
                    req_id,
                    _tool_call_names(tool_calls),
                    _tool_call_arg_keys(tool_calls),
                )
                _remember_assistant_reasoning(
                    cache_session_id,
                    "",
                    text or "",
                    thinking_mode=thinking_mode,
                    session_source=cache_session_source,
                    tool_calls=tool_calls,
                )
                text = remaining_text
            elif tool_module:
                text = _strip_raw_tool_blocks(
                    remaining_text or text or "",
                    tool_module,
                )
            raw_tool_marker_only = bool(
                tools
                and not tool_calls
                and (
                    _looks_like_raw_tool_fragment(result.get("text") or "", tool_module)
                    or "[Tool call:" in (result.get("text") or "")
                )
                and not _strip_thinking_control_markers(text or "").strip()
            )
            # Split reasoning/content for the non-streaming response too
            reasoning, content = split_thinking_text(
                text or "",
                assume_in_thinking=_enable_thinking_for_generation(thinking_mode),
            )
            if loop_force_final and not content:
                content = _tool_loop_forced_final_fallback(tool_loop_diag)
            if tools and content and _looks_like_leaked_reasoning_content(content):
                content = None
            emit_reasoning_fields = _should_emit_reasoning_fields(tools)
            if (
                TOOL_COMPAT_OVERLAY
                and tools
                and not tool_calls
                and (dropped_invalid_tool_calls or not content)
            ):
                if dropped_invalid_tool_calls:
                    content = _tool_request_fallback_content(
                        processed_messages,
                        dropped_tool_names=dropped_invalid_tool_names,
                        available_tool_names=_tool_names_from_schema(tools),
                    )
                else:
                    fallback_candidate = _strip_raw_tool_blocks(
                        _strip_thinking_control_markers(remaining_text or text or ""),
                        tool_module,
                    ).strip()
                    if (
                        not fallback_candidate
                        or _looks_like_leaked_reasoning_content(fallback_candidate)
                        or len(fallback_candidate) < 24
                    ):
                        content = _tool_request_fallback_content(
                            processed_messages,
                            empty_tool_markers=raw_tool_marker_only,
                            thinking_mode=thinking_mode,
                        )
                    else:
                        content = fallback_candidate
                    if _looks_like_leaked_reasoning_content(content):
                        content = _tool_request_fallback_content(
                            processed_messages,
                            empty_tool_markers=raw_tool_marker_only,
                            thinking_mode=thinking_mode,
                        )
                logger.warning(
                    "[rank 0] non-stream tool request %s produced no valid "
                    "tool_calls or visible content "
                    "(dropped_invalid_tool_calls=%s); sent compatibility fallback and "
                    "kept RAM prefix cache",
                    req_id,
                    dropped_invalid_tool_calls,
                )
            if content and "]<]minimax[>[" in content:
                # Raw MiniMax markup must never reach a client, including on
                # tool-less requests where no tool parser/stripper is active.
                content = content[:content.find("]<]minimax[>[")].strip()
            if content:
                content = _scrub_goal_state_echo(content)
            message = {"role": "assistant", "content": content or ""}
            if reasoning and emit_reasoning_fields:
                message["reasoning_content"] = reasoning
            if tool_calls:
                message["tool_calls"] = tool_calls
                message["content"] = None
            response_payload = {"id": req_id,
                "object": "chat.completion", "created": int(time.time()),
                "choices": [
                {"index": 0, "message": message,
                 "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "model": response_model}
            if _user_stop_requested(req_id):
                cancelled_payload = {"error": {
                    "message": "Original request was cancelled",
                    "type": "request_cancelled",
                }}
                nonstream_coalescer.complete(
                    coalesce_entry,
                    cancelled_payload,
                    status_code=499,
                )
                return JSONResponse(
                    status_code=499,
                    content=cancelled_payload,
                )
            nonstream_coalescer.complete(
                coalesce_entry,
                response_payload,
                status_code=200,
            )
            return response_payload
        except Exception as e:
            logger.error(f"[rank 0] generation FAILED, releasing memory: {e}")
            mx.clear_cache()
            gc.collect()
            error_payload = {
                "error": {
                    "message": f"Generation failed: {e}",
                    "type": "generation_error",
                }
            }
            nonstream_coalescer.complete(
                coalesce_entry,
                error_payload,
                status_code=500,
            )
            return JSONResponse(
                status_code=500,
                content=error_payload,
            )

    logger.info(f"Rank 0 serving OpenAI API on {HOST}:{PORT}")
    # ORPHAN FIX (2026-07-07): uvicorn replaces our SIGTERM handler on the
    # main thread, and its graceful drain can stall forever behind a hung
    # request — sweeps then escalate to SIGKILL, stranding ~150GB of wired
    # Metal memory (three orphans on 2026-07-06, three reboots). Two-part
    # closure: atexit runs the Metal clear on EVERY python-level exit path
    # (incl. uvicorn's own TERM shutdown), and a bounded graceful timeout
    # keeps TERM from stalling into the KILL escalation window.
    import atexit
    atexit.register(_clear_mlx_memory, "atexit")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info",
                timeout_graceful_shutdown=5)


def run_mirror(model, processor, rank):
    """Rank 1: mirror rank 0's generation in lockstep. Robust: on error,
    release memory, and keep looping for the next request (don't die)."""
    logger.info("rank 1: mirror loop (waiting for requests)")
    while True:
        req = _bcast(None, rank)  # blocks until rank 0 broadcasts
        if req is None:
            if RANK1_IDLE_SLEEP_SECONDS:
                time.sleep(RANK1_IDLE_SLEEP_SECONDS)
            continue
        if isinstance(req, dict) and req.get("shutdown"):
            logger.info("rank 1: shutdown sentinel received")
            _clear_mlx_memory("rank 1 shutdown sentinel")
            return
        if isinstance(req, dict) and req.get("op") == "reset_prompt_cache":
            reason = req.get("reason") or "admin reset"
            clear_memory = bool(req.get("clear_memory", True))
            clear_manifest = bool(req.get("clear_manifest", False))
            clear_resident = bool(req.get("clear_resident", True))
            logger.info(
                "rank 1: prompt-cache reset sentinel received (%s, clear_memory=%s, "
                "clear_manifest=%s, clear_resident=%s)",
                reason,
                clear_memory,
                clear_manifest,
                clear_resident,
            )
            if clear_memory:
                _reset_prompt_cache_and_clear_memory(reason, clear_manifest=clear_manifest,
                                                     clear_resident=clear_resident)
            else:
                _reset_prompt_cache(reason, clear_manifest=clear_manifest,
                                    clear_resident=clear_resident)
            continue
        if isinstance(req, dict) and req.get("op") == "prompt_cache_ssd_prune":
            reason = req.get("reason") or "rank 0 prune"
            logger.info("rank 1: prompt-cache SSD prune received (%s)", reason)
            with _prompt_cache_lock:
                _prompt_cache_ssd_prune_unlocked(reason=reason)
            continue
        if isinstance(req, dict) and req.get("op") == "prompt_cache_ssd_clear":
            reason = req.get("reason") or "rank 0 clear"
            logger.info("rank 1: prompt-cache SSD clear received (%s)", reason)
            with _prompt_cache_lock:
                _prompt_cache_ssd_clear_unlocked(reason=reason)
            continue
        if isinstance(req, dict) and req.get("op") == "prompt_cache_ssd_save":
            reason = req.get("reason") or "rank 0 save"
            logger.info("rank 1: prompt-cache SSD save received (%s)", reason)
            with _prompt_cache_lock:
                _prompt_cache_make_ssd_checkpoint_unlocked(reason=reason)
                _prompt_cache_ssd_save_current_unlocked(
                    model,
                    processor,
                    reason=reason,
                )
            continue
        if isinstance(req, dict) and req.get("op") == "runtime_tuning":
            values = req.get("values") or {}
            try:
                changed = _set_runtime_tuning(values)
                model_tuning = _apply_runtime_model_tuning(model)
                logger.info(
                    "rank 1: runtime tuning updated: %s model_tuning=%s",
                    changed,
                    model_tuning,
                )
            except Exception as e:
                logger.error("rank 1: runtime tuning update failed: %s", e)
            continue
        if isinstance(req, dict) and req.get("op") == "metal_warmup":
            event = _metal_warmup_touch(
                size=req.get("matrix_size") or 128,
                repeats=req.get("repeats") or 2,
                reason=req.get("reason") or "rank 0 request",
            )
            logger.info("rank 1: metal warmup event: %s", event)
            continue
        if isinstance(req, dict) and req.get("op") == "prewarm_prompt_cache":
            reason = req.get("reason") or "rank 0 prewarm"
            if req.get("visible_source"):
                reason = f"{reason}:{req.get('visible_source')}"
            if req.get("thinking_mode"):
                reason = f"{reason}:{req.get('thinking_mode')}"
            logger.info("rank 1: prompt-cache prewarm received (%s)", reason)
            _prewarm_prompt_cache(
                model,
                processor,
                req.get("prompt") or "",
                req.get("token_ids"),
                reason=reason,
                session_id=req.get("session_id"),
                session_source=req.get("session_source"),
            )
            if CLEAR_CACHE_AFTER_REQUEST:
                _clear_transient_mlx_memory("rank 1 prompt-cache prewarm")
            else:
                gc.collect()
            continue
        if isinstance(req, dict) and req.get("op") == "generation_barrier":
            logger.info(
                "rank 1: mirror barrier received for %s (stream=%s)",
                req.get("request_id"),
                bool(req.get("stream")),
            )
            continue
        # Rank 0 clears its local stop flag before broadcasting each new
        # request. Rank 1 must clear its flag too; otherwise a previous
        # distributed cancel can be observed by the next prefill/decode stop
        # check and produce an empty completion.
        _clear_stop_request()
        _clear_prefill_stop_file("rank 1 new request")
        _STOP_NONCE["value"] = req.get("stop_nonce")
        _FORCE_EOS["active"] = False
        import m3_eagle3 as _m3e3
        _m3e3.REQUEST_ACTIVE["value"] = bool(req.get("eagle3"))

        thinking_mode = req.get("thinking_mode", "adaptive")
        mirror_tools = req.get("tools")
        mirror_tool_module = _load_tool_parser(processor) if mirror_tools else None
        logger.info(f"rank 1: mirroring generation ({req['max_tokens']} tokens)")
        image_path = None
        if req.get("image_sources"):
            try:
                image_path = [_materialize_image(src) for src in req["image_sources"]]
                image_path = [p for p in image_path if p]
                if len(image_path) == 1:
                    image_path = image_path[0]
            except Exception as e:
                logger.warning(f"rank 1: image load failed: {e}")
        try:
            run_generation(model, processor, req["prompt"], req["max_tokens"], rank,
                           image=image_path, thinking_mode=thinking_mode,
                           gen_params=req.get("gen_params"),
                           token_ids=req.get("token_ids"),
                           session_id=req.get("session_id"),
                           session_source=req.get("session_source"),
                           reset_incomplete_thinking_on_limit=bool(
                               req.get("reset_incomplete_thinking_on_limit", True)
                           ),
                           tool_module=mirror_tool_module,
                           tools=mirror_tools,
                           require_tool_call=bool(
                               req.get("require_tool_call", False)
                           ),
                           action_tool_task=bool(
                               req.get("action_tool_task", False)
                           ),
                           no_call_token_budget=req.get(
                               "no_call_token_budget"
                           ))
            if CLEAR_CACHE_AFTER_REQUEST:
                _clear_transient_mlx_memory("rank 1 request complete")
            else:
                gc.collect()
            logger.info(
                "rank 1: mirror done, mlx_cache_cleared=%s",
                CLEAR_CACHE_AFTER_REQUEST,
            )
        except Exception as e:
            logger.error(f"rank 1: generation error, releasing memory: {e}")
            if CLEAR_CACHE_AFTER_ERROR:
                _clear_transient_mlx_memory("rank 1 request error")
            else:
                gc.collect()
            logger.info("rank 1: recovered from error, waiting for next request")
            # CRITICAL: do NOT exit — keep the loop alive so the cluster
            # survives errors instead of dying and orphaning memory.


# ---- distributed object broadcast (modeled on mlx_lm/server.py) ----
def _bcast(obj, rank):
    if rank == 0:
        if obj is None:
            s = mx.distributed.all_sum(mx.array(0, dtype=mx.int32))
            mx.eval(s)
            return None
        data = mx.array(pickle.dumps(obj), dtype=mx.uint8)
        sz = mx.distributed.all_sum(mx.array(int(data.size), dtype=mx.int32))
        mx.eval(sz)
        sm = mx.distributed.all_sum(data)
        mx.eval(sm)
        return obj
    else:
        s = mx.distributed.all_sum(mx.array(0, dtype=mx.int32))
        mx.eval(s)
        size = int(s.item())
        if size == 0:
            return None
        buf = mx.distributed.all_sum(mx.zeros(int(size), dtype=mx.uint8))
        mx.eval(buf)
        return pickle.loads(bytes(buf.tolist()))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        _clear_mlx_memory("process exit")
        sys.exit(0)
    except Exception:
        logger.error(f"fatal server error:\n{traceback.format_exc()}")
        _clear_mlx_memory("fatal exception")
        raise
