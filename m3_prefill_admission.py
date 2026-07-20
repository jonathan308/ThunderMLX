"""Prefill admission guard — pressure-driven memory eviction before big prefills.

Problem (2026-07-19 wedge): a large cold prefill can arrive while the GPU wired
pool is already near the OS ceiling (a long session's KV residency + other
resident caches). MLX cannot allocate the prefill working set, the forward makes
ZERO progress, the watchdog kills the process after the stall timeout, and a
process that dies inside an in-flight Metal allocation ORPHANS its wired memory —
forcing a host reboot.

Fix: before a large prefill, each rank independently estimates the working set
the prefill will add, compares it to its own wired headroom, and — if it would
crowd the ceiling — proactively frees memory via the existing eviction paths
(trim the Metal pool, drop IDLE resident cache slots that are already
SSD-checkpointed). It never blocks and never refuses; worst case it behaves
exactly like today, but in the wedge scenario it frees the headroom in ~1-2s and
the prefill proceeds normally.

SAFETY / RANK-LOCKSTEP: every action here is allocator-LOCAL to one rank (no
collectives, no cross-rank messaging). Eviction only touches IDLE resident cache
slots and the freed-buffer pool — never the live KV of the current request, and
never anything whose loss changes generated tokens. Divergent eviction between
ranks only affects future cache-HIT rates (a perf property), never correctness or
lockstep. This is what makes per-rank-local admission control safe on a
2-rank pipeline.

This module holds the PURE decision logic (unit-testable with injected numbers).
The eviction ACTIONS are supplied by the server via callbacks so this module has
no import-time dependency on the server or MLX.
"""
from __future__ import annotations

import os


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


# Default OFF. Server whitelists the env through launch_cluster.sh.
ENABLED = os.environ.get("MLX_M3_PREFILL_ADMISSION_GUARD", "0").strip() == "1"

# Only guard prefills at least this large — small prompts never approach the
# ceiling and shouldn't pay the memory-query cost.
MIN_PROMPT_TOKENS = _env_int("MLX_M3_ADMISSION_MIN_PROMPT_TOKENS", 8192)

# Use at most this fraction of the wired ceiling as the admission budget; the
# remainder is transient-allocation + macOS headroom. 0.92 of 254GB = ~234GB.
SAFETY_FRACTION = _env_float("MLX_M3_ADMISSION_SAFETY_FRACTION", 0.92)

# Per-token KV cost estimate (bytes). Measured ~87.6 KB/token on rank0 (38 of 60
# layers). Overridable; the server passes the rank-correct value when known.
DEFAULT_KV_BYTES_PER_TOKEN = _env_int(
    "MLX_M3_ADMISSION_KV_BYTES_PER_TOKEN", 90_000
)

# Fixed transient-activation reserve on top of KV (prefill chunk buffers, logits,
# routing). Deliberately generous — over-reserving costs a little extra eviction,
# under-reserving risks the wedge we are preventing.
ACTIVATION_RESERVE_BYTES = _env_int(
    "MLX_M3_ADMISSION_ACTIVATION_RESERVE_BYTES", 8 * 1024 ** 3
)

_GiB = 1024 ** 3


def admission_deficit_bytes(
    prompt_tokens,
    current_wired_bytes,
    wired_limit_bytes,
    kv_bytes_per_token=None,
    safety_fraction=None,
    activation_reserve_bytes=None,
):
    """Bytes that must be freed before this prefill fits, or <=0 if it fits.

    deficit = (current_wired + prompt_tokens*kv_per_token + activation_reserve)
              - wired_limit*safety_fraction
    """
    if kv_bytes_per_token is None:
        kv_bytes_per_token = DEFAULT_KV_BYTES_PER_TOKEN
    if safety_fraction is None:
        safety_fraction = SAFETY_FRACTION
    if activation_reserve_bytes is None:
        activation_reserve_bytes = ACTIVATION_RESERVE_BYTES
    if wired_limit_bytes <= 0:
        return 0  # unknown ceiling -> cannot reason, do nothing (fail-open)
    need = prompt_tokens * kv_bytes_per_token + activation_reserve_bytes
    budget = wired_limit_bytes * safety_fraction
    return (current_wired_bytes + need) - budget


def should_guard(prompt_tokens):
    return ENABLED and prompt_tokens >= MIN_PROMPT_TOKENS


def plan_eviction(deficit_bytes, evictables):
    """Given a byte deficit and a list of evictable items (each a dict with at
    least {'label', 'bytes'}), pick the smallest set of IDLE items whose bytes
    cover the deficit. Largest-first so we free in the fewest evictions.

    Returns (chosen_items, still_short_bytes). still_short>0 means we freed all
    we safely could and the caller should proceed anyway (fail-open) + log loud.
    """
    if deficit_bytes <= 0:
        return [], 0
    chosen = []
    freed = 0
    for item in sorted(evictables, key=lambda e: e.get("bytes", 0), reverse=True):
        if freed >= deficit_bytes:
            break
        if item.get("bytes", 0) <= 0:
            continue
        chosen.append(item)
        freed += item["bytes"]
    return chosen, max(0, deficit_bytes - freed)


def run_admission(
    prompt_tokens,
    read_wired,          # () -> current_wired_bytes
    read_limit,          # () -> wired_limit_bytes
    trim_pool,           # () -> bytes_freed  (mx.clear_cache + gc)
    list_idle_evictables,  # () -> [ {label, bytes, drop: callable}, ... ]
    kv_bytes_per_token=None,
    logger=None,
):
    """Orchestrate a per-rank admission check + least-disruptive eviction ladder.

    Returns a dict describing what happened (for telemetry/logging/tests).
    Pure control flow; all side effects are through the injected callbacks.
    """
    info = {"guarded": False, "prompt_tokens": prompt_tokens}
    if not should_guard(prompt_tokens):
        return info
    info["guarded"] = True
    limit = read_limit() or 0
    before = read_wired() or 0
    deficit = admission_deficit_bytes(
        prompt_tokens, before, limit, kv_bytes_per_token=kv_bytes_per_token
    )
    info.update({"wired_before_gib": round(before / _GiB, 2),
                 "limit_gib": round(limit / _GiB, 2),
                 "deficit_gib": round(deficit / _GiB, 2)})
    if deficit <= 0:
        return info  # fits; do nothing (the common case — zero overhead beyond 2 reads)

    actions = []
    # Rung 1: trim the freed-buffer pool (cheapest; frees nothing live).
    freed_pool = trim_pool() or 0
    actions.append({"step": "trim_pool", "freed_gib": round(freed_pool / _GiB, 2)})
    after_pool = read_wired() or before
    deficit = admission_deficit_bytes(
        prompt_tokens, after_pool, limit, kv_bytes_per_token=kv_bytes_per_token
    )

    # Rung 2: drop IDLE resident cache slots (SSD-checkpointed; cheap restore).
    if deficit > 0:
        evictables = list_idle_evictables() or []
        chosen, still_short = plan_eviction(deficit, evictables)
        for item in chosen:
            try:
                item["drop"]()
                actions.append({"step": "evict_idle",
                                "label": item.get("label"),
                                "freed_gib": round(item.get("bytes", 0) / _GiB, 2)})
            except Exception as e:  # never let eviction failure block the request
                actions.append({"step": "evict_idle_error",
                                "label": item.get("label"), "error": str(e)[:120]})
        after_evict = read_wired() or after_pool
        deficit = admission_deficit_bytes(
            prompt_tokens, after_evict, limit, kv_bytes_per_token=kv_bytes_per_token
        )

    info["actions"] = actions
    info["wired_after_gib"] = round((read_wired() or before) / _GiB, 2)
    info["residual_deficit_gib"] = round(max(0, deficit) / _GiB, 2)
    info["fits"] = deficit <= 0
    if logger is not None:
        if deficit <= 0:
            logger.warning(
                "prefill admission: freed headroom for %d-token prefill "
                "(wired %.1f->%.1f GiB, limit %.1f); actions=%s",
                prompt_tokens, info["wired_before_gib"], info["wired_after_gib"],
                info["limit_gib"], [a.get("step") for a in actions],
            )
        else:
            logger.warning(
                "prefill admission: STILL short %.1f GiB after eviction for a "
                "%d-token prefill (wired %.1f, limit %.1f) — proceeding fail-open; "
                "watchdog remains the backstop",
                info["residual_deficit_gib"], prompt_tokens,
                info["wired_after_gib"], info["limit_gib"],
            )
    return info
