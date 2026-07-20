#!/usr/bin/env python3
"""Offline end-to-end test of admission rung-3 (stale live-cache pressure release).

Imports the REAL server module (no cluster boot) and drives the real
_maybe_run_prefill_admission entry point with a fake stale holder, proving:
  1. rung-3 fires through the full wiring (flag + cold_prefill + deficit gate),
  2. it checkpoints via the autosave hook with reason=admission_pressure_release,
  3. it clears the holder cache + key state,
  4. it stays dormant when any gate says no (flag off / warm prefill / empty
     holder / no deficit).

Natural HTTP traffic cannot reach rung-3 in this tree (the one-slot switch
release and the auto-session rebuild both clear the stale holder earlier, in
_prepare_cached_prompt) — which is exactly why the backstop needs an offline
proof that the path itself is sound.

  ~/mlx-vlm064-env/bin/python3.14 ops/fable_lab/test_admission_rung3_offline.py
"""
import os
import sys

os.environ["MLX_M3_PREFILL_ADMISSION_GUARD"] = "1"
os.environ["MLX_M3_ADMISSION_DROP_STALE_LIVE"] = "1"
# Tiny synthetic ceiling -> any prefill has a positive deficit.
os.environ["MLX_M3_ADMISSION_TEST_LIMIT_GIB"] = "0.001"

_LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LAB)
os.chdir(_LAB)

import sharded_server as srv  # noqa: E402  (env must be set first)

fails = 0


def check(name, cond):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    fails += (not cond)


calls = []


def _fake_autosave(model, processor, prompt=None, reason=None, **kw):
    calls.append(reason)


srv._prompt_cache_ssd_maybe_autosave_unlocked = _fake_autosave


def stage_stale_holder():
    h = srv._prompt_cache_holder
    h["cache"] = object()          # stands in for a live KV tree
    h["cache_len"] = 4321
    h["session_id"] = "stale-session"
    h["token_ids"] = [1, 2, 3]
    h["prompt"] = "stale prompt"
    return h


fake_model, fake_proc = object(), object()

# --- 1. rung-3 fires end-to-end on a cold prefill under deficit ---
h = stage_stale_holder()
srv._maybe_run_prefill_admission(20000, 0, model=fake_model,
                                 processor=fake_proc, cold_prefill=True)
check("rung-3 cleared the stale holder cache", h.get("cache") is None)
check("rung-3 checkpointed first (autosave reason)",
      "admission_pressure_release" in calls)

# --- 2. dormant when the prefill is WARM (cached_prompt_cache existed) ---
calls.clear()
h = stage_stale_holder()
srv._maybe_run_prefill_admission(20000, 0, model=fake_model,
                                 processor=fake_proc, cold_prefill=False)
check("warm prefill leaves the live cache alone", h.get("cache") is not None)
check("warm prefill never checkpoints", "admission_pressure_release" not in calls)

# --- 3. dormant when the opt-in flag is off ---
srv._ADMISSION_DROP_STALE_LIVE = False
h = stage_stale_holder()
srv._maybe_run_prefill_admission(20000, 0, model=fake_model,
                                 processor=fake_proc, cold_prefill=True)
check("flag off -> holder untouched", h.get("cache") is not None)
srv._ADMISSION_DROP_STALE_LIVE = True

# --- 4. empty holder -> no-op, no crash ---
h["cache"] = None
srv._maybe_run_prefill_admission(20000, 0, model=fake_model,
                                 processor=fake_proc, cold_prefill=True)
check("empty holder is a clean no-op", h.get("cache") is None)

# --- 5. no deficit (real 254 ceiling) -> holder untouched ---
os.environ["MLX_M3_ADMISSION_TEST_LIMIT_GIB"] = "254"
h = stage_stale_holder()
srv._maybe_run_prefill_admission(20000, 0, model=fake_model,
                                 processor=fake_proc, cold_prefill=True)
check("no deficit -> stale holder untouched (fires only under pressure)",
      h.get("cache") is not None)

# --- 6. missing model/processor -> dormant (defensive) ---
os.environ["MLX_M3_ADMISSION_TEST_LIMIT_GIB"] = "0.001"
h = stage_stale_holder()
srv._maybe_run_prefill_admission(20000, 0, cold_prefill=True)
check("no model/processor -> dormant", h.get("cache") is not None)

print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
