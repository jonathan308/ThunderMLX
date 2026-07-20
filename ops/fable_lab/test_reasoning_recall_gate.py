#!/usr/bin/env python3
"""Offline test: degenerate reasoning must never survive the recall store.

Reproduces the 2026-07-20 opencode echo incident shape end-to-end through the
real _remember_assistant_reasoning / _recall_assistant_reasoning functions:
loop-y reasoning is refused at store time, poisoned pre-existing entries are
dropped at recall time, and healthy reasoning round-trips untouched.

  ~/mlx-vlm064-env/bin/python3.14 ops/fable_lab/test_reasoning_recall_gate.py
"""
import os
import sys

_LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LAB)
os.chdir(_LAB)

import sharded_server as srv  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    fails += (not cond)


LOOP_UNIT = (
    "Let me think about this more carefully. The user wants to add a custom "
    "background feature to the opencode desktop app. This is a real "
    "modification that requires understanding the existing code structure. "
    "Let me try a more targeted approach - fetch specific files from GitHub. "
    "Now I have a good understanding of the settings structure. "
)
LOOPED = LOOP_UNIT * 6
HEALTHY = (
    "The settings store lives in packages/desktop/src/settings.ts and uses a "
    "zod schema. I should add a backgroundImage field there, then thread it "
    "through the ThemeProvider so the chat surface can render it. The upload "
    "path needs a file picker plus a data-URI persistence fallback."
)

SESSION = "gate-test-session"
VISIBLE = "I checked the settings structure."
RAW_LOOPED = LOOPED + "</mm:think>" + VISIBLE
RAW_HEALTHY = HEALTHY + "</mm:think>" + VISIBLE

# --- 1. healthy reasoning round-trips through store + recall ---
ok = srv._remember_assistant_reasoning(
    SESSION, VISIBLE, RAW_HEALTHY, thinking_mode="enabled")
check("healthy reasoning stored", ok is True)
back = srv._recall_assistant_reasoning(SESSION, VISIBLE)
check("healthy reasoning recalled intact", back == HEALTHY)

# --- 2. looped reasoning is refused at STORE time ---
ok = srv._remember_assistant_reasoning(
    SESSION, "second visible", LOOP_UNIT * 6 + "</mm:think>second visible",
    thinking_mode="enabled")
check("copy-spiral reasoning refused at store", ok is False)
back = srv._recall_assistant_reasoning(SESSION, "second visible")
check("nothing recalled for the refused turn", back is None)

# --- 3. poisoned entry planted directly is dropped at RECALL time ---
key = srv._reasoning_recall_key("third visible", None)
skey = srv._prompt_cache_session_key(SESSION)
with srv._reasoning_recall_lock:
    srv._reasoning_recall_sessions[skey][key] = {"reasoning": LOOPED}
back = srv._recall_assistant_reasoning(SESSION, "third visible")
check("pre-existing poisoned entry dropped at recall", back is None)

# --- 4. the gates use the SAME detector as containment (single source) ---
check("detector flags the incident reasoning",
      srv._looks_like_degenerate_repetition(LOOPED))
check("detector passes the healthy reasoning",
      not srv._looks_like_degenerate_repetition(HEALTHY))

print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
