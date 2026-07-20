#!/usr/bin/env python3
"""Offline tests for the antidoom repetition detector (2026-07-20 hardening).

Covers every confirmed family from the 13-agent audit: the streamed
paragraph-loop blind spot, CJK blindness, and the benign false-fire shapes
(base64 runs, zero hashes, numeric fillers, markup walls).

  ~/mlx-vlm064-env/bin/python3.14 ops/fable_lab/test_antidoom_detector.py
"""
import os
import sys

_LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LAB)
os.chdir(_LAB)

import sharded_server as srv  # noqa: E402

det = srv._looks_like_degenerate_repetition
fails = 0


def check(name, cond):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    fails += (not cond)


PARA = (
    "Wait, I see at line 367 the handler returns early. Hmm, I don't see "
    "this branch taken in the trace. Let me re-check the call site again "
    "because the offsets look wrong. Wait, I see at line 367 something odd "
    "about the buffer length and the loop condition seems inverted here. "
    "The retry path then repeats the same claim once more. "
)
PARA = (PARA * 2)[:300]
assert len(PARA) == 300

# --- must DETECT (true loops) ---
check("paragraph x5 (motivating incident)", det(PARA * 5))
check("paragraph x30 through a 6000-char streaming tail",
      det((PARA * 30)[-6000:]))
check("shell command x10", det("rm -rf /tmp/probe && ls -la /tmp\n" * 10))
check("word loop x16 (tiny period)", det("no " * 16))
check("marker spam", det("]<]minimax[>[" * 12))
check("CJK paragraph x10", det("这个函数返回了错误的结果需要重新检查一遍代码逻辑。" * 10))
check("ab spiral x40 (above tiny-band floor)", det("ab" * 40))
check("ab x20 stays below the raised tiny-band floor (designed slack)",
      not det("ab" * 20))
check("120-char unit x10 (short band ceiling)",
      det(("x" * 60 + "the quick brown fox jumps over the lazy dog now " + "y" * 12) * 10))
check("1200-char unit x5 (long band ceiling, 6000 tail)",
      det((PARA * 4) * 5))

# --- must IGNORE (benign shapes; audit-confirmed false-fire families) ---
check("base64 A-run 200 chars", not det("A" * 200))
check("zero hash 0x000...", not det("0x" + "0" * 62))
check("zero-vector array filler", not det("0, " * 100))
check("float array filler", not det("1.0, " * 60))
check("punctuation ruler", not det("=" * 200))
check("dash ruler", not det("- " * 100))
check("<br> wall x12 (below tiny-band 16)", not det("<br>" * 12))
check("numbered list (varying lines)",
      not det("".join(f"{i}. item number {i} in the list\n" for i in range(1, 30))))
check("markdown table (varying rows)",
      not det("".join(f"| row {i} | value {i * 3} | ok |\n" for i in range(1, 25))))
check("real prose tail", not det(PARA + " Then the analysis continues with "
                                 "a different conclusion about the buffer."))
check("empty", not det(""))

# --- regression documentation: the OLD streaming bug shape ---
check("600-char tail cannot hold the paragraph loop (documents the old bug)",
      not det((PARA * 30)[-600:]))

# --- streaming-cap parity: what the live stream loop retains must detect ---
tail = ""
for tok in [PARA[i:i + 5] for i in range(0, 300, 5)] * 30:
    tail = (tail + tok)[-6000:]
check("rolling 6000-char stream buffer detects the paragraph loop", det(tail))

print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
