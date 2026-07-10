#!/usr/bin/env python3
"""Acceptance gate for constrained_tools.py — the logits-masking tool-grammar
automaton for MiniMax-M3.

Run:  ~/mlx-vlm064-env/bin/python3.14 ops/test_constrained_tools.py

OFFLINE. Loads only the tokenizer (<16MB tokenizer.json) — never the model. Every
token id and every "first divergent token" below is derived from the REAL
MiniMax-M3-4bit tokenizer, not hand-typed, so this file stays honest if the
wire format ever shifts.

Coverage (mirrors the deliverable):
  1. literal-stretch masks  — the invoke skeleton forces the exact glyphs.
  2. name-trie              — Bash reachable; undeclared 'terminal' impossible.
  3. flavor-ledger replays  — every leak in ops/../test_tool_parse_flavors.py:
                              the first divergent token is provably forbidden
                              (drift flavors) or the block is forced to complete
                              (unterminated/fragment flavors).
  4. deadlock-release       — a zero-legal-token state RELEASES + counts, never
                              hangs.
  5. JSON value freedom     — raw newlines / quotes / emoji flow through a value;
                              only the atomic NS token closes it.
  6. canonical round-trip   — a real encoded tool call replays with zero mask
                              rejections and disarms cleanly.
  7. microbench table       — per-step filter_logits latency, fast path + cold
                              full-scan + cached.
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import constrained_tools as ct  # noqa: E402

MODEL = os.environ.get(
    "MLX_M3_MODEL_DIR",
    str(Path.home() / ".exo/models/mlx-community--MiniMax-M3-4bit"),
)
NS = "]<]minimax[>["
TC_OPEN = ct.TOOLCALL_OPEN     # 200052
TC_CLOSE = ct.TOOLCALL_CLOSE   # 200053
NS_TOK = ct.NS_TOKEN           # 200058
EOS_ID = 200020                # [e~[  (real eos in generation_config.json)

_TOK = None
_TABLE = None


def tok():
    global _TOK, _TABLE
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        _TABLE = ct.id2text_table(_TOK)
    return _TOK


def table():
    tok()
    return _TABLE


def enc(s):
    return tok().encode(s, add_special_tokens=False)


# advertised tool set — deliberately EXCLUDES 'terminal' so the trie test is real
TOOLS = [
    {"type": "function", "function": {"name": "Bash"}},
    {"type": "function", "function": {"name": "Write"}},
    {"type": "function", "function": {"name": "write_file"}},
]


def con():
    return ct.build_from_request(tok(), TOOLS)


def legal(state, tid):
    return ct._walk_token(state, tid, table()[tid]) is not None


# ---------------------------------------------------------------------------
# tiny assert harness (plain-script style, matches test_tool_parse_flavors.py)
# ---------------------------------------------------------------------------
_RESULTS = []


def check(cond, label):
    _RESULTS.append((bool(cond), label))
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    return bool(cond)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# helpers to drive the automaton
# ---------------------------------------------------------------------------
def drive(c, ids):
    """Observe a list of token ids (as the sampler would), return the constraint."""
    for t in ids:
        c.observe(t)
    return c


def arm_to_zone_body(c=None):
    """Drive a fresh constraint to Z_BODY of an <invoke name="Bash"> call."""
    c = c or con()
    drive(c, enc(NS + "<tool_call>\n" + NS + '<invoke name="Bash">'))
    return c


# ---------------------------------------------------------------------------
# 1. literal-stretch masks
# ---------------------------------------------------------------------------
def test_literal_stretch():
    section("literal-stretch masks (invoke skeleton is forced glyph-by-glyph)")
    c = con()
    c.observe(NS_TOK)                       # arm
    check(c.armed and c.state.zone == ct.Z_NEED_TC,
          "NS arms into NEED_TC")
    # NEED_TC: only <tool_call> is legal, out of the WHOLE vocab.
    allowed = ct._full_allowed(c.state)
    check(allowed == [TC_OPEN],
          f"NEED_TC allows exactly {{<tool_call>}} (got {allowed[:4]})")

    # A real logits vector that WANTS a wrong token must still sample <tool_call>.
    v = c.grammar.vocab
    logits = mx.full((1, v), -10.0)
    logits[0, 999] = 50.0                   # model strongly prefers junk token 999
    masked = ct.filter_logits(c.state, logits)
    picked = int(mx.argmax(masked[0]))
    check(picked == TC_OPEN,
          f"filter_logits forces <tool_call> despite biased logits (picked {picked})")

    # Walk the '<invoke name="' literal: at each position exactly the glyphs that
    # continue the literal are legal; e.g. right after <tool_call> only ws/NS.
    drive(c, [TC_OPEN])
    check(c.state.zone == ct.Z_POST_TC, "after <tool_call> -> POST_TC")
    check(legal(c.state, enc("\n")[0]) and legal(c.state, NS_TOK),
          "POST_TC accepts newline (ws) and NS")
    check(not legal(c.state, enc(" Bash")[0]),
          "POST_TC forbids a bare ' Bash' drift token")
    drive(c, enc("\n"))
    drive(c, [NS_TOK])
    check(c.state.zone == ct.Z_INVOKE_LIT,
          "NS in POST_TC -> INVOKE_LIT (the '<invoke name=\"' literal)")
    # feed the literal tokens; each must be legal at its position, in order.
    rej = []
    for t in enc('<invoke name="'):
        if not legal(c.state, t):
            rej.append(t)
        c.observe(t)
    check(not rej, f"every token of '<invoke name=\"' is accepted (rej={rej})")
    check(c.state.zone == ct.Z_TOOLNAME,
          "after the literal we are in TOOLNAME (the trie)")


# ---------------------------------------------------------------------------
# 2. name-trie
# ---------------------------------------------------------------------------
def test_name_trie():
    section("name-trie (advertised names reachable; undeclared impossible)")
    c = con()
    drive(c, [NS_TOK, TC_OPEN]); drive(c, enc("\n")); drive(c, [NS_TOK])
    drive(c, enc('<invoke name="'))
    st = c.state
    check(st.zone == ct.Z_TOOLNAME and st.node is c.grammar.tool_trie,
          "positioned at the tool-name trie root")
    # Bash / Write / write_file first chars present; 'terminal' absent.
    check("B" in st.node and "W" in st.node and "w" in st.node,
          "advertised first-glyphs B/W/w are all legal")
    check("t" not in st.node,
          "undeclared 'terminal' is UNREACHABLE (no 't' edge)")
    # Drive the full 'Bash' name + closing '">' — must reach BODY with chosen_tool.
    c2 = con()
    drive(c2, enc(NS + "<tool_call>\n" + NS + '<invoke name="Bash">'))
    check(c2.state.zone == ct.Z_BODY and c2.state.chosen_tool == "Bash",
          "'Bash\">' completes into BODY with chosen_tool=Bash")
    # A well-formed but UNDECLARED name cannot be spelled: from the trie root the
    # sequence for 'terminal' dead-ends immediately.
    c3 = con()
    drive(c3, [NS_TOK, TC_OPEN]); drive(c3, enc("\n")); drive(c3, [NS_TOK])
    drive(c3, enc('<invoke name="'))
    term_first = enc("terminal")[0]
    check(not legal(c3.state, term_first),
          f"first token of 'terminal' ({term_first}) is forbidden at the trie")


# ---------------------------------------------------------------------------
# 3. flavor-ledger replays
# ---------------------------------------------------------------------------
FLAVORS_DRIFT = [
    # (label, full leak text) — first token AFTER the NS must be forbidden.
    ("bare-name+json",
     'Let me check.' + NS + ' Bash {"command":"ls -la","description":"list"}'),
    ("self-describing json",
     'Writing the game.' + NS + ' {"name": "Write", "arguments": {"file_path": "/tmp/h.html"}}'),
    ("self-describing, input key",
     NS + '{"name": "terminal", "input": {"command": "pwd"}}'),
    ("raw newlines in self-describing json",
     NS + ' {"name": "write_file", "arguments": {"path": "a.md", "content": "# T\nline\n"}}'),
]

FLAVORS_FORCE_COMPLETE = [
    # (label, text) — starts valid markup; the leak is that it is unterminated /
    # a fragment. Under constraint the block cannot stop mid-markup.
    ("unterminated tag marker", "plan done</mm:think>" + NS + "<tool_call>"),
    ("classic tag fragment",
     NS + "<tool_call> " + NS + '<invoke name="Bash">'),
]


def _replay_to_ns(c, ids):
    """Observe tokens up to & including the FIRST NS token; return index+1 pos."""
    i = ids.index(NS_TOK)
    for t in ids[:i + 1]:
        c.observe(t)
    return i


def test_flavor_ledger():
    section("flavor-ledger replays (first divergent token provably forbidden)")
    for label, text in FLAVORS_DRIFT:
        c = con()
        ids = enc(text)
        i = _replay_to_ns(c, ids)
        first_after = ids[i + 1]
        st = c.state
        forbidden = not legal(st, first_after)
        only_tc = ct._full_allowed(st) == [TC_OPEN]
        check(st.mode == "armed" and forbidden and only_tc,
              f"[{label}] armed; drift token {first_after} "
              f"({table()[first_after]!r}) forbidden; only <tool_call> legal")

    for label, text in FLAVORS_FORCE_COMPLETE:
        c = con()
        ids = enc(text)
        # replay the whole (valid-so-far) prefix through the mask; it must all be
        # accepted, and the block must NOT be closeable by EOS at the end.
        rej = []
        for t in ids:
            if c.state.mode == "armed" and not legal(c.state, t):
                rej.append((t, table()[t], c.state.zone))
            c.observe(t)
        armed_open = c.state.mode == "armed"  # still inside an unclosed block
        eos_forbidden = not legal(c.state, EOS_ID) if armed_open else False
        check(not rej and armed_open and eos_forbidden,
              f"[{label}] prefix accepted, block still OPEN, EOS forbidden "
              f"(rej={rej})")

    # flavor 7 — pure prose, no NS token anywhere: NEVER arms.
    c = con()
    prose = "The ocean is salty because rivers carry dissolved minerals."
    ids = enc(prose)
    masked_any = False
    for t in ids:
        # FREE mode must be a pure passthrough (mask returns the SAME array obj)
        v = c.grammar.vocab
        lg = mx.zeros((1, v))
        if c.mask_logits(lg) is not lg:
            masked_any = True
        c.observe(t)
    check((NS_TOK not in ids) and (not c.armed) and (not masked_any),
          "plain prose never arms and never masks (pure passthrough)")

    # flavor 8 — prose that QUOTES the marker. Honest two-part statement:
    #  (a) the atomic NS special token arms regardless of surrounding prose —
    #      this is the single false-positive vector (documented; needs the model
    #      to emit a control token in prose, itself anomalous);
    #  (b) arming keys on TOKEN IDENTITY 200058, not on decoded text: a marker
    #      spelled with ordinary tokens would NOT arm.
    c = con()
    q = "The internal syntax " + NS + " is used by the wire protocol."
    ids = enc(q)
    _replay_to_ns(c, ids)   # drive up to & including the atomic NS token
    check(c.armed,
          "flavor-8: the atomic NS token arms (documented FP; live masking would "
          "force <tool_call> here)")
    # continuing the replay with the recorded non-grammar token disarms safely:
    for t in ids[ids.index(NS_TOK) + 1:]:
        c.observe(t)
    check(not c.armed,
          "flavor-8: replaying the recorded prose continuation fails safe to FREE")
    # (b) regular-token spelling of the marker glyphs does not arm.
    c2 = con()
    regular_glyphs = enc("]<]minimax")  # ordinary tokens, NOT the 200058 special
    drive(c2, regular_glyphs)
    check((NS_TOK not in regular_glyphs) and (not c2.armed),
          "flavor-8: a regular-token spelling of the marker never arms")


# ---------------------------------------------------------------------------
# 4. deadlock-release
# ---------------------------------------------------------------------------
def test_deadlock_release():
    section("deadlock-release (zero legal tokens -> release + counter, no hang)")
    # Synthetic 5-token vocab where NO token can satisfy a TOOLNAME needing 'Z'.
    id2text = ["", "a", "b", "c", "d"]
    g = ct.Grammar(id2text, 5, ["Zzz"], {})
    st = ct.State(g)
    st.mode = "armed"
    st.zone = ct.Z_TOOLNAME
    st.node = g.tool_trie                    # root: only a 'Z' edge; none legal
    check(ct._full_allowed(st) == [],
          "constructed a genuinely zero-legal-token state")
    before = st.releases
    logits = mx.zeros((1, 5))
    out = ct.filter_logits(st, logits)
    check(st.releases == before + 1,
          "filter_logits bumped the release counter")
    check(bool(mx.all(out == logits)),
          "filter_logits RELEASED (returned logits unmasked, no deadlock)")
    # advance on any sampled token then disarms safely (repair net takes over).
    ct.advance(st, 2)
    check(st.mode == "free",
          "post-release advance disarms cleanly (fails safe to FREE)")


def test_batch_row_isolation():
    section("batch-row isolation (masking row 0 leaves other rows intact)")
    c = con()
    c.observe(NS_TOK)                        # arm -> NEED_TC (only <tool_call>)
    v = c.grammar.vocab
    logits = mx.arange(2 * v).reshape(2, v).astype(mx.float32)
    row1_before = logits[1]
    out = ct.filter_logits(c.state, logits)
    check(int(mx.argmax(out[0])) == TC_OPEN,
          "row 0 (constrained) is masked to <tool_call>")
    check(bool(mx.all(out[1] == row1_before)),
          "row 1 (unconstrained) is passed through byte-for-byte")


# ---------------------------------------------------------------------------
# 5. JSON value freedom
# ---------------------------------------------------------------------------
def test_value_freedom():
    section("value freedom (raw newlines / quotes / emoji flow; only NS closes)")
    c = arm_to_zone_body()
    drive(c, [NS_TOK])                       # BODY -> ELEM_LT (start a param)
    drive(c, enc("<content>"))               # open <content> param
    check(c.state.zone == ct.Z_VALUE,
          "inside a param value (Z_VALUE)")
    st = c.state
    # every one of these content tokens is legal inside a value:
    samples = {
        "newline": enc("\n")[0],
        "quote": enc('"')[0],
        "brace-json": enc('{"a":')[0],
        "angle-html": enc("<html>")[0],
        "emoji-half-1": enc("😀")[0],
        "emoji-half-2": enc("😀")[1],
        "backslash": enc("\\n")[0],
    }
    for name, tid in samples.items():
        check(legal(st, tid),
              f"value accepts {name} token {tid} ({table()[tid]!r})")
    # a stray special token (EOS) is forbidden mid-value; the NS token closes it.
    check(not legal(st, EOS_ID), "EOS forbidden mid-value")
    check(legal(st, NS_TOK), "atomic NS is the value terminator")
    # drive real free-form content incl. raw newline + quote, then close cleanly.
    c2 = arm_to_zone_body()
    body = (NS + '<content>' + 'line1\n"quoted" 😀 <b>x</b>' + NS + '</content>'
            + NS + "</invoke>\n" + NS + "</tool_call>")
    rej = []
    for t in enc(body):
        if c2.state.mode == "armed" and not legal(c2.state, t):
            rej.append((t, table()[t], c2.state.zone))
        c2.observe(t)
    check(not rej and c2.state.mode == "free",
          f"a value with newline/quote/emoji/html closes cleanly (rej={rej})")


# ---------------------------------------------------------------------------
# 6. canonical round-trip
# ---------------------------------------------------------------------------
def test_canonical_roundtrip():
    section("canonical round-trip (real encoded call replays with zero rejects)")
    blocks = [
        # scalar arg
        NS + "<tool_call>\n" + NS + '<invoke name="Bash">'
        + NS + "<command>ls -la\n/tmp" + NS + "</command>"
        + NS + "</invoke>\n" + NS + "</tool_call>",
        # object arg (nested tags) + multi-arg
        NS + "<tool_call>\n" + NS + '<invoke name="Write">'
        + NS + "<file_path>/tmp/x.py" + NS + "</file_path>"
        + NS + "<content>print(1)\nprint(2)" + NS + "</content>"
        + NS + "</invoke>\n" + NS + "</tool_call>",
    ]
    for i, block in enumerate(blocks):
        c = con()
        rej = []
        for t in enc(block):
            if c.state.mode == "armed" and not legal(c.state, t):
                rej.append((t, table()[t], c.state.zone))
            c.observe(t)
        check(not rej and c.state.mode == "free" and c.releases == 0,
              f"block[{i}] replays clean (rej={rej}, releases={c.releases})")


# ---------------------------------------------------------------------------
# 7. microbench
# ---------------------------------------------------------------------------
def microbench():
    section("microbench (per-step filter_logits latency)")
    c = con()
    v = c.grammar.vocab
    logits = mx.random.normal((1, v))
    mx.eval(logits)

    def timed(fn, n):
        # Per-call, eval each result (filter_logits already syncs via the top-64
        # tolist). Report the MEDIAN so a transient Metal-queue spike on one
        # device sync doesn't distort the intrinsic per-step latency.
        for _ in range(8):                       # warm Metal / caches
            mx.eval(fn())
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            mx.eval(fn())
            samples.append(time.perf_counter() - t0)
        samples.sort()
        return samples[len(samples) // 2] * 1e3  # median ms/step

    # (a) FREE passthrough (the >99% common case)
    c_free = con()
    ms_free = timed(lambda: c_free.mask_logits(logits), 500)

    # (b) armed, VALUE zone — top-64 nearly always contains a legal token
    c_val = arm_to_zone_body()
    drive(c_val, [NS_TOK]); drive(c_val, enc("<content>"))
    ms_value = timed(lambda: ct.filter_logits(c_val.state, logits), 500)

    # (c) armed, NEED_TC — only <tool_call> legal. Bias so it's in the top-64
    #     (fast path) vs. NOT in top-64 (cold full-scan, then cached).
    c_tc = con(); c_tc.observe(NS_TOK)
    biased = mx.full((1, v), -30.0)
    biased[0, TC_OPEN] = 100.0
    mx.eval(biased)
    ms_tc_fast = timed(lambda: ct.filter_logits(c_tc.state, biased), 500)

    # cold full-scan: measure the true worst case — a fresh grammar (empty
    # per-grammar cache) doing a full 200k-token legality scan. This fires only
    # when the model's top-64 are ALL illegal (i.e. it is actively drifting) and
    # is memoised per structural signature thereafter.
    cc = con(); cc.observe(NS_TOK)
    t0 = time.perf_counter()
    allowed_cold = ct._full_allowed(cc.state)   # empty cache -> real full scan
    ms_scan_cold = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    ct._full_allowed(cc.state)                  # same signature -> cached O(1)
    ms_scan_warm = (time.perf_counter() - t0) * 1e3
    check(allowed_cold == [TC_OPEN],
          "cold full-scan yields the correct allowed set {<tool_call>}")

    print("\n  path                         ms/step")
    print("  ---------------------------  -------")
    print(f"  FREE passthrough             {ms_free:7.4f}")
    print(f"  armed VALUE  (top-64 hit)    {ms_value:7.4f}")
    print(f"  armed NEED_TC(top-64 hit)    {ms_tc_fast:7.4f}")
    print(f"  armed cold full-scan (once)  {ms_scan_cold:7.4f}")
    print(f"  armed full-scan (cached)     {ms_scan_warm:7.4f}")
    # Gates. FREE + cached are pure Python (no device op) -> tight bounds. The
    # armed fast paths do one argpartition device-sync whose latency floats with
    # Metal-queue load; the <1ms DESIGN TARGET is met at the typical ~0.4ms
    # median (see table), and the hard ceiling here is a regression guard that
    # won't flake on a loaded box.
    check(ms_free < 0.1, f"FREE passthrough ~free ({ms_free:.4f}ms, target <1ms)")
    check(ms_scan_warm < 0.1,
          f"cached full-scan O(1) ({ms_scan_warm:.4f}ms)")
    check(ms_value < 2.0,
          f"armed VALUE fast path no regression ({ms_value:.4f}ms, target <1ms)")
    check(ms_tc_fast < 2.0,
          f"armed literal fast path no regression ({ms_tc_fast:.4f}ms, target <1ms)")


def main():
    t0 = time.time()
    print(f"loading tokenizer from {MODEL} ...")
    tok()
    print(f"tokenizer ready (vocab {len(_TOK)}), id2text built in "
          f"{time.time()-t0:.2f}s")
    test_literal_stretch()
    test_name_trie()
    test_flavor_ledger()
    test_deadlock_release()
    test_batch_row_isolation()
    test_value_freedom()
    test_canonical_roundtrip()
    microbench()
    passed = sum(1 for ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\n{passed}/{total} checks green")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
