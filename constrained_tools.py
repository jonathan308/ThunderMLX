#!/usr/bin/env python3
"""constrained_tools.py — logits-masking grammar automaton for MiniMax-M3 tool calls.

WHAT
  A per-request pushdown automaton that makes MiniMax-M3 tool-call markup drift
  IMPOSSIBLE at generation time. It masks the sampler's log-probs on rank 0 so
  that, once the model commits to a tool block, only grammar-valid tokens can be
  sampled. Outside a tool block (prose, thinking) it is a zero-cost passthrough.

THE WIRE FORMAT (discovered, not assumed)
  Verified against BOTH the chat template (chat_template.jinja lines 122-198)
  AND the real parser (mlx_vlm/tool_parsers/minimax_m3.py). The canonical form
  the model is trained on is DIRECT ARG TAGS, namespace-prefixed — NOT
  <parameter name="x">value</parameter> (that form is REJECTED by the parser,
  whose _TAG_NAME_RE fails on the space in `parameter name=...`):

    ]<]minimax[>[<tool_call>\n
    ]<]minimax[>[<invoke name="TOOL">
    ]<]minimax[>[<PARAM>value]<]minimax[>[</PARAM>          (scalar)
    ]<]minimax[>[<PARAM>]<]minimax[>[<item>..]<]minimax[>[</item>]<]minimax[>[</PARAM>  (array/obj)
    ]<]minimax[>[</invoke>\n
    ]<]minimax[>[</tool_call>

  The `]<]minimax[>[` namespace marker is a SINGLE atomic special token
  (id 200058). <tool_call>/</tool_call> are atomic too (200052/200053). Because
  the namespace marker is atomic, value termination is UNAMBIGUOUS without any
  JSON escape/quote tracking: a value ends exactly when the model samples token
  200058. Raw newlines, quotes, and emoji flow through a value as ordinary text.

HOOK (see m3_pipeline_patch.py)
  rank 0 alone samples; rank 1 mirrors the synced token. The constraint runs
  inside the existing rank0 token-sync sampler wrapper, masking `logprobs`
  (already available there) just before the sampler draws. Zero collective
  changes: the same single sampled-token send happens; we only change WHICH
  token can be drawn. rank 1 needs no automaton — it echoes rank 0's (already
  valid) token.

NEVER DEADLOCK
  If a state yields zero legal tokens, the constraint RELEASES for that step
  (returns the logits unmasked) and bumps a counter; the downstream repair net
  catches the rare leak. The automaton never hangs and never raises into decode.

Token-id map (MiniMax-M3-4bit tokenizer, vocab 200064; real tokens 0..200060):
  200058 ]<]minimax[>[    200052 <tool_call>     200053 </tool_call>
  200059 <mm:think>       200060 </mm:think>      200050/200051 <think>/</think>
  10 '\n'   60 '<'   62 '>'   1579 '</'   1925 ' name'   1139 '="'   3361 '">'
"""
from __future__ import annotations

import os
import threading

import mlx.core as mx

# ---------------------------------------------------------------------------
# Discovered token ids (single-token / atomic markers)
# ---------------------------------------------------------------------------
NS_TOKEN = 200058            # ]<]minimax[>[   (atomic namespace marker)
TOOLCALL_OPEN = 200052       # <tool_call>
TOOLCALL_CLOSE = 200053      # </tool_call>
THINK_OPEN = 200059          # <mm:think>
THINK_CLOSE = 200060         # </mm:think>
THINK_OPEN_LEGACY = 200050   # <think>
THINK_CLOSE_LEGACY = 200051  # </think>

# The added-token band. Everything here is a special token; only NS_TOKEN is
# ever legal inside a tool block (it starts every structural marker + the value
# close). Denying the rest inside a block makes EOS/BOS/pad truncation and
# stray specials impossible mid-call.
SPECIAL_LO = 200000
DEFAULT_VOCAB = 200064

# Literal skeleton the model must spell exactly (char-DFA walked, BPE-agnostic).
INVOKE_OPEN_LIT = '<invoke name="'   # after this: tool name, then ">
INVOKE_CLOSE_LIT = "invoke>"          # after a leading "</"

# Tag-name char classes, matching the parser's _TAG_NAME_RE = [A-Za-z_$][\w:.$-]*
_TAG_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$")
_TAG_CONT = _TAG_START | set("0123456789:.-")

_TERM = "\x00"  # trie terminal marker key


def _is_tag_start(ch: str) -> bool:
    return ch in _TAG_START


def _is_tag_cont(ch: str) -> bool:
    return ch in _TAG_CONT


# ---------------------------------------------------------------------------
# id -> text table (tokenizer-global, built once, module-cached)
# ---------------------------------------------------------------------------
def _bytes_to_unicode():
    """The canonical GPT-2 byte<->unicode map used by byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}  # unicode-char -> byte


_BYTE_DECODER = _bytes_to_unicode()

_ID2TEXT_CACHE = {"key": None, "table": None}
_ID2TEXT_LOCK = threading.Lock()


def _resolve_hf_tokenizer(tok):
    """Return an object exposing convert_ids_to_tokens / get_vocab, given either
    a path, an mlx TokenizerWrapper, or a bare HF tokenizer."""
    if isinstance(tok, str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tok, trust_remote_code=True)
    # mlx TokenizerWrapper wraps the HF tokenizer as ._tokenizer
    inner = getattr(tok, "_tokenizer", None)
    if inner is not None and hasattr(inner, "convert_ids_to_tokens"):
        return inner
    return tok


def _build_id2text(tok, vocab_size):
    """id -> exact text contribution, for every id in [0, vocab_size).

    Regular BPE tokens (id < 200000) are byte-decoded from their raw byte-BPE
    string; special/added tokens keep their literal content (e.g. the atomic
    ']<]minimax[>['). Padding ids past the real vocab map to '' (never legal in
    a structural zone). ~55ms once per process."""
    hf = _resolve_hf_tokenizer(tok)
    n_real = None
    for attr in ("vocab_size",):
        v = getattr(hf, attr, None)
        if isinstance(v, int):
            n_real = v
    # Real added tokens live at [SPECIAL_LO, len(hf)).
    try:
        n_tok = len(hf)
    except Exception:
        n_tok = vocab_size

    table = [""] * vocab_size
    if hasattr(hf, "convert_ids_to_tokens"):
        raw = hf.convert_ids_to_tokens(list(range(min(n_tok, vocab_size))))
        for i, s in enumerate(raw):
            if s is None:
                table[i] = ""
            elif i >= SPECIAL_LO:
                table[i] = s  # special/added token: literal content
            else:
                try:
                    table[i] = bytes(_BYTE_DECODER[c] for c in s).decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    table[i] = ""
    else:  # last-resort: per-id decode (slower but correct)
        dec = hf.decode
        for i in range(min(n_tok, vocab_size)):
            try:
                table[i] = dec([i])
            except Exception:
                table[i] = ""
    return table


def id2text_table(tok, vocab_size=DEFAULT_VOCAB):
    """Module-cached id->text table (tokenizer-global; reused across requests)."""
    key = (id(tok), vocab_size)
    c = _ID2TEXT_CACHE
    if c["key"] == key and c["table"] is not None:
        return c["table"]
    with _ID2TEXT_LOCK:
        if c["key"] == key and c["table"] is not None:
            return c["table"]
        table = _build_id2text(tok, vocab_size)
        c["key"] = key
        c["table"] = table
        return table


# ---------------------------------------------------------------------------
# Trie over a set of names (tool names, or a tool's param names)
# ---------------------------------------------------------------------------
def build_trie(names):
    root = {}
    for name in names:
        if not name:
            continue
        node = root
        for ch in name:
            node = node.setdefault(ch, {})
        node[_TERM] = True
    return root


def _tool_param_names(schema):
    """(param_name_list, strict) for a tool's parameter schema.

    strict=True  -> constrain param names to the declared properties (trie).
    strict=False -> permissive tag names (schema absent or additionalProperties
                    allowed), so a legal but undeclared key is never forbidden.
    """
    params = None
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    if isinstance(fn, dict):
        params = fn.get("parameters")
    if not isinstance(params, dict):
        return [], False
    props = params.get("properties")
    additional = params.get("additionalProperties", False)
    names = list(props.keys()) if isinstance(props, dict) else []
    strict = bool(names) and additional is False
    return names, strict


# ---------------------------------------------------------------------------
# Automaton state (a lightweight mutable object; copied per candidate test)
# ---------------------------------------------------------------------------
# Zones. FREE + the atomic-token zones gate on token identity; the rest are
# character-DFA zones walked over each candidate token's decoded text.
Z_FREE = "FREE"
Z_NEED_TC = "NEED_TC"          # atomic: only <tool_call>
Z_POST_TC = "POST_TC"          # atomic: ws* then ]<]minimax[>[
Z_INVOKE_LIT = "INVOKE_LIT"    # char literal '<invoke name="'
Z_TOOLNAME = "TOOLNAME"        # char trie over tool names, then '"'
Z_NAME_GT = "NAME_GT"          # char literal '>'
Z_BODY = "BODY"                # atomic: ws* then ]<]minimax[>[
Z_ELEM_LT = "ELEM_LT"          # char: '<'
Z_ELEM_DEC = "ELEM_DEC"        # char: '/' (invoke close) | tag-start (param)
Z_INVOKE_CLOSE = "INVOKE_CLOSE"  # char literal 'invoke>'
Z_PARAMNAME = "PARAMNAME"      # char trie (strict) over param names, then '>'
Z_PARAMNAME_P = "PARAMNAME_P"  # char permissive tag name, then '>'
Z_VALUE = "VALUE"              # token: regular content | ]<]minimax[>[
Z_VALUE_LT = "VALUE_LT"        # char: '<'
Z_VALUE_DEC = "VALUE_DEC"      # char: '/' (close top) | tag-start (nested open)
Z_VALUE_OPEN = "VALUE_OPEN"    # char permissive nested tag name, then '>'
Z_VALUE_CLOSE = "VALUE_CLOSE"  # char literal '<toptag>' close
Z_AFTER_INVOKE = "AFTER_INVOKE"  # atomic: ws* then ]<]minimax[>[
Z_ELEM2 = "ELEM2"              # atomic </tool_call> | char '<invoke name="'
Z_DONE = "DONE"                # block closed -> disarm to FREE


class State:
    __slots__ = (
        "g", "mode", "zone", "lit", "pos", "node", "partial",
        "chosen_tool", "stack", "in_think", "releases", "steps",
    )

    def __init__(self, g):
        self.g = g               # Grammar (shared, immutable)
        self.mode = "free"       # "free" | "armed"
        self.zone = Z_FREE
        self.lit = ""            # literal being matched
        self.pos = 0             # index into self.lit
        self.node = None         # current trie node
        self.partial = ""        # accumulated name / tag
        self.chosen_tool = None
        self.stack = []          # open value tags (nesting)
        self.in_think = False
        self.releases = 0
        self.steps = 0

    def copy(self):
        s = State.__new__(State)
        s.g = self.g
        s.mode = self.mode
        s.zone = self.zone
        s.lit = self.lit
        s.pos = self.pos
        s.node = self.node
        s.partial = self.partial
        s.chosen_tool = self.chosen_tool
        s.stack = self.stack  # shared until a push/pop; copy-on-write below
        s.in_think = self.in_think
        s.releases = self.releases
        s.steps = self.steps
        return s

    def _own_stack(self):
        # copy-on-write: only clone the list when we are about to mutate it
        self.stack = list(self.stack)


class Grammar:
    """Per-request shared, immutable grammar context."""

    __slots__ = ("id2text", "vocab", "tool_trie", "param_info", "fullscan_cache",
                 "_value_deny_add")

    def __init__(self, id2text, vocab, tool_names, param_info):
        self.id2text = id2text
        self.vocab = vocab
        self.tool_trie = build_trie(tool_names)
        # tool name -> (param_trie, strict)
        self.param_info = param_info
        # full-vocab allowed-set cache, keyed by structural state signature.
        # Scoped to THIS grammar (not a module global) so a recycled trie-node
        # id() from a GC'd prior request can never alias another request's tool
        # set. Freed when the grammar is.
        self.fullscan_cache = {}
        self._value_deny_add = None  # lazy: Z_VALUE additive deny mask

    def value_deny_add(self):
        """Additive logits mask for Z_VALUE (0 = legal, -1e9 = denied).

        Z_VALUE legality is state-independent: every regular token with
        nonempty text passes, NS_TOKEN passes, all other specials and
        empty-text ids are denied. Precomputing it kills the per-step
        argpartition + host sync + 64 DFA walks that made long tool values
        ~2% slower than baseline decode; the mask apply is one device add.
        Built lazily once per grammar (~vocab scan, same cost as one
        fullscan miss)."""
        if self._value_deny_add is None:
            deny = [tid for tid in range(self.vocab)
                    if not (tid == NS_TOKEN
                            or (tid < SPECIAL_LO and self.id2text[tid] != ""))]
            add = mx.zeros((self.vocab,), dtype=mx.float32)
            if deny:
                add[mx.array(deny, dtype=mx.int32)] = _NEG
            self._value_deny_add = add
        return self._value_deny_add


# ---------------------------------------------------------------------------
# Character transition (mutates st; returns False on an invalid char)
# ---------------------------------------------------------------------------
def _step_char(st, ch):
    z = st.zone

    if z == Z_INVOKE_LIT:
        if st.pos < len(st.lit) and ch == st.lit[st.pos]:
            st.pos += 1
            if st.pos == len(st.lit):
                st.zone = Z_TOOLNAME
                st.node = st.g.tool_trie
                st.partial = ""
            return True
        return False

    if z == Z_TOOLNAME:
        node = st.node
        if ch != _TERM and ch in node:
            st.node = node[ch]
            st.partial += ch
            return True
        if ch == '"' and node.get(_TERM):
            st.chosen_tool = st.partial
            st.zone = Z_NAME_GT
            return True
        return False

    if z == Z_NAME_GT:
        if ch == ">":
            st.zone = Z_BODY
            return True
        return False

    if z == Z_ELEM_LT:
        if ch == "<":
            st.zone = Z_ELEM_DEC
            return True
        return False

    if z == Z_ELEM_DEC:
        if ch == "/":
            st.zone = Z_INVOKE_CLOSE
            st.lit = INVOKE_CLOSE_LIT
            st.pos = 0
            return True
        if _is_tag_start(ch):
            trie, strict = st.g.param_info.get(st.chosen_tool, (None, False))
            if strict and trie is not None:
                if ch in trie:
                    st.zone = Z_PARAMNAME
                    st.node = trie[ch]
                    st.partial = ch
                    return True
                return False
            st.zone = Z_PARAMNAME_P
            st.partial = ch
            return True
        return False

    if z == Z_INVOKE_CLOSE:
        if st.pos < len(st.lit) and ch == st.lit[st.pos]:
            st.pos += 1
            if st.pos == len(st.lit):
                st.zone = Z_AFTER_INVOKE
            return True
        return False

    if z == Z_PARAMNAME:
        node = st.node
        if ch != _TERM and ch in node:
            st.node = node[ch]
            st.partial += ch
            return True
        if ch == ">" and node.get(_TERM):
            st._own_stack()
            st.stack.append(st.partial)
            st.zone = Z_VALUE
            return True
        return False

    if z == Z_PARAMNAME_P:
        if _is_tag_cont(ch):
            st.partial += ch
            return True
        if ch == ">" and st.partial:
            st._own_stack()
            st.stack.append(st.partial)
            st.zone = Z_VALUE
            return True
        return False

    if z == Z_VALUE_LT:
        if ch == "<":
            st.zone = Z_VALUE_DEC
            return True
        return False

    if z == Z_VALUE_DEC:
        if ch == "/":
            st.zone = Z_VALUE_CLOSE
            st.lit = (st.stack[-1] if st.stack else "") + ">"
            st.pos = 0
            return True
        if _is_tag_start(ch):
            st.zone = Z_VALUE_OPEN
            st.partial = ch
            return True
        return False

    if z == Z_VALUE_OPEN:
        if _is_tag_cont(ch):
            st.partial += ch
            return True
        if ch == ">" and st.partial:
            st._own_stack()
            st.stack.append(st.partial)
            st.zone = Z_VALUE
            return True
        return False

    if z == Z_VALUE_CLOSE:
        if st.pos < len(st.lit) and ch == st.lit[st.pos]:
            st.pos += 1
            if st.pos == len(st.lit):
                st._own_stack()
                if st.stack:
                    st.stack.pop()
                st.zone = Z_VALUE if st.stack else Z_BODY
            return True
        return False

    # --- destination zones reached MID-TOKEN after a char-literal completes ---
    # A merged BPE token can spill past a literal into a token-level zone. The
    # canonical '>\n' (id 1100) after ]<]minimax[>[</invoke> is exactly this:
    # 'invoke>' finishes Z_INVOKE_CLOSE and the trailing '\n' lands in the
    # atomic Z_AFTER_INVOKE zone. The NS marker itself is always its own atomic
    # token, so the only characters that can legally appear here mid-token are
    # trailing whitespace (Z_BODY / Z_AFTER_INVOKE) or free-form value content
    # (Z_VALUE). Without this, the model's OWN preferred tokenization of valid
    # markup would be forbidden — a BPE-boundary correctness bug.
    if z == Z_VALUE:
        return True  # value content is free until the atomic NS token
    if z in (Z_BODY, Z_AFTER_INVOKE, Z_POST_TC):
        return bool(ch) and ch.isspace()

    return False


def _char_walk(st, text):
    """Fold _step_char over text; return st (mutated copy) or None."""
    if not text:
        return None
    for ch in text:
        if not _step_char(st, ch):
            return None
    return st


# ---------------------------------------------------------------------------
# Token transition: is `tid` (text) legal from st? Return the NEW state or None.
# `st` is copied first, so the caller's state is never mutated.
# ---------------------------------------------------------------------------
def _is_ws(text):
    return bool(text) and text.strip() == ""


def _walk_token(state, tid, text):
    st = state.copy()
    z = st.zone

    # ---- atomic / token-identity zones ----
    if z == Z_NEED_TC:
        if tid == TOOLCALL_OPEN:
            st.zone = Z_POST_TC
            return st
        return None

    if z == Z_POST_TC:
        if _is_ws(text):
            return st
        if tid == NS_TOKEN:
            st.zone = Z_INVOKE_LIT
            st.lit = INVOKE_OPEN_LIT
            st.pos = 0
            return st
        return None

    if z == Z_BODY:
        if _is_ws(text):
            return st
        if tid == NS_TOKEN:
            st.zone = Z_ELEM_LT
            return st
        return None

    if z == Z_AFTER_INVOKE:
        if _is_ws(text):
            return st
        if tid == NS_TOKEN:
            st.zone = Z_ELEM2
            return st
        return None

    if z == Z_VALUE:
        if tid == NS_TOKEN:
            st.zone = Z_VALUE_LT
            return st
        # regular content token (raw newline / quote / emoji all pass); any
        # special token (EOS/BOS/pad/other markers) is forbidden mid-value.
        if 0 <= tid < SPECIAL_LO and text != "":
            return st
        return None

    if z == Z_ELEM2:
        if tid == TOOLCALL_CLOSE:
            st.zone = Z_DONE
            return st
        if text[:1] == "<":
            st.zone = Z_INVOKE_LIT
            st.lit = INVOKE_OPEN_LIT
            st.pos = 0
            return _char_walk(st, text)
        return None

    # ---- character-DFA zones ----
    # A special token can never satisfy a literal/trie zone (its text is a
    # multi-char marker); reject fast.
    if tid >= SPECIAL_LO:
        return None
    return _char_walk(st, text)


# ---------------------------------------------------------------------------
# Full-vocab allowed-set (the rare fallback), cached by state signature
# ---------------------------------------------------------------------------
def _state_sig(st):
    return (
        st.zone, st.lit, st.pos,
        id(st.node), st.chosen_tool,
        st.stack[-1] if st.stack else None,
    )


def _full_allowed(state):
    """Every legal token id from `state` (scans the vocab). Cached per signature
    (on the grammar) so repeated identical structural states are O(1)."""
    g = state.g
    cache = g.fullscan_cache
    sig = _state_sig(state)
    hit = cache.get(sig)
    if hit is not None:
        return hit
    id2text = g.id2text
    allowed = []
    for tid in range(g.vocab):
        if _walk_token(state, tid, id2text[tid]) is not None:
            allowed.append(tid)
    cache[sig] = allowed
    return allowed


# ---------------------------------------------------------------------------
# The per-step mask (module API: filter_logits(state, logits))
# ---------------------------------------------------------------------------
_NEG = -1e9
TOP_K = 64


def allowed_from_topk(state, logits, top_k=TOP_K):
    """Return (allowed_ids, used_fullscan). Lazy strategy: test the top-k
    candidates first; only on a total miss scan the full vocab."""
    row = logits[0] if logits.ndim == 2 else logits
    v = int(row.shape[-1])
    k = min(top_k, v)
    # top-k candidate ids (unordered); one small device->host sync.
    cand = mx.argpartition(row, kth=v - k)[v - k:]
    cand_ids = [int(x) for x in cand.tolist()]
    id2text = state.g.id2text
    allowed = [tid for tid in cand_ids
               if 0 <= tid < state.g.vocab
               and _walk_token(state, tid, id2text[tid]) is not None]
    if allowed:
        return allowed, False
    return _full_allowed(state), True


def filter_logits(state, logits, top_k=TOP_K):
    """Mask `logits` so only grammar-valid tokens survive. FREE mode is a pure
    passthrough. Never-deadlock: if zero tokens are legal, RELEASE (return the
    logits unchanged) and bump state.releases."""
    state.steps += 1
    if state.mode != "armed":
        return logits
    if state.zone == Z_VALUE:
        # Hot zone (~90% of a long call's tokens): state-independent deny
        # mask, no argpartition / host sync / candidate walks. Cannot
        # deadlock — the regular vocab is legal here.
        add = state.g.value_deny_add()
        if logits.ndim == 2:
            out = mx.array(logits)
            out[0] = logits[0] + add.astype(logits.dtype)
            return out
        return logits + add.astype(logits.dtype)
    allowed, _ = allowed_from_topk(state, logits, top_k=top_k)
    if not allowed:
        state.releases += 1  # never deadlock: let the repair net handle it
        return logits
    v = int(logits.shape[-1])
    idx = mx.array(allowed, dtype=mx.int32)
    if logits.ndim == 2:
        # Mask ONLY the constrained (primary) row and leave any other batched
        # sequences untouched. Decode is serialized under a single active
        # constraint (batch is 1 in practice), so row 0 is that stream; this
        # just prevents catastrophic corruption if batch>1 ever reaches here.
        row = mx.full((v,), _NEG, dtype=logits.dtype)
        row[idx] = logits[0, idx]
        out = mx.array(logits)
        out[0] = row
    else:
        out = mx.full(logits.shape, _NEG, dtype=logits.dtype)
        out[idx] = logits[idx]
    return out


def allowed_token_mask(state, vocab=None):
    """Boolean legality vector over the vocab for `state` (test/introspection
    helper; the hot path uses filter_logits' lazy top-k instead)."""
    v = vocab or state.g.vocab
    mask = [False] * v
    if state.mode != "armed":
        return [True] * v
    for tid in _full_allowed(state):
        mask[tid] = True
    return mask


# ---------------------------------------------------------------------------
# Advance (module API: advance(state, token_id))
# ---------------------------------------------------------------------------
def advance(state, token_id, text=None):
    """Walk the automaton forward on the actually-sampled token. Handles entry
    detection (arm at the tool-start namespace token), thinking-mode tracking,
    and block-close disarm. Returns `state` (mutated in place)."""
    tid = int(token_id)

    # Thinking-mode tracking: blocks stay unconstrained; a stray namespace
    # token inside <mm:think> does NOT arm.
    if tid in (THINK_OPEN, THINK_OPEN_LEGACY):
        state.in_think = True
        return state
    if tid in (THINK_CLOSE, THINK_CLOSE_LEGACY):
        state.in_think = False
        return state

    if state.mode != "armed":
        # FREE: the ONLY arming trigger is the atomic namespace token, and only
        # when not inside a thinking block. Every drift flavor (bare-name/JSON
        # after ]<]minimax[>[) begins exactly here, so arming here is what makes
        # the first divergent token impossible.
        if tid == NS_TOKEN and not state.in_think:
            state.mode = "armed"
            state.zone = Z_NEED_TC
        return state

    # Armed: the sampled token is grammar-valid (we masked to it). Fold it in.
    if text is None:
        table = state.g.id2text
        text = table[tid] if 0 <= tid < len(table) else ""
    nxt = _walk_token(state, tid, text)
    if nxt is None:
        # A released step (or an injected EOS) landed an out-of-grammar token.
        # Fail safe: disarm rather than wedge in a dead zone.
        _reset_to_free(state)
        return state
    # carry mutated fields back onto `state`
    _adopt(state, nxt)
    if state.zone == Z_DONE:
        _reset_to_free(state)
    return state


def _adopt(state, other):
    state.mode = other.mode
    state.zone = other.zone
    state.lit = other.lit
    state.pos = other.pos
    state.node = other.node
    state.partial = other.partial
    state.chosen_tool = other.chosen_tool
    state.stack = other.stack
    state.in_think = other.in_think


def _reset_to_free(state):
    state.mode = "free"
    state.zone = Z_FREE
    state.lit = ""
    state.pos = 0
    state.node = None
    state.partial = ""
    state.chosen_tool = None
    state.stack = []


# ---------------------------------------------------------------------------
# ToolGrammarConstraint: per-request object (holds the current state)
# ---------------------------------------------------------------------------
class ToolGrammarConstraint:
    """Per-request automaton. Construct once from the request's advertised tools;
    the server sets it active on rank 0 for the duration of generation."""

    def __init__(self, tokenizer_path_or_obj, tool_schemas, vocab_size=DEFAULT_VOCAB):
        id2text = id2text_table(tokenizer_path_or_obj, vocab_size)
        tool_names = []
        param_info = {}
        for schema in (tool_schemas or []):
            fn = schema.get("function", schema) if isinstance(schema, dict) else {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if not name:
                continue
            tool_names.append(name)
            names, strict = _tool_param_names(schema)
            param_info[name] = (build_trie(names) if strict else None, strict)
        self.grammar = Grammar(id2text, vocab_size, tool_names, param_info)
        self.state = State(self.grammar)
        self.tool_names = tool_names

    # --- module-API delegators (state-carrying) ---
    def filter_logits(self, logits, top_k=TOP_K):
        return filter_logits(self.state, logits, top_k=top_k)

    def advance(self, token_id, text=None):
        return advance(self.state, token_id, text=text)

    def allowed_mask(self):
        return allowed_token_mask(self.state)

    # --- convenience used by the rank0 sampler hook ---
    def mask_logits(self, logits):
        """FREE-mode fast path: return `logits` untouched (no top-k work)."""
        if self.state.mode != "armed":
            return logits
        return filter_logits(self.state, logits)

    def observe(self, token_id, text=None):
        """Fold the actually-sampled token into the automaton (every step)."""
        return advance(self.state, token_id, text=text)

    # --- introspection ---
    @property
    def armed(self):
        return self.state.mode == "armed"

    @property
    def releases(self):
        return self.state.releases

    @property
    def steps(self):
        return self.state.steps


def build_from_request(tokenizer, tools, vocab_size=DEFAULT_VOCAB):
    """Return a ToolGrammarConstraint for a request, or None when there is
    nothing to constrain (no advertised tools)."""
    schemas = []
    for t in (tools or []):
        if isinstance(t, dict) and (t.get("function") or t.get("name")):
            schemas.append(t)
    if not schemas:
        return None
    try:
        return ToolGrammarConstraint(tokenizer, schemas, vocab_size=vocab_size)
    except Exception:
        return None  # never break decode on constraint-construction failure


# ---------------------------------------------------------------------------
# Module-global active holder (rank 0 only; decode is serialized under the
# generation lock, so a single current constraint is safe). Mirrors the
# _FORCE_EOS arming pattern in sharded_server.py.
# ---------------------------------------------------------------------------
_ACTIVE = {"con": None}

CONSTRAINED_TOOLS_ENV = os.environ.get(
    "MLX_M3_CONSTRAINED_TOOLS", "0"
).strip().lower() in {"1", "true", "yes", "on"}


def env_enabled():
    return CONSTRAINED_TOOLS_ENV


def set_active(con):
    _ACTIVE["con"] = con


def clear_active():
    _ACTIVE["con"] = None


def active():
    return _ACTIVE["con"]
