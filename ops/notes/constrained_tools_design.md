# Constrained Tool Decoding — Design & Validation Plan

_Branch `feature/constrained-tools`. Offline build only: no server was started, nothing
larger than the tokenizer was loaded. Python `~/mlx-vlm064-env/bin/python3.14`._

## TL;DR

A per-request pushdown automaton (`constrained_tools.py`) masks rank 0's sampling
logits so that, once MiniMax-M3 opens a tool block, **only grammar-valid tokens are
sampleable** until the block closes. Malformed tool markup — the bare-name/JSON drift
that today costs minute-long retry-ladder rescues on big `Write` payloads — becomes
**unrepresentable at generation time** rather than detected-and-repaired after the fact.
Outside a tool block (prose, thinking) the constraint is a zero-cost passthrough.

The grammar was **verified, not assumed**, against three independent sources: the
tokenizer's `added_tokens.json`, the live `mlx_vlm/tool_parsers/minimax_m3.py` parser,
and the model's `chat_template.json` argument renderer — plus a real tokenizer
round-trip of an encoded call. Acceptance gate `ops/test_constrained_tools.py` is
**48/48 green**. Gated behind `MLX_M3_CONSTRAINED_TOOLS` (default **OFF**).

---

## 1. Token-id map (verified against the MiniMax-M3-4bit tokenizer)

`~/.exo/models/mlx-community--MiniMax-M3-4bit`, vocab 200064 (real tokens 0..200060).
Every id below was confirmed by `AutoTokenizer.encode(marker) == [id]` (single atomic
token) — see `ops/test_constrained_tools.py` and the probe transcripts.

| id      | token             | role in the grammar                              |
|---------|-------------------|--------------------------------------------------|
| 200058  | `]<]minimax[>[`   | **atomic** namespace marker — starts every structural element AND terminates every value |
| 200052  | `<tool_call>`     | atomic tool-block open                            |
| 200053  | `</tool_call>`    | atomic tool-block close                           |
| 200059  | `<mm:think>`      | thinking open (does NOT arm; sets `in_think`)     |
| 200060  | `</mm:think>`     | thinking close                                    |
| 200050 / 200051 | `<think>` / `</think>` | legacy thinking pair                    |
| 200000  | `]!p~[` (pad)     | `SPECIAL_LO`; the added-token band [200000, vocab) |
| 200019 / 200020 | bos / eos | forbidden mid-block                                |

The namespace marker being a **single atomic token** is the load-bearing fact: value
termination needs no JSON escape/quote tracking — a value ends exactly when the model
samples 200058. Raw newlines, quotes, braces, `<html>`, and emoji flow through a value
as ordinary content tokens. (Emoji tokenize to partial-UTF8 byte pieces, e.g. 😀 →
`[21557, 128]`, each decoding to `�`; harmless, because inside a value ANY non-special
token is accepted regardless of its decoded text.)

## 2. Canonical wire grammar

Confirmed against the parser (`_INVOKE_RE`, `_TAG_NAME_RE = [A-Za-z_$][\w:.$-]*`,
`_NS_TOKEN`) and the chat template `to_xml` macro. The model is trained on
**namespace-prefixed direct-arg tags** — NOT `<parameter name="x">…</parameter>`:

```
]<]minimax[>[<tool_call>\n
]<]minimax[>[<invoke name="TOOL">
]<]minimax[>[<PARAM>value]<]minimax[>[</PARAM>                                (scalar)
]<]minimax[>[<PARAM>]<]minimax[>[<item>v]<]minimax[>[</item>]<]minimax[>[</PARAM>  (array)
]<]minimax[>[<PARAM>]<]minimax[>[<k>v]<]minimax[>[</k>]<]minimax[>[</PARAM>         (object)
]<]minimax[>[</invoke>\n
]<]minimax[>[</tool_call>
```

- Scalars render as **raw `{{ val }}`** (chat-template `to_xml` else-branch) — no JSON
  escaping. Booleans render `true`/`false`. This is why value masking is JSON-string
  *safety* only, never JSON-*structure*.
- Tool names are constrained to a **per-request trie of the ADVERTISED tools**. An
  undeclared name (e.g. `terminal` when only `Bash/Write` are advertised) is
  unreachable — its first glyph has no edge from the trie root.
- Param names: **strict trie** when the tool's schema has `properties` and
  `additionalProperties:false`; otherwise a **permissive** `_TAG_NAME_RE` walk, so a
  legal-but-undeclared key is never wrongly forbidden.

**Automaton shape.** A FREE/atomic/character-DFA hybrid. Atomic zones gate on token
identity (`<tool_call>`, the NS marker). Character-DFA zones walk each candidate token's
*decoded text* glyph-by-glyph over the literal skeleton (`<invoke name="`, `invoke>`,
close tags) and the name tries — this is what makes it **BPE-boundary correct**: a
literal may be spelled by any tokenization (`<` + `invoke` + ` name` + `="`), and a
merged token that spills past a literal into the next zone (the canonical `>\n`, id
1100, after `</invoke>`) is handled by continuing the walk as trailing whitespace /
value content. The lazy candidate strategy tests the model's **top-64** tokens first
(one small device→host sync) and only full-scans the 200k vocab on a total miss
(memoised per structural signature, on the grammar).

## 3. Hook choice — why the rank0 sampled-token wrapper

**Fact (verified):** rank 0 alone samples every token; the distributed decode is kept
lockstep by a sampled-token sync that patches `mlx_vlm.generate.ar._sample_with_positions`
and broadcasts rank 0's token to the peers (`all_gather` in tensor mode
`sharded_server._install_rank0_token_sync`; `send`/`recv_like` in pipeline mode
`m3_pipeline_patch`). The existing `_FORCE_EOS` decode-stop swaps rank 0's token inside
this same wrapper — proof that this is the single point where rank 0's next token is
decided.

**Choice:** mask inside that wrapper, on rank 0 only, **before** the sampler draws
(`args[1]` is `logprobs`), then fold the sampled token into the automaton
**after** (post `_FORCE_EOS`, so an injected EOS correctly disarms). Consequences:

- **Zero collective changes.** The same single token still crosses the wire; we only
  change *which* token rank 0 can draw. rank>0 needs no automaton — it echoes rank 0's
  (already-valid) token. In pipeline mode the `observe` read is placed *after*
  `mx.eval(sends)` so the load-bearing eval ordering (the h-send/h-recv stall guard) is
  untouched.
- **Single active constraint**, a module global set per request around the decode loop
  in `run_generation` / `run_generation_stream` and cleared in the existing `finally` —
  mirroring the `_FORCE_EOS` single-owner pattern, safe because `generation_lock`
  serializes decode. Built from the request's `tools`; masks the primary batch row and
  passes any other rows through untouched.
- **Thinking is unconstrained.** `<mm:think>`/`</mm:think>` toggle `in_think`; a
  namespace token inside thinking does not arm, so the model may reason about tool
  syntax freely.
- **Fail-safe everywhere.** Every masking/observe/arm/disarm site is wrapped so any
  error degrades to fully unconstrained decode — the constraint can never wedge decode.

## 4. Never-deadlock

If a state yields zero legal tokens (top-64 AND full-scan empty), `filter_logits`
**releases** for that step (returns the logits unmasked) and bumps `state.releases`; the
next out-of-grammar token disarms the automaton and the existing repair ladder catches
the rare leak. The automaton never hangs and never raises into decode. A nonzero release
rate in production is the canary for a grammar gap (see §6).

## 5. Flavor verdict (`ops/../minimax-m3-cluster/ops/test_tool_parse_flavors.py`, 8-case ledger)

Replayed through the automaton from the real tokenizer encoding:

- **Drift-after-marker (bare-name+json, self-describing json, input-key variant, raw
  newlines):** the first token AFTER the namespace marker is **provably forbidden** —
  the only legal token at that point is `<tool_call>` (200052). The `{`/` Bash`/` {"`
  that begins each leak (ids 16673 / 64402 / 18396) cannot be sampled. **IMPOSSIBLE.**
- **Unterminated tag marker / classic tag fragment:** the valid prefix is accepted but
  the block **cannot stop mid-markup** — EOS is forbidden until `</tool_call>`, forcing
  completion. **IMPOSSIBLE to emit unterminated.**
- **Plain prose:** contains no namespace token → never arms → pure passthrough. Clean.
- **Prose quoting the marker:** arming keys on the *atomic token id 200058*, not on
  decoded text; a marker spelled with ordinary tokens does not arm. If the model emits
  the atomic control token 200058 mid-prose it *will* arm (see risk R1) — the single
  false-positive vector, guarded by `in_think`.

## 6. Risks

- **R1 — atomic-marker false positive.** Emitting special token 200058 outside thinking
  arms the constraint and forces `<tool_call>`, even in "prose". In practice the model
  reserves 200058 for real markup, so incidence is ~nil; and forcing well-formed markup
  from an already-anomalous control-token emission is a benign recovery. Mitigation: the
  `in_think` guard; default-OFF; the release/disarm fail-safe.
- **R2 — content that literally spells `]<]minimax[>[` via ordinary tokens inside a
  value.** The parser would mis-split it. This is a **pre-existing** wire-format
  ambiguity, not introduced by the constraint — the constraint strictly *improves*
  matters by guaranteeing structural markers are the atomic token. Untouched here.
- **R3 — non-ASCII tool/param names split across partial-UTF8 tokens** could dead-end
  the character walk → a release (safe) rather than a mask. Tool names and JSON keys are
  ASCII in practice. Note only.
- **R4 — batched decode (batch>1).** One global constraint masks row 0 only. Decode is
  serialized (batch=1 in practice); other rows are now passed through unmodified so a
  stray batch can't be corrupted, but a *second* concurrent tool stream would be
  unconstrained. Acceptable under the single-owner model.
- **R5 — cold full-scan latency.** ~40 ms the first time a drifting state's top-64 are
  all illegal, then memoised (0.004 ms). Fires only on active drift and is still
  vastly cheaper than the retry rescue it replaces.
- **R6 — BPE steering.** Forbidding a merged token the model preferred (e.g. `>\n`) can
  force an equivalent split spelling; identical decoded text, slight probability
  perturbation only. Verified: the canonical encoding replays with zero rejections.

## 7. Microbench (per-step `filter_logits`, M-series, offline)

| path                          | ms/step |
|-------------------------------|---------|
| FREE passthrough (>99% case)  | 0.0001  |
| armed VALUE (top-64 hit)      | ~0.44   |
| armed literal (top-64 hit)    | ~0.41   |
| armed cold full-scan (once)   | ~40     |
| armed full-scan (cached)      | 0.0036  |

All hot paths clear the <1 ms/step target. Decode is memory-bound at tens of tok/s, so
armed overhead is well under a percent of step time; FREE mode is free.

## 8. Live validation-window plan (when `MLX_M3_CONSTRAINED_TOOLS=1` is first flipped on)

Roll out shadow-safe: flag stays default-OFF; enable on rank 0 only via the env for a
bounded window, watch, then decide.

1. **Battery.** `ops/stress_battery.sh` Phase 4 (agent tool-cycles) + `ops/agent_traffic_test.py`,
   plus a big-`Write`-payload set (the markup-drift hot spot). Run OFF then ON,
   same prompts/seed.
2. **Repair-rung hit rate — the primary metric.** Count, per window:
   `_looks_like_raw_tool_fragment` trips, usable-turn **retry-ladder** firings
   (`MLX_M3_TOOL_UNUSABLE_RETRY_ATTEMPTS`), and **exhausted-ladder** forensics dumps.
   Target: the covered flavors drive fragment-trips and ladder-firings toward **zero**;
   any residual exhausted-ladder dump is a NEW flavor → add a grammar case.
3. **Constraint health.** Log `state.releases` per generation. Expected ~0; a nonzero
   rate localizes a grammar gap (which zone) without any user-visible failure.
4. **Latency delta.** Compare tok/s and TTFT OFF vs ON. Expect no measurable decode
   regression (armed steps ~0.4 ms, FREE ~0); the win is the **elimination of the
   minutes-long retry rescues**, visible as reduced p99 tool-turn latency.
5. **Correctness parity.** Diff tool-call JSON (names + arguments) OFF vs ON on the
   battery to confirm the constraint changes *validity*, not *content*.

Go/no-go: ON shows fragment-trips & ladder-firings down with zero release-storms and no
decode-latency regression → promote the default. Otherwise keep OFF; the grammar case
that leaked is the next fix.
