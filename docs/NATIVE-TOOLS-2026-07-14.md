# Native Tool Execution Gate, 2026-07-14

This note records the production qualification of ThunderMLX's MiniMax-M3
native tool path. It contains no private addresses, credentials, prompt text,
or machine-specific paths.

## Decision

Production uses the tool format and parser shipped by the pinned `mlx-vlm`
MiniMax-M3 implementation. The broad compatibility overlay remains disabled.
ThunderMLX preserves complete native calls and lets the client execute them or
return an ordinary tool error. It does not rewrite paths, invent required
fields, truncate write payloads, suppress repeated mutations, or replace a
client tool with a synthesized command.

Three narrow reliability rules remain around the native path:

1. An unclosed outer `<tool_call>` is never exposed as a partial call. A closed
   inner invoke does not make an abandoned Write atomic.
2. A schema-matching completed native block may recover a harmless envelope
   drift, such as `to="bash"` in place of `name="Bash"`, only after the official
   parser returns no calls. Recovered arguments must still satisfy the
   advertised schema.
3. When an explicit action turn ends after reasoning without emitting a call,
   the server permits one cache-friendly continuation using the same model,
   template, tools, and thinking mode. Successful native turns never enter
   this path.

## Root Cause

The final tool instability was not a missing OpenAI compatibility feature. It
was decode approximation leaking into a structured-output workload.

The reference chat profile reuses MiniMax Sparse Attention block selections
for 48 adjacent decode tokens. This is a valuable prose optimization, but a
real OpenCode A/B showed path and argument drift during long tool loops. With
reuse disabled, the same coding workload completed normally. Tool-bearing
requests therefore use exact per-token block selection by default through
`MLX_M3_TOOL_DECODE_TOPK_REUSE_TOKENS=0`; the faster chat value is restored in
a `finally` block on both ranks immediately after the request.

A second issue allowed a layer-local top-k selection to outlive its generator
invocation. Attention modules persist across sessions, retries, SSD restores,
and prewarm passes, so a new request could present a different KV backing with
a compatible length. Every generator invocation now advances a decode epoch,
and cached selections are valid only inside that epoch. Stop/start and explicit
cache-clear paths remove both historical cache attribute names.

Neither fix changes the pipeline split, JACCL/RDMA transport, model weights,
prefill kernels, cache format, or ordinary chat decode profile.

## Production Defaults

```bash
MLX_M3_TOOL_COMPAT_OVERLAY=0
MLX_M3_NATIVE_TOOL_ACTION_RETRY_ATTEMPTS=1
MLX_M3_DECODE_TOPK_REUSE_TOKENS=48
MLX_M3_TOOL_DECODE_TOPK_REUSE_TOKENS=0
MLX_M3_TOOL_PARSE_DIAGNOSTICS=0
```

`MLX_M3_TOOL_PARSE_DIAGNOSTICS=1` is a privacy-sensitive local debugging mode.
It may capture raw model output under ignored `ops/logs/` files and must not be
left enabled in production.

## Validation Evidence

All tests used the reference two-rank `38,22` Thunderbolt/JACCL cluster.

- Static parser, native-tool, cache-policy, and request-scoping suites passed.
- MSA contract and numerical tests passed on both ranks with exact grouped
  top-k selection. Steel-MMA prefill and decode stayed within the configured
  BF16 error tolerances against dense selected-attention references.
- A fresh no-thinking OpenCode project completed 14 agent steps covering
  list, read, task tracking, three file writes, shell tests, a focused edit,
  verification, and final response. Its generated suite passed 18/18 tests.
- A fresh thinking OpenCode project completed 17 steps with streamed reasoning,
  diagnosed two failed assertions, edited both files, added a follow-up feature,
  and finished with 22/22 tests passing.
- The installed ZCode 0.15.2 headless harness completed separate real goals in
  both modes. No-thinking ran a 32-message native-tool session, created and
  edited a two-file Python project, tested twice, inspected the result, and
  finished with 7/7 tests passing. Thinking repeated the workflow in an
  isolated project, added type hints through a focused edit, and finished with
  8/8 tests passing. Both artifacts were independently retested afterward.
- Four alternating extended Claude Code suites ran for about ten minutes,
  added 56 successful inference requests, exercised long Read/Bash/Edit/Write
  loops, and left the failure counter and generation lock unchanged.
- OpenAI streaming and non-streaming tools, a multi-round ZCode-shaped OpenAI
  loop, Anthropic `tool_use`, Codex Responses file writes, Codex control-tool
  schemas, and both public model modes passed.
- OpenWebUI-shaped requests carrying 34 tool schemas reused 99.05% of the
  no-thinking prefix at 0.25s server TTFT and 96.26% of the thinking prefix at
  0.65s reasoning TTFT. Both decoded at about 30.5 tok/s.
- A 49,170-token agent/tool KV entry saved to SSD, survived a RAM reset,
  restored 49,170/49,207 tokens, answered correctly, and left both ranks idle.
- Client disconnect abandoned a 47k-token prefill on both ranks. Explicit
  `/v1/stop` halted decode at token 16, retained only the valid input prefix,
  and the immediate follow-up decoded at 31.75 tok/s.
- The image smoke streamed thinking, identified both colors, decoded at
  33.44 tok/s, and did not increment failures.

## Performance Effect

Tool selection exactness does not affect prefill. Cold agent/tool prefill in
the durable-cache gate measured 367.50 prompt tok/s at 48.9k tokens. Tool
decode is context-dependent: about 28-30 tok/s at short context, about 25 tok/s
around 14k, and 22.3 tok/s around 49k in this gate.

Ordinary chat retains reuse 48. The final short control measured 31.99 tok/s,
matching the prior 30-32 tok/s baseline. Existing long-context chat/cache
ladder results remain authoritative because the new override is entered only
when a request advertises tools.

## Reproduction

```bash
python3 probes/m3_parser_smoke.py
python3 probes/m3_standard_tool_smoke.py
bin/mlx-python probes/m3_msa_contract_smoke.py
bin/mlx-python probes/m3_msa_numerical_smoke.py
python3 probes/m3_openai_multitool_live_probe.py \
  --base http://127.0.0.1:8010/v1 --model Minimax-M3-No-Think
python3 probes/m3_openai_multitool_live_probe.py \
  --base http://127.0.0.1:8010/v1 --model Minimax-M3
python3 probes/m3_openwebui_tool_cache_probe.py \
  --base http://127.0.0.1:8080 --tools 34
python3 probes/m3_image_smoke.py --base http://127.0.0.1:8080 \
  --model Minimax-M3
python3 probes/m3_cancel_probe.py --base http://127.0.0.1:8080 --live-stop
```

Run one inference probe at a time. After any failure, verify `/health` is idle
before continuing; do not stack retries on a distributed generation slot.
