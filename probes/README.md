# Probes

Repeatable benchmark and regression probes live here. They call the running
OpenAI-compatible endpoint and are intended for validation/tuning, not normal
daily use.

## Native-first release gate (2026-07-12)

The production candidate keeps normal client calls on native `mlx-vlm` tool
formatting. ThunderMLX validates the result and applies only narrow,
transcript-proven repairs for malformed output. Static tool primers,
constrained decoding, proactive no-call caps, and write scaffolding were off
during this gate.

- Five alternating extended Claude Code suites passed: `76` inference
  requests, more than `50` executed Bash/Read/Edit/Write actions, `0` request
  failures, and no leaked generation lock.
- A real OpenCode run reached `69` messages and about `34k` prompt tokens,
  recovered from client-side edit failures, wrote the requested implementation,
  and passed `26/26` generated tests. Client cancellation returned the server
  to healthy idle without orphaned wired memory.
- Codex Responses real-file writes, Anthropic streaming/non-streaming tools,
  native OpenAI tool calls in both thinking modes, OpenWebUI-shaped chat,
  image input, and disconnect-followed-by-retry all passed.
- Speed checks on the reference `38,22` cluster held short decode around
  `32.0 tok/s`. A 30,743-token cold prompt processed at `380.46 prompt tok/s`;
  its changed cached follow-up reused `30,719/30,748` tokens, reached `1.36s`
  TTFT, and decoded at `27.56 tok/s`. At 106k and 200k, cached decode measured
  `26.13` and `24.63 tok/s` respectively with at least `99.97%` reuse.
- Cold long-prefill measurements after repeated stop/start and soak cycles were
  `359.44 tok/s` at 78k, `344.21` at 106k, and `309.26` at 200k. These are
  intentionally recorded separately from warmed historical peaks. The BQ64
  long-context lane remained about 3% faster than plain Steel MMA at 200k.
- A midnight rollover changed the injected date near the prompt prefix and
  forced an 80k re-prefill. The release pins injected date text for the active
  cache-session lifetime; new or idle-expired sessions still receive the
  current date.

## Exact structured-decode gate (2026-07-14)

Real-client A/B testing found that decode top-k reuse `48` remained the fastest
chat profile but could drift inside long structured tool arguments. Production
now applies `MLX_M3_TOOL_DECODE_TOPK_REUSE_TOKENS=0` only while a request
advertises tools and restores the chat setting in `finally` on both ranks. A
per-generation epoch also prevents a layer-local block selection from crossing
session, retry, SSD-restore, or prewarm boundaries.

- Fresh OpenCode no-thinking and thinking projects completed 14 and 17 agent
  steps and finished with 18/18 and 22/22 generated tests passing.
- Actual ZCode 0.15.2 headless goals passed in both modes, executing the full
  native ZCode tool harness and finishing independently verified 7/7 and 8/8
  Python test suites.
- Four alternating extended Claude Code suites added 56 successful requests in
  about ten minutes without incrementing failures or leaking the active slot.
- A 49k agent/tool SSD roundtrip restored 99.92% of the prompt; OpenWebUI-shaped
  34-tool follow-ups reached 0.25s no-thinking and 0.65s thinking TTFT.
- Disconnect, explicit stop, image, OpenAI, Anthropic, Codex Responses, MSA
  numerical, and short-speed gates passed. No-tool short decode remained
  31.99 tok/s, so the exactness profile did not regress ordinary chat.

Full evidence and reproduction commands are in
[`docs/NATIVE-TOOLS-2026-07-14.md`](../docs/NATIVE-TOOLS-2026-07-14.md).

Common probes:

- `m3_turn_probe.py`: thinking/no-thinking multi-turn hot-cache check. The
  thinking default intentionally uses a bounded first-turn prompt with enough
  budget (`768`) to finish reasoning and visible content while staying below the
  visible-prewarm generated-token ceiling. If a lower first-turn cap produces
  reasoning-only output, the server should drop that partial KV state and the
  probe should be treated as a capped-output safety check, not a cache-speed
  regression. Larger shaped SSD probes may still need `--seed-max-tokens 1024`.
- `m3_thinking_speed_probe.py`: short thinking/no-thinking streaming speed
  comparison. Its default prompt is intentionally bounded so `--max-tokens 768`
  validates both reasoning and visible final content instead of hitting the cap
  in reasoning-only output.
- `m3_agent_cache_probe.py`: long agent-context cache preservation check.
- `m3_agent_staged_suffix_probe.py`: uneven coding-agent transcript growth
  check. Default stages are roughly 20k, then +8k, +2k, and +500 prompt tokens;
  later turns should reuse prior KV and process only the new suffix.
- `m3_incremental_context_probe.py`: grows a very large coding-agent context in
  chunks and verifies hot-cache reuse at 200k-350k+ logical prompt lengths.
  Latest stable gate: six 1900-record chunks reached ~353k logical prompt
  tokens with 0 failures; final follow-up reused ~352.9k tokens and processed a
  48-token suffix.
- `m3_high_context_decode_ab.py`: cached high-context decode A/B for cadence,
  sparse top-k, decode top-k reuse, and compact-sort experiments. Use this to
  compare 30k/80k/200k decode without rebuilding the cache for every knob.
  July 1 256k hot-cache ladder on the reference `38,22` cluster found sparse
  top-k `16`, compact sort off, cadence `1`, decode reuse `48` fastest at
  `22.36 tok/s` with 0 failures; reuse `64` regressed slightly to `22.00`.
  Promotion smoke after that measured reuse `48` at `26.00 tok/s` on a 30k
  cached decode, `24.51 tok/s` on an 80k cached decode, `21.78 tok/s` on a 200k
  cached decode, and `17.36 tok/s` on a 350k cached decode, all with 0 failures.
- `m3_multi_session_cache_probe.py`: resident multi-session KV slot restore.
- `m3_persistent_cache_probe.py`: SSD-backed prompt/KV cache save/restore
  validation. It builds the RAM cache, then calls the explicit SSD save endpoint
  so autosave can stay off during normal inference. Use `--phase roundtrip
  --target-tokens 30000` for a local reset-and-restore smoke test, or run
  `--phase build`, restart the cluster, then run `--phase restore` with the same
  `--session-id` for a true durable restart check. Repeat with
  `--target-tokens 100000` and `250000` after the 30k path is stable. Add
  `--model Minimax-M3-No-Think` for no-thinking coverage, omit it for thinking,
  `--shape openwebui-tools --session-mode metadata` for OpenWebUI-style payloads
  on an explicit stable session id, `--shape openwebui-tools --session-mode auto`
  for the separate no-metadata OpenWebUI gate, and `--shape agent-tools
  --session-mode metadata` for agent/tool-style durable sessions.
  `--cancel-after-restore` runs a stop smoke immediately after restore; it skips
  safely unless the controlled test environment has in-flight stop flags enabled.
- `m3_perf_probe.py`: short decode plus long-context prompt/decode baseline.
- `m3_prefill_ab_probe.py`: runtime `prefill_step_size` A/B, restoring the
  original setting afterward.
- `m3_prefill_shape_probe.py`: cold prefill benchmark for request/client
  shapes. It compares plain, OpenWebUI-style tool attachment, and coding-agent
  tool schemas using authoritative `/health.last_request.prompt_tps`.
- `m3_openwebui_tool_cache_probe.py`: OpenWebUI/tool-schema cache regression.
- `m3_tool_prefix_reuse_probe.py`: fresh one-turn OpenWebUI/tool-prefix reuse
  regression for clients that start separate chats with the same tool schema.
- `m3_tool_call_smoke.py`: actual OpenAI-compatible tool-call smoke. It checks
  non-stream and streaming `finish_reason=tool_calls`, verifies `tool_calls`
  are present, and fails if raw MiniMax tool markers leak into streamed chunks.
- `m3_openai_multitool_live_probe.py`: ZCode-shaped streamed OpenAI agent loop.
  It executes real Read/Write/Edit/Bash calls, verifies the resulting file, and
  requires the model to finish rather than stop after planning.
- `m3_msa_contract_smoke.py`: request-epoch and decode-selection-cache cleanup
  contract. Run with the production `bin/mlx-python` interpreter.
- `m3_msa_numerical_smoke.py`: grouped top-k plus Steel-MMA prefill/decode
  numerical comparison against dense selected-attention references. Run it on
  both ranks with the production interpreter.
- `m3_image_smoke.py`: OpenAI multimodal `image_url` VLM smoke. It sends an
  in-memory red/blue PNG data URI, verifies the response mentions both colors,
  and fails if the server failure count increases.
- `m3_multimodal_cache_smoke.py` and
  `m3_multimodal_server_policy_smoke.py`: path-independent image fingerprint
  and safe cache-reuse policy checks that do not load model weights.
- `m3_multimodal_processor_smoke.py --image /path/to/image.jpg`: loads the
  installed MiniMax processor (not model weights) and proves the anchored
  image span stays inside a follow-up turn's common token prefix.
- `m3_multimodal_cache_live_probe.py`: live two-rank gate for same-image hot
  reuse, changed/removed/reordered-image invalidation, compacted history, both
  thinking modes, and streaming plus non-streaming responses.
- `m3_multimodal_ssd_probe.py`: 10k-class image session save/RAM-reset/restore
  gate. Its `build` and `restore` phases also support a true process restart.
- `m3_multimodal_cancel_probe.py`: stops one image request during cold prefill
  and another during decode, then requires clean rank/cache idle and a healthy
  post-cancel image response.

Use the dashboard Operations tab for the safest probe entry points.
