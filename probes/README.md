# Probes

Repeatable benchmark and regression probes live here. They call the running
OpenAI-compatible endpoint and are intended for validation/tuning, not normal
daily use.

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
- `m3_image_smoke.py`: OpenAI multimodal `image_url` VLM smoke. It sends an
  in-memory red/blue PNG data URI, verifies the response mentions both colors,
  and fails if the server failure count increases.

Use the dashboard Operations tab for the safest probe entry points.
