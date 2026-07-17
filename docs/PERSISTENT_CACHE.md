# Persistent Session Cache

ThunderMLX already keeps hot prompt/KV cache in RAM and tracks session metadata
in `prompt_cache_sessions.json`. The optional SSD tier makes selected sessions
durable on local SSD so long OpenWebUI, coding-agent, and research sessions can
be resumed after TTL release, stop/start, or reboot without fully re-prefilling
the same 100k-350k+ token context when the prompt prefix still matches.

## Current State

- RAM cache is authoritative during a live server process.
- `PROMPT_CACHE_RESIDENT_SLOTS` keeps multiple sessions restorable in memory.
- `PROMPT_CACHE_SESSION_MANIFEST` is metadata-only by design: it stores session
  ids, token counts, reuse ratios, actions, and timestamps, but not prompt text,
  token ids, or KV tensors.
- A metadata-only manifest entry is useful for diagnostics but cannot rehydrate
  a model cache after TTL expiry, stop/start, or reboot. SSD artifacts are the
  durable KV tier.

### Multimodal requests and retries

Image-bearing requests currently bypass prompt/KV reuse. MLX-VLM expands image
placeholders into processor-dependent feature tokens, so reusing text KV without
also validating the image bytes, processor fingerprint, expanded feature
positions, cache shape, model, split, and rank metadata can desynchronize the
two ranks. A mismatch therefore falls back to normal cold prefill.

Exact non-stream HTTP retries are still coalesced safely: identical request
bodies share one active generation and may replay its completed response for a
short grace period. This prevents a client timeout or reconnect from serially
re-running the same large multimodal compaction without weakening the KV guard.

## Durable Cache Behavior

The opt-in SSD tier stores per-rank prompt cache shards and links them from the
existing session manifest. On a future request for the same session, the server
validates the artifact and restores the KV cache before processing only the new
suffix. If the prompt differs only after the stored prefix, ThunderMLX can trim
to the validated common prefix and restore that safe prefix.

This should behave like the existing resident-cache path:

1. Prefer the current live cache when the session is already resident.
2. Prefer an in-memory resident slot when available.
3. Optionally restore a validated SSD cache artifact.
4. Fall back to normal cold or partial-prefix prefill when validation fails.

The SSD tier must not slow the normal hot path. During the existing
`MLX_M3_PROMPT_CACHE_TTL_SECONDS` window, RAM/live and resident-slot reuse remain
first choice. SSD restore is only attempted when the requested session is not
already hot in memory, typically after TTL release, eviction, stop/start, or
reboot.

## Env Flags

- `MLX_M3_PROMPT_CACHE_SSD=0`: opt-in master switch.
- `MLX_M3_PROMPT_CACHE_SSD_RESTORE=0`: separate restore gate. Enable only after
  save-only artifacts are validated across both ranks.
- `MLX_M3_PROMPT_CACHE_SSD_THINKING_BOUNDARY_RESTORE=0`: permit the narrowly
  validated append-only thinking restore where the stored artifact differs by
  exactly one trailing `<mm:think>` marker. All other partial thinking restores
  still cold-prefill.
- `MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE=0`: save complete eligible cache states
  automatically after successful turns.
- `MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS=8192`: coalesce large
  rewrites until the durable checkpoint advances by this many tokens.
- `MLX_M3_PROMPT_CACHE_SSD_APPEND_RESERVE_TOKENS=4096`: maximum spare append
  window recreated on restore; rank-specific `_RANK0` and `_RANK1` overrides
  can reflect asymmetric memory headroom.
- `MLX_M3_BATCH_APPEND_RESERVE_TOKENS=4096`: matching bounded reserve for the
  single-cache to cancellable batch-cache bridge, also with rank overrides.
- `MLX_M3_PROMPT_CACHE_SSD_SAVE_RESERVE_TOKENS=0`: spare backing retained in a
  new SSD artifact. Zero stores logical KV contents only; restore recreates the
  bounded live append window.
- `MLX_M3_PROMPT_CACHE_SSD_DIR=~/.cache/thundermlx/prompt-kv`: default storage
  root used locally by each rank.
- `MLX_M3_PROMPT_CACHE_SSD_DIR_RANK0=`: optional rank-0 KV-cache artifact root.
- `MLX_M3_PROMPT_CACHE_SSD_DIR_RANK1=`: optional rank-1 KV-cache artifact root.
- `MLX_M3_PROMPT_CACHE_SSD_TTL_SECONDS=432000`: default 5 days.
- `MLX_M3_PROMPT_CACHE_SSD_MAX_BYTES=429496729600`: default 400 GiB cap.
- `MLX_M3_PROMPT_CACHE_SSD_MIN_TOKENS=8192`: skip tiny chats.
- `MLX_M3_PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS=10`: cache dashboard
  disk scans so `/health` polling does not walk the SSD tree every frame.
- `MLX_M3_PROMPT_CACHE_SSD_SAVE_REASONING=0`: keep visible-transcript mode as
  the default; do not persist hidden/generated-KV variants without a soak.
- `MLX_M3_PROMPT_CACHE_SSD_PRIVACY=local`: local-only, no dashboard download.

## Artifact Layout

Each rank writes only its own local prompt/KV cache artifact. This does **not**
change model sharding or layer placement. The 38/22 model split remains the
distributed inference topology; the SSD cache simply stores the already-prefilled
KV tensors for the rank that owns them. Per-device local storage is the safest
default: rank 0 restores from a primary-local path, and rank 1 restores from a
worker-local path. In the reference 38/22 setup, rank 1 owns the larger
initial-layer KV artifact, so pointing `MLX_M3_PROMPT_CACHE_SSD_DIR_RANK1` at
the worker with more/faster local SSD space captures most of the storage benefit
without adding a network filesystem to the restore path. Your best placement may
be different if you change the split, hardware, or storage layout.

```text
<ssd_dir>/
  manifest.json
  <session_key_hash>/
    session.json
    rank0/
      layer-000.safetensors
      layer-001.safetensors
      ...
    rank1/
      layer-000.safetensors
      layer-001.safetensors
      ...
```

`session.json` includes:

- schema version
- model id and model path hash
- tokenizer/chat-template hash
- sharding mode, rank count, rank id, and `M3_PIPELINE_LAYERS`
- MLX/MLX-VLM overlay version or git commit
- prompt token count, cache length, token-id hash, and optional token-id tail hash
- cache class names and tensor shapes/dtypes
- created/last-access timestamps
- total bytes

The top-level `manifest.json` is an LRU index for pruning by TTL and total byte
cap.

## Serialization

MiniMax-M3 cache objects expose a logical `state`, but MLX intentionally slices
that state to the current token offset. Persisting the sliced view makes a
restored cache exactly full. The next suffix or decode token can then reallocate
and copy the entire long-context KV tensor on every layer.

SSD schema v3 therefore stores:

- explicit KV/MSA backing arrays, cropped to logical contents plus only the
  configured save reserve (zero by default)
- logical KV and MSA index offsets separately from physical capacity
- per-layer backing layout metadata
- the normal model, tokenizer, runtime, split, rank, shape, and dtype metadata

MLX writes one safetensors file per layer. Restore installs and materializes one
layer at a time so a 350k cache does not retain both every source tensor and
every padded destination tensor at once. It crops inherited spare capacity,
then pads only to the request length plus a bounded rank-aware append reserve;
`max_tokens` remains an output safety ceiling rather than a prediction of how
much KV capacity every layer must preallocate. The single-to-batch and
batch-to-single bridge carries backing arrays and logical offsets directly
instead of routing through sliced `state`, while its conversion target uses the
same bounded policy. The batch cache keeps MLX's native 256-token growth
cadence.

Schema-v1/v2 artifacts safely miss after this change. Rebuild them under schema
v3 rather than attempting an ambiguous migration.

## Validation Gates

Never restore a durable cache unless all gates pass:

- request session id matches the artifact session key
- model id and model path hash match
- tokenizer/chat-template hash matches
- rank count and pipeline split match
- current rank has exactly one matching rank shard
- all expected layer files exist
- tensor shapes and dtypes match the cache class metadata
- token-id hash matches the new prompt prefix
- requested prompt length is at least the stored prefix length
- restored cache length agrees across both ranks

If any gate fails, log the reason, mark the manifest entry `rehydratable=false`,
and fall back to RAM cache or cold prefill.

## Distributed Restore Protocol

Rank 0 decides whether SSD restore is eligible after tokenizing the prompt, then
broadcasts a restore op containing only the session key, expected token hashes,
and restore metadata. Each rank loads its local shard from disk. Both ranks must
either restore successfully or both fall back; mixed restore/cold states can
desynchronize the pipeline.

Current implementation:

1. Save artifacts behind `MLX_M3_PROMPT_CACHE_SSD=1`.
2. Save only complete validated cache states for sessions above
   `MLX_M3_PROMPT_CACHE_SSD_MIN_TOKENS`; stopped/interrupted requests do not
   write partial cache artifacts.
3. Preserve hidden-thinking safety by creating prompt-prefix checkpoints only
   for explicit manual saves or when `MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE=1`.
   Normal OpenAI/OpenWebUI/agent turns do not trim the RAM hot cache for SSD.
   Automatic saves are coalesced until the completed cache advances by
   `MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS` (default `8192`). This
   avoids synchronously rewriting tens of GiB after every small agent turn;
   manual saves remain immediate and the in-memory cache remains the fast path.
4. Restore only behind `MLX_M3_PROMPT_CACHE_SSD_RESTORE=1` after exact metadata,
   runtime, split, rank, model, tokenizer, shape, dtype, and token-prefix checks.
   Restore is attempted only after live RAM and resident-slot reuse miss and the
   request would otherwise rebuild from prompt.
5. If SSD restore misses, the request falls back to ordinary in-memory prefill.
   A miss is expected for expired entries, metadata/runtime mismatches, missing
   artifacts, and unsafe thinking partial restores. The only penalty is TTFT for
   that request; correctness and OpenAI compatibility take priority over reuse.
6. Thinking requests allow exact durable restores and otherwise reject partial
   SSD prefix restores. With
   `MLX_M3_PROMPT_CACHE_SSD_THINKING_BOUNDARY_RESTORE=1`, one boundary-aware
   exception is allowed: an append-only continuation may crop exactly one stored
   trailing `<mm:think>` generation marker. Larger crops, different markers,
   shorter prompts, and non-prefix changes still cold-prefill.
7. Expose bytes, entries, paths, last save/restore, miss reason, prune, clear,
   and RAM-vs-SSD status in `/health` and the dashboard.
8. Prefer RAM live/resident caches first; SSD restore is only the durable fallback
   after RAM cache is cold, evicted, released, or restarted.

For validation, use dashboard **Clear RAM Cache** or
`POST /admin/prompt-cache/reset` to drop only live RAM KV. Do not use
**Clear SSD Cache** unless you intentionally want to delete durable artifacts.

Repeatable validation lives in `probes/m3_persistent_cache_probe.py`:

- `--phase build` then `--phase restore` with the same `--session-id` validates
  true stop/start or reboot durability.
- `--target-tokens 30000`, `100000`, and `250000` cover the required size
  ladder without changing the probe logic.
- `--shape openwebui-tools --session-mode metadata` exercises OpenWebUI-style
  tool-schema payloads on a stable durable session id.
- `--shape openwebui-tools --session-mode auto` exercises no-metadata OpenWebUI
  compatibility. This gate passed after moving manual SSD save onto the
  generation worker lane: a 10k-token, 12-tool auto session restored from SSD
  after RAM reset with `10,160/10,197` token reuse and `0` failures.
- `--shape agent-tools --session-mode metadata` exercises long agent/tool
  sessions with a stable durable session id. The 2026-07-01 gate passed with 12
  tool schemas, `19,136` cold prompt tokens at `390.86 prompt tok/s`, SSD
  restore reuse `19,183/19,221`, correct answer, and `0` failures.
- Thinking restore is intentionally narrower than no-thinking restore. Exact
  thinking SSD restore is validated: a 2026-07-01 patched-runtime gate restored
  an `18,167` token thinking cache, processed a `1` token suffix, reached
  `0.20s` server TTFT, decoded at `26.2 tok/s`, and kept failures at `0`.
  Partial thinking SSD restore is disabled because one partial thinking restore
  landed on an unsafe template/control boundary and wedged distributed decode.
  The patched fallback gate reprocessed the same follow-up cold at
  `365.04 prompt tok/s`, decoded at `26.79 tok/s`, and kept failures at `0`,
  with miss reason `partial_restore_disabled:*`.
- Large restore gates passed on 2026-07-01:
  - 100k-class target produced `159,270` real prompt tokens at
    `354.58 prompt tok/s`; restore reused `159,320/159,355` tokens and answered
    correctly.
  - 250k-class target produced `255,927` real prompt tokens at
    `330.15 prompt tok/s`; restore reused `255,977/256,012` tokens and answered
    correctly.
  Both gates ran with SSD restore enabled, RAM cache reset before restore, and
  `0` failed requests.
- The schema-v2 precursor capacity gate passed on 2026-07-12 with a real
  `353,608`-token
  agent/tool prompt. Cold prefill measured `273.56 prompt tok/s` and
  `19.96 decode tok/s`. The completed cache retained `354,048` physical slots
  for `354,003` logical tokens, saved locally on both ranks, cleared from RAM,
  and restored `354,003/354,039` prompt tokens. Restore reserved `356,352`
  slots, left `2,291` spare after the follow-up, answered correctly at
  `20.96 decode tok/s`, and kept failures at `0`. Clearing live RAM afterward
  returned wired memory to `6.5 GB` on the Studio rank and `3.0 GB` on the
  MacBook rank without a reboot or orphan.
- The final schema-v3 gate stores logical capacity by default and restores a
  bounded append window instead of carrying a request's full output ceiling on
  every layer. Under the final runtime fingerprint, no-thinking saved 46,588
  tokens and restored `46,588/46,625` (`99.92%`) with `0.86s` server TTFT and
  `26.49 decode tok/s`. A thinking/tool-shaped cache saved 47,366 tokens and
  restored `47,366/48,293` (`98.08%`). Its intentionally tiny first attempt
  exercised the bounded 4k internal tool-recovery budget; the capacity probe
  distinguishes that explicit bound from the rejected behavior that reserved
  the full 32k global output ceiling. Both modes ended with zero failures.
- The final safety patch changed the runtime fingerprint, so older durable
  artifacts may safely miss with a runtime mismatch. Rebuild/save under the
  current fingerprint when you need a specific long session to restore from SSD.
- `--cancel-after-restore` runs a stop smoke after restore when the controlled
  environment has synchronized in-flight stop enabled; production fast defaults
  keep those stop flags off and the probe reports a safe skip.

## Dead Ends To Avoid

- Do not persist only the metadata manifest and call it rehydratable.
- Do not persist prompt text by default; token ids and KV tensors are already
  sensitive enough and require an explicit opt-in.
- Do not restore a rank independently. Both ranks must reach the same decision.
- Do not persist full hidden-thinking/generated-KV mode until visible-transcript
  restore is soaked.
- Do not make SSD restore part of `/v1/stop`; interrupted generation should not
  write partial or unverified caches.
- Do not copy the single-cache 4096-token allocation step into `BatchKVCache`.
  A controlled 350k experiment reached about 315k tokens and triggered an M5
  IOGPUFamily `completeMemory() prepare count underflow` kernel panic despite
  healthy swap/compressor state and remaining memory headroom. Preserve backing
  capacity explicitly while retaining the native 256-token batch allocator.
