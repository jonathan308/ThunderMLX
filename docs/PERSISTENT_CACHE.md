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

MiniMax-M3 cache objects expose `state`:

- KV state: keys and values arrays from `KVCache`
- MSA index state: sparse index-key arrays from `MiniMaxM3KVCache`

MLX can write arrays with `mx.save_safetensors()` and load them with `mx.load()`.
Use one file per layer to keep failures localized and to avoid very large single
files. After loading, assign the restored tuple back through each cache object's
`state` setter, then run `mx.eval([c.state for c in cache])` before marking the
cache resident.

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
6. Thinking requests allow exact durable restores but intentionally reject
   partial SSD prefix restores for now. A partial thinking restore can land on an
   unsafe control-token/reasoning boundary in the MiniMax template, so the stable
   policy is `exact or cold prefill` until boundary-aware thinking checkpoints
   are implemented.
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
