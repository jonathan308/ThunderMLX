# Prefill and Cache-Capacity Lab, 2026-07-12

## Scope

This work was performed only in the isolated branch
`agent/prefill-optimization-20260712`. The production and public repositories
remained on their immutable golden commits while the lab was tested. The model
split stayed `38,22`; networking, JACCL/RDMA transport, quantization, native
tool behavior, MSA, and decode tuning were not changed.

## Root Cause

Long sessions could finish several turns and then wedge on a tiny suffix or the
first decode token. The final investigation found three related failure modes:

1. SSD serialization used `KVCache.state`, which slices backing tensors to the
   logical offset. A restored cache therefore had zero spare slots.
2. The single-cache to batch-cache bridge also passed through sliced `state`,
   then sliced again when returning to the single cache.
3. The first capacity fix then reserved the request's entire `max_tokens`
   ceiling during every warm-cache conversion. A normal 45k-token coding
   session with a 32k output ceiling therefore became a roughly 78k-slot cache
   on every layer even when the model generated only a short tool call.

A separate shape-specific stall was reproduced when a small incremental suffix
(`L=194` and `L=483` in captured cases) entered the custom blockwise sparse MSA
prefill kernel over a long resident KV. Cold/large prefill chunks did not show
this behavior.

At 200k-350k context, appending one token to that exact-fit cache could allocate
and copy the complete KV state on every layer. Forced teardown during the long
Metal operation was the source of the apparent orphaned wired memory.

## Fix

- SSD schema v3 stores explicit KV/MSA backing arrays and logical offsets, but
  persists only logical contents by default instead of immortalizing temporary
  spare capacity.
- Logical offsets and physical capacities are recorded separately.
- Restore crops any oversized stored backing and recreates only a bounded,
  rank-aware append window (`8192` slots on the reference Studio rank and
  `4096` on the reference MacBook rank). It materializes one layer at a time to
  bound transient memory.
- The B=1 single/batch bridge carries backing arrays and offsets directly.
- Warm batch conversion uses the same bounded reserve rather than the full
  output ceiling, and batch caches retain MLX's native 256-token allocator
  cadence.
- Small suffixes over long resident KV use mlx-vlm's native MiniMax attention
  path; large/cold prefill remains on the accelerated Steel-MMA MSA path.
- Runtime fingerprints now include the batch, pipeline, and draft integration
  modules that can change cache semantics.
- Health and probes report logical length, physical capacity, spare capacity,
  restore target capacity, and append reserve.
- The no-progress supervisor now scales its deadline with context size and asks
  both ranks to stop and drain before escalating to process termination.

## Panic Evidence And Rejected Experiment

Propagating the single-cache 4096-token growth step into `BatchKVCache` was
rejected. During a controlled 350k prefill, the M5 rank reached about 315k
tokens and macOS panicked in IOGPUFamily with
`completeMemory() prepare count underflow`. Swap and compressor state were
healthy and memory headroom remained, so this was not an ordinary OOM. The
experiment was reverted before final validation.

The worker's authoritative panic reports record two distinct incidents on
July 12, 2026:

- `09:51:16 PDT`: IOGPUFamily `completeMemory() prepare count underflow` in
  `IOGPUMemory.cpp:550`, panicked task `python3.14`. Python held about 108.7
  GiB resident, wired memory was about 110.5 GiB, and compressor/swap status
  was healthy. This is the rejected allocator experiment, not a conventional
  userspace OOM.
- `14:50:03 PDT`: a 90-second `watchdogd` check-in timeout with `python3.14`
  at about 122.1 GiB resident, roughly 126 GiB wired, and only about 14 MiB of
  free pages. This was a whole-system stall at effective unified-memory
  saturation, even though macOS did not label it memory pressure.

The final candidate avoids both signatures by retaining the native 256-token
batch allocator, bounding warm/restore append capacity, persisting no spare SSD
capacity by default, advertising a 300k operational agent window, and enforcing
a 524,288-token measured safety ceiling on the reference deployment.

## Final Measurements

### Authoritative candidate ladder

These measurements use the final cache-capacity candidate with the safe native
256-token batch allocator. The 30k-200k rows come from the same deterministic
in-memory perf probe. The 350k row is the stricter agent/tools SSD roundtrip
described below, so its TTFT is restore TTFT rather than an in-memory exact-hit
measurement.

| Class | Real prompt | Cold prompt tok/s | Cold decode tok/s | Cached long decode tok/s | Hot/restore TTFT | Failures |
|---|---:|---:|---:|---:|---:|---:|
| 30k | 30,617 | 367.94 | 26.44 | 26.25 | 0.10s exact hit | 0 |
| 80k | 81,917 | 356.93 | 18.46* | 25.48 | 0.16s exact / 0.68s appended | 0 |
| 100k | 100,817 | 353.90 | 25.11 | 25.25 | 3.65s exact hit | 0 |
| 200k | 199,817 | 337.45 | 24.01 | 23.81 | 4.58s exact / 1.07s appended | 0 |
| 350k SSD | 353,608 | 273.56 | 19.96 | 20.96 | 7.07s restore | 0 |

Short-context decode controls remained `30.45-30.51 tok/s`. The marked 80k
cold-decode sample was a one-run post-prefill/thermal outlier; its exact and
appended cache controls returned to `25.07-25.48 tok/s`, matching the earlier
`25.00-25.70` candidate range. The 200k cached decode remains above the earlier
`21.78 tok/s` reference. The candidate favors the proven cache-capacity and
coordinated-cancellation fixes over reopening the unsafe batch-allocation
experiment for a marginal prefill gain.

A second 350k request was started only after the full 30k/80k/200k ladder in
the same process. It was deliberately cancelled at `186,368/353,717` prefill
tokens when prior allocator state left the MacBook too close to its physical
ceiling. Both ranks acknowledged the stop, the request slot returned idle with
zero failures, and wired memory fell to about `6.4 GB` on the Studio and `2.9
GB` on the MacBook. This contaminated sequential stress is a cancellation and
release gate, not a replacement for the clean 353.6k row above.

### 350k durable-cache roundtrip

- Real prompt: `353,608` tokens, 12 agent tools
- Cold prefill: `273.56 prompt tok/s`
- Cold decode: `19.96 tok/s`
- Completed cache: `354,003` logical tokens in `354,048` slots
- SSD restore reuse: `354,003/354,039`
- Restored capacity: `356,352` slots
- Spare capacity after follow-up: `2,291` tokens
- Restore server TTFT: `7.07s`
- Restored decode: `20.96 tok/s`
- Correct far-tail answer, `0` failures
- Post-release wired memory: Studio `6.5 GB`, MacBook `3.0 GB`

### Agent-style staged growth

| Stage | Prompt | New suffix | Reuse | Prompt tok/s | Decode tok/s | TTFT |
|---|---:|---:|---:|---:|---:|---:|
| cold base | 21,736 | 21,736 | 0.0% | 375.08 | 26.37 | 59.71s |
| +8k | 30,480 | 8,744 | 71.31% | 1,075.91 effective | 26.75 | 29.35s |
| +2k | 32,714 | 2,204 | 93.26% | 3,944.52 effective | 26.63 | 9.30s |
| +500 | 33,318 | 574 | 98.28% | 10,927.71 effective | 26.47 | 4.00s |

### Compatibility and stability gates

- Parser regression suite: pass.
- Native OpenAI tools, no-thinking: non-stream and stream pass, hot TTFT
  `0.08s`, decode about `31.4 tok/s`, no marker leakage.
- Native OpenAI tools, thinking: non-stream and stream pass, hot TTFT `0.64s`,
  decode about `30.8 tok/s`.
- Codex Responses bridge: real file write passed on both public model IDs.
- Anthropic Messages bridge: non-stream and streamed `tool_use` passed.
- Claude Code no-thinking: simple write, Bash roundtrip, compound agent,
  multifile coding, and long-agent loop all passed. The long case executed 11
  tool operations.
- Claude Code thinking: 11-operation long-agent loop passed.
- Image-to-text: streamed red/blue recognition passed at `32.63 decode tok/s`.
- Coordinated cancellation: explicit stop, decode disconnect, 46.8k prefill
  disconnect, dashboard stop, and immediate follow-up all passed. The prefill
  stopped at `12,288/46,852` tokens and the follow-up returned normally.
- OpenWebUI-shaped stress: thinking stream, prior reasoning, disconnect,
  queued no-thinking follow-up, image input, and context lookup all passed.
- Final server state: healthy, idle, zero failed requests.

## Promotion Notes

- Promote only after user acceptance testing.
- Keep the production/public golden tag available for immediate rollback.
- `run_with_watchdog.py` uses the portable `mlx-python` launcher. The start
  script prepends the checkout's synchronized `bin/` directory, so promotion
  does not bake in this lab's absolute environment path.
- Schema-v1/v2 SSD artifacts should miss safely and be rebuilt as schema v3.
- Do not reintroduce the 4096-token batch allocator experiment.
