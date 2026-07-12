# ThunderMLX Native-First Tool Reliability Results

Validation date: 2026-07-12

## Result

The native-first MiniMax-M3 tool path is ready for promotion. Normal OpenAI
and Anthropic tool requests use the model's native `mlx-vlm` format. The
compatibility layer validates emitted calls and applies only narrow repairs
when the transcript proves a malformed path, working directory, argument
shape, or reversed Edit operation.

The validated runtime kept static tool hints, constrained decoding, proactive
no-call limits, and automatic write scaffolding disabled. One bounded retry
remained enabled for structurally unusable calls. An opened tool block that
never closes now has a separate 8,192-token safety ceiling; this does not cap
normal assistant output or complete calls.

## Fixes

1. Unknown model ids are checked against oMLX before any backend unload, so a
   typo cannot stop MiniMax-M3.
2. Agent title-generation sidecars route through a short no-thinking lane and
   no longer occupy the single-flight slot with a long generation.
3. Tool-bearing ZCode/OpenCode requests are not mistaken for ordinary
   OpenWebUI chat and do not inherit its small default output budget.
4. Command working directories are anchored only when they drift from the
   explicit client root while retaining the same final directory name.
5. Reversed Edit arguments are swapped only after a real same-path failure and
   an earlier client Read result prove the inversion.
6. Incomplete tool blocks stop through synchronized EOS and enter the existing
   bounded retry path instead of silently decoding to the global ceiling.
7. Retry guidance no longer advertises a zero-character file limit when write
   chunking is disabled.
8. Injected date context is pinned for the active cache-session lifetime, so a
   midnight rollover does not invalidate hundreds of thousands of cached
   tokens. New or idle-expired sessions still receive the current date.
9. The launcher forwards the incomplete-call and MSA long-context threshold
   settings identically to both distributed ranks.

## Reliability Gates

| Gate | Result |
| --- | --- |
| Python compile and diff whitespace | PASS |
| Parser regression suite | PASS |
| Tool-format flavor suite | 9/9 PASS |
| Constrained-decoder offline suite | 48/48 PASS |
| OpenAI streaming and non-streaming tools, both model ids | PASS |
| Codex Responses real file writes, both model ids | PASS |
| Anthropic Messages streaming and non-streaming tools | PASS |
| Claude Code extended coding loop, both model ids | PASS |
| Image/VLM input | PASS |
| OpenWebUI-shaped history and disconnect recovery | PASS |
| Five-run alternating Claude Code soak | 76 requests, 0 failures |
| Real OpenCode coding run | 69 messages, 26/26 generated tests |
| Stop/restart memory recovery | 3-4GB wired on each rank, no orphan |

## Performance Gates

Reference hardware used the documented `38,22` pipeline split, JACCL/RDMA,
prefill step 4096, sparse top-k 16, decode top-k reuse 48, and per-request
Metal cache cleanup.

| Context | Cold prompt tok/s | Cached decode tok/s | Reuse / TTFT |
| ---: | ---: | ---: | --- |
| short | n/a | 31.9-32.2 | hot tool TTFT 0.07-0.63s |
| 30,743 | 380.46 | 27.56 | 99.91%, 1.36s changed-turn TTFT |
| 77,975 | 359.44 | 26.12 | 99.96%, 0.28s exact TTFT |
| 106,235 | 344.21 | 26.13 | 99.97%, 0.36s exact TTFT |
| 199,943 | 309.26 | 24.63 | 99.99%, 2.24s changed-turn TTFT |

The 200k BQ64/compact MSA lane was about 3% faster in cold prefill than plain
Steel MMA in the same validation window. Cold figures above include repeated
stop/start, soak, and first-use kernel costs and are intentionally kept
separate from historical warmed peaks. Cache reuse and decode exceeded the
older reference ladder at every measured long-context point.

## Operational State

The release candidate uses model ids `Minimax-M3`, `Minimax-M3-No-Think`, and
`M3-Web`, with the direct OpenAI endpoint on port 8080, arbiter gateway on
8010, dashboard on 8090, and oMLX passthrough on 8000. Machine addresses,
credentials, model paths, hostfiles, local environment files, cache artifacts,
and benchmark scratch output are intentionally excluded from git.
