# ThunderMLX Setup Guide

A from-scratch walkthrough for bringing up a two-Mac MiniMax-M3 cluster.
Expect 30–60 minutes plus the model download.

## 1. What you need

| | Primary (rank 0) | Worker (rank 1) |
|---|---|---|
| Machine | Apple Silicon Mac, **≥192GB** unified memory (256GB recommended) | Apple Silicon Mac, **≥96GB** (128GB recommended) |
| Role | Serves the API, owns the deeper layer stack | Runs the first layer stack |
| Example | Mac Studio M3 Ultra 256GB | MacBook Pro M4 Max 128GB |

- **Link**: a Thunderbolt 4/5 cable between them (TB5 recommended). Wi-Fi/
  Ethernet works for control traffic but the data path wants Thunderbolt.
- **macOS** on both, same major version preferred.
- **Python 3.12+** on both, with [MLX](https://github.com/ml-explore/mlx) and
  [mlx-vlm >= 0.6.5](https://github.com/Blaizzy/mlx-vlm) installed in the same
  interpreter path on each machine.
- At least ~250GB free disk on the primary for the ~225GB model plus download
  and runtime headroom. If you enable the SSD prompt-cache tier, budget its
  configured cap separately on each rank (the portable default cap is 400GB,
  but it grows only as cache entries are written).

## 2. Network: Thunderbolt bridge

1. Connect the cable. On **both** Macs: System Settings → Network →
   Thunderbolt Bridge → Configure IPv4 → Manually.
2. Give them static addresses on a private subnet, e.g. primary `10.0.0.1`,
   worker `10.0.0.2`, netmask `255.255.255.0`.
3. Verify: `ping 10.0.0.2` from the primary (expect sub-millisecond).

## 3. SSH from primary → worker

The launcher drives the worker over SSH (key-based, no prompts):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_thundermlx -N ""
ssh-copy-id -i ~/.ssh/id_ed25519_thundermlx user@10.0.0.2
ssh -i ~/.ssh/id_ed25519_thundermlx user@10.0.0.2 true && echo OK
```

Also enable **Remote Login** on the worker (System Settings → Sharing).

## 4. Python environment (both machines)

Install the runtime into the same path on both Macs (a venv works well):

```bash
python3 -m venv ~/mlx-env
~/mlx-env/bin/pip install mlx mlx-vlm
```

Point the repo's interpreter wrapper at it (on both machines the repo expects
`bin/mlx-python` to resolve to your interpreter):

```bash
ln -sf ~/mlx-env/bin/python3 <repo>/bin/mlx-python
```

## 5. Model download (primary only)

```bash
pip install huggingface_hub
hf download mlx-community/MiniMax-M3-4bit --local-dir ~/models/MiniMax-M3-4bit
```

The worker receives its weight shards over the Thunderbolt link at load time —
it does not need its own copy on disk (first boot is slower; later boots use
the OS file cache).

## 6. Clone and configure

```bash
git clone https://github.com/jonathan308/ThunderMLX.git ~/ThunderMLX
cd ~/ThunderMLX
cp .env.example .env.local
```

Edit `.env.local`. The keys that matter on day one:

```bash
# identity / labels shown on the dashboard
M3_RANK0_LABEL="Mac Studio"
M3_RANK1_LABEL="MacBook Pro"

# how the primary reaches the worker (SSH control path)
M3_RANK1_DIRECT_SSH=10.0.0.2
M3_PEER=user@10.0.0.2
M3_SSH_KEY=~/.ssh/id_ed25519_thundermlx

# the data path (Thunderbolt bridge IPs from step 2)
M3_RANK0_DATA_IP=10.0.0.1
M3_RANK1_DATA_IP=10.0.0.2
M3_MLX_BACKEND=jaccl

# the model from step 5
MLX_M3_MODEL=/Users/you/models/MiniMax-M3-4bit
MLX_M3_MODEL_ID=mlx-community/MiniMax-M3-4bit

# Pipeline split is rank0,rank1 and must sum to 60. Rank 0 owns the final
# layers/API; rank 1 owns the initial layers. This is the 256GB/128GB reference.
M3_PIPELINE_LAYERS=38,22
```

Everything else in `.env.example` has sane defaults (32k output budget, SSD
cache tiering, keepwarm, watchdogs). Come back to them later.

`.env.local` is gitignored — it never leaves your machine.

## 7. Sync to the worker

```bash
./sync_rank1.sh
```

This copies the whitelisted runtime files to the same path on the worker and
verifies. Re-run it after every update you pull.

## 8. First boot

```bash
open ./M3_Start.command      # or double-click it in Finder
```

What it does: sweeps stale state, boots both ranks under a watchdog, arms the
crash-restart supervisor, starts the gateway (:8010) and dashboard (:8090),
and waits for the model to wire into memory. First cold boot takes a few
minutes; later boots are ~30–60 seconds warm.

Watch it come up: **http://127.0.0.1:8090** (dashboard) — both ranks should
go green, with wired memory climbing to roughly 150GB / 80GB.

Sanity check:

```bash
curl -s http://127.0.0.1:8080/health | head -c 200
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Minimax-M3","messages":[{"role":"user","content":"Say OK."}],"max_tokens":10}'
```

## 9. Connect your clients

| Client | Base URL | Notes |
|---|---|---|
| OpenWebUI / zcode / any OpenAI SDK | `http://<primary>:8010/v1` | chat completions; live thinking streams as `reasoning_content` deltas |
| codex | `http://<primary>:8010/v1` with `wire_api = "responses"` | native Responses API — thinking renders as reasoning items |
| Direct (skip the gateway) | `http://<primary>:8080/v1` | M3 only, no model merge/switch |

Models exposed: `Minimax-M3` (thinking), `Minimax-M3-No-Think`, plus anything
your oMLX instance hosts if you run one on :8000 (the gateway merges the lists
and can auto-switch backends — guarded so it never interrupts active work).

## 10. Daily operation

- **Stop**: `M3_Stop.command` — graceful teardown of everything + a check
  that no wired GPU memory was stranded.
- **The cluster is always-on by design**: crashes self-heal via the
  supervisor; nothing shuts down on idle.
- **Stops from clients** (Esc in your agent, stop button in OpenWebUI)
  cancel distributed generation within a few tokens and free the slot.
- **Dashboard cancellation**: the Sessions tab targets the displayed request
  ID. Stale clicks are rejected rather than stopping a newer request. Prefill
  stops take effect at the next synchronized prefill chunk boundary; decode
  stops take effect at the next synchronized token boundary.
- **Logs**: server `/private/tmp/minimax-m3-cluster-logs/startup.log`,
  gateway `model_gateway.log`, dashboard `cluster_gui.log`.
- **Runtime updates**: use the Models tab. Updates are staged and applied to
  both ranks as one transaction. MLX-VLM, MLX-LM, and Transformers use exact
  PyPI wheels and roll back together without replacing the custom MLX core;
  MLX and MLX-Metal are always installed as a validated pair. Successful
  updates restart the managed stack automatically. Persistent cache entries
  include a runtime fingerprint, so entries from an older runtime safely miss
  and rebuild instead of restoring incompatible KV state.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Boot dies with `[jaccl] Recv failed` | A stale server process on the worker is holding the RDMA doorway. Run `M3_Stop.command` (its sweep is thorough), wait 30s, start again. |
| Boot dies instantly, supervisor log says "orphan guard" | Wired memory from a previous run wasn't released. The guard sweeps and retries; if wired stays high with no processes, reboot the affected Mac. |
| `ping 10.0.0.2` fails | Thunderbolt bridge not configured on both ends, or the cable renegotiated. Re-seat the cable, re-check step 2. |
| First request after a restart is slow | Expected: weights fault into memory lazily and session caches rebuild (SSD-tier sessions restore in seconds; others re-prefill once). |
| Health shows `healthy` but low wired memory | Normal on warm boots — wiring completes on first use. |
| Two boots fighting / phantom instances | Never launch manually while the supervisor is armed. Use the desktop commands; they own the lifecycle. |
| Worker shows high memory pressure at big contexts | Expected jagged pattern (macOS unwire/rewire per prefill chunk). If it *stays* pegged, your resident cache budget is too high for the worker — lower `MLX_M3_PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS_RANK1`. |

## Hardware notes

- The reference pair is a 256GB rank 0 + 128GB rank 1. `M3_PIPELINE_LAYERS`
  is ordered `rank0,rank1` and must sum to 60. Rank 0 owns the final layers,
  norm, LM head, and API; rank 1 owns the initial layers and embeddings. Thus
  the reference `38,22` means final 38 layers on the primary and initial 22 on
  the worker.
- Other memory combinations can use a different split. Estimate roughly 3.7GB
  of Q4 weights per transformer layer, then leave at least 20-30GB on each Mac
  for rank-specific weights, KV cache, Metal buffers, and macOS. Treat that as
  a starting estimate only: shard-file packing and runtime allocation are not
  perfectly linear. Validate a cold prefill and both ranks' memory pressure
  before raising context or resident-cache budgets.
- The lower-level `tools/test_filter.py` diagnostic reports the exact layer
  range and approximate shard bytes selected by each rank without loading the
  weights. Normal users can instead confirm the ranges in the startup log and
  monitor both machines from the dashboard during the first cold request.
- Single 300k-token contexts fit comfortably; the practical ceiling on a
  128GB worker is ~550–650k tokens of live KV.
