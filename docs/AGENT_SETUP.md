# ThunderMLX — Agent-Guided Setup

**To the human:** give this file (or its URL) to your AI coding agent — Claude
Code, codex, or similar — running on the Mac that will be the **primary**
(rank 0). It walks the agent through discovering your hardware, installing
dependencies, configuring, and booting the cluster, asking you only the
questions that need a human.

**To the agent:** you are setting up ThunderMLX — a two-Mac Apple Silicon
cluster that serves MiniMax-M3 (456B MoE, 4-bit) over Thunderbolt RDMA as an
OpenAI-compatible endpoint. Read `README.md` and `docs/SETUP.md` first; this
file is your operating procedure. Work phase by phase, verify each phase
before the next, and ask the user rather than guess whenever a decision is
marked **ASK**.

---

## Phase 0 — Scope check

Confirm with the user before touching anything:
1. Which Mac is primary (rank 0, more RAM — serves the API) and which is the
   worker (rank 1)? You should be running on the primary.
2. Are both Macs physically connected by a Thunderbolt cable?
3. Do they want the model downloaded fresh (~130GB) or do they already have
   `mlx-community/MiniMax-M3-4bit` on disk? **ASK — never start a 130GB
   download unannounced.**

## Phase 1 — Hardware discovery (read-only)

On the primary, run and record:

```bash
sysctl -n hw.memsize | awk '{printf "RAM: %.0fGB\n", $1/1073741824}'
sysctl -n machdep.cpu.brand_string
sw_vers -productVersion
ifconfig | grep -A3 "bridge0\|en[0-9]" | grep "inet " | head -4
df -h ~ | tail -1
```

Then the same over SSH on the worker once Phase 2 establishes access.

**Go/no-go gates** (stop and tell the user if any fail):
- Combined RAM ≥ 288GB (the 4-bit model needs ~220GB of weights plus KV
  headroom; the reference pair is 256GB + 128GB)
- Both machines Apple Silicon
- ≥140GB free disk on the primary (model + headroom), more if they want a
  large SSD prompt-cache tier

## Phase 2 — Network + SSH

1. Thunderbolt bridge: both Macs need static IPs on the bridge interface
   (System Settings → Network → Thunderbolt Bridge → Manual). Convention:
   primary `10.0.0.1`, worker `10.0.0.2`. The user must click this on the
   worker — give them the exact steps, then verify with `ping -c 2 10.0.0.2`
   (expect < 1ms).
2. SSH: check `ssh <worker> true` first. If not configured, generate a
   dedicated key (**ASK before creating keys**), `ssh-copy-id` it, and have
   the user enable Remote Login on the worker. Everything the launcher does
   on the worker flows over this.

## Phase 3 — Dependencies (both machines)

Detect what exists before installing:

```bash
python3 --version                          # need 3.12+
python3 -c "import mlx.core; print(mlx.core.__version__)" 2>/dev/null
python3 -c "import mlx_vlm; print(mlx_vlm.__version__)" 2>/dev/null
```

If missing: create a venv at the **same path on both machines**
(`~/mlx-env`), `pip install mlx "mlx-vlm>=0.6.4"`, and repeat the check on
the worker over SSH. Then link the repo's interpreter wrapper on both:
`ln -sf ~/mlx-env/bin/python3 <repo>/bin/mlx-python`.

## Phase 4 — Model

**ASK** the user where the model lives. If it needs downloading:

```bash
hf download mlx-community/MiniMax-M3-4bit --local-dir ~/models/MiniMax-M3-4bit
```

Warn them about size/time. Only the primary needs the weights on disk; the
worker streams its shards over Thunderbolt at boot.

## Phase 5 — Configuration

`cp .env.example .env.local`, then set (using Phase 1–4 discoveries):

- `M3_RANK1_DIRECT_SSH`, `M3_PEER`, `M3_SSH_KEY` — the SSH path to the worker
- `M3_RANK0_DATA_IP` / `M3_RANK1_DATA_IP` — the bridge IPs
- `MLX_M3_MODEL` — the model path; `MLX_M3_MODEL_ID` — its HF id
- `M3_PIPELINE_LAYERS` — rank-ordered layer counts: `rank0,rank1`. Rank 0 is
  the primary/API process and owns the final layers; rank 1 is the worker and
  owns the initial layers. The values must sum to 60. As a rough Q4 planning
  estimate, each transformer layer accounts for about 3.7GB of weights before
  rank-specific overhead, KV cache, Metal buffers, and macOS headroom. The
  validated 256GB-primary + 128GB-worker reference is `38,22`: final 38 layers
  on rank 0, initial 22 on rank 1. For other Macs, preserve at least 20-30GB
  of working headroom on each rank, then verify actual per-rank memory under a
  cold prefill before increasing context or resident-cache budgets. Present
  the proposed split and its headroom to the user before proceeding.

`./sync_rank1.sh` to push the runtime to the worker, and verify it reports
success.

## Phase 6 — First boot + verification

1. `open ./M3_Start.command` — boots both ranks under the watchdog, arms the
   crash-restart supervisor, starts the gateway (:8010) and dashboard (:8090).
2. First cold boot takes several minutes (weights stream to the worker).
   Watch `http://127.0.0.1:8090` — both ranks green, wired memory climbing.
3. Verify, in order:

```bash
curl -s http://127.0.0.1:8080/health | head -c 300     # "healthy", ranks: 2
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Minimax-M3","messages":[{"role":"user","content":"Say OK."}],"max_tokens":10}'
```

4. Point the user's client at `http://<primary>:8010/v1` (chat completions)
   or, for codex, the same base with `wire_api = "responses"`.

Success criteria: health shows 2 ranks; a chat turn answers; the dashboard
shows both machines; a second turn in the same chat has a sub-2s first token
(cache reuse working).

## Agent safety rules (non-negotiable)

- **Never** kill cluster processes with ad-hoc `pkill` patterns — use
  `M3_Stop.command`. A missed worker process holds the RDMA doorway and the
  next boot dies with `[jaccl] Recv errno=2`.
- **Never** start a second boot while one is loading, and never launch
  manually while the supervisor is armed — stop everything first.
- The cluster is designed to be always-on. Don't restart it to "fix" things
  without diagnosing first; check `docs/SETUP.md` → Troubleshooting.
- Watch wired memory via `vm_stat` ("Pages wired down"): ~5GB idle-clean,
  ~150GB loaded on a 256GB primary. High wired with no processes = tell the
  user a reboot of that Mac is needed (do not attempt it yourself).
- Ask before: downloads over 1GB, generating SSH keys, deleting anything,
  or rebooting machines.

## When something fails

Work the Troubleshooting table in `docs/SETUP.md` first. If you're stuck,
collect for the user: the tail of the launch log
(`/private/tmp/minimax-m3-cluster-logs/startup.log`), `vm_stat` from both
machines, and the exact failing command output — that triple diagnoses
nearly everything.
