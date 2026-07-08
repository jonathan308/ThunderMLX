# ThunderMLX Docs

Start here if you are setting up or validating the cluster.

## Setup

- [SETUP.md](SETUP.md): blank-slate two-Mac setup, model path, networking,
  launch, verification, and tuning notes.
- [SECURITY.md](SECURITY.md): what should stay out of git and how to handle
  local machine config.

## Cache And Performance

- [PERSISTENT_CACHE.md](PERSISTENT_CACHE.md): SSD-backed prompt/KV cache design,
  privacy notes, restore gates, pruning, and validation commands.
- [CURRENT_GOAL_STATUS.md](CURRENT_GOAL_STATUS.md): current benchmark ledger and
  recovery notes. This is intentionally detailed and is mostly for maintainers.
- [SPECULATIVE_MTP.md](SPECULATIVE_MTP.md): speculative decoding/MTP research
  notes and audit status.

## Validation Probes

Probe documentation lives in [../probes/README.md](../probes/README.md).

The quickest compatibility pass against a running local endpoint is:

```bash
python3 probes/m3_tool_call_smoke.py --base http://127.0.0.1:8080
python3 probes/m3_image_smoke.py --base http://127.0.0.1:8080
python3 probes/m3_agent_staged_suffix_probe.py --base http://127.0.0.1:8080
```
