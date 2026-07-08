# Speculative Decoding And MTP Notes

This gateway should not expose speculative decoding as a production setting
until a compatible MiniMax-M3 drafter is found and validated on clean memory.

## Current Local Audit

Run:

```bash
python3 tools/m3_speculative_audit.py
```

Latest local result:

- `mlx`: `0.31.2`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- Target model type: `minimax_m3_vl`
- Target hidden size: `6144`
- Target layers: `60`
- Installed MLX-VLM drafter kinds: `dflash`, `eagle3`, `mtp`
- Installed drafter model-type mappings include DeepSeek, Eagle3, Gemma, and
  Qwen families.
- No local drafter config matched MiniMax-M3.

## Interpretation

MLX-VLM has the necessary speculative hooks, but the target/drafter pairing is
model-family specific. MiniMax-M3 cannot safely use a random MTP/Eagle/DFlash
checkpoint just because the runtime supports the mechanism.

Keep production speculative/MTP disabled until all of these are true:

- A MiniMax-M3-compatible drafter checkpoint is available locally.
- `tools/m3_speculative_audit.py` reports a plausible candidate.
- The drafter validates against the target without hidden-size/config mismatch.
- A clean-memory A/B run improves decode without breaking thinking, image input,
  tool calls, prompt cache reuse, or distributed rank lockstep.

## Suggested Validation Order

1. Reboot both Macs if wired memory is dirty.
2. Run `python3 tools/m3_speculative_audit.py`.
3. If a candidate appears, test in an isolated branch and keep the normal
   endpoint disabled.
4. Run `python3 probes/m3_hot_cache_probe.py`.
5. Run `python3 probes/m3_perf_probe.py --records 600`.
6. Run `python3 probes/m3_openwebui_stress.py`.
7. Compare logs with `python3 tools/m3_analyze_results.py`.

If any test wedges or increases orphaned wired memory risk, keep speculative
off and continue optimizing prompt cache, prefill, and MiniMax-specific sparse
kernels instead.
