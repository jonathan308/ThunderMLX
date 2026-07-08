# Tools

Developer and operator utilities live here. These are not required for normal
start/stop use.

- `m3_doctor.py`: local readiness/configuration check.
- `m3_analyze_results.py`: summarize probe and overnight result logs.
- `m3_overnight_runner.py`: conservative repeated validation runner.
- `m3_memory_sampler.py`: JSONL sampler for `/health` plus rank 0/rank 1
  memory, useful when Activity Monitor shows jagged memory pressure. Use
  `--output path.jsonl` to write directly to a file, and `--samples 0` for an
  until-interrupted overnight capture.
- `m3_speculative_audit.py`: metadata-only speculative/MTP readiness audit.
- `diag_shard.py`, `jaccl_smoke.py`, `smoke_test.py`, `test_filter.py`:
  lower-level diagnostics and development checks.

Run tools from the repository root, for example:

```bash
python3 tools/m3_doctor.py
python3 tools/m3_memory_sampler.py --samples 120 --stop-when-idle --output memory_prefill.jsonl
python3 tools/m3_overnight_runner.py --skip-memory-preflight --record-set 1800,4800
```
