#!/usr/bin/env python3
"""Offline unit tests for the live-tunable storage caps (dashboard Storage
card): the runtime-tuning clamp rails in sharded_server plus m3_capture's
settings-dict surface. No server, cluster, or model — module-level calls only.

Run:  mlx-vlm064-env/bin/python3.14 ops/test_storage_tuning.py
"""
import os
import sys
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# Both modules read config at import, so pin the env BEFORE importing.
_TMP = tempfile.mkdtemp(prefix="storage_tuning_test_")
GIB = 1 << 30
MIB = 1 << 20
os.environ["MLX_M3_EAGLE3_CAPTURE_ONLY"] = "1"
os.environ["MLX_M3_EAGLE3_DUMP_DIR"] = _TMP
os.environ["MLX_M3_EAGLE3_CAPTURE_MAX_MB"] = "200"
os.environ["MLX_M3_EAGLE3_CAPTURE_DIR_MAX_GB"] = "100"
os.environ["MLX_M3_PROMPT_CACHE_SSD_MAX_BYTES"] = str(300 * GIB)

import m3_capture
import sharded_server as srv

_FAILS = []


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        _FAILS.append(msg)


def _snapshot():
    return srv._runtime_tuning_status()


def _restore(previous):
    # Restore just the storage keys these tests touch; replaying a FULL
    # snapshot would re-validate unrelated keys against launch-time guards
    # (e.g. the offline default prefill_step_size sits below the safe floor).
    srv._set_runtime_tuning(
        {k: previous[k] for k in srv._STORAGE_TUNING_KEYS if k in previous})


# --------------------------------------------------------------------------
def test_boot_seeding():
    print("test_boot_seeding (env -> mutables)")
    rt = _snapshot()
    check(rt.get("prompt_cache_ssd_max_bytes") == 300 * GIB,
          "ssd cap seeded from MLX_M3_PROMPT_CACHE_SSD_MAX_BYTES")
    check(rt.get("capture_max_request_bytes") == 200 * MIB,
          "request cap seeded from MLX_M3_EAGLE3_CAPTURE_MAX_MB")
    check(rt.get("capture_max_total_bytes") == 100 * GIB,
          "total cap seeded from MLX_M3_EAGLE3_CAPTURE_DIR_MAX_GB")
    check(srv._runtime_prompt_cache_ssd_max_bytes() == 300 * GIB,
          "prune-site accessor returns the seeded value")
    caps = m3_capture.settings()
    check(caps["max_request_bytes"] == 200 * MIB
          and caps["max_total_bytes"] == 100 * GIB,
          "m3_capture settings dict seeded from env")


# --------------------------------------------------------------------------
def test_apply_and_clamp():
    print("test_apply_and_clamp (in-range applies, out-of-range clamps)")
    previous = _snapshot()
    try:
        clamped = {}
        changed = srv._set_runtime_tuning(
            {"prompt_cache_ssd_max_bytes": 120 * GIB}, clamped_out=clamped)
        check(changed.get("prompt_cache_ssd_max_bytes") == 120 * GIB,
              "in-range ssd cap lands in changed")
        check(not clamped, "in-range value is not marked clamped")
        check(srv._runtime_prompt_cache_ssd_max_bytes() == 120 * GIB,
              "prune-site accessor sees the new cap immediately")

        clamped = {}
        srv._set_runtime_tuning(
            {"prompt_cache_ssd_max_bytes": 10 * GIB}, clamped_out=clamped)
        c = clamped.get("prompt_cache_ssd_max_bytes")
        check(c == {"requested": 10 * GIB, "applied": 50 * GIB},
              "below-range ssd cap clamps up to 50 GiB")
        check(srv._runtime_prompt_cache_ssd_max_bytes() == 50 * GIB,
              "clamped value is what got applied")

        clamped = {}
        srv._set_runtime_tuning(
            {"prompt_cache_ssd_max_bytes": 500 * GIB}, clamped_out=clamped)
        check(clamped["prompt_cache_ssd_max_bytes"]["applied"] == 400 * GIB,
              "above-range ssd cap clamps down to 400 GiB")

        clamped = {}
        srv._set_runtime_tuning(
            {"capture_max_request_bytes": 1 * MIB,
             "capture_max_total_bytes": 999 * GIB},
            clamped_out=clamped)
        check(clamped["capture_max_request_bytes"]["applied"] == 50 * MIB,
              "request cap clamps up to 50 MiB")
        check(clamped["capture_max_total_bytes"]["applied"] == 200 * GIB,
              "total cap clamps down to 200 GiB")
        caps = m3_capture.settings()
        check(caps["max_request_bytes"] == 50 * MIB
              and caps["max_total_bytes"] == 200 * GIB,
              "clamped capture caps propagate into m3_capture settings")

        # clamped_out is optional (rank1 mirror / revert path passes none).
        changed = srv._set_runtime_tuning({"prompt_cache_ssd_max_bytes": 1})
        check(srv._runtime_prompt_cache_ssd_max_bytes() == 50 * GIB,
              "clamping works without a clamped_out sink")
    finally:
        _restore(previous)
    check(_snapshot() == previous, "restore returns tuning to the pre-test state")
    check(m3_capture.settings()["max_request_bytes"] == 200 * MIB,
          "restore also returns m3_capture caps")


# --------------------------------------------------------------------------
def test_rejects():
    print("test_rejects (garbage 400s, existing keys keep reject semantics)")
    previous = _snapshot()
    for key in sorted(srv._STORAGE_TUNING_KEYS):
        try:
            srv._set_runtime_tuning({key: "garbage"})
            check(False, f"{key} accepts garbage")
        except ValueError:
            check(True, f"{key} rejects a non-integer with ValueError")
    try:
        srv._set_runtime_tuning({"thinking_decode_eval_every": 99})
        check(False, "existing keys clamp instead of reject")
    except ValueError:
        check(True, "existing keys still REJECT out-of-range (no clamping)")
    changed = srv._set_runtime_tuning({"unknown_key_xyz": 1})
    check(changed == {}, "unknown keys are ignored (existing endpoint semantics)")
    check(_snapshot() == previous, "rejected requests change nothing")


# --------------------------------------------------------------------------
def test_capture_module_absent_guard():
    print("test_capture_module_absent_guard (golden tree has no m3_capture)")
    previous = _snapshot()
    real = srv._capture_module
    srv._capture_module = lambda: None
    try:
        try:
            srv._set_runtime_tuning({"capture_max_request_bytes": 100 * MIB})
            check(False, "capture tunable accepted without m3_capture")
        except ValueError as e:
            check("not deployed" in str(e),
                  "capture tunable without m3_capture -> clear ValueError")
        changed = srv._set_runtime_tuning({"prompt_cache_ssd_max_bytes": 120 * GIB})
        check(changed.get("prompt_cache_ssd_max_bytes") == 120 * GIB,
              "ssd tunable still works without m3_capture")
        status = srv._capture_corpus_status()
        check(status == {"deployed": False},
              "health capture status reports not deployed")
    finally:
        srv._capture_module = real
        _restore(previous)


# --------------------------------------------------------------------------
def test_capture_enforcement_reads_live():
    print("test_capture_enforcement_reads_live (per-flush / per-finalize)")
    import mlx.core as mx
    D = 512  # width is irrelevant to the cap logic
    previous = _snapshot()
    try:
        # Shrink the per-request cap live, mid-request: the NEXT flush caps.
        srv._set_runtime_tuning({"capture_max_request_bytes": 50 * MIB})
        m3_capture._DISABLED["value"] = False
        m3_capture._DIR_BYTES["value"] = None
        m3_capture.begin_request()
        check(m3_capture.is_capturing(), "request armed")
        m3_capture._SETTINGS["max_request_bytes"] = 1 << 10  # emulate live shrink
        m3_capture.push(mx.ones((1, 64, D), dtype=mx.float32))
        check(not m3_capture.is_capturing(),
              "flush re-reads the live cap and marks the request capped")
        m3_capture.abort_request()

        # Corpus status reflects live caps + usage estimate.
        srv._set_runtime_tuning({"capture_max_total_bytes": 20 * GIB})
        status = srv._capture_corpus_status()
        check(status.get("deployed") is True and status.get("armed") is True,
              "capture status deployed+armed under test env")
        check(status.get("max_total_bytes") == 20 * GIB,
              "capture status reports the live-tuned total cap")
        check(isinstance(status.get("total_bytes"), int),
              "capture status reports corpus usage bytes")
    finally:
        _restore(previous)
        m3_capture._WARNED["req_cap"] = False


# --------------------------------------------------------------------------
def test_broadcast_excludes_storage_keys():
    print("test_broadcast_excludes_storage_keys (rank0-local by design)")
    rt = _snapshot()
    wire = {k: v for k, v in rt.items() if k not in srv._STORAGE_TUNING_KEYS}
    check(all(k not in wire for k in srv._STORAGE_TUNING_KEYS),
          "endpoint's broadcast filter drops all storage keys")
    check("prefill_step_size" in wire,
          "broadcast filter keeps the ordinary tuning keys")


# --------------------------------------------------------------------------
def main():
    try:
        test_boot_seeding()
        test_apply_and_clamp()
        test_rejects()
        test_capture_module_absent_guard()
        test_capture_enforcement_reads_live()
        test_broadcast_excludes_storage_keys()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): " + "; ".join(_FAILS))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
