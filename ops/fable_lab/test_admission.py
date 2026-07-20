#!/usr/bin/env python3
"""Offline unit tests for the prefill admission decision logic.

No cluster, no MLX — pure logic with injected memory numbers.
  ~/mlx-vlm064-env/bin/python3.14 ops/fable_lab/test_admission.py
"""
import importlib.util, os, sys

_MOD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "m3_prefill_admission.py")
_spec = importlib.util.spec_from_file_location("m3_prefill_admission", _MOD)
adm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adm)

GiB = 1024 ** 3
fails = 0


def check(name, cond):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    fails += (not cond)


# --- admission_deficit_bytes ---
# tonight's wedge: 205GB wired, 254GB limit, 62k-token prefill.
d = adm.admission_deficit_bytes(
    prompt_tokens=62000, current_wired_bytes=205 * GiB,
    wired_limit_bytes=254 * GiB, kv_bytes_per_token=90_000,
    safety_fraction=0.92, activation_reserve_bytes=8 * GiB)
# need = 62000*90000 + 8GiB ≈ 5.2GiB + 8GiB = 13.2GiB; budget = 233.7GiB;
# projected = 205 + 13.2 = 218.2 -> fits at 254 ceiling (deficit < 0). Good:
# the raised ceiling ALONE prevents tonight's exact case.
check("wedge case at 254 ceiling fits (headroom restored the margin)", d < 0)

# same wedge at the OLD 222 ceiling: should demand eviction.
d222 = adm.admission_deficit_bytes(
    prompt_tokens=62000, current_wired_bytes=205 * GiB,
    wired_limit_bytes=222 * GiB, kv_bytes_per_token=90_000,
    safety_fraction=0.92, activation_reserve_bytes=8 * GiB)
check("wedge case at OLD 222 ceiling demands eviction (deficit>0)", d222 > 0)

# a genuinely huge prefill near a full pool still triggers at 254.
dbig = adm.admission_deficit_bytes(
    prompt_tokens=250000, current_wired_bytes=225 * GiB,
    wired_limit_bytes=254 * GiB, kv_bytes_per_token=90_000)
check("huge 250k prefill on a 225GB-full pool demands eviction", dbig > 0)

# roomy case: small pool, small prompt -> fits.
droom = adm.admission_deficit_bytes(
    prompt_tokens=20000, current_wired_bytes=120 * GiB,
    wired_limit_bytes=254 * GiB, kv_bytes_per_token=90_000)
check("roomy case fits (no eviction)", droom < 0)

# unknown ceiling -> fail-open (deficit 0).
check("unknown ceiling fails open", adm.admission_deficit_bytes(1000, 1*GiB, 0) == 0)

# --- should_guard gating ---
adm.ENABLED = True
check("small prompt below MIN not guarded", not adm.should_guard(1000))
check("large prompt guarded when enabled", adm.should_guard(50000))
adm.ENABLED = False
check("disabled -> never guarded", not adm.should_guard(50000))
adm.ENABLED = True

# --- plan_eviction ---
items = [{"label": "sessA", "bytes": 3 * GiB}, {"label": "sessB", "bytes": 9 * GiB},
         {"label": "sessC", "bytes": 1 * GiB}]
chosen, short = adm.plan_eviction(8 * GiB, items)
check("plan picks largest-first to cover deficit", [c["label"] for c in chosen] == ["sessB"])
check("plan reports zero shortfall when covered", short == 0)
chosen2, short2 = adm.plan_eviction(20 * GiB, items)
check("plan takes all when deficit exceeds total", len(chosen2) == 3)
check("plan reports residual shortfall honestly", abs(short2 - 7 * GiB) < GiB)
check("zero deficit picks nothing", adm.plan_eviction(0, items) == ([], 0))

# --- run_admission orchestration (injected callbacks) ---
state = {"wired": 205 * GiB}
dropped = []
def read_wired(): return state["wired"]
def read_limit(): return 222 * GiB  # old ceiling to force the ladder
def trim_pool():
    freed = 2 * GiB; state["wired"] -= freed; return freed
def list_idle():
    def mk(lbl, gib):
        def drop(): state["wired"] -= gib * GiB; dropped.append(lbl)
        return {"label": lbl, "bytes": gib * GiB, "drop": drop}
    return [mk("idleA", 6), mk("idleB", 12)]
import logging
info = adm.run_admission(62000, read_wired, read_limit, trim_pool, list_idle,
                         kv_bytes_per_token=90_000, logger=logging.getLogger("t"))
check("run_admission guarded the big prefill", info["guarded"])
check("run_admission trimmed pool then evicted", any(a["step"] == "evict_idle" for a in info["actions"]))
check("run_admission ended fitting", info["fits"])
check("run_admission freed wired materially", state["wired"] < 205 * GiB)

# small prompt -> untouched
info2 = adm.run_admission(1000, read_wired, read_limit, trim_pool, list_idle)
check("small prompt bypasses guard entirely", not info2["guarded"])

print(f"\n{'ALL PASS' if fails == 0 else str(fails)+' FAILED'}")
sys.exit(1 if fails else 0)
