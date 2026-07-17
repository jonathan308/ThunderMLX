#!/usr/bin/env python3
"""Deterministic regression checks for generation-lock handoff recovery."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sharded_server import _should_recover_generation_lock  # noqa: E402


def decision(**overrides):
    values = {
        "lock_locked": True,
        "active_present": False,
        "releasing_present": False,
        "owner_kind": None,
        "owner_age": None,
        "transition_age": 60.0,
    }
    values.update(overrides)
    return _should_recover_generation_lock(**values)


def main():
    checks = {
        "live_active_request_is_never_recovered": not decision(
            active_present=True,
            owner_kind="request",
            owner_age=600.0,
        ),
        "release_handoff_is_never_recovered": not decision(
            releasing_present=True,
        ),
        "fresh_ownerless_handoff_is_protected": not decision(
            transition_age=0.001,
        ),
        "fresh_request_owner_is_protected": not decision(
            owner_kind="request",
            owner_age=0.001,
        ),
        "stale_ownerless_lock_is_recovered": decision(
            transition_age=60.0,
        ),
        "stale_request_owner_is_recovered": decision(
            owner_kind="request",
            owner_age=60.0,
        ),
        "unlocked_state_is_ignored": not decision(
            lock_locked=False,
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    print(json.dumps({"ok": not failed, "checks": checks, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
