#!/usr/bin/env python3
"""Offline regression checks for large-context cache safety policies."""

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sharded_server as server


def check_autosave_delta_policy():
    holder_before = dict(server._prompt_cache_holder)
    anchors_before = server._prompt_cache_ssd_autosave_anchors.copy()
    enabled_before = server.PROMPT_CACHE_SSD_ENABLED
    autosave_before = server.PROMPT_CACHE_SSD_AUTO_SAVE
    min_tokens_before = server.PROMPT_CACHE_SSD_MIN_TOKENS
    delta_before = server.PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS
    try:
        server.PROMPT_CACHE_SSD_ENABLED = True
        server.PROMPT_CACHE_SSD_AUTO_SAVE = True
        server.PROMPT_CACHE_SSD_MIN_TOKENS = 4
        server.PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS = 8
        server._prompt_cache_ssd_autosave_anchors.clear()
        server._prompt_cache_holder.update({
            "session_id": "cache-policy-smoke",
            "session_source": "probe",
            "token_ids": list(range(16)),
        })

        due, reason = server._prompt_cache_ssd_autosave_due_unlocked(None, None)
        assert due and reason == "first_checkpoint", (due, reason)
        key = server._prompt_cache_session_key("cache-policy-smoke", "probe")
        server._prompt_cache_ssd_record_autosave_anchor_unlocked(
            key, list(range(16)), "runtime"
        )

        due, reason = server._prompt_cache_ssd_autosave_due_unlocked(None, None)
        assert not due and reason == "unchanged", (due, reason)

        server._prompt_cache_holder["token_ids"] = list(range(23))
        due, reason = server._prompt_cache_ssd_autosave_due_unlocked(None, None)
        assert not due and reason == "delta_below_threshold:7", (due, reason)

        server._prompt_cache_holder["token_ids"] = list(range(24))
        due, reason = server._prompt_cache_ssd_autosave_due_unlocked(None, None)
        assert due and reason == "delta_threshold_reached:8", (due, reason)

        server._prompt_cache_holder["token_ids"] = list(range(12))
        due, reason = server._prompt_cache_ssd_autosave_due_unlocked(None, None)
        assert due and reason == "cache_rewound", (due, reason)

        changed = list(range(16))
        changed[-1] = 999
        server._prompt_cache_holder["token_ids"] = changed
        due, reason = server._prompt_cache_ssd_autosave_due_unlocked(None, None)
        assert due and reason == "same_length_branch_changed", (due, reason)
    finally:
        server._prompt_cache_holder.clear()
        server._prompt_cache_holder.update(holder_before)
        server._prompt_cache_ssd_autosave_anchors.clear()
        server._prompt_cache_ssd_autosave_anchors.update(anchors_before)
        server.PROMPT_CACHE_SSD_ENABLED = enabled_before
        server.PROMPT_CACHE_SSD_AUTO_SAVE = autosave_before
        server.PROMPT_CACHE_SSD_MIN_TOKENS = min_tokens_before
        server.PROMPT_CACHE_SSD_AUTO_SAVE_MIN_DELTA_TOKENS = delta_before


def check_large_visible_prewarm_skips_before_generation():
    event_before = server._prompt_cache_holder.get("last_event")
    session_map_max_before = server.PROMPT_CACHE_SESSION_MAP_MAX
    prompt_cache_enabled_before = server.PROMPT_CACHE_ENABLED
    try:
        server.PROMPT_CACHE_SESSION_MAP_MAX = 0
        server.PROMPT_CACHE_ENABLED = True
        limit = server.VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS
        assert limit > 0, limit
        ok = server._prewarm_prompt_cache(
            None,
            None,
            "unused",
            [0] * (limit + 1),
            reason="cache-policy-smoke",
        )
        assert not ok
        event = server._prompt_cache_holder.get("last_event") or {}
        assert event.get("action") == "prewarm_skipped_context_limit", event
        assert event.get("prompt_tokens") == limit + 1, event
    finally:
        server._prompt_cache_holder["last_event"] = event_before
        server.PROMPT_CACHE_SESSION_MAP_MAX = session_map_max_before
        server.PROMPT_CACHE_ENABLED = prompt_cache_enabled_before


def check_stop_boundaries_cover_phase_transition():
    preparing = server._stop_boundaries_from_active({
        "tokens_emitted": 0,
        "prefill_processed_tokens": 0,
        "prefill_total_tokens": 0,
    })
    assert preparing == {
        "prefill_stop_at_tokens": None,
        "decode_stop_at_tokens": 16,
    }, preparing

    prefilling = server._stop_boundaries_from_active({
        "tokens_emitted": 0,
        "prefill_processed_tokens": 40960,
        "prefill_total_tokens": 100000,
    })
    assert prefilling["prefill_stop_at_tokens"] > 40960, prefilling
    assert prefilling["decode_stop_at_tokens"] == 16, prefilling

    decoding = server._stop_boundaries_from_active({
        "tokens_emitted": 823,
        "prefill_processed_tokens": 100000,
        "prefill_total_tokens": 100000,
    })
    assert decoding["decode_stop_at_tokens"] == 839, decoding
    payload = server._prefill_stop_payload(
        "probe",
        decoding["prefill_stop_at_tokens"],
        "any",
        "nonce",
        decoding["decode_stop_at_tokens"],
    )
    assert payload["phase"] == "any", payload
    assert payload["decode_stop_at_tokens"] == 839, payload
    assert payload["nonce"] == "nonce", payload


def main():
    check_autosave_delta_policy()
    check_large_visible_prewarm_skips_before_generation()
    check_stop_boundaries_cover_phase_transition()
    print("PASS: cache policy smoke")


if __name__ == "__main__":
    main()
