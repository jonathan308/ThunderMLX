#!/usr/bin/env python3
"""Offline server-policy checks for multimodal KV reuse and cold fallback."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sharded_server as server  # noqa: E402


class FakeCache:
    def __init__(self, offset=0):
        self.offset = int(offset)

    def is_trimmable(self):
        return True

    def trim(self, count):
        count = int(count)
        if count < 0 or count > self.offset:
            return 0
        self.offset -= count
        return count


class FakeTokenizer:
    @staticmethod
    def decode(token_ids):
        return " ".join(str(value) for value in token_ids)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def seed_holder(token_ids, fingerprint, session_id="session:mm1:fp"):
    holder = server._prompt_cache_holder
    holder.update({
        "cache": [FakeCache(len(token_ids)), FakeCache(len(token_ids))],
        "token_ids": list(token_ids),
        "cache_len": len(token_ids),
        "last_input_tokens": len(token_ids),
        "last_generated_tokens": 0,
        "last_exact_generated_ids": False,
        "last_suffix_ids": None,
        "prompt": None,
        "session_id": session_id,
        "session_source": "metadata.session_id",
        "multimodal_fingerprint": fingerprint,
        "multimodal_descriptor": {"fingerprint": fingerprint},
        "last_access_at": time.time(),
        "in_use": False,
    })
    server._prompt_cache_resident_slots.clear()


def main():
    server.PROMPT_CACHE_ENABLED = True
    server.PROMPT_CACHE_SSD_ENABLED = False
    server.PROMPT_CACHE_SSD_RESTORE_ENABLED = False
    server.PROMPT_CACHE_RESIDENT_SLOTS = 1
    server.PROMPT_CACHE_MIN_REUSE = 1
    server.PROMPT_CACHE_PROTECT_LARGE_ENABLED = False
    server.PROMPT_CACHE_SESSION_PROTECT_ENABLED = False
    server.PROMPT_CACHE_SMALL_THINKING_REBUILD_MAX_TOKENS = 0
    server._effective_prompt_cache_min_suffix_tokens = lambda *args, **kwargs: 0
    server._runtime_prompt_cache_reuse_bucket_tokens = lambda: 0

    rendered_media = server._join_model_facing_content_parts(
        [
            ("text", "Inspect this image, then use the tool."),
            ("image", "ignored-source-a"),
            ("text", "Record the result."),
            ("image", "ignored-source-b"),
        ],
        "]<]image[>[",
    )
    check(
        rendered_media
        == ("]<]image[>[" * 2)
        + "Inspect this image, then use the tool.\nRecord the result.",
        "cached multimodal rendering must preserve MiniMax image-first semantics",
    )

    def fresh_cache(_model):
        holder = server._prompt_cache_holder
        holder["cache"] = [FakeCache(), FakeCache()]
        server._clear_prompt_cache_key_state_unlocked(holder)
        return holder["cache"]

    server._get_or_build_prompt_cache_unlocked = fresh_cache
    processor = SimpleNamespace(tokenizer=FakeTokenizer())
    model = SimpleNamespace(language_model=object())

    legacy_runtime = {
        "schema": 3,
        "model": {"hash": "model-a"},
        "processor": {"hash": "processor-a"},
        "cache_impl": {
            "hash": "legacy-hash",
            "files": [
                {"path": "sharded_server.py", "sha256": "server-old"},
                {"path": "m3_multimodal_cache.py", "sha256": "cache-a"},
            ],
        },
        "cache_classes": ["MiniMaxM3KVCache"],
    }
    current_runtime = json.loads(json.dumps(legacy_runtime))
    current_runtime["hash"] = "current-hash"
    current_runtime["cache_impl"]["hash"] = "current-impl-hash"
    current_runtime["cache_impl"]["abi_version"] = 1
    current_runtime["cache_impl"]["files"][0]["sha256"] = "server-new"
    check(
        server._prompt_cache_ssd_runtime_fingerprints_compatible(
            legacy_runtime, current_runtime
        ),
        "unrelated server edits must not invalidate a stable cache ABI",
    )
    incompatible_runtime = json.loads(json.dumps(current_runtime))
    incompatible_runtime["cache_impl"]["files"][1]["sha256"] = "cache-b"
    check(
        not server._prompt_cache_ssd_runtime_fingerprints_compatible(
            legacy_runtime, incompatible_runtime
        ),
        "cache implementation changes must invalidate durable KV",
    )
    incompatible_runtime = json.loads(json.dumps(current_runtime))
    incompatible_runtime["cache_impl"]["abi_version"] = 2
    check(
        not server._prompt_cache_ssd_runtime_fingerprints_compatible(
            legacy_runtime, incompatible_runtime
        ),
        "cache ABI changes must invalidate durable KV",
    )

    old_ids = [1, 2, 99, 99, 99, 3, 4, 5]
    new_ids = old_ids + [6, 7]
    seed_holder(old_ids, "same-fingerprint")
    _suffix_prompt, cache = server._prepare_cached_prompt(
        model,
        processor,
        "unused",
        new_ids,
        session_id="session:mm1:fp",
        session_source="metadata.session_id",
        thinking_mode="disabled",
        multimodal_fingerprint="same-fingerprint",
        minimum_safe_reuse=5,
    )
    check(cache is not None, "safe text suffix should reuse the image KV")
    check(server._prompt_cache_last_suffix_ids() == [6, 7],
          "only the new text suffix should be sent")
    check(server._prompt_cache_holder["last_prepare_event"]["action"] == "reuse",
          "safe image continuation should be reported as reuse")

    seed_holder(old_ids, "same-fingerprint")
    unsafe_ids = [1, 2, 88, 88, 3, 4]
    _prompt, cache = server._prepare_cached_prompt(
        model,
        processor,
        "full-prompt",
        unsafe_ids,
        session_id="session:mm1:fp",
        session_source="metadata.session_id",
        thinking_mode="disabled",
        multimodal_fingerprint="same-fingerprint",
        minimum_safe_reuse=5,
    )
    check(cache is not None, "unsafe media overlap should have a fresh cache")
    check(
        server._prompt_cache_holder["last_prepare_event"]["action"]
        == "multimodal_media_boundary_rebuild",
        "reuse before the media boundary must cold-rebuild",
    )

    seed_holder(old_ids, "old-fingerprint")
    server.PROMPT_CACHE_RESIDENT_SLOTS = 2
    _prompt, cache = server._prepare_cached_prompt(
        model,
        processor,
        "full-prompt",
        new_ids,
        session_id="session:mm1:new",
        session_source="metadata.session_id",
        thinking_mode="disabled",
        multimodal_fingerprint="new-fingerprint",
        minimum_safe_reuse=5,
    )
    check(cache is not None, "changed image should receive a fresh cache")
    check(
        server._prompt_cache_holder["last_prepare_event"]["action"]
        == "multimodal_fingerprint_rebuild",
        "changed image bytes must not reuse the previous image KV",
    )

    # Switching away from an image-bearing cache used to cold-rebuild before
    # consulting the requested text session's durable checkpoint. Verify that
    # the exact target session can now restore while the image KV stays isolated.
    target_ids = [10, 11, 12, 13, 14, 15]
    target_session = "session:text:target"
    seed_holder(old_ids, "old-fingerprint", session_id="session:image:source")
    server.PROMPT_CACHE_SSD_ENABLED = True
    server.PROMPT_CACHE_SSD_RESTORE_ENABLED = True
    original_restore = server._prompt_cache_ssd_maybe_restore_unlocked

    def restore_target(_model, _processor, token_ids, **kwargs):
        check(kwargs.get("session_id") == target_session,
              "durable restore must use the requested session identity")
        check(kwargs.get("session_source") == "metadata.session_id",
              "durable restore must preserve the session source")
        check(kwargs.get("multimodal_fingerprint") is None,
              "a text session must not inherit the prior image fingerprint")
        holder = server._prompt_cache_holder
        holder.update({
            "cache": [FakeCache(len(target_ids)), FakeCache(len(target_ids))],
            "token_ids": list(target_ids),
            "cache_len": len(target_ids),
            "last_input_tokens": len(target_ids),
            "last_generated_tokens": 0,
            "last_exact_generated_ids": False,
            "last_suffix_ids": None,
            "prompt": None,
            "session_id": target_session,
            "session_source": "metadata.session_id",
            "multimodal_fingerprint": None,
            "multimodal_descriptor": None,
        })
        return {"restored_ssd": True}

    server._prompt_cache_ssd_maybe_restore_unlocked = restore_target
    try:
        _suffix_prompt, cache = server._prepare_cached_prompt(
            model,
            processor,
            "unused",
            target_ids + [16, 17],
            session_id=target_session,
            session_source="metadata.session_id",
            thinking_mode="disabled",
            multimodal_fingerprint=None,
            minimum_safe_reuse=0,
        )
    finally:
        server._prompt_cache_ssd_maybe_restore_unlocked = original_restore
        server.PROMPT_CACHE_SSD_ENABLED = False
        server.PROMPT_CACHE_SSD_RESTORE_ENABLED = False
    check(cache is not None, "text session should receive its restored durable KV")
    check(server._prompt_cache_holder["session_id"] == target_session,
          "restored text KV must retain the requested session identity")
    check(server._prompt_cache_last_suffix_ids() == [16, 17],
          "restored text KV should process only the new suffix")
    check(server._prompt_cache_holder["multimodal_fingerprint"] is None,
          "restored text KV must not leak the previous image fingerprint")
    check(
        server._prompt_cache_holder["last_prepare_event"].get("restored_ssd_cache"),
        "the final request telemetry must retain the SSD restore decision",
    )

    context = {
        "token_ids": new_ids,
        "input_ids": object(),
        "pixel_values": object(),
        "mask": object(),
        "data_kwargs": {"image_grid_thw": object()},
        "media_token_ids": (99,),
    }
    kwargs = {}
    reused = server._apply_multimodal_generation_inputs(
        kwargs,
        context,
        [FakeCache(8)],
        [6, 7],
    )
    check(reused == 8, "physical reuse count should match the cached prefix")
    check(context["pixel_values"] is None,
          "a cache hit should release duplicate pixel tensors before decode")
    check(context["physical_cache_hit"],
          "telemetry must identify the physical image-cache hit")

    print(json.dumps({
        "ok": True,
        "safe_suffix_reuse": True,
        "media_boundary_cold_fallback": True,
        "changed_image_cold_fallback": True,
        "image_to_text_ssd_restore": True,
        "cross_session_fingerprint_isolation": True,
        "stable_runtime_fingerprint": True,
        "hit_releases_pixel_tensor": True,
        "native_image_first_rendering": True,
    }, indent=2))


if __name__ == "__main__":
    main()
