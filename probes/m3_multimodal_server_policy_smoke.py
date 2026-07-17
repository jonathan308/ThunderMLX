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
        "hit_releases_pixel_tensor": True,
        "native_image_first_rendering": True,
    }, indent=2))


if __name__ == "__main__":
    main()
