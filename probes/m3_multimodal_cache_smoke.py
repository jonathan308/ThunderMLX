#!/usr/bin/env python3
"""Offline contract checks for exact multimodal prompt-cache identities."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import m3_multimodal_cache as mm  # noqa: E402


class MockTokenizer:
    pass


class MockImageProcessor:
    def __init__(self, patch_size=14):
        self.size = {"longest_edge": 2048}
        self.patch_size = patch_size
        self.temporal_patch_size = 2
        self.merge_size = 2
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]


class MockProcessor:
    def __init__(self, patch_size=14):
        self.tokenizer = MockTokenizer()
        self.image_processor = MockImageProcessor(patch_size)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    with tempfile.TemporaryDirectory(prefix="m3-mm-cache-") as tmp:
        root = Path(tmp)
        first = root / "first.png"
        copied = root / "renamed-anywhere.jpg"
        second = root / "second.png"
        changed = root / "changed.png"
        first.write_bytes(b"exact-image-one")
        copied.write_bytes(first.read_bytes())
        second.write_bytes(b"exact-image-two")
        changed.write_bytes(b"exact-image-one-changed")

        first_manifest = mm.source_manifest(str(first))
        copied_manifest = mm.source_manifest(str(copied))
        changed_manifest = mm.source_manifest(str(changed))
        ordered = mm.source_manifest([str(first), str(second)])
        reversed_order = mm.source_manifest([str(second), str(first)])

        check(first_manifest["hash"] == copied_manifest["hash"],
              "same bytes at a new path must keep the same identity")
        check(first_manifest["hash"] != changed_manifest["hash"],
              "changed bytes must invalidate the identity")
        check(ordered["hash"] != reversed_order["hash"],
              "multi-image order must be part of the identity")
        check(str(root) not in json.dumps(first_manifest),
              "cache metadata must not contain the source path")

        config = SimpleNamespace(
            model_type="minimax_m3_vl",
            image_token_id=99,
            image_token_index=99,
            video_token_id=100,
            video_token_index=100,
            vision_config=SimpleNamespace(
                image_size=2048,
                patch_size=14,
                temporal_patch_size=2,
                spatial_merge_size=2,
            ),
        )
        processor_a = mm.processor_fingerprint(MockProcessor(14), config)
        processor_b = mm.processor_fingerprint(MockProcessor(16), config)
        check(processor_a["hash"] != processor_b["hash"],
              "processor geometry changes must invalidate the identity")

        token_ids = [1, 2, 99, 99, 99, 3, 4, 5]
        media_ids = mm.media_token_ids(config)
        check(mm.media_token_spans(token_ids, media_ids) == ((2, 5),),
              "media span must cover every expanded image token")
        check(mm.media_safe_prefix_min(token_ids, media_ids) == 5,
              "safe reuse must begin after the final media token")
        check(not mm.prefix_is_media_safe(token_ids, 4, media_ids),
              "a prefix ending inside media tokens must be rejected")
        check(mm.prefix_is_media_safe(token_ids, 5, media_ids),
              "the first text-only suffix boundary must be accepted")

        descriptor = mm.build_descriptor(
            source=first_manifest,
            processor=processor_a,
            token_ids=token_ids,
            media_ids=media_ids,
            image_grid_thw=[[1, 4, 4]],
            pixel_values_shape=[16, 1176],
            pixel_values_dtype="float32",
        )
        prompt_changed = mm.build_descriptor(
            source=first_manifest,
            processor=processor_a,
            token_ids=token_ids + [6],
            media_ids=media_ids,
            image_grid_thw=[[1, 4, 4]],
            pixel_values_shape=[16, 1176],
            pixel_values_dtype="float32",
        )
        image_changed = mm.build_descriptor(
            source=changed_manifest,
            processor=processor_a,
            token_ids=token_ids,
            media_ids=media_ids,
            image_grid_thw=[[1, 4, 4]],
            pixel_values_shape=[16, 1176],
            pixel_values_dtype="float32",
        )
        check(descriptor["fingerprint"] == prompt_changed["fingerprint"],
              "text suffix changes must not alter the image identity")
        check(descriptor["plan_hash"] != prompt_changed["plan_hash"],
              "the expanded prompt plan must still detect text changes")
        check(descriptor["fingerprint"] != image_changed["fingerprint"],
              "changed image bytes must alter the cache identity")
        check(
            mm.cache_session_id("chat-1", descriptor["fingerprint"])
            != mm.cache_session_id("chat-1", image_changed["fingerprint"]),
            "changed images must not share a resident/SSD session key",
        )
        check(mm.consensus_vector(descriptor) != mm.consensus_vector(prompt_changed),
              "rank consensus must cover the complete expanded prompt plan")
        check(len(mm.consensus_vector(descriptor)) == 9,
              "consensus vector schema changed unexpectedly")

        print(json.dumps({
            "ok": True,
            "schema": mm.SCHEMA_VERSION,
            "same_bytes_path_independent": True,
            "changed_bytes_invalidated": True,
            "ordered_multi_image": True,
            "media_safe_prefix_min": descriptor["media_safe_prefix_min"],
            "fingerprint": descriptor["fingerprint_short"],
        }, indent=2))


if __name__ == "__main__":
    main()
