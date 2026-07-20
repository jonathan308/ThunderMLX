"""Pure metadata helpers for exact multimodal prompt-cache reuse.

The language KV cache may be reused after an image only when the image bytes,
preprocessing contract, image order, and expanded media-token layout are all
identical. This module deliberately stores hashes and shape metadata, never
raw image bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass
    return repr(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def source_manifest(image_paths: Any) -> dict[str, Any]:
    """Return an order-sensitive, path-independent image-byte manifest."""
    paths = image_paths if isinstance(image_paths, list) else [image_paths]
    items = []
    for path in paths:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"image source is not a local path: {type(path)!r}")
        digest, size = _sha256_file(os.fspath(path))
        items.append({"sha256": digest, "bytes": size})
    manifest = {
        "schema": SCHEMA_VERSION,
        "image_count": len(items),
        "items": items,
    }
    manifest["hash"] = stable_hash(manifest)
    return manifest


def processor_fingerprint(processor: Any, model_config: Any) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    image_attrs = {}
    for name in (
        "size",
        "min_pixels",
        "max_pixels",
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "resample",
        "rescale_factor",
        "image_mean",
        "image_std",
    ):
        if image_processor is not None and hasattr(image_processor, name):
            image_attrs[name] = _json_value(getattr(image_processor, name))
    fields = {
        "schema": SCHEMA_VERSION,
        "processor_class": (
            processor.__class__.__module__ + "." + processor.__class__.__name__
        ),
        "image_processor_class": (
            image_processor.__class__.__module__
            + "."
            + image_processor.__class__.__name__
            if image_processor is not None
            else None
        ),
        "tokenizer_class": (
            tokenizer.__class__.__module__ + "." + tokenizer.__class__.__name__
        ),
        "image_processor": image_attrs,
        "image_token_id": getattr(model_config, "image_token_id", None),
        "image_token_index": getattr(model_config, "image_token_index", None),
        "vision_start_token_id": getattr(processor, "vision_start_token_id", None),
        "vision_end_token_id": getattr(processor, "vision_end_token_id", None),
        "model_type": getattr(model_config, "model_type", None),
        "vision_config": _json_value(
            {
                name: getattr(getattr(model_config, "vision_config", None), name, None)
                for name in (
                    "image_size",
                    "patch_size",
                    "temporal_patch_size",
                    "spatial_merge_size",
                )
            }
        ),
    }
    fields["hash"] = stable_hash(fields)
    return fields


def media_token_ids(model_config: Any) -> tuple[int, ...]:
    values = set()
    for name in (
        "image_token_id",
        "image_token_index",
        "video_token_id",
        "video_token_index",
    ):
        value = getattr(model_config, name, None)
        if value is not None:
            values.add(int(value))
    return tuple(sorted(values))


def media_token_spans(
    token_ids: Sequence[int], media_ids: Iterable[int]
) -> tuple[tuple[int, int], ...]:
    wanted = {int(token_id) for token_id in media_ids}
    spans: list[tuple[int, int]] = []
    start = None
    for index, token_id in enumerate(token_ids):
        if int(token_id) in wanted:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(token_ids)))
    return tuple(spans)


def media_safe_prefix_min(token_ids: Sequence[int], media_ids: Iterable[int]) -> int:
    spans = media_token_spans(token_ids, media_ids)
    return max((end for _start, end in spans), default=0)


def prefix_is_media_safe(
    token_ids: Sequence[int], prefix_len: int, media_ids: Iterable[int]
) -> bool:
    return int(prefix_len) >= media_safe_prefix_min(token_ids, media_ids)


def build_descriptor(
    *,
    source: dict[str, Any],
    processor: dict[str, Any],
    token_ids: Sequence[int],
    media_ids: Iterable[int],
    image_grid_thw: Any,
    pixel_values_shape: Sequence[int] | None,
    pixel_values_dtype: str | None,
) -> dict[str, Any]:
    ids = [int(token_id) for token_id in token_ids]
    media_ids = tuple(int(token_id) for token_id in media_ids)
    spans = media_token_spans(ids, media_ids)
    grids = _json_value(image_grid_thw)
    cache_identity = {
        "schema": SCHEMA_VERSION,
        "source_hash": source.get("hash"),
        "processor_hash": processor.get("hash"),
        "image_count": int(source.get("image_count") or 0),
        "image_grid_thw": grids,
        "media_token_ids": list(media_ids),
        "media_token_counts": [end - start for start, end in spans],
        "pixel_values_shape": list(pixel_values_shape or []),
        "pixel_values_dtype": pixel_values_dtype,
    }
    fingerprint = stable_hash(cache_identity)
    request_plan = {
        "fingerprint": fingerprint,
        "expanded_prompt_tokens": len(ids),
        "expanded_prompt_hash": stable_hash(ids),
        "media_spans": [list(span) for span in spans],
        "media_safe_prefix_min": media_safe_prefix_min(ids, media_ids),
    }
    request_plan["plan_hash"] = stable_hash(request_plan)
    return {
        "schema": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_short": fingerprint[:16],
        "source_hash": source.get("hash"),
        "processor_hash": processor.get("hash"),
        "image_count": cache_identity["image_count"],
        "image_grid_thw": grids,
        "media_token_ids": list(media_ids),
        "media_token_count": sum(end - start for start, end in spans),
        **request_plan,
    }


def prefix_manifest_hash(items: Sequence[dict], count: int) -> str:
    """Hash the first `count` manifest items exactly as source_manifest would.

    Reconstructs the manifest shape for a PREFIX of the image list so an old
    descriptor's aggregate source_hash can be verified against the leading
    images of a new, longer request — the append-aware reuse check. Order
    sensitivity is inherited from the item list.
    """
    manifest = {
        "schema": SCHEMA_VERSION,
        "image_count": int(count),
        "items": [
            {"sha256": item.get("sha256"), "bytes": item.get("bytes")}
            for item in list(items)[: int(count)]
        ],
    }
    return stable_hash(manifest)


def cache_session_id(session_id: str | None, fingerprint: str) -> str:
    base = str(session_id or "__default__")
    return f"{base}:mm{SCHEMA_VERSION}:{fingerprint}"


def consensus_vector(descriptor: dict[str, Any] | None) -> tuple[int, ...]:
    if not descriptor:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    plan_hash = str(descriptor.get("plan_hash") or "").ljust(32, "0")
    return (
        1,
        int(descriptor.get("expanded_prompt_tokens") or 0),
        int(descriptor.get("media_safe_prefix_min") or 0),
        int(descriptor.get("image_count") or 0),
        int(descriptor.get("media_token_count") or 0),
        int(plan_hash[0:8], 16),
        int(plan_hash[8:16], 16),
        int(plan_hash[16:24], 16),
        int(plan_hash[24:32], 16),
    )
