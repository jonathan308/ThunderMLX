#!/usr/bin/env python3
"""Offline tests for append-aware image cache reuse (v0.3.5 feature).

Exercises the real _multimodal_append_reuse_plan verdict and the mode-3 path
of _apply_multimodal_generation_inputs with synthetic descriptors/tensors.

  ~/mlx-vlm064-env/bin/python3.14 ops/fable_lab/test_image_append_cache.py
"""
import os
import sys

os.environ["MLX_M3_IMAGE_PROMPT_CACHE_APPEND"] = "1"

_LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LAB)
os.chdir(_LAB)

import mlx.core as mx  # noqa: E402
import sharded_server as srv  # noqa: E402
import m3_multimodal_cache as mm  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    fails += (not cond)


ITEMS = [
    {"sha256": "a" * 64, "bytes": 111},
    {"sha256": "b" * 64, "bytes": 222},
]
OLD_GRIDS = [[1, 4, 4]]
NEW_GRIDS = [[1, 4, 4], [1, 6, 6]]
MEDIA_ID = 99
OLD_IDS = [1, 2, 3] + [MEDIA_ID] * 4 + [4, 5, 6, 7]
NEW_IDS = OLD_IDS + [8, 9] + [MEDIA_ID] * 9 + [10, 11]


def old_desc():
    return {
        "image_count": 1,
        "processor_hash": "prochash",
        "source_hash": mm.prefix_manifest_hash(ITEMS, 1),
        "image_grid_thw": [list(g) for g in OLD_GRIDS],
        "media_safe_prefix_min": 7,  # end of the old media span
    }


def new_ctx(**over):
    ctx = {
        "descriptor": {
            "image_count": 2,
            "processor_hash": "prochash",
            "image_grid_thw": [list(g) for g in NEW_GRIDS],
        },
        "source_manifest": {"items": [dict(i) for i in ITEMS]},
        "token_ids": list(NEW_IDS),
        "media_token_ids": (MEDIA_ID,),
    }
    ctx.update(over)
    return ctx


# --- verdict: the clean append accepts ---
plan = srv._multimodal_append_reuse_plan(old_desc(), OLD_IDS, new_ctx())
check("clean append accepted", bool(plan))
check("floor = cached media region end", plan and plan["minimum_safe_reuse"] == 7)
check("pixel row start = prod(old grid) = 16",
      plan and plan["appended_pixel_row_start"] == 16)

# --- verdict rejections ---
d = old_desc(); d["processor_hash"] = "other"
check("processor change rejected",
      srv._multimodal_append_reuse_plan(d, OLD_IDS, new_ctx()) is None)
bad_items = [{"sha256": "X" * 64, "bytes": 111}, ITEMS[1]]
check("changed first image bytes rejected",
      srv._multimodal_append_reuse_plan(
          old_desc(), OLD_IDS, new_ctx(source_manifest={"items": bad_items})) is None)
bad_grids = new_ctx()
bad_grids["descriptor"]["image_grid_thw"] = [[1, 8, 8], [1, 6, 6]]
check("changed first image grid rejected",
      srv._multimodal_append_reuse_plan(old_desc(), OLD_IDS, bad_grids) is None)
same_count = new_ctx()
same_count["descriptor"]["image_count"] = 1
check("equal image count rejected",
      srv._multimodal_append_reuse_plan(old_desc(), OLD_IDS, same_count) is None)
check("diverged cached ids rejected",
      srv._multimodal_append_reuse_plan(
          old_desc(), [9] * len(OLD_IDS), new_ctx()) is None)
srv.IMAGE_PROMPT_CACHE_APPEND_ENABLED = False
check("env gate off -> always None",
      srv._multimodal_append_reuse_plan(old_desc(), OLD_IDS, new_ctx()) is None)
srv.IMAGE_PROMPT_CACHE_APPEND_ENABLED = True

# --- mode-3 apply: suffix + sliced pixels ---
ROWS, DIM = 16 + 36, 8
ctx = new_ctx(
    append_reuse=dict(plan),
    pixel_values=mx.arange(ROWS * DIM).reshape(ROWS, DIM),
    data_kwargs={"image_grid_thw": [list(g) for g in NEW_GRIDS]},
    input_ids=mx.array([NEW_IDS]),
    mask=None,
)
reuse_boundary = len(OLD_IDS)  # common prefix = full old sequence
suffix = NEW_IDS[reuse_boundary:]
gk = {}
reuse = srv._apply_multimodal_generation_inputs(gk, ctx, object(), suffix)
check("mode-3 reuse tokens = boundary", reuse == reuse_boundary)
check("suffix ids attached", gk["input_ids"].shape == (1, len(suffix)))
check("pixels sliced to appended image (36 rows)",
      gk["pixel_values"].shape == (36, DIM))
check("grids sliced to appended image", gk["image_grid_thw"] == [[1, 6, 6]])
check("first sliced row is image-2's first row",
      int(gk["pixel_values"][0][0]) == 16 * DIM)

# --- mode-3 floor violation raises (crash-don't-corrupt) ---
ctx2 = new_ctx(
    append_reuse=dict(plan),
    pixel_values=mx.zeros((ROWS, DIM)),
    data_kwargs={"image_grid_thw": [list(g) for g in NEW_GRIDS]},
)
try:
    srv._apply_multimodal_generation_inputs(
        {}, ctx2, object(), NEW_IDS[3:])  # boundary 3 < floor 7
    check("boundary below floor raises", False)
except RuntimeError:
    check("boundary below floor raises", True)

# --- non-append unsafe reuse still raises exactly as before ---
ctx3 = new_ctx(pixel_values=mx.zeros((ROWS, DIM)),
               data_kwargs={"image_grid_thw": [list(g) for g in NEW_GRIDS]})
try:
    srv._apply_multimodal_generation_inputs({}, ctx3, object(), NEW_IDS[3:])
    check("legacy media-boundary violation still raises", False)
except RuntimeError:
    check("legacy media-boundary violation still raises", True)

# --- text-suffix mode untouched (reuse past ALL media) ---
ctx4 = new_ctx(pixel_values=mx.zeros((ROWS, DIM)),
               data_kwargs={"image_grid_thw": [list(g) for g in NEW_GRIDS]})
safe_boundary = 3 + 4 + 2 + 9 + 3  # past the last media span
gk4 = {}
srv._apply_multimodal_generation_inputs(
    gk4, ctx4, object(), NEW_IDS[len(NEW_IDS) - 1:])
check("legacy text-suffix mode drops pixels entirely",
      "pixel_values" not in gk4 and ctx4["pixel_values"] is None)

print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
