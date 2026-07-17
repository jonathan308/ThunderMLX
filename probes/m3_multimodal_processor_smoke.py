#!/usr/bin/env python3
"""Verify MiniMax keeps an anchored image prefix stable across chat turns.

This probe loads only the processor and config. It does not load model weights
or contact either cluster rank.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import (
    load_config,
    load_image_processor,
    load_processor,
    prepare_inputs,
)

import m3_multimodal_cache as mm_cache
import sharded_server as server


def _common_prefix(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _render(processor, config, messages, *, thinking: bool) -> str:
    return apply_chat_template(
        processor,
        config,
        messages,
        add_generation_prompt=True,
        num_images=0,
        enable_thinking=thinking,
    )


def _expanded_ids(processor, config, prompt: str, image: str) -> list[int]:
    inputs = prepare_inputs(
        processor,
        images=image,
        prompts=prompt,
        image_token_index=config.get("image_token_index"),
        add_special_tokens=True,
        padding=True,
        padding_side="left",
    )
    return [int(token) for token in inputs["input_ids"].flatten().tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "MLX_M3_MODEL",
            str(
                Path.home()
                / ".exo/models/mlx-community--MiniMax-M3-4bit"
            ),
        ),
    )
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image = str(Path(args.image).expanduser().resolve())
    if not Path(image).is_file():
        raise SystemExit(f"image not found: {image}")

    server.OMLX_MINIMAX_OVERLAY = True
    if not server._install_omlx_minimax_overlay():
        raise AssertionError("MiniMax MSA/processor overlay was not installed")
    model_path = Path(args.model).expanduser().resolve()
    config = load_config(model_path)
    processor = load_processor(model_path)
    image_processor = load_image_processor(model_path)
    if image_processor is not None:
        processor.image_processor = image_processor
    marker = server._image_prompt_marker(processor)
    first_messages = [
        {
            "role": "user",
            "content": f"{marker}\nDescribe this image precisely.",
        }
    ]
    followup_messages = [
        *first_messages,
        {"role": "assistant", "content": "It contains a visible scene."},
        {"role": "user", "content": "What detail did you notice first?"},
    ]

    config_object = type("Config", (), config)()
    media_ids = mm_cache.media_token_ids(config_object)
    if not media_ids:
        raise AssertionError("model config exposes no image token id")

    for thinking in (False, True):
        first = _expanded_ids(
            processor,
            config,
            _render(processor, config, first_messages, thinking=thinking),
            image,
        )
        followup = _expanded_ids(
            processor,
            config,
            _render(processor, config, followup_messages, thinking=thinking),
            image,
        )
        shared = _common_prefix(first, followup)
        safe_boundary = mm_cache.media_safe_prefix_min(first, media_ids)
        assert safe_boundary > 0
        assert shared >= safe_boundary, (shared, safe_boundary)
        assert shared >= len(first) - 1, (len(first), len(followup), shared)
        print(
            "PASS",
            "thinking" if thinking else "no-thinking",
            f"first={len(first)}",
            f"followup={len(followup)}",
            f"shared={shared}",
            f"media_safe={safe_boundary}",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
