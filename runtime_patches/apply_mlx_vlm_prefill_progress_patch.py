#!/usr/bin/env python3
"""Install the MiniMax-M3 serving prefill-progress hook into MLX-VLM.

The public MLX-VLM AR generator does chunked prefill internally but does not
surface progress until the first decoded token. For long distributed prompts
that makes a healthy prefill indistinguishable from a wedge. This patch adds an
optional ``prefill_progress_callback(processed_tokens, total_tokens)`` kwarg to
``mlx_vlm.generate.ar.generate_step`` and calls it after each prefill chunk.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys


MARKER = "# MiniMax-M3 cluster patch: prefill_progress_callback"


def main() -> int:
    spec = importlib.util.find_spec("mlx_vlm.generate.ar")
    if spec is None or spec.origin is None:
        print("mlx_vlm.generate.ar not found", file=sys.stderr)
        return 2
    path = pathlib.Path(spec.origin)
    text = path.read_text()
    if MARKER in text:
        print(f"prefill progress patch already installed: {path}")
        return 0

    old_sig = (
        "    prompt_cache_checkpoint: Optional[Callable[[int, List[Any]], None]] = None,\n"
        "    prompt_cache_checkpoint_len: Optional[int] = None,\n"
        "    seed: Optional[int] = None,\n"
    )
    new_sig = (
        "    prompt_cache_checkpoint: Optional[Callable[[int, List[Any]], None]] = None,\n"
        "    prompt_cache_checkpoint_len: Optional[int] = None,\n"
        "    prefill_progress_callback: Optional[Callable[[int, int], None]] = None,\n"
        "    seed: Optional[int] = None,\n"
    )
    if old_sig not in text:
        print("signature patch anchor not found", file=sys.stderr)
        return 3
    text = text.replace(old_sig, new_sig, 1)

    old_loop = (
        "                    mx.eval([c.state for c in prompt_cache])\n"
        "                    processed_tokens += n_to_process\n"
        "                    if (\n"
    )
    new_loop = (
        "                    mx.eval([c.state for c in prompt_cache])\n"
        "                    processed_tokens += n_to_process\n"
        f"                    {MARKER}\n"
        "                    if prefill_progress_callback is not None:\n"
        "                        prefill_progress_callback(processed_tokens, total_tokens)\n"
        "                    if (\n"
    )
    if old_loop not in text:
        print("prefill loop patch anchor not found", file=sys.stderr)
        return 4
    text = text.replace(old_loop, new_loop, 1)

    path.write_text(text)
    print(f"installed prefill progress patch: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
