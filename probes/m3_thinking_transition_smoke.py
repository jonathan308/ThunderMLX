#!/usr/bin/env python3
"""Offline checks for MiniMax reasoning-to-artifact channel repair."""

from __future__ import annotations

import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLX_M3_THINKING_ARTIFACT_TRANSITION", "1")
os.environ.setdefault("MLX_M3_THINKING_ARTIFACT_TRANSITION_MIN_TOKENS", "64")

import sharded_server as server  # noqa: E402


EXACT_LIKE_PROMPT = """Create a complete, single-file, fully interactive HTML
website called Shadow Syntax. Output the complete HTML file with embedded
Tailwind and all JS inline."""


def check_request_scope() -> None:
    assert server._thinking_artifact_request(
        EXACT_LIKE_PROMPT,
        "enabled",
        [],
    )
    assert server._thinking_artifact_request(
        EXACT_LIKE_PROMPT,
        "adaptive",
        [],
    )
    assert not server._thinking_artifact_request(
        EXACT_LIKE_PROMPT,
        "disabled",
        [],
    )
    assert not server._thinking_artifact_request(
        EXACT_LIKE_PROMPT,
        "enabled",
        [{"type": "function", "function": {"name": "write_file"}}],
    )
    assert not server._thinking_artifact_request(
        "Explain how a complete HTML document works.",
        "enabled",
        [],
    )


def check_thinking_sampling_defaults() -> None:
    defaults = server._request_generation_params(
        {"thinking_mode": "enabled"},
        tools=None,
    )
    assert defaults["temperature"] == 1.0, defaults
    assert defaults["top_p"] == 0.95, defaults
    assert defaults["top_k"] == 0, defaults
    assert defaults["min_p"] == 0.0, defaults
    assert defaults["repetition_penalty"] == 1.0, defaults

    explicit = server._request_generation_params(
        {
            "thinking_mode": "enabled",
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.03,
        },
        tools=None,
    )
    assert explicit["temperature"] == 0.7, explicit
    assert explicit["top_p"] == 0.8, explicit
    assert explicit["repetition_penalty"] == 1.03, explicit


def check_phase_detection() -> None:
    examples = (
        "Plan complete. Let me start coding:\n```html",
        "I have the structure. Let me code this now:\n<!DOCTYPE html>",
        "Next, let me structure the HTML:\n<html>",
        "Now let me start writing the complete implementation:\n```html",
    )
    for text in examples:
        match = server._thinking_artifact_phase_match(text, 128)
        assert match is not None, text
        assert text[match.end():].lstrip().startswith(("```html", "<!DOCTYPE", "<html"))
    assert server._thinking_artifact_phase_match(examples[0], 63) is None
    assert server._thinking_artifact_phase_match(
        "Let me inspect the existing files before deciding.",
        128,
    ) is None
    long_prefix = "planning detail. " * 200
    long_text = long_prefix + "Let me start coding:\n<!DOCTYPE html>"
    long_match = server._thinking_artifact_phase_match(long_text, 512)
    assert long_match is not None
    assert long_text[long_match.end():].lstrip().startswith("<!DOCTYPE html>")


def check_synchronized_close_queue() -> None:
    original_batch = server._BATCH_PATH_ACTIVE.get("value")
    original_close_ids = list(
        server._FORCE_TOKEN_SEQUENCE.get("thinking_close_ids") or []
    )
    try:
        server._BATCH_PATH_ACTIVE["value"] = True
        server._FORCE_TOKEN_SEQUENCE["thinking_close_ids"] = [200060]
        server._reset_forced_token_sequence()
        assert not server._arm_rank0_thinking_close(1, "test", 100)
        assert server._arm_rank0_thinking_close(0, "test", 100)
        assert not server._arm_rank0_thinking_close(0, "duplicate", 101)
        assert server._consume_rank0_forced_token() == 200060
        assert server._consume_rank0_forced_token() is None
        assert not server._FORCE_TOKEN_SEQUENCE["active"]
    finally:
        server._reset_forced_token_sequence()
        server._FORCE_TOKEN_SEQUENCE["thinking_close_ids"] = original_close_ids
        server._BATCH_PATH_ACTIVE["value"] = original_batch


def check_preamble_cleanup() -> None:
    complete = "```html\n<!DOCTYPE html>\n<html><body>ok</body></html>\n```"
    repeated = "```html\n<!DOCTYPE html>\n```html\n<!DOCTYPE html>\n<html></html>"
    cleaned = server._collapse_repeated_html_preamble(repeated)
    assert cleaned.count("```html") == 1, cleaned
    assert cleaned.lower().count("<!doctype html>") == 1, cleaned
    assert server._collapse_repeated_html_preamble(complete) == complete
    embedded = "prefix\n" + repeated
    assert server._collapse_repeated_html_preamble(embedded) == embedded


def check_repetition_guard_scope() -> None:
    assert not server._looks_like_degenerate_repetition("=" * 160)
    assert not server._looks_like_degenerate_repetition("/* " + "-" * 160 + " */")
    assert server._looks_like_degenerate_repetition("repeat-this " * 24)


def check_artifact_completion_boundary() -> None:
    state = server._new_thinking_artifact_completion_state()
    assert not server._thinking_artifact_completion_reached(
        state,
        "```html\n<!DOCTYPE html><html><body>hello",
        100,
    )
    assert server._thinking_artifact_completion_reached(
        state,
        "</body></html>\n<!DOCTYPE html>",
        120,
    )
    assert server._trim_thinking_artifact_completion_delta(
        state,
        "</body></html>\n<!DOCTYPE html>",
    ) == "</body></html>"
    assert server._thinking_artifact_completion_reached(state, "\n```", 121)
    assert server._trim_thinking_artifact_completion_text(
        "<mm:think>plan</mm:think>\n"
        "<!DOCTYPE html><html><body>ok</body></html>\n"
        "<!DOCTYPE html>"
    ).endswith("</body></html>")

    unfenced = server._new_thinking_artifact_completion_state()
    assert server._thinking_artifact_completion_reached(
        unfenced,
        "<!DOCTYPE html><html><body>ok</body></html>",
        200,
    )

    literal = server._new_thinking_artifact_completion_state()
    assert not server._thinking_artifact_completion_reached(
        literal,
        "const sample = '</html>';",
        300,
    )


def main() -> None:
    check_request_scope()
    check_thinking_sampling_defaults()
    check_phase_detection()
    check_synchronized_close_queue()
    check_preamble_cleanup()
    check_repetition_guard_scope()
    check_artifact_completion_boundary()
    print("PASS: thinking artifact transition smoke")


if __name__ == "__main__":
    main()
