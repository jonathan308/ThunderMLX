#!/usr/bin/env python3
"""Smoke checks for the ThunderMLX/oMLX model gateway helpers."""

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_gateway import (  # noqa: E402
    DEFAULT_MODEL_ID,
    backend_for_model,
    normalize_openai_json_body,
)


def check_empty_model_defaults_to_m3_agent_model():
    body, model, changed = normalize_openai_json_body(
        json.dumps({
            "model": "",
            "messages": [{"role": "user", "content": "hi"}],
        }).encode("utf-8")
    )
    payload = json.loads(body)
    assert changed is True, (model, payload)
    assert model == DEFAULT_MODEL_ID, (model, payload)
    assert payload["model"] == DEFAULT_MODEL_ID, payload
    assert backend_for_model(model) == "m3", model


def check_explicit_omlx_model_is_preserved():
    body, model, changed = normalize_openai_json_body(
        json.dumps({
            "model": "DeepSeek-V4-Flash-4bit",
            "messages": [{"role": "user", "content": "hi"}],
        }).encode("utf-8")
    )
    payload = json.loads(body)
    assert changed is False, (model, payload)
    assert model == "DeepSeek-V4-Flash-4bit", model
    assert payload["model"] == "DeepSeek-V4-Flash-4bit", payload
    assert backend_for_model(model) == "omlx", model


def main():
    check_empty_model_defaults_to_m3_agent_model()
    check_explicit_omlx_model_is_preserved()
    print("PASS")


if __name__ == "__main__":
    main()
