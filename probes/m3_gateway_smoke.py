#!/usr/bin/env python3
"""Smoke checks for the ThunderMLX/oMLX model gateway helpers."""

import asyncio
import json
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_gateway as gateway  # noqa: E402
from model_gateway import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _responses_model_prefers_reasoning_heartbeat,
    backend_for_model,
    canonical_m3_model_id,
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


def check_m3_case_and_path_aliases_are_canonicalized():
    for alias, expected in (
        ("MiniMax-M3", "Minimax-M3"),
        ("MINIMAX-M3-NO-THINK", "Minimax-M3-No-Think"),
        ("mlx-community--MiniMax-M3-4bit", "Minimax-M3"),
    ):
        body, model, changed = normalize_openai_json_body(
            json.dumps({
                "model": alias,
                "messages": [{"role": "user", "content": "hi"}],
            }).encode("utf-8")
        )
        payload = json.loads(body)
        assert changed is True, (alias, model, payload)
        assert model == expected, (alias, model, payload)
        assert payload["model"] == expected, (alias, payload)
        assert canonical_m3_model_id(alias) == expected, alias
        assert backend_for_model(alias) == "m3", alias


class _ConnectedRequest:
    async def is_disconnected(self):
        return False


class _FakeUpstream:
    def __init__(self, chunks):
        self.status_code = 200
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_text(self):
        await asyncio.sleep(0.03)
        for chunk in self._chunks:
            yield chunk


class _FakeClient:
    chunks = []

    def __init__(self, *args, **kwargs):
        pass

    def stream(self, *args, **kwargs):
        return _FakeUpstream(list(self.chunks))

    async def aclose(self):
        pass


def _decode_responses_events(chunks):
    events = []
    for chunk in chunks:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        event_name = None
        payload = None
        for line in text.splitlines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if event_name and payload:
            events.append((event_name, payload))
    return events


async def _run_stream_fixture(model, deltas):
    old_client = gateway.httpx.AsyncClient
    old_ensure = gateway.ensure_backend
    old_hb = os.environ.get("M3_GATEWAY_RESPONSES_HEARTBEAT_SECONDS")

    async def _ready(_backend):
        return {"ok": True}

    _FakeClient.chunks = [
        "data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n"
        for delta in deltas
    ] + ["data: [DONE]\n\n"]
    gateway.httpx.AsyncClient = _FakeClient
    gateway.ensure_backend = _ready
    os.environ["M3_GATEWAY_RESPONSES_HEARTBEAT_SECONDS"] = "0.01"
    try:
        response = await gateway._responses_stream_live(
            {
                "model": model,
                "input": "heartbeat regression",
                "stream": True,
            },
            _ConnectedRequest(),
        )
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return _decode_responses_events(chunks)
    finally:
        gateway.httpx.AsyncClient = old_client
        gateway.ensure_backend = old_ensure
        if old_hb is None:
            os.environ.pop("M3_GATEWAY_RESPONSES_HEARTBEAT_SECONDS", None)
        else:
            os.environ["M3_GATEWAY_RESPONSES_HEARTBEAT_SECONDS"] = old_hb


def _added_items(events):
    return [
        (payload["output_index"], payload["item"]["type"], payload["item"]["id"])
        for name, payload in events
        if name == "response.output_item.added"
    ]


def check_responses_heartbeat_model_selection():
    assert _responses_model_prefers_reasoning_heartbeat("Minimax-M3")
    assert _responses_model_prefers_reasoning_heartbeat("M3-Web")
    assert not _responses_model_prefers_reasoning_heartbeat("Minimax-M3-No-Think")


def check_thinking_heartbeat_keeps_reasoning_separate():
    events = asyncio.run(_run_stream_fixture(
        "Minimax-M3",
        [
            {"reasoning_content": "private reasoning"},
            {"content": "visible answer"},
        ],
    ))
    added = _added_items(events)
    assert [(idx, kind) for idx, kind, _ in added] == [
        (0, "reasoning"),
        (1, "message"),
    ], added
    assert len({item_id for _, _, item_id in added}) == 2, added

    reasoning_deltas = [
        payload for name, payload in events
        if name == "response.reasoning_summary_text.delta" and payload.get("delta")
    ]
    text_deltas = [
        payload for name, payload in events
        if name == "response.output_text.delta" and payload.get("delta")
    ]
    assert [item["delta"] for item in reasoning_deltas] == ["private reasoning"]
    assert [item["output_index"] for item in reasoning_deltas] == [0]
    assert [item["delta"] for item in text_deltas] == ["visible answer"]
    assert [item["output_index"] for item in text_deltas] == [1]

    completed = [
        payload["response"] for name, payload in events
        if name == "response.completed"
    ]
    assert len(completed) == 1, completed
    assert [item["type"] for item in completed[0]["output"]] == [
        "reasoning",
        "message",
    ], completed[0]


def check_no_think_heartbeat_reuses_message_item():
    events = asyncio.run(_run_stream_fixture(
        "Minimax-M3-No-Think",
        [{"content": "visible answer"}],
    ))
    added = _added_items(events)
    assert [(idx, kind) for idx, kind, _ in added] == [(0, "message")], added
    text_deltas = [
        payload for name, payload in events
        if name == "response.output_text.delta" and payload.get("delta")
    ]
    assert [item["delta"] for item in text_deltas] == ["visible answer"]
    assert [item["output_index"] for item in text_deltas] == [0]


def main():
    check_empty_model_defaults_to_m3_agent_model()
    check_explicit_omlx_model_is_preserved()
    check_m3_case_and_path_aliases_are_canonicalized()
    check_responses_heartbeat_model_selection()
    check_thinking_heartbeat_keeps_reasoning_separate()
    check_no_think_heartbeat_reuses_message_item()
    print("PASS")


if __name__ == "__main__":
    main()
