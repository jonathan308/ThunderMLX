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
    _normalize_session_title,
    _normalize_session_title_sse,
    _normalize_zcode_goal_verifier_json,
    _responses_model_prefers_reasoning_heartbeat,
    _sse_keepalive_comment,
    backend_for_model,
    canonical_m3_model_id,
    model_ids_from_catalog,
    normalize_openai_json_body,
    resolve_requested_model,
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


def check_zcode_title_sidecar_is_short_and_visible():
    body, model, changed = normalize_openai_json_body(
        json.dumps({
            "model": "Minimax-M3",
            "messages": [
                {
                    "role": "system",
                    "content": "Generate a concise title for this coding session.",
                },
                {"role": "user", "content": "Build an interactive transformer demo."},
            ],
            "stream": False,
        }).encode("utf-8")
    )
    payload = json.loads(body)
    assert changed is True, payload
    assert model == "Minimax-M3-No-Think", payload
    assert payload["model"] == "Minimax-M3-No-Think", payload
    assert payload["thinking_mode"] == "disabled", payload
    assert payload["max_tokens"] == 24, payload
    assert payload["temperature"] == 0, payload
    assert payload["stream"] is False, payload
    assert len(payload["messages"]) == 2, payload
    assert "metadata, not an action" in payload["messages"][0]["content"], payload
    assert "Build an interactive transformer demo." in payload["messages"][1]["content"], payload
    assert "Output only the title" in payload["messages"][1]["content"], payload


def check_opencode_title_sidecar_is_short_streaming_and_single():
    body, model, changed = normalize_openai_json_body(
        json.dumps({
            "model": "Minimax-M3-No-Think",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a title generator. You output ONLY a thread title.",
                },
                {
                    "role": "user",
                    "content": "Generate a title for this conversation:\n",
                },
                {
                    "role": "user",
                    "content": "Download Gemma weights for a local inference server.",
                },
            ],
            "stream": True,
            "max_tokens": 32000,
        }).encode("utf-8")
    )
    payload = json.loads(body)
    assert changed is True, payload
    assert model == "Minimax-M3-No-Think", payload
    assert payload["thinking_mode"] == "disabled", payload
    assert payload["max_tokens"] == 24, payload
    assert payload["temperature"] == 0, payload
    assert payload["stream"] is True, payload
    assert len(payload["messages"]) == 2, payload
    assert "exactly one" in payload["messages"][0]["content"], payload
    assert "Download Gemma weights" in payload["messages"][1]["content"], payload
    assert "Generate a title for this conversation" not in payload["messages"][1]["content"], payload

    long_title = _normalize_session_title(
        "Test Fixture: Probe Format Function with List Input and Self-Check Assertions"
    )
    assert len(long_title) <= 50, long_title
    assert len(long_title.split()) <= 7, long_title

    clipped_title = _normalize_session_title(
        "Add summarize_bounds helper to opencode_gui_probe.py with dictionary return"
    )
    assert clipped_title == "Add summarize_bounds helper", clipped_title

    multi_title = _normalize_session_title(
        "Here are a few title options for this conversation:\n"
        "1. Debugging OpenCode title generation\n"
        "2. Improving background labels"
    )
    assert multi_title == "Debugging OpenCode title generation", multi_title

    raw = b"".join([
        b'data: {"id":"title-1","object":"chat.completion.chunk","created":1,"model":"Qwen","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        b'data: {"id":"title-1","object":"chat.completion.chunk","created":1,"model":"Qwen","choices":[{"index":0,"delta":{"content":"Here is a thinking process:"},"finish_reason":null}]}\n\n',
        b'data: [DONE]\n\n',
    ])
    normalized = _normalize_session_title_sse(raw, payload).decode("utf-8")
    assert "Here is a thinking process" not in normalized, normalized
    assert "Download Gemma weights local inference server" in normalized, normalized
    assert normalized.endswith("data: [DONE]\n\n"), normalized

    qwen_raw = b"".join([
        b'data: {"id":"title-2","object":"chat.completion.chunk","created":0,"model":"keepalive","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        b'data: {"id":"title-2","object":"chat.completion.chunk","created":0,"model":"keepalive","choices":[{"index":0,"delta":{"content":"**Analyze User Input:**\\n- **Input Request:**"},"finish_reason":null}]}\n\n',
        b'data: [DONE]\n\n',
    ])
    qwen_normalized = _normalize_session_title_sse(qwen_raw, payload).decode("utf-8")
    assert "Analyze User Input" not in qwen_normalized, qwen_normalized
    assert '"model": "Minimax-M3-No-Think"' in qwen_normalized, qwen_normalized
    assert "Download Gemma weights local inference server" in qwen_normalized, qwen_normalized


def check_zcode_goal_verifier_is_short_no_think_and_preserves_context():
    verifier = (
        "Verify whether the active session goal is actually complete.\n\n"
        "This is a verification request only. Do not continue implementation "
        "work, do not write files, and do not call tools.\n"
        "Return only a JSON object with this exact shape:\n"
        '{"passed": boolean, "reason": string, "nextAction": string}'
    )
    messages = [
        {
            "role": "system",
            "content": "Generate a concise title for this coding session.",
        },
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Build and test the notes app."},
        {"role": "assistant", "content": "All tests pass."},
        {"role": "user", "content": verifier},
    ]
    body, model, changed = normalize_openai_json_body(
        json.dumps({
            "model": "Minimax-M3",
            "messages": messages,
            "max_tokens": 64000,
            "stream": False,
        }).encode("utf-8")
    )
    payload = json.loads(body)
    assert changed is True, payload
    assert model == "Minimax-M3-No-Think", payload
    assert payload["model"] == "Minimax-M3-No-Think", payload
    assert payload["thinking_mode"] == "disabled", payload
    assert payload["temperature"] == 0, payload
    assert payload["max_tokens"] == 256, payload
    assert payload["_metadata_request"] == "zcode_goal_verification", payload
    assert payload["messages"] == messages, payload
    assert payload["stream"] is False, payload

    # Ordinary completion questions and real tool requests are not metadata.
    ordinary = json.dumps({
        "model": "Minimax-M3",
        "messages": [{"role": "user", "content": "Is the work complete?"}],
    }).encode("utf-8")
    ordinary_body, ordinary_model, ordinary_changed = normalize_openai_json_body(ordinary)
    assert ordinary_changed is False, ordinary_body
    assert ordinary_model == "Minimax-M3", ordinary_model

    historical_title = json.dumps({
        "model": "Minimax-M3",
        "messages": [
            {
                "role": "system",
                "content": "Generate a concise title for this coding session.",
            },
            {"role": "user", "content": "Build the notes app."},
            {"role": "assistant", "content": "The notes app is ready."},
            {"role": "user", "content": "Summarize the implementation."},
        ],
    }).encode("utf-8")
    normalized_history, history_model, history_changed = normalize_openai_json_body(
        historical_title
    )
    assert history_changed is False, normalized_history
    assert history_model == "Minimax-M3", history_model

    tool_body, tool_model, tool_changed = normalize_openai_json_body(
        json.dumps({
            "model": "Minimax-M3",
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": {"name": "Read", "parameters": {"type": "object"}},
            }],
        }).encode("utf-8")
    )
    assert tool_changed is False, tool_body
    assert tool_model == "Minimax-M3", tool_model

    omlx_body = json.dumps({
        "model": "DeepSeek-V4-Flash-4bit",
        "messages": messages,
        "max_tokens": 64000,
    }).encode("utf-8")
    normalized_omlx, omlx_model, omlx_changed = normalize_openai_json_body(omlx_body)
    assert omlx_changed is False, normalized_omlx
    assert omlx_model == "DeepSeek-V4-Flash-4bit", omlx_model

    valid_response = json.dumps({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": (
                    'Preamble. {"passed":true,"reason":"All 90 tests pass.",'
                    '"nextAction":""} Trailing prose.'
                ),
                "reasoning_content": "hidden",
            },
            "finish_reason": "length",
        }],
    }).encode("utf-8")
    normalized_valid = json.loads(_normalize_zcode_goal_verifier_json(valid_response))
    valid_choice = normalized_valid["choices"][0]
    valid_verdict = json.loads(valid_choice["message"]["content"])
    assert valid_verdict == {
        "passed": True,
        "reason": "All 90 tests pass.",
        "nextAction": "",
    }, valid_verdict
    assert "reasoning_content" not in valid_choice["message"], valid_choice
    assert valid_choice["finish_reason"] == "stop", valid_choice

    malformed_response = json.dumps({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": '{"passed": false, "reason": "missing next action"}',
            },
            "finish_reason": "stop",
        }],
    }).encode("utf-8")
    normalized_bad = json.loads(_normalize_zcode_goal_verifier_json(malformed_response))
    bad_verdict = json.loads(normalized_bad["choices"][0]["message"]["content"])
    assert bad_verdict["passed"] is False, bad_verdict
    assert set(bad_verdict) == {"passed", "reason", "nextAction"}, bad_verdict


def check_unknown_models_cannot_trigger_an_omlx_switch():
    catalog = {
        "object": "list",
        "data": [
            {"id": "DeepSeek-V4-Flash-4bit"},
            {"id": "Qwen3.6-35B-A3B-MLX-8bit"},
        ],
    }
    assert model_ids_from_catalog(catalog) == [
        "DeepSeek-V4-Flash-4bit",
        "Qwen3.6-35B-A3B-MLX-8bit",
    ]

    old_get_json = gateway.get_json

    async def _catalog(_url):
        return catalog

    gateway.get_json = _catalog
    try:
        valid = asyncio.run(resolve_requested_model("deepseek-v4-flash-4bit"))
        assert valid == {
            "ok": True,
            "backend": "omlx",
            "model": "DeepSeek-V4-Flash-4bit",
        }, valid

        typo = asyncio.run(resolve_requested_model("Minimax-M3-No-Thik"))
        assert typo["ok"] is False, typo
        assert typo["status_code"] == 404, typo
        assert typo["type"] == "model_not_found", typo
    finally:
        gateway.get_json = old_get_json


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


async def _run_stream_fixture(model, deltas, *, include_raw=False):
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
        events = _decode_responses_events(chunks)
        return (events, chunks) if include_raw else events
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
    assert _sse_keepalive_comment() == ": keepalive\n\n"
    assert _responses_model_prefers_reasoning_heartbeat("Minimax-M3")
    assert _responses_model_prefers_reasoning_heartbeat("M3-Web")
    assert not _responses_model_prefers_reasoning_heartbeat("Minimax-M3-No-Think")


def check_thinking_heartbeat_keeps_reasoning_separate():
    events, raw_chunks = asyncio.run(_run_stream_fixture(
        "Minimax-M3",
        [
            {"reasoning_content": "private reasoning"},
            {"content": "visible answer"},
        ],
        include_raw=True,
    ))
    assert any(str(chunk).startswith(": keepalive") for chunk in raw_chunks), raw_chunks
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


def check_anthropic_buffered_prefill_emits_transport_keepalive():
    old_messages = gateway.anthropic_messages
    old_seconds = gateway.SSE_KEEPALIVE_SECONDS

    async def _delayed_messages(_payload):
        await asyncio.sleep(0.03)
        return gateway.JSONResponse({
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "Minimax-M3-No-Think",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 1},
        })

    async def _collect():
        response = await gateway.anthropic_messages_stream({
            "model": "Minimax-M3-No-Think",
            "stream": True,
        })
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(str(chunk))
        return response, chunks

    gateway.anthropic_messages = _delayed_messages
    gateway.SSE_KEEPALIVE_SECONDS = 0.01
    try:
        response, chunks = asyncio.run(_collect())
    finally:
        gateway.anthropic_messages = old_messages
        gateway.SSE_KEEPALIVE_SECONDS = old_seconds

    assert chunks[0] == ": keepalive\n\n", chunks
    assert any("event: message_start" in chunk for chunk in chunks), chunks
    assert response.headers.get("x-accel-buffering") == "no", response.headers


def main():
    check_empty_model_defaults_to_m3_agent_model()
    check_explicit_omlx_model_is_preserved()
    check_m3_case_and_path_aliases_are_canonicalized()
    check_zcode_title_sidecar_is_short_and_visible()
    check_opencode_title_sidecar_is_short_streaming_and_single()
    check_zcode_goal_verifier_is_short_no_think_and_preserves_context()
    check_unknown_models_cannot_trigger_an_omlx_switch()
    check_responses_heartbeat_model_selection()
    check_thinking_heartbeat_keeps_reasoning_separate()
    check_no_think_heartbeat_reuses_message_item()
    check_anthropic_buffered_prefill_emits_transport_keepalive()
    print("PASS")


if __name__ == "__main__":
    main()
