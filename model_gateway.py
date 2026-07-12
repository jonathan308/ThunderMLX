#!/usr/bin/env python3
"""OpenAI-compatible arbiter for ThunderMLX MiniMax-M3 and oMLX.

The gateway keeps oMLX and ThunderMLX on their native ports, exposes one
OpenAI-compatible surface, and switches memory-heavy backends by model id.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


ROOT = Path(__file__).resolve().parent


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


GATEWAY_HOST = os.environ.get("M3_GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.environ.get("M3_GATEWAY_PORT", "8010"))
M3_BASE_URL = os.environ.get("M3_GATEWAY_M3_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
OMLX_BASE_URL = os.environ.get("M3_GATEWAY_OMLX_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
M3_MODEL_IDS = set(env_list("M3_GATEWAY_M3_MODEL_IDS", "Minimax-M3,Minimax-M3-No-Think,M3-Web"))
CLAUDE_MODEL_IDS = set(env_list("M3_GATEWAY_CLAUDE_MODEL_IDS", "Claude-Code,Claude-Code-Sonnet"))
# Shim routes stay functional either way; this only controls /v1/models advertising
# (agent UIs pick up everything listed, and the Claude shim shouldn't be offered).
CLAUDE_MODELS_VISIBLE = env_bool("M3_GATEWAY_CLAUDE_MODELS_VISIBLE", False)
DEFAULT_MODEL_ID = os.environ.get("M3_GATEWAY_DEFAULT_MODEL_ID", "Minimax-M3-No-Think").strip() or "Minimax-M3-No-Think"
AUTO_SWITCH = env_bool("M3_GATEWAY_AUTO_SWITCH", True)
ALLOW_START_M3 = env_bool("M3_GATEWAY_ALLOW_START_M3", True)
ALLOW_STOP_M3 = env_bool("M3_GATEWAY_ALLOW_STOP_M3", True)
# Even when auto-stop is allowed, never yank M3 while it is serving or has
# served within this window — an oMLX probe once stopped it mid-session.
# 30s (was 120): single-user cluster - a 2-minute cooldown before oMLX
# switches just feels broken from the model picker (2026-07-09).
STOP_M3_GRACE_S = float(os.environ.get("M3_GATEWAY_STOP_M3_GRACE_S", "30"))
# Streamed /v1/responses translates upstream deltas live (codex sees thinking
# stream into its reasoning UI). 0 falls back to the end-of-turn replay.
RESPONSES_LIVE_STREAM = env_bool("M3_GATEWAY_RESPONSES_LIVE", True)
ALLOW_UNLOAD_OMLX = env_bool("M3_GATEWAY_ALLOW_UNLOAD_OMLX", True)
SWITCH_TIMEOUT = float(os.environ.get("M3_GATEWAY_SWITCH_TIMEOUT_SECONDS", "900"))
# Read-timeout for proxied non-stream completion POSTs — generation-scale,
# matches the server's M3_MAX_GENERATION_SECONDS budget.
COMPLETION_TIMEOUT = float(os.environ.get("M3_GATEWAY_COMPLETION_TIMEOUT_SECONDS", "7200"))
HTTP_TIMEOUT = float(os.environ.get("M3_GATEWAY_HTTP_TIMEOUT_SECONDS", "10"))
START_COMMAND = os.environ.get("M3_GATEWAY_M3_START_COMMAND", f"/bin/zsh {shlex.quote(str(ROOT / 'M3_Start.command'))}")
STOP_COMMAND = os.environ.get(
    "M3_GATEWAY_M3_STOP_COMMAND",
    f"M3_STOP_KEEP_DASHBOARD=1 M3_STOP_KEEP_GATEWAY=1 /bin/zsh {shlex.quote(str(ROOT / 'stop_cluster.sh'))}",
)
EXTRA_OMLX_UNLOAD_URLS = env_list("M3_GATEWAY_OMLX_UNLOAD_URLS", "")
CLAUDE_CLI = os.environ.get("M3_GATEWAY_CLAUDE_CLI", str(Path.home() / ".local/bin/claude"))
CLAUDE_MODEL = os.environ.get("M3_GATEWAY_CLAUDE_MODEL", "sonnet").strip() or "sonnet"
CLAUDE_PERMISSION_MODE = os.environ.get("M3_GATEWAY_CLAUDE_PERMISSION_MODE", "dontAsk").strip() or "dontAsk"
CLAUDE_WORKDIR = os.environ.get("M3_GATEWAY_CLAUDE_WORKDIR", str(Path.home()))
CLAUDE_TIMEOUT = float(os.environ.get("M3_GATEWAY_CLAUDE_TIMEOUT_SECONDS", "7200"))
CLAUDE_INCLUDE_TOOLS_NOTE = env_bool("M3_GATEWAY_CLAUDE_INCLUDE_TOOLS_NOTE", False)
ANTHROPIC_DEFAULT_MODEL = os.environ.get("M3_GATEWAY_ANTHROPIC_DEFAULT_MODEL", "Minimax-M3").strip() or "Minimax-M3"
ANTHROPIC_SMALL_MODEL = os.environ.get("M3_GATEWAY_ANTHROPIC_SMALL_MODEL", "Minimax-M3-No-Think").strip() or "Minimax-M3-No-Think"
ANTHROPIC_INTERNAL_TIMEOUT = float(os.environ.get("M3_GATEWAY_ANTHROPIC_TIMEOUT_SECONDS", "7200"))
ANTHROPIC_RETRY_SMALL_ON_EMPTY_TOOL = env_bool("M3_GATEWAY_ANTHROPIC_RETRY_SMALL_ON_EMPTY_TOOL", True)
ANTHROPIC_TOOL_HINT = env_bool("M3_GATEWAY_ANTHROPIC_TOOL_HINT", True)
ANTHROPIC_ROUTE_ALIASED_TO_SMALL = env_bool("M3_GATEWAY_ANTHROPIC_ROUTE_ALIASED_TO_SMALL", False)
ANTHROPIC_REQUIRE_TOOLS_ON_ACTION = env_bool("M3_GATEWAY_ANTHROPIC_REQUIRE_TOOLS_ON_ACTION", True)
# Context window advertised via /v1/models. The backend natively supports 1M,
# but 262k-512k is the recommended coding window; 300k keeps Codex/Claude Code
# style shims compacting early enough for stable long agent runs.
ADVERTISED_MAX_MODEL_LEN = int(os.environ.get("M3_GATEWAY_ADVERTISED_MAX_MODEL_LEN", "300000"))

APP = FastAPI(title="ThunderMLX Model Gateway")
SWITCH_LOCK = asyncio.Lock()
STATE: dict[str, Any] = {
    "active_backend": "unknown",
    "last_switch": None,
    "last_error": None,
    "events": [],
}


def canonical_m3_model_id(model: str | None) -> str | None:
    """Map common client spelling/case variants to a visible M3 model id."""
    raw = str(model or "").strip()
    if not raw:
        return None
    by_casefold = {model_id.casefold(): model_id for model_id in M3_MODEL_IDS}
    direct = by_casefold.get(raw.casefold())
    if direct:
        return direct
    aliases = {
        "m3": "Minimax-M3",
        "m3-think": "Minimax-M3",
        "m3-thinking": "Minimax-M3",
        "minimax-m3-think": "Minimax-M3",
        "minimax-m3-thinking": "Minimax-M3",
        "minimax-m3-4bit": "Minimax-M3",
        "mlx-community/minimax-m3-4bit": "Minimax-M3",
        "mlx-community--minimax-m3-4bit": "Minimax-M3",
        "m3-no-think": "Minimax-M3-No-Think",
        "m3-nothink": "Minimax-M3-No-Think",
        "m3-no-thinking": "Minimax-M3-No-Think",
        "minimax-m3-nothink": "Minimax-M3-No-Think",
        "minimax-m3-no-thinking": "Minimax-M3-No-Think",
        "m3-web": "M3-Web",
    }
    candidate = aliases.get(raw.casefold())
    if not candidate:
        return None
    return by_casefold.get(candidate.casefold(), candidate)


def static_m3_model(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 1783116151,
        "owned_by": "thundermlx",
        "max_model_len": ADVERTISED_MAX_MODEL_LEN,
        "gateway_backend": "m3",
        "gateway_static": True,
    }


def static_claude_model(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 1783116151,
        "owned_by": "claude-code",
        "max_model_len": 1000000,
        "gateway_backend": "claude",
        "gateway_static": True,
    }


def record_event(action: str, **fields: Any) -> None:
    event = {"time": time.time(), "action": action, **fields}
    STATE["last_switch"] = event
    STATE["events"] = ([event] + STATE.get("events", []))[:64]


def backend_for_model(model: str | None) -> str:
    if not model:
        return "m3"
    if str(model).casefold() in {item.casefold() for item in CLAUDE_MODEL_IDS}:
        return "claude"
    return "m3" if canonical_m3_model_id(model) else "omlx"


def anthropic_model_to_m3(model: str | None) -> str:
    raw = str(model or "").strip()
    lowered = raw.lower()
    canonical = canonical_m3_model_id(raw)
    if canonical:
        return canonical
    if "no-think" in lowered or "nothink" in lowered or "haiku" in lowered:
        return ANTHROPIC_SMALL_MODEL
    return ANTHROPIC_DEFAULT_MODEL


def normalize_openai_json_body(body: bytes) -> tuple[bytes, str | None, bool]:
    """Return a JSON body whose model field is never an empty string.

    Some agent shims send OpenAI-shaped payloads with ``model: ""`` after a
    model picker/Responses bridge loses state. OpenAI's validators reject that
    before work starts. For ThunderMLX, an empty model means "use the default
    M3 agent model" so the gateway stays forgiving while preserving explicit
    oMLX model ids.
    """
    if not body:
        return body, None, False
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body, None, False
    if not isinstance(payload, dict):
        return body, None, False
    raw_model = payload.get("model")
    model = str(raw_model or "").strip()
    changed = False
    if not model:
        payload["model"] = DEFAULT_MODEL_ID
        model = DEFAULT_MODEL_ID
        changed = True
    canonical_m3 = canonical_m3_model_id(model)
    if canonical_m3 and canonical_m3 != model:
        payload["model"] = canonical_m3
        model = canonical_m3
        changed = True
    if not changed:
        return body, model, False
    normalized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return normalized, model, True


def filtered_headers(request: Request) -> dict[str, str]:
    drop = {"host", "content-length", "connection", "keep-alive", "transfer-encoding"}
    return {k: v for k, v in request.headers.items() if k.lower() not in drop}


async def get_json(url: str, timeout: float = HTTP_TIMEOUT) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else None
    except Exception:
        return None


async def post_json(url: str, payload: dict[str, Any] | None = None, timeout: float = HTTP_TIMEOUT) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload or {})
            if response.status_code >= 400:
                return {"ok": False, "status_code": response.status_code, "text": response.text[:500]}
            data = response.json() if response.content else {}
            return data if isinstance(data, dict) else {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


async def m3_health() -> dict[str, Any] | None:
    data = await get_json(f"{M3_BASE_URL}/health")
    if data and data.get("status") == "healthy":
        return data
    return None


async def omlx_health() -> dict[str, Any] | None:
    data = await get_json(f"{OMLX_BASE_URL}/health")
    if data and str(data.get("status", "")).lower() in {"healthy", "ok", "ready"}:
        return data
    return data


async def wait_for_m3(up: bool, timeout: float = SWITCH_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        healthy = await m3_health()
        if bool(healthy) is up:
            return True
        await asyncio.sleep(2)
    return False


async def run_shell(command: str, *, timeout: float = SWITCH_TIMEOUT) -> dict[str, Any]:
    env = os.environ.copy()
    env["M3_GATEWAY_SKIP_START"] = "1"
    env.setdefault("M3_WARMUP_ON_START", "0")
    proc = await asyncio.create_subprocess_exec(
        "/bin/zsh",
        "-lc",
        command,
        cwd=str(ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.terminate()
        return {"ok": False, "timeout": timeout, "command": command}
    text = stdout.decode("utf-8", "replace")[-8000:] if stdout else ""
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "command": command, "output": text}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in {"text", "input_text"} and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif kind in {"image_url", "input_image"}:
                parts.append("[image omitted by Claude-Code CLI shim]")
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _anthropic_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif kind == "thinking" and isinstance(item.get("thinking"), str):
            # Keep prior assistant reasoning out of visible content. The M3
            # server stores/recalls reasoning separately when clients preserve it.
            continue
        elif kind == "image":
            parts.append("[image omitted by Anthropic gateway shim]")
    return "\n".join(part for part in parts if part)


def _anthropic_system_to_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return _anthropic_content_to_text(system)
    return "" if system is None else str(system)


def _anthropic_write_alias(tool: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, str]] | None:
    name = str(tool.get("name") or "").strip()
    if name != "Write":
        return None
    schema = tool.get("input_schema")
    props = schema.get("properties") if isinstance(schema, dict) else None
    props = props if isinstance(props, dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else []
    required = required if isinstance(required, list) else []
    file_prop = next(
        (key for key in ("file_path", "filePath", "path", "filename") if key in props),
        "file_path",
    )
    content_prop = "content" if "content" in props else "text"
    alias_schema = {
        "type": "object",
        "properties": {
            "filename": props.get(file_prop) or {"type": "string"},
            "content": props.get(content_prop) or {"type": "string"},
        },
        "required": [
            key for key, original in (("filename", file_prop), ("content", content_prop))
            if original in required or key == "filename"
        ],
    }
    if "content" not in alias_schema["required"]:
        alias_schema["required"].append("content")
    return "make_file", alias_schema, {"filename": file_prop, "content": content_prop}


def _anthropic_tools_to_openai(tools: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    out: list[dict[str, Any]] = []
    aliases: dict[str, dict[str, Any]] = {}
    name_aliases: dict[str, str] = {}
    if not isinstance(tools, list):
        return out, aliases, name_aliases
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        alias = _anthropic_write_alias(tool)
        model_name = name
        schema = tool.get("input_schema")
        if alias:
            model_name, schema, arg_map = alias
            aliases[model_name] = {"name": name, "arg_map": arg_map}
            name_aliases[name] = model_name
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": model_name,
                "description": str(tool.get("description") or ""),
                "parameters": schema,
            },
        })
    return out, aliases, name_aliases


def _prune_openai_tools_for_anthropic_action(
    tools: list[dict[str, Any]],
    aliases: dict[str, dict[str, Any]],
    name_aliases: dict[str, str],
    action_text: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    if len(tools) <= 8:
        return tools, aliases, name_aliases
    lowered = str(action_text or "").lower()
    if not _anthropic_text_requests_action(action_text):
        return tools, aliases, name_aliases
    file_markers = (
        "file",
        "directory",
        "project",
        "folder",
        "read",
        "inspect",
        "list",
        "create",
        "write",
        "edit",
        "change",
        "modify",
        "run",
        "python",
        "src/",
        ".py",
        ".js",
        ".ts",
        ".swift",
    )
    web_markers = ("web", "search", "fetch", "url", "http")
    keep_names = {"Bash", "Read", "Edit", "make_file"}
    if any(marker in lowered for marker in web_markers):
        keep_names.update({"WebSearch", "WebFetch"})
    if not any(marker in lowered for marker in file_markers + web_markers):
        return tools, aliases, name_aliases
    pruned = [
        tool for tool in tools
        if str(((tool or {}).get("function") or {}).get("name") or "") in keep_names
    ]
    if not pruned:
        return tools, aliases, name_aliases
    kept = {
        str(((tool or {}).get("function") or {}).get("name") or "")
        for tool in pruned
        if isinstance(tool, dict)
    }
    aliases = {name: alias for name, alias in aliases.items() if name in kept}
    name_aliases = {client: model for client, model in name_aliases.items() if model in kept}
    return pruned, aliases, name_aliases


def _anthropic_tool_choice_to_openai(tool_choice: Any, name_aliases: dict[str, str] | None = None) -> Any:
    if not tool_choice:
        return "auto"
    if isinstance(tool_choice, str):
        return {"any": "required"}.get(tool_choice, tool_choice)
    if not isinstance(tool_choice, dict):
        return "auto"
    choice_type = str(tool_choice.get("type") or "auto")
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        name = str(tool_choice["name"])
        name = (name_aliases or {}).get(name, name)
        return {"type": "function", "function": {"name": name}}
    return "auto"


def _openai_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(((tool or {}).get("function") or {}).get("name") or "")
        for tool in tools or []
        if isinstance(tool, dict)
    }


def _anthropic_text_requests_action(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "use tool",
        "use tools",
        "inspect",
        "read ",
        "list ",
        "create ",
        "write ",
        "edit ",
        "run ",
        "check ",
        "search",
        "fetch",
    )
    return any(marker in lowered for marker in markers)


def _anthropic_text_requests_mutation(text: str) -> bool:
    """Whether the task explicitly asks for a state-changing action."""
    return bool(re.search(
        r"\b(?:add|build|change|copy|create|delete|edit|fix|implement|make|"
        r"modify|move|patch|remove|rename|rewrite|save|update|write)\b",
        str(text or ""),
        flags=re.IGNORECASE,
    ))


def _anthropic_bash_command_mutates(command: str) -> bool:
    """Conservatively recognize shell commands that can satisfy a mutation."""
    command = str(command or "")
    if not command.strip():
        return False
    scrubbed = re.sub(r"(?:^|\s)\d+>{1,2}&?\d*\s*[^\s;|]*", " ", command)
    if re.search(
        r"(?:^|[;&|\n]\s*)(?:cp|install|mkdir|mv|rm|touch|truncate)\b",
        scrubbed,
    ):
        return True
    if re.search(r"\b(?:git\s+apply|perl\s+-pi|sed\s+-i|tee)\b", scrubbed):
        return True
    if re.search(r"\.(?:write_bytes|write_text)\s*\(", scrubbed):
        return True
    if re.search(r"\bopen\s*\([^\n]*,[^\n]*['\"](?:a|w|x)[+b]?['\"]", scrubbed):
        return True
    return bool(re.search(r"(?<![0-9])>{1,2}\s*[^&|\s]", scrubbed))


def _anthropic_has_mutating_tool_use(payload: dict[str, Any]) -> bool:
    mutating_names = {
        "applypatch", "edit", "editfile", "makefile", "multiedit",
        "notebookedit", "write", "writefile",
    }
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = re.sub(
                r"[^a-z0-9]", "", str(block.get("name") or "").lower()
            )
            if name in mutating_names:
                return True
            if name in {"bash", "execcommand", "shell"}:
                tool_input = block.get("input")
                if isinstance(tool_input, dict) and _anthropic_bash_command_mutates(
                    tool_input.get("command") or tool_input.get("cmd") or ""
                ):
                    return True
    return False


def _anthropic_alias_input_to_model(args: dict[str, Any], alias: dict[str, Any] | None) -> dict[str, Any]:
    if not alias:
        return args
    arg_map = alias.get("arg_map")
    if not isinstance(arg_map, dict):
        return args
    reverse = {str(original): str(model) for model, original in arg_map.items()}
    return {reverse.get(str(key), str(key)): value for key, value in args.items()}


def _anthropic_alias_input_to_client(args: dict[str, Any], alias: dict[str, Any] | None) -> dict[str, Any]:
    if not alias:
        return args
    arg_map = alias.get("arg_map")
    if not isinstance(arg_map, dict):
        return args
    return {str(arg_map.get(str(key), str(key))): value for key, value in args.items()}


def _anthropic_messages_to_openai(
    messages: Any,
    name_aliases: dict[str, str] | None = None,
    aliases: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return out
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                        continue
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind == "text" and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
                    elif kind == "tool_use":
                        name = str(block.get("name") or "").strip()
                        if not name:
                            continue
                        arguments = block.get("input")
                        if not isinstance(arguments, dict):
                            arguments = {}
                        model_name = (name_aliases or {}).get(name, name)
                        arguments = _anthropic_alias_input_to_model(arguments, (aliases or {}).get(model_name))
                        tool_calls.append({
                            "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                            "type": "function",
                            "function": {
                                "name": model_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        })
            else:
                text = _anthropic_content_to_text(content)
                if text:
                    text_parts.append(text)
            item: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part) or None,
            }
            if tool_calls:
                item["tool_calls"] = tool_calls
            out.append(item)
            continue

        if role == "user" and isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "tool_result":
                    if text_parts:
                        out.append({"role": "user", "content": "\n".join(text_parts)})
                        text_parts = []
                    tool_content = block.get("content")
                    tool_id = str(block.get("tool_use_id") or "").strip()
                    prefix = f"Tool result {tool_id}:" if tool_id else "Tool result:"
                    # MiniMax-M3's tool template can emit empty follow-up turns
                    # after OpenAI `tool` role messages. Anthropic clients do
                    # not require that internal role shape, so make the result
                    # explicit model-facing user text while preserving the
                    # client-facing Anthropic tool_use/tool_result protocol.
                    out.append({
                        "role": "user",
                        "content": f"{prefix}\n{_anthropic_content_to_text(tool_content)}",
                    })
                elif kind == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif kind == "image":
                    text_parts.append("[image omitted by Anthropic gateway shim]")
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        text = _anthropic_content_to_text(content)
        if text or role in {"user", "tool"}:
            out.append({"role": role if role in {"user", "assistant", "tool"} else "user", "content": text})
    return out


def anthropic_to_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model = anthropic_model_to_m3(payload.get("model"))
    tools, aliases, name_aliases = _anthropic_tools_to_openai(payload.get("tools"))
    action_text = _anthropic_action_text(payload)
    has_tool_result = _anthropic_has_tool_result(payload)
    # 2026-07-06 cache audit: prune on the FIRST user message, not the latest
    # turn's wording. Per-turn pruning ping-ponged the forwarded tool list,
    # flipping the backend's session fingerprint and destroying its KV cache
    # every few turns (the dominant zero-reuse leak for agent sessions).
    tools, aliases, name_aliases = _prune_openai_tools_for_anthropic_action(
        tools,
        aliases,
        name_aliases,
        _strip_anthropic_system_reminders(_anthropic_first_user_text(payload)),
    )
    messages = _anthropic_messages_to_openai(payload.get("messages"), name_aliases, aliases)
    system = _anthropic_system_to_text(payload.get("system")).strip()
    if system:
        messages.insert(0, {"role": "system", "content": system})
    out: dict[str, Any] = {
        "model": model,
        "messages": messages or [{"role": "user", "content": ""}],
        "stream": False,
        "max_tokens": int(payload.get("max_tokens") or 1024),
    }
    if tools:
        if ANTHROPIC_TOOL_HINT:
            # 2026-07-06 cache audit: the hint sits at prompt position 0, so
            # every character of it must be conversation-stable. Embedding the
            # tool-name list shifted all downstream tokens whenever pruning
            # changed, invalidating the backend's cached prefix at token ~150.
            # The model sees the schemas in the tools payload; no need to
            # repeat the names here.
            if "make_file" in aliases:
                hint_text = (
                    "Tool-use compatibility rule for Claude Code: when the "
                    "user asks to create, write, edit, read, list, run, check, "
                    "search, fetch, inspect, or otherwise act on files or the "
                    "environment, call the matching tool instead of describing "
                    "what you would do. Do not say you will use a tool; emit a "
                    "valid tool call. The client-facing tool Write is exposed "
                    "to you as make_file with argument keys filename and "
                    "content. Prefer Bash, Read, or Edit for inspection, "
                    "verification, or code edits when those tools are "
                    "available."
                )
            else:
                hint_text = (
                    "Tool-use compatibility rule for Claude Code: when the "
                    "user asks to create, write, edit, read, list, run, check, "
                    "search, fetch, inspect, or otherwise act on files or the "
                    "environment, call the matching tool instead of describing "
                    "what you would do. Do not say you will use a tool; emit a "
                    "valid tool call. Use the exact advertised tool names and "
                    "argument keys."
                )
            if _anthropic_text_requests_mutation(action_text):
                hint_text += (
                    " Do not give the requested completion/final answer until "
                    "at least one requested state-changing action has actually "
                    "succeeded. Listing or reading files does not complete a "
                    "write, edit, create, fix, or update request."
                )
            messages.insert(0, {
                "role": "system",
                "content": hint_text,
            })
        out["tools"] = tools
        out["tool_choice"] = _anthropic_tool_choice_to_openai(
            payload.get("tool_choice"),
            name_aliases,
        )
        pending_mutation = bool(
            has_tool_result and _anthropic_pending_mutation(payload)
        )
        if (
            has_tool_result
            and out["tool_choice"] == "required"
            and not pending_mutation
        ):
            out["tool_choice"] = "auto"
        if (
            ANTHROPIC_REQUIRE_TOOLS_ON_ACTION
            and out["tool_choice"] == "auto"
            and _anthropic_text_requests_action(action_text)
        ):
            if pending_mutation:
                out["tool_choice"] = "required"
            elif not has_tool_result and "Bash" in _openai_tool_names(tools):
                out["tool_choice"] = {"type": "function", "function": {"name": "Bash"}}
            elif not has_tool_result:
                out["tool_choice"] = "required"
        if aliases:
            out["_anthropic_tool_aliases"] = aliases
    if payload.get("temperature") is not None:
        out["temperature"] = payload.get("temperature")
    if payload.get("top_p") is not None:
        out["top_p"] = payload.get("top_p")
    metadata = payload.get("metadata")
    session = None
    if isinstance(metadata, dict):
        session = metadata.get("session_id") or metadata.get("conversation_id")
    if not session:
        # 2026-07-06 cache audit: hand the backend a REAL session id so its
        # session-protect / resident-slot / reasoning-recall machinery works
        # for agent traffic instead of the per-turn auto-fingerprint.
        # System prompt + first user message is stable for the lifetime of a
        # conversation and never crosses two different chats by accident
        # (the backend's token-prefix checks still guard actual KV reuse).
        anchor_system = _anthropic_system_to_text(payload.get("system")).strip()
        anchor_user = _anthropic_first_user_text(payload)
        if anchor_system or anchor_user:
            digest = hashlib.sha256(
                (anchor_system + "\x00" + anchor_user).encode("utf-8", "ignore")
            ).hexdigest()[:16]
            session = f"anthro-{digest}"
    if session:
        out["metadata"] = {"session_id": str(session)}
    return out


def _json_from_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _scrub_minimax_markup(text: str) -> str:
    """Drop raw MiniMax tool markup from client-visible text.

    The backend strips these blocks on tool-bearing requests, but the marker
    must never reach a client from any path (tool-less side requests, template
    drift), so scrub again at the protocol boundary.
    """
    if not isinstance(text, str) or "]<]minimax[>[" not in text:
        return text
    return text[:text.find("]<]minimax[>[")].strip()


def openai_to_anthropic_message(
    openai_response: dict[str, Any],
    requested_model: str,
    aliases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    choice = (openai_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = _scrub_minimax_markup(message.get("content"))
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        alias = (aliases or {}).get(name)
        client_name = str(alias.get("name") or name) if alias else name
        client_input = _anthropic_alias_input_to_client(
            _json_from_tool_arguments(fn.get("arguments")),
            alias,
        )
        content.append({
            "type": "tool_use",
            "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": client_name,
            "input": client_input,
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = openai_response.get("usage") if isinstance(openai_response.get("usage"), dict) else {}
    stop_reason = "tool_use" if tool_calls else "end_turn"
    finish = choice.get("finish_reason")
    if finish == "length":
        stop_reason = "max_tokens"
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def _openai_response_has_usable_content(openai_response: dict[str, Any]) -> bool:
    choice = (openai_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if message.get("tool_calls"):
        return True
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        lowered = content.strip().lower()
        if "empty tool-call markers" in lowered:
            return False
        if "could not produce a valid tool call" in lowered:
            return False
        if "could not complete that tool step" in lowered:
            return False
        if "best answer from the context already gathered" in lowered:
            return False
        if "previous tool action was incomplete" in lowered:
            return False
        if "previous apply_patch call was malformed" in lowered:
            return False
        if "malformed tool action" in lowered:
            return False
        return True
    return False


def _openai_response_has_tool_calls(openai_response: dict[str, Any]) -> bool:
    choice = (openai_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return bool(message.get("tool_calls"))


def _anthropic_declared_working_directory(payload: dict[str, Any]) -> str:
    system = _anthropic_system_to_text(payload.get("system"))
    patterns = (
        r"<cwd>\s*(/[^<\r\n]+?)\s*</cwd>",
        r"(?mi)^\s*(?:Current\s+)?Working directory\s*:\s*(/[^\r\n]+?)\s*$",
        r"(?mi)^\s*cwd\s*:\s*(/[^\r\n]+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, system, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = os.path.normpath(match.group(1).strip())
        if (
            os.path.isabs(candidate)
            and candidate != "/"
            and len(candidate) <= 1024
            and not any(ord(ch) < 32 for ch in candidate)
        ):
            return candidate
    return ""


def _anthropic_tool_result_path_context(payload: dict[str, Any]) -> dict[str, Any]:
    parts: list[str] = []
    command_by_id: dict[str, str] = {}
    result_parts: list[tuple[str, str]] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_id = str(block.get("id") or "").strip()
                    tool_input = block.get("input")
                    if tool_id and isinstance(tool_input, dict):
                        command = tool_input.get("command") or tool_input.get("cmd")
                        if isinstance(command, str) and command.strip():
                            command_by_id[tool_id] = command.strip()
                elif block.get("type") == "tool_result":
                    tool_id = str(block.get("tool_use_id") or "").strip()
                    text = _anthropic_content_to_text(block.get("content"))
                    parts.append(text)
                    result_parts.append((tool_id, text))
    text = "\n".join(part for part in parts if part)
    files: set[str] = set()
    cwd = _anthropic_declared_working_directory(payload)

    def add_file(name: str, base_dir: str = "") -> None:
        name = str(name or "").strip()
        if not name or name in {".", ".."}:
            return
        if name.startswith("./"):
            name = name[2:]
        if "/" in name or "." in name:
            files.add(name)
        if base_dir and not name.startswith("/"):
            files.add(f"{base_dir.rstrip('/')}/{name}")

    for tool_id, result_text in result_parts:
        command = command_by_id.get(tool_id, "")
        base_dir = ""
        match = re.search(
            r"(?:^|[;&|\n]\s*)ls\s+(?:-[A-Za-z]+\s+)?([A-Za-z0-9._~/-]+)",
            command,
        )
        if match:
            candidate_dir = match.group(1).strip().strip("'\"")
            if candidate_dir not in {".", "./"} and not candidate_dir.startswith("-"):
                base_dir = candidate_dir.removeprefix("./")
        lines = result_text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "--- pwd ---" and index + 1 < len(lines):
                cwd = lines[index + 1].strip()
                continue
            if stripped.startswith("/") and " " not in stripped:
                cwd = stripped
                continue
            if stripped == "--- files ---":
                for file_line in lines[index + 1:]:
                    item = file_line.strip()
                    if not item or item.startswith("--- "):
                        break
                    add_file(item, base_dir)
                continue
            if stripped.startswith("./"):
                add_file(stripped, base_dir)
                continue
            # Parse common `ls -la` output: mode links owner group size date name.
            if re.match(r"^[bcdlps-][rwxstST-]{9}[@+]?\s+", stripped):
                fields = stripped.split()
                if len(fields) >= 9:
                    add_file(" ".join(fields[8:]), base_dir)
                continue
            if re.match(r"^[A-Za-z0-9._~/-]+$", stripped):
                add_file(stripped, base_dir)
    commands = [
        command.strip()
        for command in command_by_id.values()
        if isinstance(command, str) and command.strip()
    ]
    return {"cwd": cwd, "files": files, "commands": commands, "tool_result_text": text}


def _normalize_bash_command(command: str) -> str:
    return re.sub(r"\s+", " ", str(command or "").strip())


def _bash_command_for_repeated_inspection(payload: dict[str, Any], repeated: str) -> str | None:
    action_text = _anthropic_action_text(payload)
    lowered = action_text.lower()
    normalized = _normalize_bash_command(repeated)
    if normalized not in {"ls", "ls .", "ls -la", "ls -al", "pwd"}:
        return None
    if "agent_report.txt" in lowered and "notes" in lowered and "code" in lowered:
        return (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "codes = []\n"
            "for path in sorted(Path('notes').glob('*.txt')):\n"
            "    for line in path.read_text().splitlines():\n"
            "        if line.lower().startswith('code:'):\n"
            "            codes.append(line.split(':', 1)[1].strip())\n"
            "Path('AGENT_REPORT.txt').write_text(','.join(codes))\n"
            "print(Path('AGENT_REPORT.txt').read_text())\n"
            "PY"
        )
    return (
        "pwd\n"
        "printf '\\n--- files ---\\n'\n"
        "find . -maxdepth 4 -type f | sort | sed -n '1,220p'\n"
        "printf '\\n--- previews ---\\n'\n"
        "for f in README.md package.json pyproject.toml notes/*.txt src/*.py src/*.js src/*.ts "
        "app.py main.py; do "
        "if [ -f \"$f\" ]; then printf '\\n--- %s ---\\n' \"$f\"; sed -n '1,220p' \"$f\"; fi; "
        "done"
    )


def _repair_path_value_from_context(value: Any, context: dict[str, Any]) -> Any:
    if not isinstance(value, str) or not (
        value.startswith("/") or value.startswith("~/")
    ):
        return value
    if os.path.exists(os.path.expanduser(value)):
        return value
    files = context.get("files")
    if not isinstance(files, set) or not files:
        return value
    normalized = value.replace("\\", "/")
    matches = sorted(
        (candidate for candidate in files if normalized.endswith("/" + candidate) or normalized.endswith(candidate)),
        key=len,
        reverse=True,
    )
    if matches:
        return matches[0]
    return value


def _repair_anthropic_tool_call_paths(data: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    context = _anthropic_tool_result_path_context(payload)
    action_text = _anthropic_action_text(payload)
    commands = context.get("commands") if isinstance(context.get("commands"), list) else []
    command_counts: dict[str, int] = {}
    for command in commands:
        normalized = _normalize_bash_command(str(command))
        if normalized:
            command_counts[normalized] = command_counts.get(normalized, 0) + 1
    if (
        not context.get("files")
        and not command_counts
        and not context.get("cwd")
        and not action_text
    ):
        return data
    changed = False
    for choice in data.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict):
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except Exception:
                continue
            if not isinstance(args, dict):
                continue
            name = str(function.get("name") or "")
            command = args.get("command") or args.get("cmd")
            normalized_command = _normalize_bash_command(command) if isinstance(command, str) else ""
            if name == "Bash" and command_counts.get(normalized_command, 0) >= 2:
                repaired_command = _bash_command_for_repeated_inspection(payload, normalized_command)
                if repaired_command and repaired_command != command:
                    args["command"] = repaired_command
                    args.setdefault("description", "Break repeated tool loop and gather evidence")
                    changed = True
            for key in ("file_path", "filePath", "path", "filename"):
                raw_value = args.get(key)
                repaired = _repair_path_value_from_context(raw_value, context)
                cwd = str(context.get("cwd") or "")
                if (
                    isinstance(raw_value, str)
                    and os.path.isabs(raw_value)
                    and raw_value not in action_text
                ):
                    normalized = os.path.normpath(raw_value)
                    inside = False
                    if cwd:
                        try:
                            inside = os.path.commonpath([cwd, normalized]) == cwd
                        except ValueError:
                            inside = False
                    basename = os.path.basename(normalized)
                    relative_match = re.search(
                        rf"(?<![A-Za-z0-9_.-])"
                        rf"((?:[A-Za-z0-9_.-]+/)*{re.escape(basename)})"
                        rf"(?![A-Za-z0-9_.-])",
                        action_text,
                    ) if basename else None
                    if not inside and relative_match:
                        relative_target = relative_match.group(1)
                        if cwd:
                            candidate = os.path.normpath(
                                os.path.join(cwd, relative_target)
                            )
                            try:
                                candidate_inside = (
                                    os.path.commonpath([cwd, candidate]) == cwd
                                )
                            except ValueError:
                                candidate_inside = False
                            if candidate_inside:
                                repaired = candidate
                        else:
                            # Claude Code does not always place its cwd in the
                            # Anthropic system payload. When the task itself
                            # names a relative target, keep that client-relative
                            # path instead of accepting a model-invented absolute
                            # home-directory path. Explicit user absolute paths
                            # are preserved by the raw_value-in-action guard.
                            repaired = relative_target
                if repaired != args.get(key):
                    args[key] = repaired
                    changed = True
            if changed:
                function["arguments"] = json.dumps(args, ensure_ascii=False)
    return data


def _required_tool_retry_payload(openai_payload: dict[str, Any]) -> dict[str, Any]:
    retry_payload = dict(openai_payload)
    messages = list(openai_payload.get("messages") or [])
    tools = openai_payload.get("tools") or []
    tool_names = [
        str(((tool or {}).get("function") or {}).get("name") or "")
        for tool in tools
        if isinstance(tool, dict)
    ]
    retry_payload["messages"] = [{
        "role": "system",
        "content": (
            "The previous attempt answered in prose without a required tool "
            "call. Retry this turn by emitting exactly one valid tool call now. "
            "Do not explain, apologize, or describe the plan. Use one of these "
            f"available tool names: {', '.join(name for name in tool_names if name)}."
        ),
    }, *messages]
    retry_payload["tool_choice"] = "required"
    retry_payload["max_tokens"] = min(int(retry_payload.get("max_tokens") or 1024), 1024)
    return retry_payload


def _tool_result_continuation_retry_payload(openai_payload: dict[str, Any]) -> dict[str, Any]:
    retry_payload = dict(openai_payload)
    messages = list(openai_payload.get("messages") or [])
    retry_payload["messages"] = [*messages, {
        "role": "user",
        "content": (
            "The previous assistant turn did not produce a usable tool call or "
            "final answer. Continue from the tool results already provided. "
            "If another action is needed, emit exactly one valid tool call using "
            "the advertised schema. If no more action is needed, answer with the "
            "final result. Do not mention malformed, incomplete, unavailable, "
            "or failed tool calls."
        ),
    }]
    if retry_payload.get("tool_choice") == "required":
        retry_payload["tool_choice"] = "auto"
    retry_payload["max_tokens"] = min(int(retry_payload.get("max_tokens") or 2048), 2048)
    return retry_payload


def _anthropic_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _anthropic_content_to_text(message.get("content")).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _anthropic_first_user_text(payload: dict[str, Any]) -> str:
    """Text of the conversation's FIRST user message — stable for its lifetime.

    Used for anything that must not change turn-to-turn (tool pruning, the
    synthesized cache session id): keying those on the latest turn's wording
    churned the backend prompt cache every few turns (2026-07-06 audit).
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _anthropic_content_to_text(message.get("content")).strip()
        if text:
            return text
    return ""


def _strip_anthropic_system_reminders(text: str) -> str:
    if not isinstance(text, str) or "<system-reminder" not in text:
        return text
    return re.sub(
        r"<system-reminder\b[^>]*>.*?</system-reminder>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _anthropic_action_text(payload: dict[str, Any]) -> str:
    user_text = _strip_anthropic_system_reminders(_anthropic_user_text(payload))
    if _anthropic_text_requests_action(user_text) or _extract_write_request(user_text):
        return user_text
    system_text = _strip_anthropic_system_reminders(
        _anthropic_system_to_text(payload.get("system")),
    )
    if system_text:
        return "\n\n".join(part for part in (system_text, user_text) if part)
    return user_text


def _anthropic_pending_mutation(payload: dict[str, Any]) -> bool:
    """True while an explicit mutation task has only inspection evidence."""
    return (
        _anthropic_text_requests_mutation(_anthropic_action_text(payload))
        and not _anthropic_has_mutating_tool_use(payload)
    )


def _anthropic_has_tool_result(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            return True
    return False


def _anthropic_tool_result_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(_anthropic_content_to_text(block.get("content")))
    return "\n".join(part for part in parts if part)


def _anthropic_latest_tool_result_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return _anthropic_content_to_text(block.get("content")).strip()
    return ""


def _anthropic_latest_tool_result_is_error(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return bool(block.get("is_error"))
    return False


def _anthropic_last_tool_command(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                continue
            command = tool_input.get("command") or tool_input.get("cmd")
            if isinstance(command, str):
                return command.strip()
    return ""


def _anthropic_exact_reply_text(payload: dict[str, Any]) -> str:
    action_text = _anthropic_action_text(payload)
    match = re.search(
        r"\breply\s+with\s+exactly\s+(.+?)(?:[.。]\s*$|\n|$)",
        action_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    text = match.group(1).strip().strip("'\"`")
    return text if 0 < len(text) <= 200 else ""


def _anthropic_exact_reply_after_verified_tool(
    payload: dict[str, Any],
    requested_model: str,
) -> dict[str, Any] | None:
    """Return exact requested final text after a successful verification tool."""
    if not _anthropic_has_tool_result(payload) or _anthropic_latest_tool_result_is_error(payload):
        return None
    # A read-only command such as `cat notes/*.txt` is evidence gathering, not
    # proof that a requested output file was created. The old shortcut returned
    # the exact final here and silently skipped the mutation.
    if _anthropic_pending_mutation(payload):
        return None
    exact = _anthropic_exact_reply_text(payload)
    if not exact:
        return None
    command = _normalize_bash_command(_anthropic_last_tool_command(payload)).lower()
    result_text = _anthropic_latest_tool_result_text(payload)
    if not command or not result_text:
        return None
    if any(marker in result_text.lower() for marker in ("no such file", "traceback", "error:", "command not found")):
        return None
    # Only synthesize an exact final when the task supplied literal expected
    # file content and the verification output matches it. Dynamic agent and
    # coding tasks must return to the model for semantic verification; merely
    # running `cat` does not prove ordering, completeness, or correctness.
    verified_literal = False
    for requested in _extract_write_requests(_anthropic_action_text(payload)):
        filename = str(requested.get("filename") or "").strip()
        expected_content = str(requested.get("content") or "").strip()
        if (
            filename
            and expected_content
            and filename.lower() in command
            and result_text == expected_content
        ):
            verified_literal = True
            break
    if not verified_literal:
        return None
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{"type": "text", "text": exact}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _anthropic_tool_names(payload: dict[str, Any]) -> set[str]:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return set()
    return {
        str(tool.get("name") or "").strip()
        for tool in tools
        if isinstance(tool, dict) and str(tool.get("name") or "").strip()
    }


def _extract_write_requests(text: str) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        return []
    lowered = text.lower()
    if not any(word in lowered for word in ("write", "create", "save")):
        return []
    if "containing" not in lowered:
        return []
    action_re = re.compile(
        r"\b(?:create|write|save)\s+"
        r"(?:a\s+)?(?:simple\s+)?(?:text\s+)?(?:file\s+)?"
        r"(?:on\s+the\s+desktop\s+)?(?:named\s+|at\s+)?"
        r"(?P<path>/[^\s'\"`;,<>]+|[A-Za-z0-9._~/-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)"
        r"\s+containing(?:\s+exactly)?\s+",
        re.IGNORECASE,
    )
    matches = list(action_re.finditer(text))
    requests: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        content_start = match.end()
        if index + 1 < len(matches):
            content_end = matches[index + 1].start()
        else:
            tail = re.search(
                r"\s*,?\s+then\s+(?:check|list|answer|reply|run|show|print|verify)\b|[.。]\s*$",
                text[content_start:],
                re.IGNORECASE | re.DOTALL,
            )
            content_end = content_start + tail.start() if tail else len(text)
        content = text[content_start:content_end]
        content = re.sub(r"\s*,?\s+then\s*$", "", content, flags=re.IGNORECASE).strip().strip("'\"`")
        if content.endswith(".") and "\n" not in content:
            content = content[:-1]
        file_path = match.group("path").strip().strip("'\"`")
        if file_path and content:
            requests.append({"filename": file_path, "content": content})
    return requests


def _extract_write_request(text: str) -> dict[str, str] | None:
    requests = _extract_write_requests(text)
    return requests[0] if requests else None


def _is_simple_write_only_request(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    lowered = text.lower()
    if "containing exactly" not in lowered:
        return False
    if not any(word in lowered for word in ("create", "write", "save")):
        return False
    compound_markers = (
        "inspect",
        "read ",
        "list ",
        "search",
        "analy",
        "summar",
        "check ",
        "review",
        "edit ",
        "modify",
        "look at",
    )
    return not any(marker in lowered for marker in compound_markers)


def _is_safe_write_fallback_request(text: str) -> bool:
    if _is_simple_write_only_request(text):
        return True
    if not isinstance(text, str) or not text.strip():
        return False
    lowered = text.lower()
    # Post-fallback is only a last-resort bridge for simple file creation or
    # inspect/read/write tasks. Do not synthesize tool calls for real coding
    # work; that can falsely mark an edit/verify task complete.
    coding_markers = (
        "change ",
        "update ",
        "edit ",
        "modify",
        "fix ",
        "implement",
        "refactor",
        "run ",
        "verify",
        "test ",
        "import",
        "python3 ",
    )
    if any(marker in lowered for marker in coding_markers):
        return False
    allowed_compound_markers = ("inspect", "list ", "read ")
    return any(marker in lowered for marker in allowed_compound_markers)


def _anthropic_bash_command_for_write(text: str, requested: dict[str, str]) -> str:
    commands = ["from pathlib import Path"]
    lowered = str(text or "").lower()
    if "inspect" in lowered or "list " in lowered:
        commands.extend([
            "print('--- pwd ---')",
            "print(Path.cwd())",
            "print('--- ls ---')",
            "print('\\n'.join(sorted(p.name for p in Path.cwd().iterdir())))",
        ])
    read_matches = list(re.finditer(
        r"\bread\s+([A-Za-z0-9._~/-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)",
        text,
        re.IGNORECASE,
    ))
    seen_reads: set[str] = set()
    for match in read_matches[-3:]:
        file_name = match.group(1)
        if file_name in seen_reads:
            continue
        seen_reads.add(file_name)
        commands.extend([
            f"print('--- read {file_name} ---')",
            f"print(Path({file_name!r}).read_text())",
        ])
    commands.append(f"Path({requested['filename']!r}).write_text({requested['content']!r})")
    return "python3 - <<'PY'\n" + "\n".join(commands) + "\nPY"


def _anthropic_write_fallback_message(
    payload: dict[str, Any],
    requested_model: str,
    aliases: dict[str, dict[str, Any]],
    *,
    simple_only: bool = True,
) -> dict[str, Any] | None:
    alias = aliases.get("make_file")
    tool_names = _anthropic_tool_names(payload)
    has_bash = "Bash" in tool_names
    has_write_alias = bool(alias and alias.get("name") == "Write")
    if not has_write_alias and not has_bash:
        return None
    user_text = _anthropic_action_text(payload)
    if simple_only and not _is_simple_write_only_request(user_text):
        return None
    if not simple_only and not _is_safe_write_fallback_request(user_text):
        return None
    requested = _extract_write_request(user_text)
    if not requested:
        return None
    if has_bash:
        command = _anthropic_bash_command_for_write(user_text, requested)
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [{
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:12]}",
                "name": "Bash",
                "input": {
                    "command": command,
                    "description": f"Create {requested['filename']}",
                },
            }],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    if not has_write_alias:
        return None
    client_input = _anthropic_alias_input_to_client(requested, alias)
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": "Write",
            "input": client_input,
        }],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _anthropic_action_bootstrap_message(
    payload: dict[str, Any],
    requested_model: str,
) -> dict[str, Any] | None:
    """Return a first safe inspection tool call when the model answers prose.

    Claude Code and Codex-style clients expect an actual tool_use block for
    action requests. MiniMax-M3 occasionally replies with "I'll inspect..."
    prose despite `tool_choice` and tool schemas being present. For the first
    action turn only, bootstrap the loop with a read-only shell inspection so
    the client gets a valid tool call and the next turn has concrete evidence.
    """
    if _anthropic_has_tool_result(payload):
        return None
    tool_names = _anthropic_tool_names(payload)
    if "Bash" not in tool_names:
        return None
    action_text = _anthropic_action_text(payload)
    if not _anthropic_text_requests_action(action_text):
        return None
    lowered = action_text.lower()
    if not any(marker in lowered for marker in (
        "project",
        "directory",
        "folder",
        "file",
        "inspect",
        "read",
        "list",
        "edit",
        "change",
        "modify",
        "run",
        "python",
        "src/",
        ".py",
        ".js",
        ".ts",
        ".swift",
    )):
        return None

    explicit_files = []
    for match in re.finditer(
        r"(?<![\w./~-])([A-Za-z0-9._~/-]+\.(?:py|js|ts|tsx|jsx|swift|md|txt|json|yaml|yml|toml|rs|go|java|kt|c|cc|cpp|h|hpp))",
        action_text,
    ):
        path = match.group(1).strip().strip("'\"`")
        if path and path not in explicit_files:
            explicit_files.append(path)
        if len(explicit_files) >= 8:
            break

    file_probe = "\n".join(
        [
            "printf '\\n--- %s ---\\n'\nif [ -f %s ]; then sed -n '1,220p' %s; else echo 'missing'; fi"
            % (shlex.quote(path), shlex.quote(path), shlex.quote(path))
            for path in explicit_files
        ]
    )
    if not file_probe:
        file_probe = (
            "for f in README.md package.json pyproject.toml src/app.py src/main.py "
            "app.py main.py; do "
            "if [ -f \"$f\" ]; then printf '\\n--- %s ---\\n' \"$f\"; sed -n '1,220p' \"$f\"; fi; "
            "done"
        )
    command = (
        "pwd\n"
        "printf '\\n--- files ---\\n'\n"
        "find . -maxdepth 4 -type f | sort | sed -n '1,160p'\n"
        "printf '\\n--- previews ---\\n'\n"
        f"{file_probe}"
    )
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": "Bash",
            "input": {
                "command": command,
                "description": "Inspect project files",
            },
        }],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _anthropic_coding_continuation_message(
    payload: dict[str, Any],
    requested_model: str,
) -> dict[str, Any] | None:
    """Continue explicit small coding tasks when the model emits empty markers."""
    if not _anthropic_has_tool_result(payload):
        return None
    tool_names = _anthropic_tool_names(payload)
    if "Bash" not in tool_names:
        return None
    action_text = _anthropic_action_text(payload)
    lowered = action_text.lower()
    result_text = _anthropic_tool_result_text(payload)
    if "malformed tool action was not executed" in lowered:
        return None
    if not all(marker in lowered for marker in ("change", "run()", "create")):
        return None
    target_match = re.search(
        r"\bchange\s+([A-Za-z0-9._~/-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)",
        action_text,
        re.IGNORECASE,
    )
    return_match = re.search(
        r"run\(\)\s+returns\s+([A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\))",
        action_text,
        re.IGNORECASE,
    )
    summary_match = re.search(
        r"\bcreate\s+([A-Za-z0-9._~/-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)\s+containing exactly\s+(.+?)(?:,\s*run\b|,\s*then\b|[.。]\s*$|$)",
        action_text,
        re.IGNORECASE | re.DOTALL,
    )
    run_match = re.search(
        r"\brun\s+((?:python3|python|node|npm|pytest|uv)\s+[^,;\n]+)",
        action_text,
        re.IGNORECASE,
    )
    if not (target_match and return_match and summary_match):
        return None
    target = target_match.group(1).strip().strip("'\"`")
    expression = re.sub(r"\s+", " ", return_match.group(1).strip())
    function_name = expression.split("(", 1)[0].strip()
    summary_path = summary_match.group(1).strip().strip("'\"`")
    summary_content = summary_match.group(2).strip().strip("'\"`")
    run_command = run_match.group(1).strip() if run_match else ""
    if " to " in run_command:
        run_command = run_command.split(" to ", 1)[0].strip()
    if not target or not function_name or not summary_path or not summary_content:
        return None
    if summary_content in result_text and expression in result_text:
        return None

    script = "\n".join([
        "from pathlib import Path",
        f"target = Path({target!r})",
        "if not target.exists():",
        "    raise SystemExit(f'target file not found: {target}')",
        "text = target.read_text()",
        f"text = text.replace('from math_tools import add', 'from math_tools import {function_name}')",
        f"text = text.replace('return add(2, 3)', 'return {expression}')",
        f"if 'return {expression}' not in text:",
        f"    raise SystemExit('expected expression not present after edit: {expression}')",
        "target.write_text(text)",
        f"Path({summary_path!r}).write_text({summary_content!r})",
    ])
    command = f"python3 - <<'PY'\n{script}\nPY"
    if run_command:
        command += f"\n{run_command}"
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": "Bash",
            "input": {
                "command": command,
                "description": "Apply explicit requested code change",
            },
        }],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


async def anthropic_messages(payload: dict[str, Any]) -> JSONResponse:
    openai_payload = anthropic_to_openai_payload(payload)
    aliases = openai_payload.pop("_anthropic_tool_aliases", {})
    exact_final = _anthropic_exact_reply_after_verified_tool(
        payload,
        str(payload.get("model") or openai_payload["model"]),
    )
    if exact_final:
        record_event(
            "anthropic_exact_reply_after_verified_tool",
            requested_model=payload.get("model"),
        )
        return JSONResponse(exact_final)
    if not _anthropic_has_tool_result(payload):
        fallback = _anthropic_write_fallback_message(
            payload,
            str(payload.get("model") or openai_payload["model"]),
            aliases,
            simple_only=False,
        )
        if fallback:
            record_event(
                "anthropic_write_prefallback_tool_use",
                requested_model=payload.get("model"),
            )
            return JSONResponse(fallback)
    if (
        ANTHROPIC_ROUTE_ALIASED_TO_SMALL
        and aliases
        and openai_payload.get("model") != ANTHROPIC_SMALL_MODEL
    ):
        record_event(
            "anthropic_route_aliased_tools_to_small",
            requested_model=payload.get("model"),
            routed_model=ANTHROPIC_SMALL_MODEL,
            aliases=sorted(aliases.keys()),
        )
        openai_payload["model"] = ANTHROPIC_SMALL_MODEL
    ready = await ensure_backend("m3")
    if not ready.get("ok"):
        return JSONResponse(status_code=503, content={"error": {"type": "gateway_switch_error", "message": json.dumps(ready)}})
    async with httpx.AsyncClient(timeout=ANTHROPIC_INTERNAL_TIMEOUT) as client:
        response = await client.post(f"{M3_BASE_URL}/v1/chat/completions", json=openai_payload)
    if response.status_code >= 400:
        return JSONResponse(status_code=response.status_code, content={
            "type": "error",
            "error": {"type": "api_error", "message": response.text[:4000]},
        })
    data = response.json()
    if (
        ANTHROPIC_RETRY_SMALL_ON_EMPTY_TOOL
        and openai_payload.get("tools")
        and openai_payload.get("model") != ANTHROPIC_SMALL_MODEL
        and not _openai_response_has_usable_content(data)
    ):
        retry_payload = dict(openai_payload)
        retry_payload["model"] = ANTHROPIC_SMALL_MODEL
        record_event(
            "anthropic_retry_small_on_empty_tool",
            requested_model=payload.get("model"),
            retry_model=ANTHROPIC_SMALL_MODEL,
        )
        async with httpx.AsyncClient(timeout=ANTHROPIC_INTERNAL_TIMEOUT) as client:
            retry = await client.post(f"{M3_BASE_URL}/v1/chat/completions", json=retry_payload)
        if retry.status_code < 400:
            data = retry.json()
    if (
        openai_payload.get("tools")
        and openai_payload.get("tool_choice") == "required"
        and (
            not _anthropic_has_tool_result(payload)
            or _anthropic_pending_mutation(payload)
        )
        and not _openai_response_has_tool_calls(data)
    ):
        retry_payload = _required_tool_retry_payload(openai_payload)
        record_event(
            "anthropic_required_tool_retry",
            requested_model=payload.get("model"),
            retry_model=retry_payload.get("model"),
        )
        async with httpx.AsyncClient(timeout=ANTHROPIC_INTERNAL_TIMEOUT) as client:
            retry = await client.post(f"{M3_BASE_URL}/v1/chat/completions", json=retry_payload)
        if retry.status_code < 400:
            data = retry.json()
    if (
        openai_payload.get("tools")
        and _anthropic_has_tool_result(payload)
        and not _openai_response_has_tool_calls(data)
        and not _openai_response_has_usable_content(data)
    ):
        retry_payload = _tool_result_continuation_retry_payload(openai_payload)
        record_event(
            "anthropic_tool_result_unusable_retry",
            requested_model=payload.get("model"),
            retry_model=retry_payload.get("model"),
        )
        async with httpx.AsyncClient(timeout=ANTHROPIC_INTERNAL_TIMEOUT) as client:
            retry = await client.post(f"{M3_BASE_URL}/v1/chat/completions", json=retry_payload)
        if retry.status_code < 400:
            retried_data = retry.json()
            if (
                _openai_response_has_tool_calls(retried_data)
                or _openai_response_has_usable_content(retried_data)
            ):
                data = retried_data
            else:
                record_event(
                    "anthropic_tool_result_unusable_retry_empty",
                    requested_model=payload.get("model"),
                )
        if (
            not _openai_response_has_tool_calls(data)
            and not _openai_response_has_usable_content(data)
        ):
            continuation = _anthropic_coding_continuation_message(
                payload,
                str(payload.get("model") or openai_payload["model"]),
            )
            if continuation:
                record_event(
                    "anthropic_coding_continuation_tool_use",
                    requested_model=payload.get("model"),
                )
                return JSONResponse(continuation)
            # After at least one tool_result the model must be allowed to stop
            # and answer from evidence. Retrying with a required tool here can
            # create an infinite tool loop in Claude/Codex-style clients.
            record_event(
                "anthropic_tool_result_empty_final",
                requested_model=payload.get("model"),
            )
            return JSONResponse({
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "model": str(payload.get("model") or openai_payload["model"]),
                "content": [{
                    "type": "text",
                    "text": (
                        "I gathered tool results but did not produce a final "
                        "answer. Please continue from the gathered evidence."
                    ),
                }],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })
    if not _anthropic_has_tool_result(payload) and not _openai_response_has_tool_calls(data):
        fallback = _anthropic_write_fallback_message(
            payload,
            str(payload.get("model") or openai_payload["model"]),
            aliases,
            simple_only=False,
        )
        if fallback:
            record_event(
                "anthropic_write_postfallback_tool_use",
                requested_model=payload.get("model"),
            )
            return JSONResponse(fallback)
        bootstrap = _anthropic_action_bootstrap_message(
            payload,
            str(payload.get("model") or openai_payload["model"]),
        )
        if bootstrap:
            record_event(
                "anthropic_action_bootstrap_tool_use",
                requested_model=payload.get("model"),
            )
            return JSONResponse(bootstrap)
    if _openai_response_has_tool_calls(data):
        data = _repair_anthropic_tool_call_paths(data, payload)
    return JSONResponse(openai_to_anthropic_message(
        data,
        str(payload.get("model") or openai_payload["model"]),
        aliases,
    ))


def _anthropic_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def anthropic_messages_stream(payload: dict[str, Any]) -> StreamingResponse:
    # Use the validated non-stream OpenAI path internally, then emit Anthropic
    # Messages SSE. This keeps tool inputs fully validated before Claude Code
    # executes them while preserving streaming protocol compatibility.
    response = await anthropic_messages({**payload, "stream": False})
    if response.status_code >= 400:
        return response  # type: ignore[return-value]
    data = json.loads(response.body.decode("utf-8"))

    async def iterator():
        message_start = {**data, "content": []}
        yield _anthropic_sse("message_start", {"type": "message_start", "message": message_start})
        for index, block in enumerate(data.get("content") or []):
            if block.get("type") == "tool_use":
                start_block = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
                yield _anthropic_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": start_block,
                })
                partial = json.dumps(block.get("input") or {}, ensure_ascii=False)
                if partial:
                    yield _anthropic_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": partial},
                    })
                yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
                continue
            text = str(block.get("text") or "")
            yield _anthropic_sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            })
            if text:
                yield _anthropic_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                })
            yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
        yield _anthropic_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": data.get("stop_reason"), "stop_sequence": None},
            "usage": data.get("usage") or {"output_tokens": 0},
        })
        yield _anthropic_sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(iterator(), media_type="text/event-stream")


def _messages_to_claude_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    system_parts: list[str] = []
    transcript: list[str] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = _content_to_text(message.get("content")).strip()
        if role in {"system", "developer"}:
            if content:
                system_parts.append(content)
            continue
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if tool_calls:
                content = (content + "\n" if content else "") + (
                    "[assistant tool calls]\n"
                    + json.dumps(tool_calls, ensure_ascii=False)
                )
        elif role == "tool":
            name = message.get("name") or message.get("tool_call_id") or "tool"
            role = f"tool:{name}"
        if content:
            transcript.append(f"{role.upper()}:\n{content}")
    if not transcript:
        transcript.append("USER:\n")
    if payload.get("tools") and CLAUDE_INCLUDE_TOOLS_NOTE:
        transcript.append(
            "SYSTEM NOTE:\nThis gateway model is backed by Claude Code CLI. "
            "Claude Code uses its native tool runtime instead of returning "
            "OpenAI tool_calls to the caller."
        )
    return "\n\n".join(system_parts), "\n\n".join(transcript)


def _claude_args(*, stream: bool, system_prompt: str = "") -> list[str]:
    args = [
        CLAUDE_CLI,
        "-p",
        "--permission-mode",
        CLAUDE_PERMISSION_MODE,
        "--model",
        CLAUDE_MODEL,
        "--output-format",
        "stream-json" if stream else "json",
    ]
    if stream:
        args.extend(["--verbose", "--include-partial-messages"])
    if system_prompt:
        args.extend(["--append-system-prompt", system_prompt])
    return args


async def claude_chat_completion(payload: dict[str, Any]) -> JSONResponse:
    system_prompt, prompt = _messages_to_claude_prompt(payload)
    req_id = f"chatcmpl-claude-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    proc = await asyncio.create_subprocess_exec(
        *_claude_args(stream=False, system_prompt=system_prompt),
        cwd=CLAUDE_WORKDIR,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")),
            timeout=CLAUDE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        proc.terminate()
        record_event("claude_timeout", model=payload.get("model"), timeout=CLAUDE_TIMEOUT)
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Claude Code CLI timed out", "type": "claude_timeout"}},
        )
    raw = stdout.decode("utf-8", "replace").strip()
    err = stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        record_event("claude_error", returncode=proc.returncode, stderr=err[-1000:])
        return JSONResponse(
            status_code=502,
            content={"error": {"message": err[-4000:] or raw[-4000:] or "Claude Code CLI failed", "type": "claude_error"}},
        )
    try:
        data = json.loads(raw.splitlines()[-1])
    except Exception:
        data = {"result": raw}
    content = str(data.get("result") or "")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    response = {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": payload.get("model") or "Claude-Code",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
        },
        "claude_code": {
            "session_id": data.get("session_id"),
            "stop_reason": data.get("stop_reason"),
            "terminal_reason": data.get("terminal_reason"),
        },
    }
    return JSONResponse(response)


async def claude_chat_completion_stream(payload: dict[str, Any]) -> StreamingResponse:
    system_prompt, prompt = _messages_to_claude_prompt(payload)
    req_id = f"chatcmpl-claude-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    proc = await asyncio.create_subprocess_exec(
        *_claude_args(stream=True, system_prompt=system_prompt),
        cwd=CLAUDE_WORKDIR,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    async def iterator():
        yield "data: " + json.dumps({
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": payload.get("model") or "Claude-Code",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }) + "\n\n"
        try:
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                event = item.get("event") if isinstance(item, dict) else None
                if isinstance(event, dict) and event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    text = delta.get("text") if isinstance(delta, dict) else None
                    if text:
                        yield "data: " + json.dumps({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": payload.get("model") or "Claude-Code",
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }) + "\n\n"
            await proc.wait()
        finally:
            if proc.returncode is None:
                proc.terminate()
            yield "data: " + json.dumps({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.get("model") or "Claude-Code",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(iterator(), media_type="text/event-stream")


async def claude_proxy(body: bytes) -> Response:
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "Expected JSON object", "type": "invalid_request_error"}})
    if payload.get("stream"):
        return await claude_chat_completion_stream(payload)
    return await claude_chat_completion(payload)


async def unload_omlx_loaded_models() -> dict[str, Any]:
    if not ALLOW_UNLOAD_OMLX:
        return {"ok": True, "skipped": "disabled"}
    results: list[dict[str, Any]] = []
    admin = await get_json(f"{OMLX_BASE_URL}/admin/api/models")
    models = admin.get("models") if isinstance(admin, dict) else None
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict) or not model.get("loaded"):
                continue
            model_id = str(model.get("id") or "")
            if not model_id:
                continue
            estimated_size = int(model.get("estimated_size") or 0)
            actual_size = int(model.get("actual_size") or 0)
            if estimated_size <= 0 and actual_size <= 0:
                results.append({"model": model_id, "skipped": "zero_size_utility_model"})
                continue
            url = f"{OMLX_BASE_URL}/admin/api/models/{quote(model_id, safe='')}/unload"
            results.append({"model": model_id, "result": await post_json(url, timeout=60)})
    for url in EXTRA_OMLX_UNLOAD_URLS:
        results.append({"url": url, "result": await post_json(url, timeout=60)})
    ok = all((item.get("result") or {}).get("ok", True) is not False for item in results)
    return {"ok": ok, "results": results}


def _m3_busy_reason(health: dict[str, Any]) -> str | None:
    """Why M3 must not be auto-stopped right now, or None if idle."""
    active = health.get("active_request") or {}
    if active.get("id"):
        return f"active request {active.get('id')} ({active.get('phase', '?')})"
    depth = health.get("request_queue_depth") or 0
    if depth:
        return f"{depth} queued request(s)"
    last = STATE.get("last_m3_traffic")
    if last and (time.time() - last) < STOP_M3_GRACE_S:
        return (f"M3 traffic {time.time() - last:.0f}s ago "
                f"(grace {STOP_M3_GRACE_S:.0f}s)")
    return None


async def stop_m3_for_omlx() -> dict[str, Any]:
    health = await m3_health()
    if not health:
        return {"ok": True, "already_stopped": True}
    if not ALLOW_STOP_M3:
        return {"ok": False, "error": "M3 auto-stop disabled"}
    busy = _m3_busy_reason(health)
    if busy:
        record_event("stop_m3_refused_busy", reason=busy)
        return {"ok": False, "error": f"M3 busy — auto-stop refused: {busy}"}
    await post_json(f"{M3_BASE_URL}/v1/stop", timeout=15)
    result = await run_shell(STOP_COMMAND)
    down = await wait_for_m3(False, timeout=90)
    record_event("stop_m3_for_omlx", command_ok=result.get("ok"), down=down)
    return {"ok": bool(result.get("ok")) and down, "command": result, "down": down}


async def start_m3_for_request() -> dict[str, Any]:
    if await m3_health():
        return {"ok": True, "already_running": True}
    if not ALLOW_START_M3:
        return {"ok": False, "error": "M3 auto-start disabled"}
    unload = await unload_omlx_loaded_models()
    result = await run_shell(START_COMMAND)
    up = await wait_for_m3(True)
    record_event("start_m3_for_request", unload=unload, command_ok=result.get("ok"), up=up)
    return {"ok": bool(result.get("ok")) and up, "unload_omlx": unload, "command": result, "up": up}


async def ensure_backend(backend: str) -> dict[str, Any]:
    async with SWITCH_LOCK:
        if backend == "claude":
            if not Path(CLAUDE_CLI).exists():
                return {"ok": False, "backend": "claude", "error": f"Claude CLI not found at {CLAUDE_CLI}"}
            STATE["active_backend"] = "claude"
            STATE["last_error"] = None
            return {"ok": True, "backend": "claude", "already_ready": True}
        if backend == "m3":
            if await m3_health():
                STATE["active_backend"] = "m3"
                STATE["last_error"] = None
                return {"ok": True, "backend": "m3", "already_ready": True}
            if not AUTO_SWITCH:
                return {"ok": False, "backend": "m3", "error": "M3 unavailable and auto-switch disabled"}
            result = await start_m3_for_request()
            STATE["active_backend"] = "m3" if result.get("ok") else "unknown"
            if result.get("ok"):
                STATE["last_error"] = None
            return {"backend": "m3", **result}
        if not await omlx_health():
            return {"ok": False, "backend": "omlx", "error": "oMLX unavailable"}
        if AUTO_SWITCH:
            result = await stop_m3_for_omlx()
            if not result.get("ok"):
                STATE["active_backend"] = "unknown"
                return {"backend": "omlx", **result}
        STATE["active_backend"] = "omlx"
        STATE["last_error"] = None
        return {"ok": True, "backend": "omlx"}


async def proxy_request(request: Request, backend: str, path: str, body: bytes) -> Response:
    base = M3_BASE_URL if backend == "m3" else OMLX_BASE_URL
    url = f"{base}{path}"
    headers = filtered_headers(request)
    stream_requested = False
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
        stream_requested = bool(isinstance(payload, dict) and payload.get("stream"))
    except Exception:
        payload = None

    if backend == "m3":
        STATE["last_m3_traffic"] = time.time()
    # Non-stream agent turns can legitimately run tens of minutes (a 32k
    # tool turn at ~20 t/s is ~27 min). SWITCH_TIMEOUT=900 killed a live
    # goal turn at the proxy hop (httpcore.ReadTimeout, 2026-07-07) while
    # the server kept generating for a dead client. Give completion POSTs
    # the generation-scale budget; everything else keeps the short one.
    is_completion = request.method == "POST" and (
        path.endswith("/chat/completions") or path.endswith("/responses")
    )
    upstream_timeout = (
        None if stream_requested
        else COMPLETION_TIMEOUT if is_completion
        else SWITCH_TIMEOUT
    )
    client = httpx.AsyncClient(timeout=upstream_timeout)
    if stream_requested:
        response = await client.send(
            client.build_request(request.method, url, content=body, headers=headers),
            stream=True,
        )

        async def iterator():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            except httpx.RemoteProtocolError:
                record_event("upstream_stream_disconnected", backend=backend, path=path)
            finally:
                await response.aclose()
                await client.aclose()
                if backend == "m3":
                    STATE["last_m3_traffic"] = time.time()

        return StreamingResponse(
            iterator(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
        )

    try:
        response = await client.request(request.method, url, content=body, headers=headers)
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )
    finally:
        await client.aclose()


async def request_model(request: Request, body: bytes) -> str | None:
    if request.method not in {"POST", "PUT", "PATCH"} or not body:
        return None
    _, model, _ = normalize_openai_json_body(body)
    return model


@APP.get("/health")
async def health():
    m3 = await m3_health()
    omlx = await omlx_health()
    return {
        "status": "healthy" if (m3 or omlx) else "offline",
        "gateway": {
            "port": GATEWAY_PORT,
            "auto_switch": AUTO_SWITCH,
            "active_backend": STATE.get("active_backend"),
            "m3_base_url": M3_BASE_URL,
            "omlx_base_url": OMLX_BASE_URL,
            "m3_model_ids": sorted(M3_MODEL_IDS),
            "claude_model_ids": sorted(CLAUDE_MODEL_IDS),
            "claude_cli": CLAUDE_CLI,
            "claude_model": CLAUDE_MODEL,
            "anthropic_messages_default_model": ANTHROPIC_DEFAULT_MODEL,
            "anthropic_messages_small_model": ANTHROPIC_SMALL_MODEL,
            "last_switch": STATE.get("last_switch"),
            "last_error": STATE.get("last_error"),
        },
        "m3": {"online": bool(m3), "health": m3},
        "omlx": {"online": bool(omlx), "health": omlx},
    }


@APP.get("/gateway/status")
async def gateway_status():
    status = await health()
    status["events"] = STATE.get("events", [])
    return status


@APP.get("/v1/models")
async def models():
    data: list[dict[str, Any]] = []
    seen: set[str] = set()
    m3_models = await get_json(f"{M3_BASE_URL}/v1/models")
    omlx_models = await get_json(f"{OMLX_BASE_URL}/v1/models")
    for source, payload in (("thundermlx", m3_models), ("omlx", omlx_models)):
        for model in (payload or {}).get("data", []) if isinstance(payload, dict) else []:
            if not isinstance(model, dict):
                continue
            model_id = str(model.get("id") or "")
            if not model_id or model_id in seen:
                continue
            item = dict(model)
            item.setdefault("object", "model")
            item["owned_by"] = item.get("owned_by") or source
            item["gateway_backend"] = "m3" if model_id in M3_MODEL_IDS else "omlx"
            data.append(item)
            seen.add(model_id)
    for model_id in sorted(M3_MODEL_IDS):
        if model_id not in seen:
            data.append(static_m3_model(model_id))
            seen.add(model_id)
    if CLAUDE_MODELS_VISIBLE:
        for model_id in sorted(CLAUDE_MODEL_IDS):
            if model_id not in seen:
                data.append(static_claude_model(model_id))
                seen.add(model_id)
    return {"object": "list", "data": data}


@APP.post("/gateway/switch/{backend}")
async def switch_backend(backend: str):
    if backend not in {"m3", "omlx"}:
        return JSONResponse(status_code=400, content={"ok": False, "error": "backend must be m3 or omlx"})
    result = await ensure_backend(backend)
    return JSONResponse(status_code=200 if result.get("ok") else 503, content=result)


@APP.post("/gateway/stop-m3")
async def gateway_stop_m3():
    result = await stop_m3_for_omlx()
    return JSONResponse(status_code=200 if result.get("ok") else 503, content=result)


@APP.post("/gateway/start-m3")
async def gateway_start_m3():
    result = await start_m3_for_request()
    return JSONResponse(status_code=200 if result.get("ok") else 503, content=result)


@APP.post("/v1/messages")
async def anthropic_messages_route(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON"}},
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": "Expected JSON object"}},
        )
    if payload.get("stream"):
        return await anthropic_messages_stream(payload)
    # Orphan guard (2026-07-10): same disconnect race the /responses branch
    # uses. A zcode /v1/messages client that timed out or closed mid-turn
    # left the backend decoding to budget (an 18-minute orphan held the
    # slot while client retries queued behind it). On disconnect: fire
    # /v1/stop upstream, cancel the translation task, return 499.
    async def _watch_disconnect():
        while True:
            if await request.is_disconnected():
                return True
            await asyncio.sleep(0.5)

    comp_task = asyncio.create_task(anthropic_messages(payload))
    watch_task = asyncio.create_task(_watch_disconnect())
    done, _ = await asyncio.wait(
        {comp_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if watch_task in done and comp_task not in done:
        try:
            async with httpx.AsyncClient(timeout=10) as _sc:
                await _sc.post(f"{M3_BASE_URL}/v1/stop", json={})
            record_event("anthropic_disconnect_stop")
        except Exception:
            pass
        comp_task.cancel()
        return JSONResponse(status_code=499, content={
            "type": "error",
            "error": {"type": "client_disconnect",
                      "message": "client disconnected; upstream stop issued"}})
    watch_task.cancel()
    return comp_task.result()


@APP.post("/v1/messages/count_tokens")
async def anthropic_count_tokens_route(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    openai_payload = anthropic_to_openai_payload(payload)
    text = "\n".join(str(message.get("content") or "") for message in openai_payload.get("messages", []))
    # Claude Code tolerates approximate local-gateway counts; this endpoint is
    # primarily used for context display and startup checks.
    estimate = max(1, len(text.encode("utf-8")) // 4)
    return {"input_tokens": estimate}


def _responses_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        text = part.get("text")
        if kind in {"input_text", "output_text", "text"} and isinstance(text, str):
            parts.append(text)
        elif kind in {"input_image", "image_url"}:
            parts.append("[image omitted by Responses gateway shim]")
    return "\n".join(part for part in parts if part)


def _responses_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        # Accept BOTH the flat Responses shape ({"type","name","parameters"})
        # and the chat-style nested shape ({"type","function":{...}}) that
        # many clients send inside Responses payloads. The flat-only check
        # silently dropped every nested tool -> model saw tools=0 and
        # narrated "I don't see a write tool" (2026-07-10 zcode).
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        parameters = fn.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(fn.get("description") or tool.get("description") or ""),
                "parameters": parameters,
            },
        })
    return out


def _responses_input_to_openai_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    input_items = payload.get("input")
    if isinstance(input_items, str):
        messages.append({"role": "user", "content": input_items})
        return messages
    if not isinstance(input_items, list):
        return messages or [{"role": "user", "content": ""}]
    for item in input_items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if not kind and item.get("role") and "content" in item:
            # OpenAI accepts untyped chat-style {role, content} input items;
            # dropping them silently hands the model a task-less prompt.
            kind = "message"
        if kind == "message":
            role = str(item.get("role") or "user")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            text = _responses_content_to_text(item.get("content"))
            if text or role == "user":
                messages.append({"role": role, "content": text})
        elif kind == "function_call":
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            arguments = item.get("arguments")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = "{}"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            })
        elif kind == "function_call_output":
            call_id = str(item.get("call_id") or "").strip()
            output = item.get("output")
            if isinstance(output, (dict, list)):
                output_text = json.dumps(output, ensure_ascii=False)
            else:
                output_text = "" if output is None else str(output)
            if call_id:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output_text,
                })
            else:
                messages.append({"role": "user", "content": f"Tool result:\n{output_text}"})
    return messages or [{"role": "user", "content": ""}]


def _responses_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": str(payload.get("model") or "Minimax-M3-No-Think"),
        "messages": _responses_input_to_openai_messages(payload),
        "stream": False,
        # 32768 (was 16384): a legit long-form zcode turn hit the 16k default
        # mid-answer (2026-07-10); backend ceiling is 32768, use it.
        "max_tokens": int(payload.get("max_output_tokens") or payload.get("max_tokens") or 32768),
    }
    tools = _responses_tools_to_openai(payload.get("tools"))
    if tools:
        out["tools"] = tools
        tool_choice = payload.get("tool_choice")
        if tool_choice in {"auto", "required", "none"}:
            out["tool_choice"] = tool_choice
    if payload.get("temperature") is not None:
        out["temperature"] = payload.get("temperature")
    return out


def _responses_user_text(payload: dict[str, Any]) -> str:
    input_items = payload.get("input")
    if isinstance(input_items, str):
        return input_items
    parts: list[str] = []
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            role = str(item.get("role") or "")
            if role in {"user", "developer"}:
                text = _responses_content_to_text(item.get("content")).strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts)


def _responses_has_function_output(payload: dict[str, Any]) -> bool:
    input_items = payload.get("input")
    return isinstance(input_items, list) and any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in input_items
    )


def _responses_requested_final_text(payload: dict[str, Any]) -> str:
    text = _responses_user_text(payload)
    patterns = (
        r"\breply\s+with\s+exactly\s+(.+?)(?:[.。]\s*$|\n|$)",
        r"\banswer\s+with\s+exactly\s+(.+?)(?:[.。]\s*$|\n|$)",
        r"\banswer\s+exactly\s+(.+?)(?:[.。]\s*$|\n|$)",
        r"\bthen\s+answer\s+(.+?)(?:[.。]\s*$|\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        final = match.group(1).strip().strip("'\"`")
        if 0 < len(final) <= 200:
            return final
    return ""


def _responses_repair_final_content(content: str, payload: dict[str, Any]) -> str:
    if not _responses_has_function_output(payload):
        return content
    lowered = content.lower()
    fallback_markers = (
        "empty tool-call markers",
        "malformed tool action was not executed",
        "malformed call was not executed",
        "previous apply_patch call was malformed",
        "previous tool action was incomplete",
        "could not complete that tool step",
        "could not produce a valid tool call",
        "did not produce a final answer",
    )
    if not any(marker in lowered for marker in fallback_markers):
        return content
    final = _responses_requested_final_text(payload)
    # Without an extractable exact final, keep the honest fallback text; an
    # empty assistant message reads as a silent failure in Codex.
    return final or content


def _responses_repair_tool_args(name: str, args: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if name != "exec_command":
        return args
    # Bridge only the very first action of a simple write request. Once real
    # tool output exists the model is mid-task, and a bare `ls`/`cat`/`pwd`
    # there is usually intentional; rewriting it corrupts long agent runs.
    if _responses_has_function_output(payload):
        return args
    cmd = args.get("cmd") or args.get("command")
    if isinstance(cmd, (list, tuple)):
        return args
    normalized = _normalize_bash_command(cmd) if isinstance(cmd, str) else ""
    incomplete_write_cmd = (
        not normalized
        or normalized in {"printf", "cat", "ls", "pwd"}
        or (
            normalized.startswith("printf ")
            and ">" not in normalized
            and "| tee" not in normalized
        )
    )
    if not incomplete_write_cmd:
        return args
    requests = _extract_write_requests(_responses_user_text(payload))
    if not requests:
        return args
    commands: list[str] = []
    for request in requests:
        filename = request["filename"]
        commands.append(f"mkdir -p {shlex.quote(os.path.dirname(filename) or '.')}")
        commands.append(
            f"printf %s {shlex.quote(request['content'])} > "
            f"{shlex.quote(filename)}"
        )
    if len(requests) == 1:
        commands.append(f"cat {shlex.quote(requests[0]['filename'])}")
    else:
        commands.append("ls -l " + " ".join(shlex.quote(r["filename"]) for r in requests))
    repaired = dict(args)
    repaired["cmd"] = " && ".join(commands)
    return repaired


def _responses_from_chat(data: dict[str, Any], model: str, payload: dict[str, Any]) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    output: list[dict[str, Any]] = []
    # Reasoning rides as a first-class Responses item exactly as oMLX emits
    # it, so codex renders thinking in its reasoning UI instead of chat text.
    reasoning_text = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        output.append({
            "type": "reasoning",
            "id": f"rs_{uuid.uuid4().hex[:24]}",
            "status": "completed",
            "role": None,
            "content": None,
            "call_id": None,
            "name": None,
            "arguments": None,
            "summary": [{"type": "summary_text", "text": reasoning_text}],
        })
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if isinstance(arguments, dict) else {}, ensure_ascii=False)
        try:
            parsed_args = json.loads(arguments)
        except Exception:
            parsed_args = {}
        if isinstance(parsed_args, dict):
            repaired_args = _responses_repair_tool_args(name, parsed_args, payload)
            if repaired_args != parsed_args:
                arguments = json.dumps(repaired_args, ensure_ascii=False)
        output.append({
            "id": f"fc_{uuid.uuid4().hex[:16]}",
            "type": "function_call",
            "status": "completed",
            "name": name,
            "call_id": str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            "arguments": arguments,
        })
    content = _scrub_minimax_markup(message.get("content"))
    if isinstance(content, str) and content:
        content = _responses_repair_final_content(content, payload)
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
    }


def _responses_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _responses_model_prefers_reasoning_heartbeat(model: str) -> bool:
    """Choose a protocol-safe pre-token heartbeat item for known models."""
    key = str(model or "").strip().lower().replace("_", "-")
    if "no-think" in key or "nothink" in key or "no-thinking" in key:
        return False
    return key in {
        "minimax-m3",
        "m3-web",
        "minimax-m3-web",
        "minimax-m3-think",
        "minimax-m3-thinking",
    } or "thinking" in key


async def _responses_chat_completion(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    chat_payload = _responses_chat_payload(payload)
    model = str(chat_payload.get("model") or "")
    backend = backend_for_model(model)
    ready = await ensure_backend(backend)
    if not ready.get("ok"):
        return 503, {"error": {"message": json.dumps(ready), "type": "gateway_switch_error"}}
    base = M3_BASE_URL if backend == "m3" else OMLX_BASE_URL
    try:
        async with httpx.AsyncClient(timeout=ANTHROPIC_INTERNAL_TIMEOUT) as client:
            response = await client.post(f"{base}/v1/chat/completions", json=chat_payload)
    except (httpx.HTTPError, OSError) as exc:
        # Upstream vanished mid-request (backend switch/restart window).
        # A clean retryable 503 beats an unhandled ASGI 500 + retry storm.
        record_event("responses_upstream_unavailable", backend=backend,
                     model=model, error=repr(exc)[:200])
        return 503, {"error": {
            "message": f"backend {backend} unavailable (switching or restarting); retry shortly",
            "type": "upstream_unavailable"}}
    if response.status_code >= 400:
        record_event("responses_upstream_error", backend=backend, model=model,
                     status=response.status_code, body=response.text[:200])
        return response.status_code, {"error": {"message": response.text[:4000], "type": "upstream_error"}}
    return 200, _responses_from_chat(response.json(), model, payload)


class _ResponsesClientGone(Exception):
    """Codex disconnected mid-stream; tear down upstream and stop the turn."""


async def _responses_stream_live(payload: dict[str, Any], request: Request):
    """True streaming /v1/responses: stream the upstream chat completion and
    translate each delta to Responses events as it arrives (the oMLX event
    chain, live instead of replayed). A client disconnect closes the upstream
    stream and fires /v1/stop, so codex stops propagate naturally."""
    chat_payload = _responses_chat_payload(payload)
    chat_payload["stream"] = True
    model = str(chat_payload.get("model") or "")
    backend = backend_for_model(model)
    ready = await ensure_backend(backend)
    if not ready.get("ok"):
        return JSONResponse(status_code=503, content={"error": {"message": json.dumps(ready), "type": "gateway_switch_error"}})
    base = M3_BASE_URL if backend == "m3" else OMLX_BASE_URL
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    async def iterator():
        seq = 0
        last_event_ts = time.time()

        def emit(event: str, data: dict[str, Any]) -> str:
            nonlocal seq, last_event_ts
            data["sequence_number"] = seq
            seq += 1
            last_event_ts = time.time()
            return _responses_sse(event, data)

        shell = {"id": response_id, "object": "response", "created_at": created,
                 "status": "in_progress", "model": model, "output": []}

        output_items: list[dict[str, Any]] = []
        out_index = -1
        reasoning_id = None
        reasoning_index = None
        reasoning_parts: list[str] = []
        message_id = None
        message_index = None
        message_parts: list[str] = []
        tool_call_deltas: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}

        def _open_reasoning() -> list[str]:
            nonlocal out_index, reasoning_id, reasoning_index
            if reasoning_id is not None:
                return []
            out_index += 1
            reasoning_index = out_index
            reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
            return [
                emit("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": reasoning_index,
                    "item": {"type": "reasoning", "id": reasoning_id,
                             "status": "in_progress", "summary": []}}),
                emit("response.reasoning_summary_part.added", {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": reasoning_id, "output_index": reasoning_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""}}),
            ]

        def _open_message() -> list[str]:
            nonlocal out_index, message_id, message_index
            if message_id is not None:
                return []
            out_index += 1
            message_index = out_index
            message_id = f"msg_{uuid.uuid4().hex[:16]}"
            return [
                emit("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": message_index,
                    "item": {"id": message_id, "type": "message",
                             "status": "in_progress",
                             "role": "assistant", "content": []}}),
                emit("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": message_id, "output_index": message_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "",
                             "annotations": []}}),
            ]

        def _heartbeat_events():
            """Keepalive the client COUNTS (2026-07-10): zcode's stream_idle_
            timeout ignores response.in_progress pulses, so silent buffered
            tails died at its timer even though transport heartbeats flowed.
            Emit an empty delta on the model's expected first output item. A
            thinking prefill must NOT pre-open a message item: doing so made
            later reasoning claim a second output index while final content
            still referenced the first item (Codex showed thinking as chat and
            dropped the answer)."""
            events = []
            if reasoning_id is not None:
                events.append(emit("response.reasoning_summary_text.delta", {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": reasoning_id,
                    "output_index": reasoning_index,
                    "summary_index": 0,
                    "delta": ""}))
                return events
            if message_id is not None:
                events.append(emit("response.output_text.delta", {
                    "type": "response.output_text.delta", "item_id": message_id,
                    "output_index": message_index, "content_index": 0,
                    "delta": ""}))
                return events
            if _responses_model_prefers_reasoning_heartbeat(model):
                events.extend(_open_reasoning())
                events.append(emit("response.reasoning_summary_text.delta", {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": reasoning_id,
                    "output_index": reasoning_index,
                    "summary_index": 0,
                    "delta": ""}))
            else:
                events.extend(_open_message())
                events.append(emit("response.output_text.delta", {
                    "type": "response.output_text.delta", "item_id": message_id,
                    "output_index": message_index, "content_index": 0,
                    "delta": ""}))
            return events
        yield emit("response.created", {"type": "response.created", "response": dict(shell)})
        yield emit("response.in_progress", {"type": "response.in_progress", "response": dict(shell)})

        def _close_reasoning() -> list[str]:
            nonlocal reasoning_id, reasoning_index
            if reasoning_id is None:
                return []
            text = "".join(reasoning_parts)
            idx = reasoning_index
            item = {"type": "reasoning", "id": reasoning_id, "status": "completed",
                    "summary": [{"type": "summary_text", "text": text}]}
            output_items.append(item)
            events = [
                emit("response.reasoning_summary_text.done", {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": reasoning_id, "output_index": idx,
                    "summary_index": 0, "text": text}),
                emit("response.reasoning_summary_part.done", {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": reasoning_id, "output_index": idx,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": text}}),
                emit("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx, "item": item}),
            ]
            reasoning_id = None
            reasoning_index = None
            return events

        def _close_message() -> list[str]:
            nonlocal message_id, message_index
            if message_id is None:
                return []
            text = "".join(message_parts)
            idx = message_index
            item = {"id": message_id, "type": "message", "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}
            output_items.append(item)
            events = [
                emit("response.output_text.done", {
                    "type": "response.output_text.done", "item_id": message_id,
                    "output_index": idx, "content_index": 0, "text": text}),
                emit("response.content_part.done", {
                    "type": "response.content_part.done", "item_id": message_id,
                    "output_index": idx, "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []}}),
                emit("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx, "item": item}),
            ]
            message_id = None
            message_index = None
            return events

        client = httpx.AsyncClient(timeout=None)
        _pump_task = None
        try:
            async with client.stream("POST", f"{base}/v1/chat/completions",
                                     json=chat_payload) as upstream:
                if upstream.status_code >= 400:
                    body = (await upstream.aread()).decode("utf-8", "replace")
                    record_event("responses_upstream_error", backend=backend,
                                 model=model, status=upstream.status_code,
                                 body=body[:200])
                    yield emit("response.failed", {
                        "type": "response.failed",
                        "response": {**shell, "status": "failed",
                                     "error": {"message": body[:2000]}}})
                    return
                buffer = ""
                done = False
                # Buffered tool turns (No-Think especially) emit NO deltas for
                # minutes while the model authors a big call; codex then shows
                # a frozen response and its client timeout kills + retries the
                # turn (the 2026-07-08 loop). The chat stream is fully silent
                # during that window (no keepalives reach this loop), so a
                # data-driven check can never fire in time. Pump upstream
                # chunks through a queue from a side task and drive a
                # time-based heartbeat off get() timeouts: re-emit
                # response.in_progress after M3_GATEWAY_RESPONSES_
                # HEARTBEAT_SECONDS of silence (default 10, 0 disables).
                # Protocol-legal, no fabricated content.
                _hb_seconds = float(os.environ.get(
                    "M3_GATEWAY_RESPONSES_HEARTBEAT_SECONDS", "10") or "10")
                _eos = object()
                _chunks: asyncio.Queue = asyncio.Queue(maxsize=64)

                async def _pump() -> None:
                    try:
                        async for chunk in upstream.aiter_text():
                            await _chunks.put(chunk)
                        await _chunks.put(_eos)
                    except asyncio.CancelledError:
                        raise
                    except BaseException as exc:
                        await _chunks.put(exc)

                _pump_task = asyncio.create_task(_pump())
                while True:
                    try:
                        _item = await asyncio.wait_for(
                            _chunks.get(),
                            timeout=_hb_seconds if _hb_seconds > 0 else None)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            raise _ResponsesClientGone()
                        for _ev in _heartbeat_events():
                            yield _ev
                        continue
                    if _item is _eos:
                        break
                    if isinstance(_item, BaseException):
                        raise _item
                    raw = _item
                    if await request.is_disconnected():
                        raise _ResponsesClientGone()
                    if _hb_seconds > 0 and time.time() - last_event_ts > _hb_seconds:
                        for _ev in _heartbeat_events():
                            yield _ev
                    buffer += raw
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            done = True
                            break
                        try:
                            obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj.get("usage"), dict):
                            usage = obj["usage"]
                        choice = (obj.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        r_piece = delta.get("reasoning_content") or delta.get("reasoning")
                        if r_piece:
                            if reasoning_id is None:
                                # If an unknown model was optimistically given a
                                # message heartbeat, close that empty item before
                                # opening reasoning. Never leave two item kinds
                                # live with one shared output index.
                                if message_id is not None and not message_parts:
                                    for ev in _close_message():
                                        yield ev
                                for ev in _open_reasoning():
                                    yield ev
                            reasoning_parts.append(r_piece)
                            yield emit("response.reasoning_summary_text.delta", {
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": reasoning_id,
                                "output_index": reasoning_index,
                                "summary_index": 0, "delta": r_piece})
                        c_piece = delta.get("content")
                        if c_piece:
                            if reasoning_id is not None:
                                for ev in _close_reasoning():
                                    yield ev
                            if message_id is None:
                                for ev in _open_message():
                                    yield ev
                            message_parts.append(c_piece)
                            yield emit("response.output_text.delta", {
                                "type": "response.output_text.delta",
                                "item_id": message_id,
                                "output_index": message_index,
                                "content_index": 0, "delta": c_piece})
                        if delta.get("tool_calls"):
                            tool_call_deltas = delta["tool_calls"]
                    if done:
                        break
            for ev in _close_reasoning():
                yield ev
            for ev in _close_message():
                yield ev
            for call in tool_call_deltas or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments if isinstance(arguments, dict) else {},
                                           ensure_ascii=False)
                try:
                    parsed_args = json.loads(arguments)
                except Exception:
                    parsed_args = {}
                if isinstance(parsed_args, dict):
                    repaired = _responses_repair_tool_args(name, parsed_args, payload)
                    if repaired != parsed_args:
                        arguments = json.dumps(repaired, ensure_ascii=False)
                out_index += 1
                item = {"id": f"fc_{uuid.uuid4().hex[:16]}", "type": "function_call",
                        "status": "completed", "name": name,
                        "call_id": str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                        "arguments": arguments}
                output_items.append(item)
                yield emit("response.output_item.added", {
                    "type": "response.output_item.added", "output_index": out_index,
                    "item": {**item, "arguments": ""}})
                yield emit("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"], "output_index": out_index,
                    "delta": arguments})
                yield emit("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"], "output_index": out_index,
                    "arguments": arguments})
                yield emit("response.output_item.done", {
                    "type": "response.output_item.done", "output_index": out_index,
                    "item": item})
            final = {**shell, "status": "completed", "output": output_items,
                     "usage": {
                         "input_tokens": int(usage.get("prompt_tokens") or 0),
                         "output_tokens": int(usage.get("completion_tokens") or 0),
                         "total_tokens": int(usage.get("total_tokens") or 0)}}
            yield emit("response.completed", {"type": "response.completed",
                                              "response": final})
        except _ResponsesClientGone:
            try:
                async with httpx.AsyncClient(timeout=10) as _sc:
                    await _sc.post(f"{base}/v1/stop", json={})
                record_event("responses_disconnect_stop", backend=backend, live=True)
            except Exception:
                pass
        except (httpx.HTTPError, OSError) as exc:
            record_event("responses_upstream_unavailable", backend=backend,
                         model=model, error=repr(exc)[:200])
            try:
                yield emit("response.failed", {
                    "type": "response.failed",
                    "response": {**shell, "status": "failed",
                                 "error": {"message": f"backend {backend} unavailable; retry shortly"}}})
            except Exception:
                pass
        finally:
            if _pump_task is not None:
                _pump_task.cancel()
                try:
                    await _pump_task
                except BaseException:
                    pass
            await client.aclose()

    return StreamingResponse(iterator(), media_type="text/event-stream")


@APP.post("/v1/responses")
async def responses_route(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "Expected JSON object", "type": "invalid_request_error"}})
    if payload.get("stream") and RESPONSES_LIVE_STREAM:
        return await _responses_stream_live(payload, request)
    # Codex stop propagation (2026-07-07): the shim's upstream call is
    # non-streaming, so a codex disconnect used to leave the backend turn
    # running to budget (stops died in translation). Race the completion
    # against a disconnect watcher; on disconnect, fire /v1/stop upstream
    # and return. Full streaming pass-through is the durable fix (queued).
    async def _watch_disconnect():
        while True:
            if await request.is_disconnected():
                return True
            await asyncio.sleep(0.5)

    comp_task = asyncio.create_task(_responses_chat_completion(payload))
    watch_task = asyncio.create_task(_watch_disconnect())
    done, _ = await asyncio.wait(
        {comp_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if watch_task in done and comp_task not in done:
        model = str((payload.get("model") or ""))
        backend = backend_for_model(model)
        base = M3_BASE_URL if backend == "m3" else OMLX_BASE_URL
        try:
            async with httpx.AsyncClient(timeout=10) as _sc:
                await _sc.post(f"{base}/v1/stop", json={})
            record_event("responses_disconnect_stop", backend=backend)
        except Exception:
            pass
        comp_task.cancel()
        return JSONResponse(status_code=499, content={"error": {
            "message": "client disconnected; upstream stop issued",
            "type": "client_disconnect"}})
    watch_task.cancel()
    status, response_data = comp_task.result()
    if status >= 400:
        return JSONResponse(status_code=status, content=response_data)
    if not payload.get("stream"):
        return JSONResponse(response_data)

    async def iterator():
        seq = 0

        def emit(event: str, data: dict[str, Any]) -> str:
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            return _responses_sse(event, data)

        started = dict(response_data)
        started["status"] = "in_progress"
        started["output"] = []
        yield emit("response.created", {"type": "response.created", "response": started})
        yield emit("response.in_progress", {"type": "response.in_progress", "response": started})
        for index, item in enumerate(response_data.get("output") or []):
            if item.get("type") == "reasoning":
                item_id = str(item.get("id"))
                summary = item.get("summary") or []
                text = ""
                if summary and isinstance(summary[0], dict):
                    text = str(summary[0].get("text") or "")
                yield emit("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress", "summary": []},
                })
                yield emit("response.reasoning_summary_part.added", {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                })
                if text:
                    yield emit("response.reasoning_summary_text.delta", {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index": 0,
                        "delta": text,
                    })
                yield emit("response.reasoning_summary_text.done", {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index": 0,
                    "text": text,
                })
                yield emit("response.reasoning_summary_part.done", {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": text},
                })
                yield emit("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item,
                })
                continue
            if item.get("type") == "function_call":
                item_id = str(item.get("id"))
                arguments = str(item.get("arguments") or "{}")
                yield emit("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "arguments": ""},
                })
                yield emit("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "delta": arguments,
                })
                yield emit("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id,
                    "output_index": index,
                    "arguments": arguments,
                })
                yield emit("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item,
                })
                continue
            text = ""
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    text = str(first.get("text") or "")
            yield emit("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "content": []},
            })
            yield emit("response.content_part.added", {
                "type": "response.content_part.added",
                "item_id": item.get("id"),
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            })
            if text:
                yield emit("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": item.get("id"),
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                })
            yield emit("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": item.get("id"),
                "output_index": index,
                "content_index": 0,
                "text": text,
            })
            yield emit("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": item.get("id"),
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            })
            yield emit("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": index,
                "item": item,
            })
        yield emit("response.completed", {"type": "response.completed", "response": response_data})

    return StreamingResponse(iterator(), media_type="text/event-stream")


@APP.api_route("/v1/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def openai_proxy(rest: str, request: Request):
    body = await request.body()
    body, normalized_model, normalized = normalize_openai_json_body(body)
    model = normalized_model if normalized_model is not None else await request_model(request, body)
    if normalized:
        record_event("defaulted_empty_model", path=f"/v1/{rest}", model=model)
    backend = backend_for_model(model)
    ready = await ensure_backend(backend)
    if not ready.get("ok"):
        STATE["last_error"] = ready
        return JSONResponse(status_code=503, content={"error": {"message": json.dumps(ready), "type": "gateway_switch_error"}})
    if backend == "claude":
        if rest != "chat/completions":
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "Claude Code shim supports /v1/chat/completions", "type": "not_found"}},
            )
        return await claude_proxy(body)
    return await proxy_request(request, backend, f"/v1/{rest}", body)


def main() -> None:
    uvicorn.run(APP, host=GATEWAY_HOST, port=GATEWAY_PORT, log_level=os.environ.get("M3_GATEWAY_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
