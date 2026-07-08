#!/usr/bin/env python3
"""Cold prefill benchmark for client/request shapes.

Use this to separate core MSA prefill speed from request-shape artifacts such
as OpenWebUI always-attached tools or coding-agent tool schemas. It reports the
authoritative server-side `/health.last_request.prompt_tps`.
"""
import argparse
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def request_json(method, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(timeout=5):
    return request_json("GET", "/health", timeout=timeout)


def reset_cache(reason):
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": reason, "clear_memory": False},
        timeout=30,
    )


def dummy_tools(count, shape):
    tools = []
    for index in range(int(count or 0)):
        if shape == "agent-tools":
            params = {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path to inspect or edit.",
                    },
                    "command": {
                        "type": "string",
                        "description": "Shell command to run for project inspection.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Patch, replacement text, or structured agent payload.",
                    },
                },
                "required": ["path"] if index % 3 == 0 else [],
            }
            description = (
                "Coding-agent tool schema used for local project exploration, "
                "file reads, edits, terminal commands, and validation."
            )
        else:
            params = {"type": "object", "properties": {}}
            description = "No-op OpenWebUI compatibility probe tool."
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"{shape.replace('-', '_')}_tool_{index}",
                    "description": description,
                    "parameters": params,
                },
            }
        )
    return tools


def build_context(target_tokens):
    records = max(64, int(int(target_tokens) / 18))
    return "\n".join(
        f"shape_prefill/file_{i:06d}.py :: symbol_{i}(value) returns value + {i}; owner=agent; priority={i % 17}"
        for i in range(records)
    )


def messages_for_shape(target_tokens, shape):
    context = build_context(target_tokens)
    if shape == "plain":
        return [
            {
                "role": "user",
                "content": context + "\n\nSummarize the prefill benchmark context briefly.",
            }
        ]
    if shape == "openwebui-tools":
        return [
            {
                "role": "system",
                "content": (
                    "You are a concise local assistant in OpenWebUI. Tools may be "
                    "listed for compatibility, but answer directly."
                ),
            },
            {
                "role": "user",
                "content": context + "\n\nOpenWebUI shape benchmark: answer briefly.",
            },
        ]
    if shape == "agent-tools":
        return [
            {
                "role": "system",
                "content": (
                    "You are a coding agent with tool schemas available. For this "
                    "benchmark, do not call tools; reason from the workspace."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Workspace snapshot follows. Treat every line as a file/symbol fact.\n"
                    + context
                    + "\n\nAgent benchmark: state one cache/stability implication briefly."
                ),
            },
        ]
    raise ValueError(f"unsupported shape={shape}")


def wait_idle(before_completed, timeout=180):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health(timeout=5)
        pcache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and not pcache.get("in_use")
            and int(last.get("requests_completed") or 0) > before_completed
        ):
            return last
        time.sleep(0.25)
    return last or health(timeout=5)


def stream_chat(label, *, model, target_tokens, shape, tools, max_tokens, timeout):
    reset_cache(f"prefill shape probe {label}")
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages_for_shape(target_tokens, shape),
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metadata": {
            "session_id": f"prefill-shape-{label}-{int(time.time())}",
            "source": "m3_prefill_shape_probe",
        },
    }
    if shape.endswith("tools"):
        payload["tools"] = dummy_tools(tools, shape)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_piece_s = None
    chunks = 0
    chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            obj = json.loads(item)
            chunks += 1
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            piece = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
            )
            if piece and first_piece_s is None:
                first_piece_s = time.time() - started
            chars += len(piece)

    final = wait_idle(before, timeout=max(180, timeout // 2))
    last = final.get("last_request") or {}
    prepare = last.get("prompt_cache_prepare") or {}
    request_shape = last.get("request_shape") or {}
    kernels = final.get("kernel_stats") or {}
    row = {
        "label": label,
        "model": model,
        "shape": shape,
        "tools_requested": tools if shape.endswith("tools") else 0,
        "tools_count": request_shape.get("tools_count"),
        "target_tokens": target_tokens,
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "client_elapsed_s": round(time.time() - started, 3),
        "chunks": chunks,
        "output_chars": chars,
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_ttft_s": last.get("first_token_s"),
        "server_decode_tps": last.get("decode_tps"),
        "server_tokens": last.get("tokens"),
        "cache_action": prepare.get("action"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "last_msa_k1_impl": kernels.get("last_msa_k1_impl"),
        "prefill_blockwise_topk": kernels.get("prefill_blockwise_topk"),
        "prefill_standard_topk": kernels.get("prefill_standard_topk"),
        "topk_native_error": kernels.get("topk_native_error"),
        "failed": final.get("requests_failed"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--models", default="Minimax-M3-No-Think")
    parser.add_argument("--shapes", default="plain,openwebui-tools,agent-tools")
    parser.add_argument("--target-tokens", default="30000")
    parser.add_argument("--tools", type=int, default=34)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--min-prompt-tps", type=float, default=0.0)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    print(
        json.dumps(
            {
                "initial": {
                    "status": initial.get("status"),
                    "failed": initial.get("requests_failed"),
                    "runtime_tuning": (initial.get("generation_defaults") or {}).get("runtime_tuning"),
                    "prefill_step_size": (initial.get("generation_defaults") or {}).get("prefill_step_size"),
                    "prefill_stop_check_every": (initial.get("generation_defaults") or {}).get("prefill_stop_check_every"),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows = []
    failures = []
    for model in [item.strip() for item in args.models.split(",") if item.strip()]:
        for target in [int(item.strip()) for item in args.target_tokens.split(",") if item.strip()]:
            for shape in [item.strip() for item in args.shapes.split(",") if item.strip()]:
                label = f"{model}-{shape}-{target}"
                row = stream_chat(
                    label,
                    model=model,
                    target_tokens=target,
                    shape=shape,
                    tools=args.tools,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                rows.append(row)
                if args.min_prompt_tps and (row.get("server_prompt_tps") or 0.0) < args.min_prompt_tps:
                    failures.append(f"{label} prompt_tps={row.get('server_prompt_tps')}")

    print(
        json.dumps(
            {
                "summary": {
                    "rows": len(rows),
                    "failures": failures,
                    "prompt_tps": [
                        [row.get("model"), row.get("shape"), row.get("target_tokens"), row.get("server_prompt_tokens"), row.get("server_prompt_tps")]
                        for row in rows
                    ],
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit("; ".join(failures))
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
