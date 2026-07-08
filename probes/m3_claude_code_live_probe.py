#!/usr/bin/env python3
"""Live Claude Code compatibility probe for the ThunderMLX gateway.

This exercises the Anthropic Messages route through the real Claude Code CLI.
It verifies that Claude-facing tool calls execute and that the gateway does not
leak blank/prose-only turns for file/action requests.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_BASE = "http://127.0.0.1:8010"
DEFAULT_CLAUDE = str(Path.home() / ".local/bin/claude")


def save_raw(args: argparse.Namespace, workdir: Path, name: str, raw: str) -> None:
    if args.keep_workdir:
        (workdir / f"{name}.stream.jsonl").write_text(raw)


def run_claude(
    *,
    claude: str,
    base: str,
    model: str,
    workdir: Path,
    prompt: str,
    allowed_tools: str,
    timeout: int,
) -> tuple[list[dict[str, Any]], str]:
    env = os.environ.copy()
    env.update({
        "ANTHROPIC_BASE_URL": base.rstrip("/"),
        "ANTHROPIC_API_KEY": "local",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Minimax-M3-No-Think",
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    cmd = [
        claude,
        "-p",
        "--model",
        model,
        "--permission-mode",
        "bypassPermissions",
        "--add-dir",
        str(workdir),
        f"--allowedTools={allowed_tools}",
        "--output-format=stream-json",
        "--verbose",
        prompt,
    ]
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    events: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("⚠"):
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            events.append({"type": "raw", "text": line})
    if proc.returncode != 0:
        raise RuntimeError(f"Claude Code exited {proc.returncode}: {proc.stdout[-2000:]}")
    return events, proc.stdout


def tool_names(events: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for event in events:
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append(str(block.get("name") or ""))
    return out


def final_result(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "result":
            return str(event.get("result") or "")
    return ""


def run_simple_write(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    workdir = root / "simple_write"
    workdir.mkdir()
    events, raw = run_claude(
        claude=args.claude,
        base=args.base,
        model=args.model,
        workdir=workdir,
        allowed_tools="Write,Bash",
        timeout=args.timeout,
        prompt=(
            "In the current directory, create result.txt containing exactly "
            "thundermlx-tool-ok, then reply with exactly done."
        ),
    )
    save_raw(args, workdir, "simple_write", raw)
    target = workdir / "result.txt"
    if not target.exists() or target.read_text() != "thundermlx-tool-ok":
        raise AssertionError("simple write file was not created correctly")
    result = final_result(events)
    if "done" not in result.lower():
        raise AssertionError(f"simple write final result was unexpected: {result!r}")
    names = tool_names(events)
    if not names:
        raise AssertionError("simple write did not execute a tool")
    return {"case": "simple_write", "tools": names, "result": result}


def run_bash_roundtrip(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    workdir = root / "bash_roundtrip"
    workdir.mkdir()
    events, raw = run_claude(
        claude=args.claude,
        base=args.base,
        model=args.model,
        workdir=workdir,
        allowed_tools="Bash",
        timeout=args.timeout,
        prompt="Use Bash to run pwd, then reply with the exact output path only.",
    )
    save_raw(args, workdir, "bash_roundtrip", raw)
    result = final_result(events)
    if str(workdir) not in result:
        raise AssertionError(f"bash final result was unexpected: {result!r}")
    names = tool_names(events)
    if "Bash" not in names:
        raise AssertionError(f"bash tool was not used: {names}")
    return {"case": "bash_roundtrip", "tools": names, "result": result}


def run_compound_agent(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    workdir = root / "compound_agent"
    workdir.mkdir()
    (workdir / "notes.txt").write_text(
        "ThunderMLX agent probe\n"
        "- endpoint: local MiniMax-M3 gateway\n"
        "- requirement: tools must execute and final answer must summarize evidence\n"
    )
    events, raw = run_claude(
        claude=args.claude,
        base=args.base,
        model=args.model,
        workdir=workdir,
        allowed_tools="Bash,Read,Write",
        timeout=args.timeout,
        prompt=(
            "Use tools to inspect this directory, read notes.txt, create "
            "report.txt containing exactly agent-probe-ok, then reply with one "
            "sentence naming the created file."
        ),
    )
    save_raw(args, workdir, "compound_agent", raw)
    target = workdir / "report.txt"
    if not target.exists() or target.read_text() != "agent-probe-ok":
        raise AssertionError("compound agent report was not created correctly")
    names = tool_names(events)
    if not names:
        raise AssertionError("compound agent did not execute a tool")
    return {"case": "compound_agent", "tools": names, "result": final_result(events)}


def run_multifile_coding(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    workdir = root / "multifile_coding"
    src = workdir / "src"
    src.mkdir(parents=True)
    (src / "math_tools.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )
    (src / "app.py").write_text(
        "from math_tools import add\n"
        "\n"
        "\n"
        "def run():\n"
        "    return add(2, 3)\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    print(run())\n"
    )
    (workdir / "README.md").write_text(
        "# Probe Project\n\n"
        "Update the app to use multiplication and document the result.\n"
    )
    events, raw = run_claude(
        claude=args.claude,
        base=args.base,
        model=args.model,
        workdir=workdir,
        allowed_tools="Bash,Read,Write,Edit",
        timeout=args.timeout,
        prompt=(
            "Use tools to inspect this small Python project. Change src/app.py "
            "so run() returns multiply(6, 7), keep imports correct, create "
            "SUMMARY.txt containing exactly product=42, run python3 src/app.py "
            "to verify it prints 42, then reply with exactly coding-probe-ok."
        ),
    )
    save_raw(args, workdir, "multifile_coding", raw)
    app_text = (src / "app.py").read_text()
    if "multiply" not in app_text or "multiply(6, 7)" not in app_text:
        raise AssertionError(
            "app.py was not updated correctly; "
            f"tools={tool_names(events)} result={final_result(events)!r}\n{app_text}"
        )
    summary = workdir / "SUMMARY.txt"
    if not summary.exists() or summary.read_text() != "product=42":
        raise AssertionError("SUMMARY.txt was not created correctly")
    proc = subprocess.run(
        [sys.executable, "src/app.py"],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
        check=False,
    )
    if proc.returncode != 0 or proc.stdout.strip() != "42":
        raise AssertionError(
            f"project verification failed rc={proc.returncode} "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    names = tool_names(events)
    if len(names) < 2:
        raise AssertionError(f"expected multiple tool uses, got {names}")
    result = final_result(events)
    if "coding-probe-ok" not in result:
        raise AssertionError(f"coding final result was unexpected: {result!r}")
    return {"case": "multifile_coding", "tools": names, "result": result}


def run_long_agent_loop(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    workdir = root / "long_agent_loop"
    notes = workdir / "notes"
    notes.mkdir(parents=True)
    expected = []
    for index in range(1, 9):
        code = f"AX-{index:02d}"
        expected.append(code)
        (notes / f"file_{index:02d}.txt").write_text(
            f"title: probe note {index:02d}\n"
            f"code: {code}\n"
            f"status: ready\n"
        )
    events, raw = run_claude(
        claude=args.claude,
        base=args.base,
        model=args.model,
        workdir=workdir,
        allowed_tools="Bash,Read,Write",
        timeout=args.timeout,
        prompt=(
            "Use tools to inspect the notes directory. Read the note files, "
            "create AGENT_REPORT.txt containing exactly the comma-separated "
            "codes in ascending file order, then run cat AGENT_REPORT.txt to "
            "verify it. Reply with exactly long-agent-ok."
        ),
    )
    save_raw(args, workdir, "long_agent_loop", raw)
    report = workdir / "AGENT_REPORT.txt"
    expected_text = ",".join(expected)
    if not report.exists() or report.read_text().strip() != expected_text:
        raise AssertionError(
            "long agent report was not created correctly; "
            f"tools={tool_names(events)} result={final_result(events)!r} "
            f"expected={expected_text!r} actual={report.read_text() if report.exists() else '<missing>'!r}"
        )
    names = tool_names(events)
    if len(names) < 2:
        raise AssertionError(f"expected multiple tool uses, got {names}")
    result = final_result(events)
    if "long-agent-ok" not in result:
        raise AssertionError(f"long agent final result was unexpected: {result!r}")
    return {"case": "long_agent_loop", "tools": names, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--model", default="Minimax-M3")
    parser.add_argument("--claude", default=DEFAULT_CLAUDE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "simple_write",
            "bash_roundtrip",
            "compound_agent",
            "multifile_coding",
            "long_agent_loop",
        ],
        default="all",
    )
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    if not shutil.which(args.claude) and not Path(args.claude).exists():
        raise SystemExit(f"Claude Code CLI not found: {args.claude}")

    root = Path(tempfile.mkdtemp(prefix="thundermlx_claude_probe_"))
    try:
        case_fns = {
            "simple_write": run_simple_write,
            "bash_roundtrip": run_bash_roundtrip,
            "compound_agent": run_compound_agent,
            "multifile_coding": run_multifile_coding,
            "long_agent_loop": run_long_agent_loop,
        }
        if args.case == "all":
            selected = ["simple_write", "bash_roundtrip", "compound_agent"]
            if args.extended:
                selected.extend(["multifile_coding", "long_agent_loop"])
        else:
            selected = [args.case]
        rows = [case_fns[name](args, root) for name in selected]
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        print("PASS")
    finally:
        if args.keep_workdir:
            print(json.dumps({"workdir": str(root)}, sort_keys=True))
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
