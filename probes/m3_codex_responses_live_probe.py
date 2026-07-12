#!/usr/bin/env python3
"""Live Codex CLI compatibility probe for the ThunderMLX Responses bridge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


DEFAULT_CODEX = (
    shutil.which("codex")
    or "/Applications/ChatGPT.app/Contents/Resources/codex"
)
DEFAULT_BASE = "http://127.0.0.1:8010/v1"


def run_codex_case(
    *,
    codex: str,
    base: str,
    model: str,
    workdir: Path,
    filename: str,
    content: str,
    timeout: int,
) -> dict[str, Any]:
    prompt = (
        f"In the current directory, create {filename} containing exactly "
        f"{content}, then answer done."
    )
    cmd = [
        codex,
        "-a",
        "never",
        "-c",
        'model_provider="thundermlx"',
        "-c",
        'model_providers.thundermlx.name="ThunderMLX"',
        "-c",
        f'model_providers.thundermlx.base_url="{base.rstrip("/")}"',
        "-c",
        'model_providers.thundermlx.env_key="OPENAI_API_KEY"',
        "-c",
        'model_providers.thundermlx.wire_api="responses"',
        "-c",
        "model_providers.thundermlx.requires_openai_auth=false",
        "-m",
        model,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
        "--ephemeral",
        "-C",
        str(workdir),
        prompt,
    ]
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        env={"OPENAI_API_KEY": "local", **dict(os.environ)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    target = workdir / filename
    actual = target.read_text() if target.exists() else None
    if proc.returncode != 0 or actual != content:
        raise AssertionError(
            f"Codex probe failed model={model} rc={proc.returncode} "
            f"expected={content!r} actual={actual!r}\n{proc.stdout[-4000:]}"
        )
    if '"type":"turn.completed"' not in proc.stdout and '"type": "turn.completed"' not in proc.stdout:
        raise AssertionError(f"Codex probe did not complete cleanly:\n{proc.stdout[-4000:]}")
    return {
        "model": model,
        "file": filename,
        "content": actual,
        "workdir": str(workdir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--codex", default=DEFAULT_CODEX)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    models = args.model or ["Minimax-M3-No-Think", "Minimax-M3"]
    root = Path(tempfile.mkdtemp(prefix="thundermlx_codex_probe_"))
    try:
        for index, model in enumerate(models, start=1):
            case_dir = root / f"case_{index}"
            case_dir.mkdir()
            result = run_codex_case(
                codex=args.codex,
                base=args.base,
                model=model,
                workdir=case_dir,
                filename=f"codex_probe_{index}.txt",
                content=f"codex-ok-{index}",
                timeout=args.timeout,
            )
            print(result)
        print("PASS")
    finally:
        if args.keep_workdir:
            print({"workdir": str(root)})
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
