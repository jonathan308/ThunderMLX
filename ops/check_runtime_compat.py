#!/usr/bin/env python3
"""Validate the synchronized ThunderMLX Python runtime after an update.

The production MLX/JACCL wheel is built from the 0.32 development line and is
recorded in ``mlx_variants.json``.  PyPI packages correctly see its PEP 440
version as older than the final 0.32.0 release even though this exact paired
build is the cluster's validated runtime.  Permit only that known metadata
mismatch; every other ``pip check`` issue remains fatal.
"""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys


MLX_REQUIREMENT_MISMATCH = re.compile(
    r"^mlx-vlm \S+ has requirement mlx>=0\.32\.0, "
    r"but you have mlx (?P<version>\S+)\.$"
)


def validated_mlx_pair(manifest_path: Path) -> tuple[bool, dict]:
    try:
        manifest = json.loads(manifest_path.read_text())
        label = str(manifest.get("recommended") or "").strip()
        record = (manifest.get("variants") or {}).get(label) or {}
        expected = str(record.get("version") or "").strip()
        mlx_version = metadata.version("mlx")
        metal_version = metadata.version("mlx-metal")
        variant_dir = manifest_path.parent / "variants" / label
        mlx_wheels = list(variant_dir.glob(f"mlx-{expected}-*.whl"))
        metal_wheels = list(variant_dir.glob(f"mlx_metal-{expected}-*.whl"))
        approved = str(record.get("status") or "").lower() in {
            "production",
            "validated",
        }
        ok = bool(
            label
            and expected
            and approved
            and mlx_version == expected
            and metal_version == expected
            and len(mlx_wheels) == 1
            and len(metal_wheels) == 1
        )
        return ok, {
            "label": label,
            "expected": expected,
            "mlx": mlx_version,
            "mlx_metal": metal_version,
            "approved": approved,
            "artifacts_present": len(mlx_wheels) == 1 and len(metal_wheels) == 1,
        }
    except Exception as exc:
        return False, {"error": str(exc)}


def main() -> int:
    manifest_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "runtime_patches/mlx_variants.json"
    ).expanduser()
    pair_ok, pair = validated_mlx_pair(manifest_path)

    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    issues = [
        line.strip()
        for line in check.stdout.splitlines()
        if line.strip() and line.strip() != "No broken requirements found."
    ]
    if check.returncode and not issues:
        detail = check.stderr.strip() or f"pip check exited {check.returncode}"
        issues.append(detail)
    rejected: list[str] = []
    accepted: list[str] = []
    for issue in issues:
        match = MLX_REQUIREMENT_MISMATCH.fullmatch(issue)
        if match and pair_ok and match.group("version") == pair.get("mlx"):
            accepted.append(issue)
        else:
            rejected.append(issue)

    import_errors: dict[str, str] = {}
    for module in (
        "mlx",
        "mlx.core",
        "mlx.core.distributed",
        "mlx_lm",
        "mlx_vlm",
        "transformers",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            import_errors[module] = f"{type(exc).__name__}: {exc}"

    result = {
        "ok": not rejected and not import_errors,
        "validated_mlx_pair": pair,
        "accepted_metadata_mismatches": accepted,
        "rejected_dependency_issues": rejected,
        "import_errors": import_errors,
        "versions": {
            package: metadata.version(package)
            for package in ("mlx", "mlx-metal", "mlx-lm", "mlx-vlm", "transformers")
        },
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
