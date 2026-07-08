#!/usr/bin/env python3
"""Audit speculative/MTP readiness for the local MiniMax-M3 gateway.

This is intentionally a metadata-only check. It reads package versions and
config.json files, but does not load target or drafter weights.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "mlx-community/MiniMax-M3-4bit"
DEFAULT_SCAN_ROOTS = [
    "~/.exo/models",
    "~/.lmstudio/models",
    "~/.cache/m3-models",
    "~/.cache/huggingface/hub",
]


def load_env():
    env = os.environ.copy()
    for name in (".env.local", "m3_cluster.env", ".env", ".env.example"):
        path = ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in env:
                env[key] = value
        break
    return env


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_config(path):
    try:
        with (Path(path) / "config.json").open() as f:
            return json.load(f)
    except Exception:
        return None


def nested_get(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def hidden_size(config):
    if not isinstance(config, dict):
        return None
    return (
        nested_get(config, "text_config", "hidden_size")
        or config.get("hidden_size")
        or config.get("target_hidden_size")
        or config.get("backbone_hidden_size")
    )


def num_layers(config):
    if not isinstance(config, dict):
        return None
    return nested_get(config, "text_config", "num_hidden_layers") or config.get("num_hidden_layers")


def model_type(config):
    if not isinstance(config, dict):
        return None
    architectures = config.get("architectures") or []
    if any("eagle3" in str(a).lower() for a in architectures):
        return "eagle3"
    return config.get("speculators_model_type") or config.get("model_type")


def summarize_config(path, config):
    if config is None:
        return {"path": str(path), "exists": Path(path).exists(), "config": "missing"}
    return {
        "path": str(path),
        "model_type": model_type(config),
        "architectures": config.get("architectures"),
        "hidden_size": hidden_size(config),
        "num_hidden_layers": num_layers(config),
        "vocab_size": nested_get(config, "text_config", "vocab_size") or config.get("vocab_size"),
        "has_dflash_config": bool(config.get("dflash_config")),
        "has_mtp_keys": any(str(k).lower().startswith("mtp") or "mtp" in str(k).lower() for k in config),
    }


def possible_model_paths(value):
    paths = []
    if not value:
        return paths
    raw = Path(value).expanduser()
    if raw.exists():
        paths.append(raw)
    safe = str(value).replace("/", "--")
    for root in DEFAULT_SCAN_ROOTS:
        p = Path(root).expanduser() / safe
        if p.exists():
            paths.append(p)
    return list(dict.fromkeys(paths))


def local_drafter_registry():
    out = {
        "known_kinds": [],
        "kind_by_model_type": {},
        "available_modules": [],
        "error": None,
    }
    try:
        mod = importlib.import_module("mlx_vlm.speculative.drafters")
        out["known_kinds"] = sorted(getattr(mod, "KNOWN_DRAFTER_KINDS", []))
        out["kind_by_model_type"] = dict(getattr(mod, "DRAFTER_KIND_BY_MODEL_TYPE", {}))
    except Exception as exc:
        out["error"] = repr(exc)
    pkg = Path(importlib.import_module("mlx_vlm.speculative.drafters").__file__).parent
    for child in sorted(pkg.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists() and not child.name.startswith("__"):
            out["available_modules"].append(child.name)
    return out


def scan_drafter_candidates(roots, registry, target_cfg, limit=5000):
    target_hidden = hidden_size(target_cfg)
    candidates = []
    seen = set()
    checked = 0
    kind_by_type = registry.get("kind_by_model_type") or {}
    for root in roots:
        base = Path(root).expanduser()
        if not base.exists():
            continue
        for cfg_path in base.rglob("config.json"):
            checked += 1
            if checked > limit:
                break
            parent = cfg_path.parent
            if parent in seen:
                continue
            seen.add(parent)
            cfg = read_config(parent)
            mt = model_type(cfg)
            expected = kind_by_type.get(mt)
            if expected is None and mt and "mtp" in str(mt).lower():
                expected = "mtp"
            looks_drafter = bool(
                expected
                or mt in {"eagle3"}
                or (mt and any(s in str(mt).lower() for s in ("assistant", "dflash", "mtp", "eagle")))
                or (cfg and cfg.get("dflash_config"))
            )
            if not looks_drafter:
                continue
            dh = hidden_size(cfg)
            hidden_match = None
            if target_hidden is not None and dh is not None:
                hidden_match = int(target_hidden) == int(dh)
            candidates.append({
                "path": str(parent),
                "model_type": mt,
                "expected_kind": expected or "dflash",
                "hidden_size": dh,
                "target_hidden_size": target_hidden,
                "hidden_size_match": hidden_match,
                "architectures": cfg.get("architectures") if cfg else None,
            })
        if checked > limit:
            break
    return candidates


def recommendation(target, registry, candidates):
    target_type = target.get("model_type")
    notes = []
    ready = False
    if not registry.get("known_kinds"):
        notes.append("mlx-vlm speculative drafter registry is unavailable.")
    if str(target_type).lower() in {"minimax_m3_vl", "minimax-m3", "minimax_m3"}:
        notes.append("Target is MiniMax-M3; no installed local drafter family is explicitly mapped to MiniMax-M3.")
    compatible = [c for c in candidates if c.get("hidden_size_match") is not False]
    if compatible:
        notes.append(f"Found {len(compatible)} plausible drafter candidate(s); load-test one only on clean memory.")
        ready = True
    else:
        notes.append("No compatible local drafter config was found in scanned model roots.")
    if not ready:
        notes.append("Keep speculative/MTP disabled in the production gateway for now.")
    return {"speculative_ready": ready, "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Target model path or id; defaults to MLX_M3_MODEL")
    parser.add_argument("--scan-root", action="append", default=[], help="Additional root to scan for drafter config.json files")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    env = load_env()
    model_value = args.model or env.get("MLX_M3_MODEL") or DEFAULT_MODEL_ID
    model_paths = possible_model_paths(model_value)
    target_path = model_paths[0] if model_paths else Path(model_value).expanduser()
    target_cfg = read_config(target_path)
    registry = local_drafter_registry()
    roots = args.scan_root or DEFAULT_SCAN_ROOTS
    candidates = scan_drafter_candidates(roots, registry, target_cfg)
    result = {
        "packages": {
            "mlx": package_version("mlx"),
            "mlx-vlm": package_version("mlx-vlm"),
            "mlx-lm": package_version("mlx-lm"),
        },
        "target": summarize_config(target_path, target_cfg),
        "registry": registry,
        "scan_roots": [str(Path(r).expanduser()) for r in roots],
        "candidates": candidates,
    }
    result["recommendation"] = recommendation(result["target"], registry, candidates)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("# Speculative / MTP Readiness Audit\n")
    print("Packages:")
    for key, value in result["packages"].items():
        print(f"- {key}: {value or 'missing'}")
    print("\nTarget:")
    for key in ("path", "model_type", "architectures", "hidden_size", "num_hidden_layers", "has_mtp_keys", "has_dflash_config"):
        print(f"- {key}: {result['target'].get(key)}")
    print("\nInstalled drafter support:")
    print(f"- known kinds: {', '.join(registry.get('known_kinds') or []) or 'none'}")
    print(f"- drafter model types: {', '.join(sorted((registry.get('kind_by_model_type') or {}).keys())) or 'none'}")
    print("\nCandidates:")
    if not candidates:
        print("- none found")
    for c in candidates:
        verdict = "hidden ok" if c.get("hidden_size_match") else "hidden unknown" if c.get("hidden_size_match") is None else "hidden mismatch"
        print(f"- {c['path']} [{c.get('expected_kind')}, {c.get('model_type')}, {verdict}]")
    print("\nRecommendation:")
    print(f"- speculative_ready: {result['recommendation']['speculative_ready']}")
    for note in result["recommendation"]["notes"]:
        print(f"- {note}")


if __name__ == "__main__":
    main()
