#!/usr/bin/env python3
"""Preflight/readiness doctor for the ThunderMLX MiniMax-M3 cluster."""
import argparse
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
REQUIRED_FILES = [
    ".env.example",
    "M3_Start.command",
    "M3_Stop.command",
    "README.md",
    "docs/SETUP.md",
    "docs/SECURITY.md",
    "launch_cluster.sh",
    "sync_rank1.sh",
    "stop_cluster.sh",
    "auto_restart.sh",
    "sharded_server.py",
    "cluster_gui.py",
    "probes/m3_perf_probe.py",
    "probes/m3_hot_cache_probe.py",
    "tools/m3_analyze_results.py",
    "tools/m3_overnight_runner.py",
    "tools/m3_speculative_audit.py",
    "tools/m3_doctor.py",
]
PY_CHECK = [
    "sharded_server.py",
    "cluster_gui.py",
    "probes/m3_perf_probe.py",
    "probes/m3_hot_cache_probe.py",
    "tools/m3_analyze_results.py",
    "tools/m3_overnight_runner.py",
    "tools/m3_speculative_audit.py",
    "tools/m3_doctor.py",
]
SH_CHECK = [
    "launch_cluster.sh",
    "sync_rank1.sh",
    "stop_cluster.sh",
    "auto_restart.sh",
    "scripts/download_model.sh",
]


def load_env():
    env = os.environ.copy()
    loaded = None
    for name in (".env.local", "m3_cluster.env", ".env", ".env.example"):
        path = ROOT / name
        if not path.exists():
            continue
        loaded = name
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
    return env, loaded


def run(cmd, *, timeout=20):
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {"ok": p.returncode == 0, "code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "code": 124, "stdout": exc.stdout or "", "stderr": "timeout"}
    except Exception as exc:
        return {"ok": False, "code": 1, "stdout": "", "stderr": repr(exc)}


def parse_vm_stat(text):
    def pages(label):
        m = re.search(rf"{re.escape(label)}:\s+([0-9]+)", text)
        return int(m.group(1)) if m else 0

    page_size = 16384
    m = re.search(r"page size of ([0-9]+) bytes", text)
    if m:
        page_size = int(m.group(1))
    free = pages("Pages free") + pages("Pages speculative")
    inactive = pages("Pages inactive")
    purgeable = pages("Pages purgeable")
    wired = pages("Pages wired down")
    return {
        "wired_gb": round(wired * page_size / 1024**3, 1),
        "available_gb": round((free + inactive + purgeable) * page_size / 1024**3, 1),
    }


def vm_stat_local():
    out = run(["vm_stat"], timeout=5)
    data = parse_vm_stat(out["stdout"]) if out["stdout"] else {}
    data.update({"ok": out["ok"], "error": out["stderr"]})
    return data


def resolve_peer(env):
    return (
        env.get("M3_PEER")
        or env.get("M3_RANK1_DIRECT_SSH")
        or env.get("M3_RANK1_FALLBACK_SSH")
        or env.get("M3_TAILSCALE_PEER")
        or ""
    )


def vm_stat_remote(peer):
    if not peer:
        return {"ok": False, "error": "peer not configured"}
    out = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", peer, "vm_stat"], timeout=12)
    data = parse_vm_stat(out["stdout"]) if out["stdout"] else {}
    data.update({"ok": out["ok"], "error": out["stderr"]})
    return data


def port_open(host, port):
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except Exception:
        return False


def health(endpoint):
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=2) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "offline", "error": str(exc)}


def package_versions():
    out = {}
    for pkg in ("mlx", "mlx-vlm", "mlx-lm"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def model_status(env):
    model = env.get("MLX_M3_MODEL", "mlx-community/MiniMax-M3-4bit")
    p = Path(model).expanduser()
    is_pathish = model.startswith(("/", "~", "."))
    return {
        "value": model,
        "is_path": is_pathish,
        "exists": p.exists() if is_pathish else None,
        "config_json": (p / "config.json").exists() if is_pathish else None,
    }


def syntax_status():
    py = run([sys.executable, "-m", "py_compile", *PY_CHECK], timeout=60)
    shell_results = {name: run(["bash", "-n", name], timeout=20) for name in SH_CHECK}
    sh = {
        "ok": all(result["ok"] for result in shell_results.values()),
        "files": shell_results,
    }
    return {"python": py, "shell": sh}


def git_status():
    tracked = run(["git", "status", "--short"], timeout=10)
    ignored = run(["git", "status", "--ignored", "--short"], timeout=10)
    return {"status": tracked["stdout"], "ignored": ignored["stdout"], "ok": tracked["ok"] and ignored["ok"]}


def check_required_files():
    return {name: (ROOT / name).exists() for name in REQUIRED_FILES}


def orphan_thresholds(env):
    r0 = float(env.get("M3_ORPHAN_RANK0_WIRED_GB") or env.get("M3_ORPHAN_STUDIO_WIRED_GB") or 30)
    r1 = float(env.get("M3_ORPHAN_RANK1_WIRED_GB") or env.get("M3_ORPHAN_MACBOOK_WIRED_GB") or 20)
    return r0, r1


def build_report(args):
    env, loaded_env = load_env()
    peer = resolve_peer(env)
    endpoint = args.endpoint or env.get("M3_ENDPOINT") or DEFAULT_ENDPOINT
    gui_host = env.get("M3_GUI_HOST", "127.0.0.1")
    gui_port = int(env.get("M3_GUI_PORT", "8090"))
    api_port = int(env.get("MLX_M3_PORT", "8080"))
    local_mem = vm_stat_local()
    remote_mem = vm_stat_remote(peer) if args.remote else {"ok": None, "error": "remote check skipped"}
    r0_limit, r1_limit = orphan_thresholds(env)
    rank0_dirty = local_mem.get("wired_gb") is not None and local_mem["wired_gb"] > r0_limit
    rank1_dirty = (
        remote_mem.get("wired_gb") is not None
        and remote_mem["wired_gb"] > r1_limit
    )
    h = health(endpoint)
    report = {
        "env_file": loaded_env,
        "packages": package_versions(),
        "required_files": check_required_files(),
        "syntax": syntax_status(),
        "git": git_status(),
        "model": model_status(env),
        "peer": peer or None,
        "memory": {
            "rank0": local_mem,
            "rank1": remote_mem,
            "rank0_limit_gb": r0_limit,
            "rank1_limit_gb": r1_limit,
            "dirty": bool(rank0_dirty or rank1_dirty),
        },
        "ports": {
            "api": {"port": api_port, "listening": port_open("127.0.0.1", api_port)},
            "dashboard": {"host": gui_host, "port": gui_port, "listening": port_open("127.0.0.1", gui_port)},
        },
        "health": h,
    }
    return report


def status_label(ok):
    return "OK" if ok else "FAIL"


def print_report(report):
    print("# M3 Cluster Doctor\n")
    print(f"Config: {report['env_file'] or 'none'}")
    print("Packages: " + ", ".join(f"{k}={v or 'missing'}" for k, v in report["packages"].items()))

    files_ok = all(report["required_files"].values())
    print(f"Required files: {status_label(files_ok)}")
    missing = [k for k, v in report["required_files"].items() if not v]
    if missing:
        print("  missing: " + ", ".join(missing))

    py_ok = report["syntax"]["python"]["ok"]
    sh_ok = report["syntax"]["shell"]["ok"]
    print(f"Syntax: python={status_label(py_ok)} shell={status_label(sh_ok)}")
    if not py_ok:
        print("  python: " + report["syntax"]["python"]["stderr"][-500:])
    if not sh_ok:
        bad = [
            f"{name}: {result.get('stderr') or result.get('stdout') or 'failed'}"
            for name, result in report["syntax"]["shell"].get("files", {}).items()
            if not result.get("ok")
        ]
        print("  shell: " + "; ".join(bad)[-500:])

    git_dirty = bool(report["git"]["status"].strip())
    print(f"Git worktree: {'DIRTY' if git_dirty else 'clean'}")

    model = report["model"]
    model_msg = model["value"]
    if model["is_path"]:
        model_msg += f" exists={model['exists']} config_json={model['config_json']}"
    print(f"Model: {model_msg}")

    mem = report["memory"]
    print(
        "Memory: "
        f"rank0 wired={mem['rank0'].get('wired_gb')}GB limit={mem['rank0_limit_gb']}GB; "
        f"rank1 wired={mem['rank1'].get('wired_gb')}GB limit={mem['rank1_limit_gb']}GB; "
        f"{'DIRTY - reboot before launch' if mem['dirty'] else 'clean'}"
    )
    if mem["rank1"].get("ok") is False:
        print("  rank1: " + str(mem["rank1"].get("error") or "unreachable"))

    print(
        "Ports: "
        f"api:{report['ports']['api']['port']}={'up' if report['ports']['api']['listening'] else 'down'} "
        f"dashboard:{report['ports']['dashboard']['port']}={'up' if report['ports']['dashboard']['listening'] else 'down'}"
    )
    print(f"Endpoint health: {report['health'].get('status')}")
    if report["health"].get("error"):
        print("  " + report["health"]["error"])

    ready_to_launch = (
        files_ok
        and py_ok
        and sh_ok
        and not mem["dirty"]
        and not git_dirty
    )
    print(f"\nReady to launch: {status_label(ready_to_launch)}")
    if not ready_to_launch:
        print("Next action:")
        if mem["dirty"]:
            print("- Reboot affected machines before starting the cluster.")
        elif git_dirty:
            print("- Commit/stash local changes or knowingly continue with a dirty worktree.")
        elif not files_ok or not py_ok or not sh_ok:
            print("- Fix missing files or syntax errors.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=None, help="Endpoint root, default from env or http://127.0.0.1:8080")
    parser.add_argument("--remote", action="store_true", help="Check rank 1 wired memory over SSH")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    files_ok = all(report["required_files"].values())
    syntax_ok = report["syntax"]["python"]["ok"] and report["syntax"]["shell"]["ok"]
    return 1 if report["memory"]["dirty"] or not files_ok or not syntax_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
