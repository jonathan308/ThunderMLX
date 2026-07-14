#!/usr/bin/env python3
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
import uvicorn


def load_env_file(path):
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_str(key, default=""):
    value = os.environ.get(key)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def env_int(key, default):
    value = env_str(key, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


BASE_DIR = Path(__file__).resolve().parent
for env_name in (".env.local", "m3_cluster.env", ".env"):
    load_env_file(BASE_DIR / env_name)


CLUSTER = Path(env_str("M3_CLUSTER_DIR", str(BASE_DIR))).expanduser()
PEER = (
    env_str("M3_PEER")
    or env_str("M3_RANK1_DIRECT_SSH")
    or env_str("M3_RANK1_FALLBACK_SSH")
    or ""
)
ENDPOINT = env_str("M3_ENDPOINT", "http://127.0.0.1:8080")
GUI_HOST = env_str("M3_GUI_HOST", "0.0.0.0")
GUI_PORT = env_int("M3_GUI_PORT", 8090)
DEFAULT_MODEL_ID = "mlx-community/MiniMax-M3-4bit"
LOCAL_ENV_PATH = CLUSTER / ".env.local"

APP = FastAPI()
SNAPSHOT_TTL_SECONDS = float(os.environ.get("M3_DASHBOARD_SNAPSHOT_TTL_SECONDS", "3"))
_snapshot_cache = {
    "at": 0.0,
    "memory": None,
    "processes": None,
}


def host_from_url(value):
    try:
        return urllib.parse.urlparse(str(value or "")).hostname or ""
    except Exception:
        return ""


def dashboard_public_url():
    explicit = os.environ.get("M3_GUI_PUBLIC_URL", "").strip()
    if explicit:
        return explicit
    host = os.environ.get("M3_GUI_PUBLIC_HOST", "").strip()
    if not host:
        host = host_from_url(os.environ.get("M3_PUBLIC_BASE_URL", ""))
    if host:
        return f"http://{host}:{GUI_PORT}"
    if GUI_HOST not in ("0.0.0.0", "::", ""):
        return f"http://{GUI_HOST}:{GUI_PORT}"
    return f"http://127.0.0.1:{GUI_PORT}"


def public_config():
    model = os.environ.get("MLX_M3_MODEL", DEFAULT_MODEL_ID)
    model_id = os.environ.get("MLX_M3_MODEL_ID", DEFAULT_MODEL_ID)
    return {
        "cluster_dir": str(CLUSTER),
        "backend": os.environ.get("M3_MLX_BACKEND", "jaccl"),
        "sharding_mode": os.environ.get("M3_SHARDING_MODE", "pipeline"),
        "pipeline_layers": os.environ.get("M3_PIPELINE_LAYERS", "38,22"),
        "rank0_label": os.environ.get("M3_RANK0_LABEL", "Primary Mac"),
        "rank1_label": os.environ.get("M3_RANK1_LABEL", "Worker Mac"),
        "rank0_data_ip": os.environ.get("M3_RANK0_DATA_IP", "not configured"),
        "rank1_data_ip": os.environ.get("M3_RANK1_DATA_IP", "not configured"),
        "rank0_rdma": os.environ.get("M3_RANK0_RDMA", ""),
        "rank1_rdma": os.environ.get("M3_RANK1_RDMA", ""),
        "public_base_url": os.environ.get("M3_PUBLIC_BASE_URL", f"{ENDPOINT}/v1"),
        "gateway_port": os.environ.get("M3_GATEWAY_PORT", "8010"),
        "gateway_base_url": os.environ.get(
            "M3_GATEWAY_PUBLIC_BASE_URL",
            f"http://{host_from_url(os.environ.get('M3_PUBLIC_BASE_URL', '')) or '127.0.0.1'}:{os.environ.get('M3_GATEWAY_PORT', '8010')}/v1",
        ),
        "model": model,
        "model_id": model_id,
        "model_info": model_info(model, model_id),
        "omlx_port": os.environ.get("M3_OMLX_PORT", "8000"),
        "api_port": os.environ.get("MLX_M3_PORT", "8080"),
        "rank0_total_gb": os.environ.get("M3_RANK0_TOTAL_GB", "256"),
        "rank1_total_gb": os.environ.get("M3_RANK1_TOTAL_GB", "128"),
        "gui_host": GUI_HOST,
        "gui_port": str(GUI_PORT),
        "dashboard_url": dashboard_public_url(),
    }


def is_local_model_path(value):
    text = str(value or "").strip()
    if not text:
        return False
    if Path(text).expanduser().exists():
        return True
    return bool(
        text.startswith("/")
        or text.startswith("~")
        or text.startswith(".")
    )


def default_model_download_dir(model_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id or DEFAULT_MODEL_ID).strip("-")
    return str(Path.home() / ".cache" / "m3-models" / safe)


def model_info(model=None, model_id=None):
    model = model or os.environ.get("MLX_M3_MODEL", DEFAULT_MODEL_ID)
    model_id = model_id or os.environ.get("MLX_M3_MODEL_ID", DEFAULT_MODEL_ID)
    is_local = is_local_model_path(model)
    expanded = str(Path(model).expanduser()) if is_local else ""
    exists = Path(expanded).exists() if expanded else False
    return {
        "model": model,
        "model_id": model_id,
        "is_local_path": is_local,
        "expanded_path": expanded,
        "exists": exists,
        "download_target": expanded or default_model_download_dir(model_id),
        "loaded_requires_restart": True,
    }


def run(cmd, *, timeout=12):
    try:
        p = subprocess.run(
            cmd,
            cwd=str(CLUSTER),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "ok": p.returncode == 0,
            "code": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "code": 124, "stdout": e.stdout or "", "stderr": "timeout"}
    except Exception as e:
        return {"ok": False, "code": 1, "stdout": "", "stderr": repr(e)}


def sh(script, *, timeout=12):
    return run(["/bin/zsh", "-lc", script], timeout=timeout)


def start_background_job(name, script, log_name):
    """Start a dashboard job without relying on screen's login-shell behavior."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "job"
    log_path = CLUSTER / log_name
    pid_path = CLUSTER / f"{safe_name}.pid"
    try:
        with log_path.open("a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {name} =====\n")
            log.flush()
            proc = subprocess.Popen(
                ["/bin/zsh", "-lc", script],
                cwd=str(CLUSTER),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(proc.pid))
        return {"ok": True, "pid": proc.pid, "log": log_name, "pid_file": pid_path.name}
    except Exception as e:
        return {"ok": False, "error": repr(e), "log": log_name}


def health():
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=8) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"status": "offline", "error": str(e)}


def reset_prompt_cache(*, clear_manifest=False):
    payload = {"clear_manifest": bool(clear_manifest)}
    req = urllib.request.Request(
        f"{ENDPOINT}/admin/prompt-cache/reset",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return {"ok": False, "status_code": e.code, **body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def reset_request_history():
    req = urllib.request.Request(
        f"{ENDPOINT}/admin/request-history/reset",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return {"ok": False, "status_code": e.code, **body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def metal_warmup():
    payload = json.dumps({
        "reason": "dashboard warm endpoint",
        "matrix_size": 128,
        "repeats": 2,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{ENDPOINT}/admin/metal-warmup",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return {"ok": False, "status_code": e.code, **body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def chat_warmup():
    script = CLUSTER / "m3_warmup.py"
    if not script.exists():
        return {"ok": False, "error": f"missing warmup script: {script}"}
    env = os.environ.copy()
    env.setdefault("M3_WARMUP_BASE", ENDPOINT)
    env["M3_WARMUP_SKIP_AFTER_COMPLETED"] = "-1"
    env.setdefault("M3_WARMUP_TIMEOUT_SECONDS", "300")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(CLUSTER),
            env=env,
            text=True,
            capture_output=True,
            timeout=int(env.get("M3_WARMUP_TIMEOUT_SECONDS") or "300") + 30,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "error": f"warmup timed out: {e}"}
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    parsed = None
    if stdout:
        try:
            parsed = json.loads(stdout.splitlines()[-1])
        except Exception:
            parsed = None
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "result": parsed,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
    }


def persistent_cache_action(action):
    if action not in {"clear", "prune", "save"}:
        return {"ok": False, "error": f"unsupported persistent cache action {action}"}
    payload = {"reason": f"dashboard {action}"}
    req = urllib.request.Request(
        f"{ENDPOINT}/admin/prompt-cache/ssd/{action}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return {"ok": False, "status_code": e.code, **body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stop_generation():
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/stop",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return {"ok": False, "status_code": e.code, **body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def runtime_tuning(values):
    payload = json.dumps({"values": values}).encode("utf-8")
    req = urllib.request.Request(
        f"{ENDPOINT}/admin/runtime-tuning",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return {"ok": False, "status_code": e.code, **body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def parse_vm_stat(text):
    def pages(label):
        m = re.search(rf"{re.escape(label)}:\s+([0-9]+)", text)
        return int(m.group(1)) if m else 0

    page_size = 16384
    m = re.search(r"page size of ([0-9]+) bytes", text)
    if m:
        page_size = int(m.group(1))
    free = pages("Pages free") + pages("Pages speculative")
    purgeable = pages("Pages purgeable")
    wired = pages("Pages wired down")
    active = pages("Pages active")
    inactive = pages("Pages inactive")
    wired_active = wired + active
    available = free + inactive + purgeable
    return {
        "page_size": page_size,
        "free_gb": round(free * page_size / 1024**3, 1),
        "available_gb": round(available * page_size / 1024**3, 1),
        "purgeable_gb": round(purgeable * page_size / 1024**3, 1),
        "wired_gb": round(wired * page_size / 1024**3, 1),
        "active_gb": round(active * page_size / 1024**3, 1),
        "wired_active_gb": round(wired_active * page_size / 1024**3, 1),
        "inactive_gb": round(inactive * page_size / 1024**3, 1),
    }


def memory_local():
    out = sh("vm_stat", timeout=5)
    return parse_vm_stat(out["stdout"]) | {"ok": out["ok"]}


def memory_remote():
    if not PEER:
        return {"ok": False, "error": "M3_PEER is not configured"}
    out = sh(f"ssh -o BatchMode=yes -o ConnectTimeout=5 {PEER!r} vm_stat", timeout=8)
    data = parse_vm_stat(out["stdout"]) if out["stdout"] else {}
    data.update({"ok": out["ok"], "error": out["stderr"]})
    return data


def proc_snapshot():
    local = sh(
        "ps -axo pid,ppid,stat,rss,command | "
        "egrep -i 'mlx.launch|run_with_watchdog|sharded_server|auto_restart|omlx|python.*8080|python.*8000' | "
        "grep -v egrep || true",
        timeout=5,
    )
    if PEER:
        remote = sh(
            f"ssh -o BatchMode=yes -o ConnectTimeout=5 {PEER!r} "
            "'ps -axo pid,ppid,stat,rss,command | "
            "egrep -i \"mlx.launch|run_with_watchdog|sharded_server|auto_restart|python.*8080\" | "
            "grep -v egrep || true'",
            timeout=8,
        )
    else:
        remote = {"ok": False, "stdout": "", "stderr": "M3_PEER is not configured"}
    ports = sh(
        "lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null; "
        "lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null; "
        "lsof -nP -iTCP:8090 -sTCP:LISTEN 2>/dev/null",
        timeout=5,
    )
    return {
        "local": local["stdout"].splitlines(),
        "remote": remote["stdout"].splitlines(),
        "remote_ok": remote["ok"],
        "ports": ports["stdout"].splitlines(),
    }


def cached_dashboard_snapshot(*, prefer_stale=False):
    now = time.time()
    if (
        _snapshot_cache["memory"] is not None
        and _snapshot_cache["processes"] is not None
        and (
            prefer_stale
            or now - float(_snapshot_cache["at"] or 0.0) < SNAPSHOT_TTL_SECONDS
        )
    ):
        return _snapshot_cache["memory"], _snapshot_cache["processes"]
    memory = {"studio": memory_local(), "macbook": memory_remote()}
    processes = proc_snapshot()
    _snapshot_cache.update({"at": now, "memory": memory, "processes": processes})
    return memory, processes


def settings():
    text = (CLUSTER / "launch_cluster.sh").read_text()
    envs = {}
    for key in (
        "MLX_M3_MODEL",
        "MLX_M3_MODEL_ID",
        "M3_PIPELINE_LAYERS",
        "M3_SHARDING_MODE",
        "M3_WATCHDOG_PREFILL_TIMEOUT",
        "M3_WATCHDOG_DECODE_TIMEOUT",
        "M3_MAX_GENERATION_SECONDS",
        "MLX_M3_DEFAULT_MAX_TOKENS",
        "MLX_M3_NONSTREAM_DEFAULT_MAX_TOKENS",
        "MLX_M3_DEFAULT_TEMPERATURE",
        "MLX_M3_DEFAULT_TOP_P",
        "MLX_M3_DEFAULT_TOP_K",
        "MLX_M3_DEFAULT_MIN_P",
        "MLX_M3_TOOL_DEFAULT_TEMPERATURE",
        "MLX_M3_TOOL_DEFAULT_TOP_P",
        "MLX_M3_TOOL_DEFAULT_TOP_K",
        "MLX_M3_TOOL_DEFAULT_MIN_P",
        "MLX_M3_DEFAULT_REPETITION_PENALTY",
        "MLX_M3_DEFAULT_PRESENCE_PENALTY",
        "MLX_M3_DEFAULT_FREQUENCY_PENALTY",
        "MLX_M3_THINKING_BUDGET",
        "MLX_M3_ALLOW_THINKING_BUDGET",
        "MLX_M3_STREAM_MODE",
        "MLX_M3_MAX_CONCURRENT_REQUESTS",
        "MLX_M3_REQUEST_HISTORY_MAX",
        "MLX_M3_DECODE_EVAL_EVERY",
        "MLX_M3_DECODE_EVAL_AFTER_TOKENS",
        "MLX_M3_DECODE_EVAL_AFTER_EVERY",
        "MLX_M3_THINKING_DECODE_EVAL_EVERY",
        "MLX_M3_THINKING_RAW_SILENT_LIMIT",
        "MLX_M3_LONG_CONTEXT_DECODE_EVAL_TOKENS",
        "MLX_M3_LONG_CONTEXT_DECODE_EVAL_EVERY",
        "MLX_M3_ADAPTIVE_LONG_CONTEXT_DECODE_EVAL",
        "MLX_M3_MID_CONTEXT_DECODE_EVAL_TOKENS",
        "MLX_M3_MID_CONTEXT_DECODE_EVAL_EVERY",
        "MLX_M3_HIGH_CONTEXT_DECODE_EVAL_TOKENS",
        "MLX_M3_HIGH_CONTEXT_DECODE_EVAL_EVERY",
        "MLX_M3_ALLOW_UNSAFE_RUNTIME_TUNING",
        "MLX_M3_RANK0_ONLY_LOGITS",
        "MLX_M3_RANK0_DECODE_OWNER",
        "MLX_M3_IMAGE_DEFAULT_MAX_TOKENS",
        "MLX_M3_WIRED_LIMIT_GB",
        "MLX_M3_WIRED_LIMIT_GB_RANK0",
        "MLX_M3_WIRED_LIMIT_GB_RANK1",
        "MLX_M3_MEMORY_LIMIT_GB",
        "MLX_M3_CACHE_LIMIT_GB",
        "MLX_MAX_OPS_PER_BUFFER",
        "MLX_MAX_MB_PER_BUFFER",
        "MLX_M3_PREFILL_STEP_SIZE",
        "MLX_M3_MAX_KV_SIZE",
        "MLX_M3_KV_QUANT_ENABLED",
        "MLX_M3_KV_BITS",
        "MLX_M3_KV_GROUP_SIZE",
        "MLX_M3_KV_QUANT_SCHEME",
        "MLX_M3_QUANTIZED_KV_START",
        "MLX_M3_KV_CACHE_STEP",
        "MLX_M3_PROMPT_CACHE",
        "MLX_M3_PROMPT_CACHE_THINKING",
        "MLX_M3_PROMPT_CACHE_THINKING_MODE",
        "MLX_M3_PROMPT_CACHE_DIRECT_SUFFIX_IDS",
        "MLX_M3_PROMPT_CACHE_MIN_REUSE",
        "MLX_M3_PROMPT_CACHE_MIN_SUFFIX_TOKENS",
        "MLX_M3_PROMPT_CACHE_REUSE_BUCKET_TOKENS",
        "MLX_M3_REASONING_RECALL",
        "MLX_M3_REASONING_RECALL_MAX_SESSIONS",
        "MLX_M3_REASONING_RECALL_MAX_ITEMS",
        "MLX_M3_REASONING_RECALL_MAX_CHARS",
        "MLX_M3_PROMPT_CACHE_TTL_SECONDS",
        "MLX_M3_PROMPT_CACHE_MAX_TOKENS",
        "MLX_M3_PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS",
        "MLX_M3_PROMPT_CACHE_PROTECT_LARGE",
        "MLX_M3_PROMPT_CACHE_PROTECT_MIN_TOKENS",
        "MLX_M3_PROMPT_CACHE_PROTECT_BYPASS_MAX_TOKENS",
        "MLX_M3_PROMPT_CACHE_SESSION_PROTECT",
        "MLX_M3_PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS",
        "MLX_M3_PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS",
        "MLX_M3_PROMPT_CACHE_RESIDENT_SLOTS",
        "MLX_M3_PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS",
        "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST",
        "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST_MAX",
        "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST_PATH",
        "MLX_M3_PROMPT_CACHE_SSD",
        "MLX_M3_PROMPT_CACHE_SSD_RESTORE",
        "MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE",
        "MLX_M3_PROMPT_CACHE_SSD_DIR",
        "MLX_M3_PROMPT_CACHE_SSD_DIR_RANK0",
        "MLX_M3_PROMPT_CACHE_SSD_DIR_RANK1",
        "MLX_M3_PROMPT_CACHE_SSD_TTL_SECONDS",
        "MLX_M3_PROMPT_CACHE_SSD_MAX_BYTES",
        "MLX_M3_PROMPT_CACHE_SSD_MIN_TOKENS",
        "MLX_M3_PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS",
        "MLX_M3_PROMPT_CACHE_SSD_SAVE_REASONING",
        "MLX_M3_PROMPT_CACHE_SSD_PRIVACY",
        "MLX_M3_PROMPT_CACHE_KEEPWARM",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_MODE",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_IDLE_AFTER_SECONDS",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_MATRIX_SIZE",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_LARGE_CACHE_TOKENS",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_LARGE_INTERVAL_SECONDS",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_SLOW_BACKOFF_SECONDS",
        "MLX_M3_CLEAR_CACHE_AFTER_REQUEST",
        "MLX_M3_CLEAR_CACHE_AFTER_ERROR",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_BLOCKING",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_SUFFIX_TOKENS",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS",
        "MLX_M3_MAX_TOKENS_CEILING",
        "MLX_M3_LAYER_EVAL_EVERY",
        "MLX_M3_OMLX_MINIMAX_OVERLAY",
        "MLX_M3_DISABLE_SPARSE_INDEX",
        "MLX_M3_KERNEL_STATS",
        "MLX_M3_MSA_K1_IMPL",
        "MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_MIN_KV_LEN",
        "MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_BLOCK_CHUNK",
        "MLX_M3_SPARSE_TOPK_BLOCKS_OVERRIDE",
        "MLX_M3_DECODE_TOPK_REUSE_TOKENS",
        "MLX_M3_COMPACT_DECODE_SORT_TOPK",
        "MLX_M3_USE_DIRECT_DECODE_KERNEL",
        "MLX_M3_DIRECT_DECODE_EVAL_MODE",
    ):
        m = re.search(rf"--env {re.escape(key)}=(\"[^\"]*\"|'[^']*'|[^\\\n ]+)", text)
        if m:
            value = m.group(1).strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            default = re.fullmatch(rf"\$\{{{re.escape(key)}:-([^}}]*)\}}", value)
            envs[key] = default.group(1) if default else value
    m = re.search(r'BACKEND="\$\{M3_MLX_BACKEND:-([^}]+)\}"', text)
    envs["M3_MLX_BACKEND"] = m.group(1) if m else "unknown"
    envs.setdefault("MLX_M3_MODEL", os.environ.get("MLX_M3_MODEL", DEFAULT_MODEL_ID))
    envs.setdefault("MLX_M3_MODEL_ID", os.environ.get("MLX_M3_MODEL_ID", DEFAULT_MODEL_ID))
    local_env = {}
    for candidate in (CLUSTER / ".env.local", CLUSTER / "m3_cluster.env", CLUSTER / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                local_env[key] = value
        break
    for key in envs:
        if key in os.environ:
            envs[key] = os.environ[key]
    for key in envs:
        if key in local_env:
            envs[key] = local_env[key]
    return envs


def quote_env_value(value):
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:@,+%=-]+", text):
        return text
    return shlex.quote(text)


def runtime_python():
    bundled = CLUSTER / "bin" / "mlx-python"
    return str(bundled) if bundled.exists() else sys.executable


def mlx_vlm_versions():
    py = runtime_python()
    script = r"""
import importlib.metadata
import json
import ssl
import urllib.request

try:
    installed = importlib.metadata.version("mlx-vlm")
except Exception:
    installed = None
latest = None
try:
    context = None
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    with urllib.request.urlopen("https://pypi.org/pypi/mlx-vlm/json", timeout=10, context=context) as r:
        latest = json.loads(r.read().decode("utf-8")).get("info", {}).get("version")
except Exception as e:
    print(json.dumps({"ok": False, "installed": installed, "latest": latest, "error": str(e)}))
    raise SystemExit(0)
print(json.dumps({"ok": True, "installed": installed, "latest": latest, "update_available": bool(installed and latest and installed != latest)}))
"""
    out = run([py, "-c", script], timeout=18)
    try:
        data = json.loads(out["stdout"].splitlines()[-1])
    except Exception:
        data = {"ok": False, "error": out["stderr"] or out["stdout"] or "version check failed"}
    if not data.get("ok"):
        pip = run([py, "-m", "pip", "index", "versions", "mlx-vlm"], timeout=30)
        text = "\n".join([pip.get("stdout", ""), pip.get("stderr", "")])
        latest = installed = None
        m = re.search(r"mlx-vlm \(([^)]+)\)", text)
        if m:
            latest = m.group(1)
        m = re.search(r"INSTALLED:\s*([^\s]+)", text)
        if m:
            installed = m.group(1)
        m = re.search(r"LATEST:\s*([^\s]+)", text)
        if m:
            latest = m.group(1)
        if installed or latest:
            data = {
                "ok": True,
                "installed": installed,
                "latest": latest,
                "update_available": bool(installed and latest and installed != latest),
                "source": "pip index",
            }
    data["python"] = py
    return data


# The serving stack is four packages: mlx + mlx-metal (core array framework,
# Metal kernels, jaccl/ring distributed — PINNED to a custom-built wheel, a
# pip update would silently replace it with an older release), mlx-lm
# (text-LM building blocks + some tool parsers; dependency of mlx-vlm) and
# mlx-vlm (the model runtime: MiniMax-M3-VL implementation, generation loop,
# minimax_m3 tool parser, vision path).
RUNTIME_PACKAGES = ("mlx", "mlx-metal", "mlx-lm", "mlx-vlm")
RUNTIME_UPDATABLE = {"mlx", "mlx-lm", "mlx-vlm"}
MLX_VARIANTS_MANIFEST = CLUSTER / "runtime_patches" / "mlx_variants.json"

_VERSIONS_SNIPPET = r"""
import importlib.metadata, json
out = {}
for p in ("mlx", "mlx-metal", "mlx-lm", "mlx-vlm"):
    try:
        out[p] = importlib.metadata.version(p)
    except Exception:
        out[p] = None
print(json.dumps(out))
"""


def _pypi_latest(package):
    script = (
        "import json, ssl, urllib.request\n"
        "context = None\n"
        "try:\n"
        "    import certifi\n"
        "    context = ssl.create_default_context(cafile=certifi.where())\n"
        "except Exception:\n"
        "    pass\n"
        f"with urllib.request.urlopen('https://pypi.org/pypi/{package}/json', timeout=10, context=context) as r:\n"
        "    print(json.loads(r.read().decode())['info']['version'])\n"
    )
    out = run([runtime_python(), "-c", script], timeout=15)
    value = (out.get("stdout") or "").strip().splitlines()
    return value[-1] if value else None


def validated_mlx_variant():
    try:
        manifest = json.loads(MLX_VARIANTS_MANIFEST.read_text())
        label = str(manifest.get("recommended") or "").strip()
        record = (manifest.get("variants") or {}).get(label) or {}
    except Exception as e:
        return {"ok": False, "error": str(e), "label": "", "version": None}
    variant_dir = CLUSTER / "runtime_patches" / "variants" / label
    mlx_wheels = list(variant_dir.glob("mlx-[0-9]*.whl"))
    metal_wheels = list(variant_dir.glob("mlx_metal-[0-9]*.whl"))
    complete = len(mlx_wheels) == 1 and len(metal_wheels) == 1
    approved = str(record.get("status") or "").lower() in {"production", "validated"}
    return {
        "ok": bool(label and complete and approved),
        "label": label,
        "version": record.get("version"),
        "status": record.get("status"),
        "note": record.get("note"),
        "artifacts_present": complete,
        "error": (
            None if complete and approved
            else f"validated production wheel pair is unavailable for {label or 'recommended variant'}"
        ),
    }


def runtime_stack_versions():
    py = runtime_python()
    local = {}
    out = run([py, "-c", _VERSIONS_SNIPPET], timeout=18)
    try:
        local = json.loads(out["stdout"].splitlines()[-1])
    except Exception:
        local = {"error": out.get("stderr") or "local version check failed"}
    remote = {}
    if PEER:
        remote_command = (
            f"cd {shlex.quote(str(CLUSTER))} && "
            f"./bin/mlx-python -c {shlex.quote(_VERSIONS_SNIPPET)}"
        )
        rout = run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", PEER,
             remote_command],
            timeout=25,
        )
        try:
            remote = json.loads(rout["stdout"].splitlines()[-1])
        except Exception:
            remote = {"error": rout.get("stderr") or "worker unreachable"}
    latest = {}
    for package in ("mlx", "mlx-lm", "mlx-vlm"):
        try:
            latest[package] = _pypi_latest(package)
        except Exception:
            latest[package] = None
    mlx_variant = validated_mlx_variant()
    packages = []
    for package in RUNTIME_PACKAGES:
        installed = local.get(package)
        pinned = bool(installed and ("dev" in str(installed) or "+" in str(installed)))
        if package == "mlx":
            updatable = bool(mlx_variant.get("ok"))
            target_version = mlx_variant.get("version")
            update_mode = "validated-pair"
            note = (
                f"validated paired MLX/MLX-Metal build ({mlx_variant.get('label')})"
                if updatable else mlx_variant.get("error")
            )
        elif package == "mlx-metal":
            updatable = False
            target_version = mlx_variant.get("version")
            update_mode = "paired-with-mlx"
            note = "updated atomically with MLX; never install this package alone"
        else:
            updatable = package in RUNTIME_UPDATABLE
            target_version = latest.get(package)
            update_mode = "pypi-exact"
            note = "dependency of mlx-vlm" if package == "mlx-lm" else "model runtime"
        packages.append({
            "package": package,
            "rank0": installed,
            "rank1": remote.get(package),
            "ranks_match": (installed == remote.get(package)) if remote and "error" not in remote else None,
            "latest_release": latest.get(package if package != "mlx-metal" else "mlx"),
            "target_version": target_version,
            "update_available": bool(updatable and installed and target_version and installed != target_version),
            "updatable": updatable,
            "update_mode": update_mode,
            "pinned_custom_build": pinned,
            "note": note,
        })
    return {"ok": "error" not in local, "python": py, "packages": packages,
            "worker": PEER or "", "worker_error": remote.get("error"),
            "mlx_variant": mlx_variant}


def write_local_env_setting(key, value):
    allowed_keys = {"MLX_M3_MODEL", "MLX_M3_MODEL_ID"}
    if key not in allowed_keys:
        raise ValueError(f"unsupported local env setting {key}")
    value = str(value).strip()
    if not value:
        raise ValueError(f"{key} cannot be empty")
    if any(ch in value for ch in "\r\n\0"):
        raise ValueError(f"{key} contains an invalid character")
    LOCAL_ENV_PATH.touch(mode=0o600, exist_ok=True)
    lines = LOCAL_ENV_PATH.read_text().splitlines()
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    replacement = f"{key}={quote_env_value(value)}"
    updated = False
    for i, line in enumerate(lines):
        if pattern.match(line) and not line.lstrip().startswith("#"):
            lines[i] = replacement
            updated = True
            break
    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(replacement)
    LOCAL_ENV_PATH.write_text("\n".join(lines) + "\n")
    os.environ[key] = value


def write_setting(key, value):
    if key in {"MLX_M3_MODEL", "MLX_M3_MODEL_ID"}:
        write_local_env_setting(key, value)
        return
    allowed = {
        "M3_PIPELINE_LAYERS": r"[0-9]+,[0-9]+",
        "M3_SHARDING_MODE": r"(tensor|pipeline)",
        "M3_WATCHDOG_PREFILL_TIMEOUT": r"[0-9]+",
        "M3_WATCHDOG_DECODE_TIMEOUT": r"[0-9]+",
        "M3_MAX_GENERATION_SECONDS": r"[0-9]+",
        "MLX_M3_DEFAULT_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_NONSTREAM_DEFAULT_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_DEFAULT_TEMPERATURE": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_DEFAULT_TOP_P": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_DEFAULT_TOP_K": r"[0-9]+",
        "MLX_M3_DEFAULT_MIN_P": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_TOOL_DEFAULT_TEMPERATURE": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_TOOL_DEFAULT_TOP_P": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_TOOL_DEFAULT_TOP_K": r"[0-9]+",
        "MLX_M3_TOOL_DEFAULT_MIN_P": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_DEFAULT_REPETITION_PENALTY": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_DEFAULT_PRESENCE_PENALTY": r"-?[0-9]+(\\.[0-9]+)?",
        "MLX_M3_DEFAULT_FREQUENCY_PENALTY": r"-?[0-9]+(\\.[0-9]+)?",
        "MLX_M3_THINKING_BUDGET": r"[0-9]+",
        "MLX_M3_ALLOW_THINKING_BUDGET": r"[01]",
        "MLX_M3_IMAGE_DEFAULT_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_WIRED_LIMIT_GB": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_WIRED_LIMIT_GB_RANK0": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_WIRED_LIMIT_GB_RANK1": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_MEMORY_LIMIT_GB": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_CACHE_LIMIT_GB": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_MAX_CONCURRENT_REQUESTS": r"[1-9][0-9]*",
        "MLX_M3_REQUEST_HISTORY_MAX": r"[0-9]+",
        "MLX_M3_DECODE_EVAL_EVERY": r"[0-9]+",
        "MLX_M3_DECODE_EVAL_AFTER_TOKENS": r"[0-9]+",
        "MLX_M3_DECODE_EVAL_AFTER_EVERY": r"[0-9]+",
        "MLX_M3_THINKING_DECODE_EVAL_EVERY": r"[0-9]+",
        "MLX_M3_THINKING_RAW_SILENT_LIMIT": r"[0-9]+",
        "MLX_M3_LONG_CONTEXT_DECODE_EVAL_TOKENS": r"[0-9]+",
        "MLX_M3_LONG_CONTEXT_DECODE_EVAL_EVERY": r"[0-9]+",
        "MLX_M3_ADAPTIVE_LONG_CONTEXT_DECODE_EVAL": r"[01]",
        "MLX_M3_MID_CONTEXT_DECODE_EVAL_TOKENS": r"[0-9]+",
        "MLX_M3_MID_CONTEXT_DECODE_EVAL_EVERY": r"[0-9]+",
        "MLX_M3_HIGH_CONTEXT_DECODE_EVAL_TOKENS": r"[0-9]+",
        "MLX_M3_HIGH_CONTEXT_DECODE_EVAL_EVERY": r"[0-9]+",
        "MLX_M3_ALLOW_UNSAFE_RUNTIME_TUNING": r"[01]",
        "MLX_M3_RANK0_ONLY_LOGITS": r"[01]",
        "MLX_M3_RANK0_DECODE_OWNER": r"[01]",
        "MLX_MAX_OPS_PER_BUFFER": r"[0-9]+",
        "MLX_MAX_MB_PER_BUFFER": r"[0-9]+",
        "MLX_M3_PREFILL_STEP_SIZE": r"[0-9]+",
        "MLX_M3_MAX_KV_SIZE": r"[0-9]+",
        "MLX_M3_KV_QUANT_ENABLED": r"[01]",
        "MLX_M3_KV_BITS": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_KV_GROUP_SIZE": r"[0-9]+",
        "MLX_M3_KV_QUANT_SCHEME": r"(uniform|turboquant)",
        "MLX_M3_QUANTIZED_KV_START": r"[0-9]+",
        "MLX_M3_KV_CACHE_STEP": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE": r"[01]",
        "MLX_M3_PROMPT_CACHE_THINKING": r"[01]",
        "MLX_M3_PROMPT_CACHE_THINKING_MODE": r"(off|visible|full|0|1)",
        "MLX_M3_PROMPT_CACHE_DIRECT_SUFFIX_IDS": r"[01]",
        "MLX_M3_PROMPT_CACHE_MIN_REUSE": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_MIN_SUFFIX_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_REUSE_BUCKET_TOKENS": r"[0-9]+",
        "MLX_M3_REASONING_RECALL": r"[01]",
        "MLX_M3_REASONING_RECALL_MAX_SESSIONS": r"[0-9]+",
        "MLX_M3_REASONING_RECALL_MAX_ITEMS": r"[0-9]+",
        "MLX_M3_REASONING_RECALL_MAX_CHARS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_TTL_SECONDS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_GENERATED_REUSE_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_PROTECT_LARGE": r"[01]",
        "MLX_M3_PROMPT_CACHE_PROTECT_MIN_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_PROTECT_BYPASS_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_SESSION_PROTECT": r"[01]",
        "MLX_M3_PROMPT_CACHE_SESSION_PROTECT_MIN_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_SESSION_PROTECT_BYPASS_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_RESIDENT_SLOTS": r"[1-9][0-9]*",
        "MLX_M3_PROMPT_CACHE_RESIDENT_MAX_TOTAL_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST": r"[01]",
        "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST_MAX": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_SESSION_MANIFEST_PATH": r"[^\\n\\r]+",
        "MLX_M3_PROMPT_CACHE_SSD": r"[01]",
        "MLX_M3_PROMPT_CACHE_SSD_RESTORE": r"[01]",
        "MLX_M3_PROMPT_CACHE_SSD_AUTO_SAVE": r"[01]",
        "MLX_M3_PROMPT_CACHE_SSD_DIR": r"[^\\n\\r]+",
        "MLX_M3_PROMPT_CACHE_SSD_DIR_RANK0": r"[^\\n\\r]*",
        "MLX_M3_PROMPT_CACHE_SSD_DIR_RANK1": r"[^\\n\\r]*",
        "MLX_M3_PROMPT_CACHE_SSD_TTL_SECONDS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_SSD_MAX_BYTES": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_SSD_MIN_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_SSD_STATUS_SCAN_INTERVAL_SECONDS": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_PROMPT_CACHE_SSD_SAVE_REASONING": r"[01]",
        "MLX_M3_PROMPT_CACHE_SSD_PRIVACY": r"(local)",
        "MLX_M3_PROMPT_CACHE_KEEPWARM": r"[01]",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_MODE": r"(metal|prewarm)",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_INTERVAL_SECONDS": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_IDLE_AFTER_SECONDS": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_MATRIX_SIZE": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_LARGE_CACHE_TOKENS": r"[0-9]+",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_LARGE_INTERVAL_SECONDS": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_PROMPT_CACHE_KEEPWARM_SLOW_BACKOFF_SECONDS": r"[0-9]+(\\.[0-9]+)?",
        "MLX_M3_CLEAR_CACHE_AFTER_REQUEST": r"[01]",
        "MLX_M3_CLEAR_CACHE_AFTER_ERROR": r"[01]",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM": r"[01]",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_BLOCKING": r"[01]",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MIN_GENERATED": r"[0-9]+",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_TOKENS": r"[0-9]+",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_SUFFIX_TOKENS": r"[0-9]+",
        "MLX_M3_VISIBLE_TRANSCRIPT_PREWARM_MAX_GENERATED_TOKENS": r"[0-9]+",
        "MLX_M3_MAX_TOKENS_CEILING": r"[0-9]+",
        "MLX_M3_LAYER_EVAL_EVERY": r"[0-9]+",
        "MLX_M3_OMLX_MINIMAX_OVERLAY": r"[01]",
        "MLX_M3_DISABLE_SPARSE_INDEX": r"[01]",
        "MLX_M3_KERNEL_STATS": r"[01]",
        "MLX_M3_MSA_K1_IMPL": r"[A-Za-z0-9_]+",
        "MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_MIN_KV_LEN": r"[0-9]+",
        "MLX_M3_MSA_PREFILL_BLOCKWISE_TOPK_BLOCK_CHUNK": r"[0-9]+",
        "MLX_M3_SPARSE_TOPK_BLOCKS_OVERRIDE": r"[0-9]+",
        "MLX_M3_DECODE_TOPK_REUSE_TOKENS": r"[0-9]+",
        "MLX_M3_COMPACT_DECODE_SORT_TOPK": r"[01]",
        "MLX_M3_USE_DIRECT_DECODE_KERNEL": r"[01]",
        "MLX_M3_DIRECT_DECODE_EVAL_MODE": r"(small|all|full|0|1|true|false|yes|no|on|off)",
        "M3_MLX_BACKEND": r"(jaccl|jaccl-ring|ring)",
    }
    if key not in allowed or not re.fullmatch(allowed[key], str(value)):
        raise ValueError(f"invalid setting {key}={value!r}")
    path = CLUSTER / "launch_cluster.sh"
    text = path.read_text()
    if key == "M3_MLX_BACKEND":
        text, n = re.subn(
            r'BACKEND="\$\{M3_MLX_BACKEND:-[^}]+}"',
            f'BACKEND="${{M3_MLX_BACKEND:-{value}}}"',
            text,
        )
    else:
        replacement_value = (
            f'"${{{key}:-}}"'
            if str(value) == ""
            else quote_env_value(value)
        )
        text, n = re.subn(
            rf"--env {re.escape(key)}=(\"[^\"]*\"|'[^']*'|[^\\\n ]+)",
            f"--env {key}={replacement_value}",
            text,
        )
    if n != 1:
        raise ValueError(f"could not update {key}")
    path.write_text(text)


@APP.get("/", response_class=HTMLResponse)
def index():
    return dashboard_html()


@APP.get("/legacy", response_class=HTMLResponse)
def legacy_dashboard():
    """Original full-featured dashboard (chat tab, settings, probes) — kept
    reachable while the v2 redesign reaches feature parity."""
    legacy = CLUSTER / "dashboard_legacy.html"
    if legacy.exists():
        return legacy.read_text()
    return "<h3>dashboard_legacy.html not found</h3>"


@APP.get("/api/status")
def api_status():
    h = health()
    memory, processes = cached_dashboard_snapshot(prefer_stale=bool(h.get("active_request")))
    return {
        "time": time.time(),
        "endpoint": ENDPOINT,
        "config": public_config(),
        "health": h,
        "memory": memory,
        "processes": processes,
        "settings": settings(),
    }


@APP.post("/api/start")
def api_start():
    (CLUSTER / ".stop_requested").unlink(missing_ok=True)
    sync = run(["/bin/zsh", str(CLUSTER / "sync_rank1.sh")], timeout=30)
    if not sync["ok"]:
        return JSONResponse(status_code=500, content={"ok": False, "step": "sync", "result": sync})
    sh("/usr/bin/screen -S minimax_m3 -X quit >/dev/null 2>&1 || true", timeout=5)
    start = sh(f"/usr/bin/screen -dmS minimax_m3 /bin/zsh {str(CLUSTER / 'auto_restart.sh')!r}", timeout=5)
    return {"ok": start["ok"], "sync": sync, "start": start}


@APP.post("/api/stop")
def api_stop():
    result = run(["/bin/zsh", str(CLUSTER / "stop_cluster.sh")], timeout=75)
    return {"ok": result["ok"], "result": result}


@APP.post("/api/sync")
def api_sync():
    result = run(["/bin/zsh", str(CLUSTER / "sync_rank1.sh")], timeout=30)
    return {"ok": result["ok"], "result": result}


@APP.post("/api/probe/text")
def api_probe_text():
    result = start_background_job(
        "m3_text_probe",
        "python3 probes/m3_no_think_probe.py",
        "gui_text_probe.log",
    )
    return {"ok": result["ok"], "log": "gui_text_probe.log", "result": result}


@APP.post("/api/probe/full")
def api_probe_full():
    result = start_background_job(
        "m3_full_probe",
        "python3 probes/m3_openwebui_stress.py",
        "gui_full_probe.log",
    )
    return {"ok": result["ok"], "log": "gui_full_probe.log", "result": result}


@APP.post("/api/probe/tool-cache")
def api_probe_tool_cache():
    result = start_background_job(
        "m3_tool_cache_probe",
        "python3 probes/m3_openwebui_tool_cache_probe.py",
        "gui_tool_cache_probe.log",
    )
    return {"ok": result["ok"], "log": "gui_tool_cache_probe.log", "result": result}


@APP.post("/api/probe/cache-map")
def api_probe_cache_map():
    result = start_background_job(
        "m3_cache_map_probe",
        "python3 probes/m3_cache_map_probe.py",
        "gui_cache_map_probe.log",
    )
    return {"ok": result["ok"], "log": "gui_cache_map_probe.log", "result": result}


@APP.post("/api/probe/agent-cache")
def api_probe_agent_cache():
    result = start_background_job(
        "m3_agent_cache_probe",
        "python3 probes/m3_agent_cache_probe.py "
        "--records 600 "
        "--model Minimax-M3-No-Think "
        "--first-max-tokens 96 "
        "--followup-max-tokens 96 "
        "--session-id gui-agent-A "
        "--interleave-short-session gui-agent-B",
        "gui_agent_cache_probe.log",
    )
    return {"ok": result["ok"], "log": "gui_agent_cache_probe.log", "result": result}


@APP.post("/api/probe/persistent-cache")
def api_probe_persistent_cache():
    result = start_background_job(
        "m3_persistent_cache_probe",
        "python3 probes/m3_persistent_cache_probe.py "
        "--phase roundtrip "
        "--target-tokens 30000 "
        "--session-id gui-persistent-cache-30k",
        "gui_persistent_cache_probe.log",
    )
    return {"ok": result["ok"], "log": "gui_persistent_cache_probe.log", "result": result}


@APP.post("/api/probe/prefill-ab")
def api_probe_prefill_ab():
    result = start_background_job(
        "m3_prefill_ab_probe",
        "python3 probes/m3_prefill_ab_probe.py "
        "--records 1200 "
        "--steps 4096,5120,6144 "
        "--max-tokens 32",
        "gui_prefill_ab_probe.log",
    )
    return {"ok": result["ok"], "log": "gui_prefill_ab_probe.log", "result": result}


@APP.post("/api/cache/reset")
def api_cache_reset():
    result = reset_prompt_cache()
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/cache/manifest/clear")
def api_cache_manifest_clear():
    result = reset_prompt_cache(clear_manifest=True)
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/cache/ssd/prune")
def api_cache_ssd_prune():
    result = persistent_cache_action("prune")
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/cache/ssd/save")
def api_cache_ssd_save():
    result = persistent_cache_action("save")
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/cache/ssd/clear")
def api_cache_ssd_clear():
    result = persistent_cache_action("clear")
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/request-history/reset")
def api_request_history_reset():
    result = reset_request_history()
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/warmup/metal")
def api_metal_warmup():
    result = metal_warmup()
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/warmup/chat")
def api_chat_warmup():
    result = chat_warmup()
    return result if result.get("ok") else JSONResponse(status_code=502, content=result)


@APP.post("/api/generation/stop")
def api_generation_stop():
    result = stop_generation()
    if result.get("ok") is False:
        return JSONResponse(status_code=502, content=result)
    return result


@APP.post("/api/runtime/profile")
async def api_runtime_profile(payload: dict):
    profile = str(payload.get("profile") or "").strip()
    profiles = {
        "thinking-safe": {
            "label": "Thinking stable cadence 1",
            "values": {"thinking_decode_eval_every": 1},
        },
        "thinking-ab-3": {
            "label": "Thinking A/B cadence 3",
            "values": {"thinking_decode_eval_every": 3},
        },
    }
    selected = profiles.get(profile)
    if not selected:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"unknown runtime profile {profile!r}"},
        )
    result = runtime_tuning(selected["values"])
    status = health()
    content = {
        "ok": bool(result.get("ok")),
        "profile": profile,
        "label": selected["label"],
        "values": selected["values"],
        "result": result,
        "health": status,
    }
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=content)
    return content


@APP.post("/api/chat/stream")
async def api_chat_stream(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"ok": False, "error": "payload must be an object"})
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse(status_code=400, content={"ok": False, "error": "messages must be a non-empty list"})
    body = {
        "model": str(payload.get("model") or "Minimax-M3-No-Think"),
        "messages": messages,
        "stream": True,
    }
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "metadata",
    ):
        if key in payload:
            body[key] = payload[key]
    data = json.dumps(body).encode("utf-8")
    upstream = f"{ENDPOINT}/v1/chat/completions"

    def event_stream():
        req = urllib.request.Request(
            upstream,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                while True:
                    chunk = r.readline()
                    if not chunk:
                        break
                    yield chunk
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:
                detail = str(e)
            yield f"event: error\ndata: {json.dumps({'ok': False, 'status_code': e.code, 'error': detail})}\n\n".encode("utf-8")
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'ok': False, 'error': str(e)})}\n\n".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@APP.post("/api/settings")
async def api_settings(payload: dict):
    changed = {}
    try:
        for key, value in payload.items():
            write_setting(key, str(value))
            changed[key] = str(value)
        sync = run(["/bin/zsh", str(CLUSTER / "sync_rank1.sh")], timeout=30)
        return {"ok": True, "changed": changed, "sync": sync, "settings": settings()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})


@APP.post("/api/model/download")
async def api_model_download(payload: dict):
    repo_id = str(payload.get("model_id") or os.environ.get("MLX_M3_MODEL_ID") or DEFAULT_MODEL_ID).strip()
    target = str(payload.get("target") or default_model_download_dir(repo_id)).strip()
    include_worker = bool(payload.get("include_worker", True))
    try:
        write_local_env_setting("MLX_M3_MODEL_ID", repo_id)
        write_local_env_setting("MLX_M3_MODEL", target)
        sync = run(["/bin/zsh", str(CLUSTER / "sync_rank1.sh")], timeout=30)
        worker_flag = "1" if include_worker else "0"
        cmd = (
            f"cd {str(CLUSTER)!r} && "
            f"M3_DOWNLOAD_ON_WORKER={worker_flag} "
            f"/bin/zsh {str(CLUSTER / 'scripts' / 'download_model.sh')!r} {shlex.quote(repo_id)} {shlex.quote(target)} "
            "> model_download.log 2>&1"
        )
        result = sh(f"/usr/bin/screen -dmS m3_model_download /bin/zsh -lc {cmd!r}", timeout=5)
        return {
            "ok": result["ok"],
            "repo_id": repo_id,
            "target": target,
            "include_worker": include_worker,
            "sync": sync,
            "log": "model_download.log",
            "result": result,
            "note": "Restart the cluster after the download finishes so the endpoint loads this model path.",
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})


@APP.get("/api/runtime/mlx-vlm/check")
def api_mlx_vlm_check():
    return mlx_vlm_versions()


@APP.get("/api/runtime/versions")
def api_runtime_versions():
    return runtime_stack_versions()


@APP.post("/api/runtime/update")
async def api_runtime_update(payload: dict):
    package = str(payload.get("package") or "").strip()
    if package not in RUNTIME_UPDATABLE:
        return JSONResponse(status_code=400, content={
            "ok": False,
            "error": f"{package!r} is not an independently updatable runtime package",
        })
    if payload.get("include_worker", True) is False:
        return JSONResponse(status_code=400, content={
            "ok": False,
            "error": "distributed runtime updates must update both ranks atomically",
        })
    status = runtime_update_status()
    if status.get("running"):
        return JSONResponse(status_code=409, content={
            "ok": False,
            "error": "another runtime update is already running",
            "status": status,
        })
    current_health = health()
    if current_health.get("active_request"):
        return JSONResponse(status_code=409, content={
            "ok": False,
            "error": "wait for the active inference request to finish before updating",
        })
    restart = bool(payload.get("restart", True))
    dry_run = bool(payload.get("dry_run", False))
    variant = str(payload.get("variant") or "").strip()
    if package == "mlx":
        approved_variant = validated_mlx_variant()
        if not approved_variant.get("ok"):
            return JSONResponse(status_code=409, content={
                "ok": False,
                "error": approved_variant.get("error") or "no validated MLX build is available",
            })
        approved_label = str(approved_variant.get("label") or "")
        if variant and variant != approved_label:
            return JSONResponse(status_code=400, content={
                "ok": False,
                "error": f"MLX variant {variant!r} is not the approved production pair",
            })
        variant = approved_label
    command = " ".join([
        f"M3_RUNTIME_UPDATE_DRY_RUN={'1' if dry_run else '0'}",
        "/bin/zsh",
        shlex.quote(str(CLUSTER / "scripts" / "update_runtime.sh")),
        shlex.quote(package),
        "1" if restart else "0",
        shlex.quote(variant),
    ])
    if dry_run:
        result = sh(command, timeout=180)
    else:
        result = start_background_job(
            "m3_runtime_update",
            command,
            "runtime_update.log",
        )
    return {
        "ok": result["ok"],
        "package": package,
        "include_worker": True,
        "worker": PEER,
        "python": runtime_python(),
        "restart": restart,
        "dry_run": dry_run,
        "variant": variant,
        "log": "runtime_update.log",
        "result": result,
        "note": "Both ranks are staged, validated, and restarted as one transaction.",
    }


@APP.post("/api/runtime/mlx-vlm/update")
async def api_mlx_vlm_update(payload: dict):
    forwarded = dict(payload or {})
    forwarded["package"] = "mlx-vlm"
    return await api_runtime_update(forwarded)


def runtime_update_status():
    pid_path = CLUSTER / "m3_runtime_update.pid"
    log_path = CLUSTER / "runtime_update.log"
    pid = None
    running = False
    try:
        pid = int(pid_path.read_text().strip())
        running = run(["ps", "-p", str(pid), "-o", "pid="], timeout=3).get("ok", False)
    except Exception:
        pass
    tail = ""
    if log_path.exists():
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        except Exception:
            tail = ""
    match = re.findall(r"RUNTIME_UPDATE_RESULT\s+([^\n]+)", tail)
    result = match[-1] if match else None
    return {
        "ok": True,
        "running": running,
        "pid": pid,
        "log": log_path.name,
        "result": result,
        "tail": tail,
    }


@APP.get("/api/runtime/update/status")
def api_runtime_update_status():
    return runtime_update_status()


@APP.get("/api/logs", response_class=PlainTextResponse)
def api_logs(lines: int = 160):
    lines = max(20, min(int(lines), 500))
    log_dir = Path(os.environ.get("M3_LOG_DIR", str(CLUSTER))).expanduser()
    candidates = [
        log_dir / "startup.log",
        log_dir / "restart.log",
        CLUSTER / "model_download.log",
        CLUSTER / "mlx_vlm_update.log",
        CLUSTER / "runtime_update.log",
        CLUSTER / "gui_text_probe.log",
        CLUSTER / "gui_full_probe.log",
        CLUSTER / "gui_tool_cache_probe.log",
        CLUSTER / "gui_cache_map_probe.log",
        CLUSTER / "gui_agent_cache_probe.log",
        CLUSTER / "gui_persistent_cache_probe.log",
        CLUSTER / "gui_prefill_ab_probe.log",
    ]
    paths = " ".join(shlex.quote(str(p)) for p in candidates)
    out = sh(
        f"tail -n {lines} {paths} 2>/dev/null",
        timeout=5,
    )
    return out["stdout"] or out["stderr"]


# The dashboard UI lives in dashboard.html next to this file. It is read from
# disk (cached by mtime) so UI iterations do not require restarting the GUI
# process; a minimal fallback page keeps the API usable if the file is missing.
DASHBOARD_FILE = CLUSTER / "dashboard.html"
_DASHBOARD_CACHE = {"mtime": None, "html": ""}
FALLBACK_HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>ThunderMLX Console</title></head>
<body style='font-family:system-ui;background:#0d0d0d;color:#fff;padding:40px'>
<h2>ThunderMLX Console</h2>
<p>dashboard.html is missing next to cluster_gui.py; the JSON API is still live at
<a href='/api/status' style='color:#3987e5'>/api/status</a>.</p>
</body></html>"""


def dashboard_html():
    try:
        mtime = DASHBOARD_FILE.stat().st_mtime
    except OSError:
        return FALLBACK_HTML
    if _DASHBOARD_CACHE["mtime"] != mtime:
        try:
            _DASHBOARD_CACHE["html"] = DASHBOARD_FILE.read_text()
            _DASHBOARD_CACHE["mtime"] = mtime
        except OSError:
            return FALLBACK_HTML
    return _DASHBOARD_CACHE["html"]


if __name__ == "__main__":
    uvicorn.run(APP, host=GUI_HOST, port=GUI_PORT, log_level="warning")
