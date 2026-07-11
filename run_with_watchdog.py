#!/usr/bin/env python3
"""
run_with_watchdog.py — wraps sharded_server.py with a heartbeat watchdog.

THE PROBLEM THIS SOLVES
  When the *partner* rank dies (GPU timeout, OOM, segfault), the local rank
  hangs FOREVER at a distributed collective barrier (MLX distributed has no
  timeout API). That orphans ~170GB of Metal memory -> forces a reboot.

THE WATCHDOG
  A heartbeat proves the local rank is making progress. If generation is
  ACTIVE and no heartbeat for HEARTBEAT_TIMEOUT seconds, the watchdog calls
  os._exit(1) -> clean process death -> Metal memory released. The auto-
  restart loop then brings the cluster back.

CRITICAL DESIGN (the bug this version fixes)
  Earlier versions only ticked the heartbeat on each emitted TOKEN. But the
  FIRST request's prefill can take longer than the timeout (the prompt is
  processed in 2048-token chunks, and the first request after load is slow).
  No token is produced until prefill finishes -> no heartbeat ticks -> the
  watchdog killed BOTH ranks mid-prefill at 90s, even though prefill was
  progressing normally. Pure self-inflicted crash, zero errors in the log.

  FIX: sharded_server calls heartbeat_tick() on every decoded token on BOTH
  ranks. The watchdog tolerates long prefills via a generous stall timeout, but
  once tokens have started it catches real no-progress stalls. A separate hard
  max-generation timer prevents a disconnected client or wedged VLM request
  from holding the single distributed slot indefinitely.

  Memory safety: we call mx.clear_cache() right before os._exit, because
  os._exit skips finally/atexit handlers — without this the cache clear in
  sharded_server never runs and memory orphans (forces a reboot).

USAGE: this replaces sharded_server.py as the script mlx.launch runs.
"""
import os
import signal
import subprocess
import sys
import threading
import time

# EAGLE3-window diagnostic (this dev tree only): `kill -USR2 <worker pid>`
# dumps every python thread's stack to stderr -> startup.log. This is the
# python-level view `sample` cannot give (hardened runtime blocks py-spy),
# and the tool the 2026-07-09 verify-forward deadlock hunt was missing.
try:
    import faulthandler
    faulthandler.register(signal.SIGUSR2, all_threads=True, chain=False)
except Exception:
    pass

HEARTBEAT_TIMEOUT = int(os.environ.get("M3_WATCHDOG_TIMEOUT", "240"))
PREFILL_TIMEOUT = int(os.environ.get("M3_WATCHDOG_PREFILL_TIMEOUT", str(HEARTBEAT_TIMEOUT)))
# A LIVE prefill ticks the heartbeat every chunk (~13s at 4096); total
# prefill time may legitimately be long, but chunk SILENCE may not. The
# 11:53 wedge sat frozen mid-prefill for 18+ min because only the 1-hour
# phase timeout applied. Silence past this bound is a frozen step.
PREFILL_STALL_FATAL = int(os.environ.get("M3_WATCHDOG_PREFILL_STALL_FATAL", "240"))
# FIX A (2026-07-07): a large prefill can legitimately run longer than
# PREFILL_STALL_FATAL between ticks — a single big prompt_step blocks in the
# jaccl recv for the whole chunk, so ~zero heartbeats arrive during it. The
# 42k-suffix wedge was the 240s watchdog firing BEFORE jaccl's own
# ProgressGuard (30 min) had given up — a premature kill of a slow-but-live
# prefill. Scale the prefill fatal window by the known prompt size:
#   effective = max(PREFILL_STALL_FATAL, prompt_tokens / PREFILL_MIN_TPS + margin)
# so small prefills keep fast wedge detection and 300k prefills get room.
# PREFILL_MIN_TPS is a conservative FLOOR (well under the ~285 t/s baseline)
# that still tolerates memory-pressure slowdowns; a genuinely dead partner is
# caught by jaccl ProgressGuard (JACCL_PROGRESS_TIMEOUT_MS) and the hard
# MAX_GENERATION_SECONDS backstop, which now release memory cleanly (fix C).
PREFILL_MIN_TPS = float(os.environ.get("M3_WATCHDOG_PREFILL_MIN_TPS", "60"))
PREFILL_STALL_MARGIN = int(os.environ.get("M3_WATCHDOG_PREFILL_STALL_MARGIN", "120"))
DECODE_TIMEOUT = int(os.environ.get("M3_WATCHDOG_DECODE_TIMEOUT", str(HEARTBEAT_TIMEOUT)))
# Fire live_wedge_capture (spindumps both ranks) at this many seconds of stall,
# well before the fatal timeout kills the evidence. 0 disables.
STALL_CAPTURE_AT = int(os.environ.get("M3_WATCHDOG_STALL_CAPTURE_AT", "60"))
MAX_GENERATION_SECONDS = int(os.environ.get("M3_MAX_GENERATION_SECONDS", "240"))

_lock = threading.Lock()
_last_heartbeat = None   # None = idle (no generation in progress)
_last_token_progress = None
_generation_started_at = None
_generation_active = False
_tokens_seen = 0
_progress_tokens_seen = 0
_stall_capture_fired = False  # one capture per stall episode
_prefill_token_budget = 0     # prompt tokens to prefill this turn (fix A)


def note_prefill_budget(tokens):
    """sharded_server calls this before prefill with the prompt token count so
    the watchdog can size the prefill stall window to the work (fix A)."""
    global _prefill_token_budget
    try:
        with _lock:
            _prefill_token_budget = max(0, int(tokens or 0))
    except Exception:
        pass


def _effective_prefill_timeout(budget):
    """Prefill stall-fatal window scaled to the prompt size (fix A)."""
    base = min(PREFILL_TIMEOUT, PREFILL_STALL_FATAL)
    if budget and PREFILL_MIN_TPS > 0:
        return max(base, int(budget / PREFILL_MIN_TPS) + PREFILL_STALL_MARGIN)
    return base


def clear_mlx_cache(reason):
    """Best-effort Metal release before a force-exit.

    clear_cache() only drops the Metal *cache pool* — it does NOT free the
    wired model weights (~160GB), which os._exit then strands in kernel wired
    memory and forces a machine reboot (observed 2026-07-07: a watchdog kill
    left 176GB wired with no owning process). set_wired_limit(0) unwires the
    model allocation so the OS reclaims it on process death. Both calls run
    under the caller's SIGALRM guard, so if a wedged Metal queue makes either
    hang, the alarm hard-exits rather than blocking the release path.
    """
    try:
        import mlx.core as mx
        try:
            # Unwire first: this is what actually releases the 160GB the
            # weights hold. Safe no-op if already unwired.
            mx.set_wired_limit(0)
            sys.stderr.write(f"[WATCHDOG] wired limit set to 0 ({reason})\n")
            sys.stderr.flush()
        except Exception as we:
            sys.stderr.write(f"[WATCHDOG] set_wired_limit(0) failed ({reason}): {we}\n")
            sys.stderr.flush()
        mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        sys.stderr.write(f"[WATCHDOG] cache cleared ({reason})\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[WATCHDOG] cache clear failed ({reason}): {e}\n")
        sys.stderr.flush()


def install_signal_handlers():
    def _hard_exit_after_signal(signum, _frame):
        name = signal.Signals(signum).name
        sys.stderr.write(f"[WATCHDOG] {name} while clearing cache; hard-exiting\n")
        sys.stderr.flush()
        os._exit(128 + signum)

    try:
        signal.signal(signal.SIGALRM, _hard_exit_after_signal)
    except Exception:
        pass

    def _handle_signal(signum, _frame):
        name = signal.Signals(signum).name
        sys.stderr.write(f"[WATCHDOG] received {name}; clearing cache before exit\n")
        sys.stderr.flush()
        try:
            signal.alarm(int(os.environ.get("M3_SIGNAL_CLEAR_TIMEOUT", "10") or "10"))
        except Exception:
            pass
        clear_mlx_cache(name)
        try:
            signal.alarm(0)
        except Exception:
            pass
        os._exit(128 + signum)

    for sig_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except Exception:
                pass


def heartbeat_start():
    """Called when a generation begins. Arms the watchdog."""
    global _last_heartbeat, _last_token_progress
    global _generation_started_at, _generation_active, _tokens_seen, _progress_tokens_seen
    global _stall_capture_fired
    with _lock:
        now = time.time()
        _last_heartbeat = now
        _last_token_progress = None
        _generation_started_at = now
        _generation_active = True
        _tokens_seen = 0
        _progress_tokens_seen = 0
        _stall_capture_fired = False


def heartbeat_tick(progress=True):
    """Record liveness separately from client-visible token progress.

    Decode-eval drains prove the Metal/JACCL loop is alive. Visible stream
    progress is tracked separately for diagnostics, but it must not be used as
    the fatal watchdog condition: a bad template or hidden-token run can be
    silent while both ranks are still decoding in lockstep. Killing that live
    decode path is what orphans wired Metal memory.
    """
    global _last_heartbeat, _last_token_progress, _tokens_seen, _progress_tokens_seen
    with _lock:
        now = time.time()
        _last_heartbeat = now
        _tokens_seen += 1
        if progress:
            _last_token_progress = now
            _progress_tokens_seen += 1


def heartbeat_stop():
    """Called when generation ends normally. Disarms the watchdog."""
    global _generation_active, _generation_started_at, _last_token_progress
    global _tokens_seen, _progress_tokens_seen, _prefill_token_budget
    with _lock:
        _generation_active = False
        _generation_started_at = None
        _last_token_progress = None
        _tokens_seen = 0
        _progress_tokens_seen = 0
        _prefill_token_budget = 0  # clear per-turn budget (fix A)


def start_watchdog():
    """Watchdog thread: if generation is active and stalled, force-exit."""
    sys.stderr.write(
        f"[WATCHDOG] stall-capture armed (capture_at={STALL_CAPTURE_AT}s, "
        f"rank={os.environ.get('MLX_RANK', 'unset')})\n"
    )
    sys.stderr.flush()

    def _watch():
        while True:
            time.sleep(5)
            with _lock:
                active = _generation_active
                last = _last_heartbeat
                last_progress = _last_token_progress
                started = _generation_started_at
                tokens_seen = _tokens_seen
                progress_tokens_seen = _progress_tokens_seen
                prefill_budget = _prefill_token_budget
            if not active or last is None:
                continue
            now = time.time()
            elapsed = time.time() - started if started is not None else 0
            stall = now - last
            if progress_tokens_seen > 0:
                visible_stall = now - (last_progress or last)
                timeout = DECODE_TIMEOUT
                stall_kind = (
                    f"decode-liveness, visible_silent={visible_stall:.0f}s"
                )
            else:
                timeout = _effective_prefill_timeout(prefill_budget)
                stall_kind = (
                    f"prefill/liveness (chunk-silence fatal, "
                    f"budget={prefill_budget}tok, window={timeout}s)"
                )
            if MAX_GENERATION_SECONDS > 0 and elapsed > MAX_GENERATION_SECONDS:
                sys.stderr.write(
                    f"\n[WATCHDOG] Generation exceeded max duration "
                    f"{elapsed:.0f}s>{MAX_GENERATION_SECONDS}s. Force-exiting "
                    f"to release Metal memory and unblock the endpoint.\n"
                )
                sys.stderr.flush()
                try:
                    clear_mlx_cache("watchdog max generation exit")
                except Exception:
                    pass
                os._exit(1)
            # Pre-exit forensics: every wedge autopsy so far died with the
            # process (force-exit beats spindump every time). Fire the live
            # capture at the still-spinning ranks well before the fatal
            # timeout — a stall that later resolves just costs a harmless
            # snapshot. Fire-and-forget; never blocks the watchdog.
            global _stall_capture_fired
            if stall > 30 and not _stall_capture_fired:
                # breadcrumb: proves this branch was reached with these values
                # (the 06:39 wedge fired the fatal path but never the capture,
                # with no surviving explanation — never again)
                try:
                    _bc = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "ops", "logs", "watchdog_stall_breadcrumb",
                    )
                    with open(_bc, "a") as bf:
                        bf.write(
                            f"{time.time():.0f} stall={stall:.0f} cap_at={STALL_CAPTURE_AT} "
                            f"rank={os.environ.get('MLX_RANK', '?')} fired={_stall_capture_fired}\n"
                        )
                except Exception:
                    pass
            if (
                STALL_CAPTURE_AT > 0
                and not _stall_capture_fired
                and stall > STALL_CAPTURE_AT
                # capture orchestrates from rank 0 (reaches rank 1 via ssh);
                # rank 1 firing its own would duplicate and race it
                and os.environ.get("MLX_RANK", "0") == "0"
            ):
                _stall_capture_fired = True
                sys.stderr.write(
                    f"\n[WATCHDOG] stall {stall:.0f}s > {STALL_CAPTURE_AT}s — "
                    f"firing live wedge capture (spindumps both ranks)\n"
                )
                sys.stderr.flush()
                try:
                    capture = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "ops",
                        "live_wedge_capture.sh",
                    )
                    log_path = os.path.join(
                        os.path.dirname(capture), "logs", "autostall_capture.log"
                    )
                    with open(log_path, "a") as lf:
                        lf.write(
                            f"[{time.strftime('%F %T')}] spawning wedge "
                            f"capture (stall={stall:.0f}s rank0)\n"
                        )
                        lf.flush()
                        subprocess.Popen(
                            ["/bin/zsh", capture, time.strftime("autostall_%H%M%S")],
                            stdout=lf,
                            stderr=lf,
                            start_new_session=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(f"[WATCHDOG] capture spawn failed: {exc}\n")
                    sys.stderr.flush()
            # Fatal timeout is based on heartbeat/liveness, not visible stream
            # progress. Thinking/model-control output can be intentionally
            # silent to the client while both ranks are still decoding in
            # lockstep. The hard max-generation timer remains the upper bound
            # for those silent-but-live cases.
            if timeout > 0 and stall > timeout:
                sys.stderr.write(
                    f"\n[WATCHDOG] Generation stalled {stall:.0f}s "
                    f"({stall_kind}, eval_ticks={tokens_seen}, "
                    f"progress_tokens={progress_tokens_seen}, timeout={timeout}s) — "
                    f"partner rank likely dead or stream progress wedged. "
                    f"Force-exiting to release Metal memory.\n"
                )
                sys.stderr.flush()
                # CRITICAL: clear the Metal cache BEFORE os._exit. os._exit
                # skips all finally/atexit handlers, so the sharded_server's
                # `finally: mx.clear_cache()` never runs -> Metal memory
                # orphans in wired kernel memory -> forces a reboot.
                # Doing it here, right before exit, releases the ~160GB of
                # model/activation buffers cleanly so the machine doesn't
                # need a reboot after a watchdog-kill.
                try:
                    _crash = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "ops", "logs",
                        time.strftime("crash_%Y%m%d_%H%M%S.txt"),
                    )
                    with open(_crash, "w") as cf:
                        cf.write(
                            f"watchdog fatal: stall={stall:.0f}s kind={stall_kind}\n"
                            f"eval_ticks={tokens_seen} progress_tokens={progress_tokens_seen}\n"
                            f"elapsed={elapsed:.0f}s timeout={timeout}s "
                            f"rank={os.environ.get('MLX_RANK', '?')}\n"
                        )
                except Exception:
                    pass
                try:
                    clear_mlx_cache("watchdog forced exit")
                except Exception:
                    pass
                os._exit(1)  # immediate process death -> Metal memory released

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    sys.stderr.write(
        f"[WATCHDOG] armed (prefill_timeout={PREFILL_TIMEOUT}s, "
        f"decode_timeout={DECODE_TIMEOUT}s, legacy_timeout={HEARTBEAT_TIMEOUT}s, "
        f"max_generation={MAX_GENERATION_SECONDS}s)\n"
    )
    sys.stderr.flush()


def main():
    install_signal_handlers()
    cluster_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, cluster_dir)
    import sharded_server
    sharded_server._WATCHDOG_TICK = heartbeat_tick
    sharded_server._WATCHDOG_PREFILL_BUDGET = note_prefill_budget  # fix A

    # Patch generation functions to arm/disarm the watchdog. The background
    # sharded_server calls heartbeat_tick() from both streaming and non-streaming
    # generation loops on both ranks.
    _orig_run = sharded_server.run_generation
    _orig_stream = sharded_server.run_generation_stream

    def _patched_run(*a, **kw):
        heartbeat_start()
        try:
            return _orig_run(*a, **kw)
        finally:
            heartbeat_stop()

    def _patched_stream(*a, **kw):
        heartbeat_start()
        try:
            for chunk in _orig_stream(*a, **kw):
                yield chunk
        finally:
            heartbeat_stop()

    sharded_server.run_generation = _patched_run
    sharded_server.run_generation_stream = _patched_stream

    start_watchdog()
    sharded_server.main()


if __name__ == "__main__":
    main()
