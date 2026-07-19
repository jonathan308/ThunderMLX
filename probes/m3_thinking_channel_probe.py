#!/usr/bin/env python3
"""Stress MiniMax-M3 thinking-to-content channel transitions over SSE."""

import argparse
import json
import pathlib
import re
import time
import urllib.request


DEFAULT_BASE = "http://127.0.0.1:8080"
SHADOW_SYNTAX_PROMPT = '''Create a complete, single-file, fully interactive HTML + Tailwind CSS + vanilla JavaScript website called "Shadow Syntax" — an experimental typography studio and digital manifesto exploring how type is a living, imperfect, deconstructed thing.

Core aesthetic:
- Brutalist/minimalist with high-contrast black typography on off-white or concrete-textured backgrounds
- Heavy use of experimental, deconstructed, glitchy, and "living" typography
- Fluid mouse-driven interactions, subtle GSAP-style animations, parallax, and hover distortions
- Scrolling background of dense, semi-transparent code characters that react to mouse movement
- Glitch effects, scan lines, and "keystroke trace" particle effects on text
- Overall feeling: avant-garde, poetic, slightly dystopian design studio from the future

Pages / Sections (smooth SPA navigation):
1. Hero / Landing: Massive centered text "TYPE IS A LIVING THING" with heavy glitch + distortion that reacts to mouse. Subtle concrete wall background that shifts.
2. Manifesto: Text "We believe in the beauty of imperfection. Every keystroke leaves a trace." with animated typing effect and shadow particles.
3. Gallery: Masonry/grid of large typographic posters with titles like:
   - CONSTRUCT, CAST FORM, NEW FORM, SHADOW
   - DECONSTRUCTED DELAY
   - GEOMETRY EXPERIMENTAL FORMS
   - ECHOES
   - FLUX FORM
   - A/R D/R A/F K/V R/F (broken letter art)
   Each poster should have hover effects that break apart the letters or cast dynamic shadows.
4. The Terminal / Editor: Section titled "THE TERMINAL" with big headline that you can interactively type into (live preview): e.g. "MAKE YOUR MARK." → "T_<E YOUR MARK." → "TYPE *-UR MARK." etc. Big "START EXPERIMENT" button that triggers a full-screen interactive type playground.
5. Navigation: Top bar — SHADOW SYNTAX | MANIFESTO | GALLERY | EDITOR
6. Footer: "SHADOW SYNTAX 2024" + "EXPERIMENTAL TYPOGRAPHY"

Technical requirements:
- Fully responsive
- No external dependencies except Tailwind via CDN
- Smooth 60fps animations (use GSAP via CDN if needed, otherwise requestAnimationFrame)
- Mouse-following effects, scroll-triggered animations, hover distortions on all text
- Dark/light mode toggle that feels like a terminal command
- Performance optimized for long sessions

Make it feel like a real, production-ready digital art piece that a cutting-edge typography studio would actually launch. Output the complete HTML file with embedded Tailwind and all JS inline. Add helpful comments in the code.'''
FINAL_ARTIFACT_HINT = (
    "\n\nImportant response-channel requirement: use reasoning only for concise "
    "planning. Close the native reasoning block before the first code fence, "
    "DOCTYPE, or final artifact byte. Put the entire complete HTML deliverable "
    "in the final answer channel; never write final code inside reasoning."
)


def get_json(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_idle(base, completed_before, timeout=120):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = get_json(base, "/health", timeout=5)
        if (
            last.get("active_request") is None
            and int(last.get("requests_completed") or 0) > completed_before
        ):
            return last
        time.sleep(0.25)
    raise TimeoutError(f"server did not return idle: {last}")


def stream_probe(args):
    initial = get_json(args.base, "/health")
    completed_before = int(initial.get("requests_completed") or 0)
    prompt = (
        pathlib.Path(args.prompt_file).read_text(encoding="utf-8")
        if args.prompt_file
        else SHADOW_SYNTAX_PROMPT
    )
    if args.artifact_transition_hint:
        prompt += FINAL_ARTIFACT_HINT
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking_mode": args.thinking_mode,
        "metadata": {"session_id": args.session_id},
    }
    if not args.omit_max_tokens:
        payload["max_tokens"] = args.max_tokens
    if not args.omit_sampling:
        payload.update(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
        )
        if args.seed is not None:
            payload["seed"] = args.seed

    request = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    reasoning = []
    content = []
    first_reasoning_s = None
    first_content_s = None
    finish_reason = None
    usage = None
    events = 0

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            events += 1
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning_piece = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                content_piece = delta.get("content") or ""
                if reasoning_piece:
                    if first_reasoning_s is None:
                        first_reasoning_s = time.monotonic() - started
                    reasoning.append(reasoning_piece)
                if content_piece:
                    if first_content_s is None:
                        first_content_s = time.monotonic() - started
                        print(
                            json.dumps(
                                {
                                    "event": "content_started",
                                    "elapsed_s": round(first_content_s, 3),
                                    "reasoning_chars": sum(map(len, reasoning)),
                                }
                            ),
                            flush=True,
                        )
                    content.append(content_piece)
                finish_reason = choice.get("finish_reason") or finish_reason
            if events % 256 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "events": events,
                            "reasoning_chars": sum(map(len, reasoning)),
                            "content_chars": sum(map(len, content)),
                            "elapsed_s": round(time.monotonic() - started, 3),
                        }
                    ),
                    flush=True,
                )

    final = wait_idle(args.base, completed_before)
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    doctype_count = len(re.findall(r"<!doctype\s+html>", content_text, re.I))
    html_open_count = len(re.findall(r"<html\b", content_text, re.I))
    html_close_count = len(re.findall(r"</html\s*>", content_text, re.I))
    last = final.get("last_request") or {}
    result = {
        "model": args.model,
        "thinking_mode": args.thinking_mode,
        "artifact_transition_hint": args.artifact_transition_hint,
        "session_id": args.session_id,
        "sampling": {
            "omitted": args.omit_sampling,
            "temperature": None if args.omit_sampling else args.temperature,
            "top_p": None if args.omit_sampling else args.top_p,
            "top_k": None if args.omit_sampling else args.top_k,
            "min_p": None if args.omit_sampling else args.min_p,
            "repetition_penalty": (
                None if args.omit_sampling else args.repetition_penalty
            ),
            "seed": None if args.omit_sampling else args.seed,
        },
        "max_tokens": None if args.omit_max_tokens else args.max_tokens,
        "max_tokens_omitted": args.omit_max_tokens,
        "finish_reason": finish_reason,
        "usage": usage,
        "events": events,
        "first_reasoning_s": round(first_reasoning_s or 0, 3),
        "first_content_s": round(first_content_s or 0, 3),
        "reasoning_chars": len(reasoning_text),
        "content_chars": len(content_text),
        "content_started": bool(content_text),
        "has_complete_html": (
            "<html" in content_text.lower()
            and "</html>" in content_text.lower()
        ),
        "html_document_count": {
            "doctype": doctype_count,
            "open": html_open_count,
            "close": html_close_count,
        },
        "has_one_complete_html": (
            doctype_count == 1
            and html_open_count == 1
            and html_close_count == 1
        ),
        "server": {
            "request_id": last.get("id"),
            "tokens": last.get("tokens"),
            "ttft_s": last.get("first_token_s"),
            "prompt_tps": last.get("prompt_tps"),
            "decode_tps": last.get("decode_tps"),
            "ok": last.get("ok"),
            "failures": final.get("requests_failed"),
        },
    }
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "result": result,
                    "reasoning": reasoning_text,
                    "content": content_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["content_started"] and result["has_one_complete_html"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--model", default="Minimax-M3")
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "adaptive", "disabled"),
        default="enabled",
    )
    parser.add_argument("--prompt-file")
    parser.add_argument("--artifact-transition-hint", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--session-id", default="thinking-shadow-syntax-probe")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--omit-max-tokens",
        action="store_true",
        help="Omit max_tokens to exercise the server/client-shaped default.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--omit-sampling",
        action="store_true",
        help="Let the server apply its normal model/mode defaults.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    args.base = args.base.rstrip("/")
    raise SystemExit(stream_probe(args))


if __name__ == "__main__":
    main()
