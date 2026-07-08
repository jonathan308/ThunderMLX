#!/usr/bin/env python3
"""Analyze MiniMax-M3 probe and overnight validation logs.

Accepts raw JSON-line probe logs such as perf_probe_*.log and overnight
events.jsonl files. The goal is to make hot-cache and PPT regressions obvious
without hand-reading every probe row.
"""
import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PATTERNS = [
    "perf_probe*.log",
    "hot_cache*.log",
    "overnight_results/*/events.jsonl",
]


def as_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(*values):
    for value in values:
        n = as_number(value)
        if n is not None:
            return n
    return None


def iter_json_lines(path):
    with path.open("r", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                continue


def unwrap_probe_row(obj):
    if obj.get("event") == "probe_output":
        data = obj.get("data")
        if isinstance(data, dict):
            return data
        return None
    if "initial" in obj or "final" in obj:
        return None
    return obj


def row_name(row, fallback):
    return str(row.get("name") or row.get("label") or fallback)


def cache_reason(row):
    return (
        row.get("cache_miss_reason")
        or row.get("miss_reason")
        or row.get("cache_action")
        or row.get("cache_reason")
    )


def generated_reuse(row):
    return first_number(
        row.get("cache_generated_reuse_ratio"),
        row.get("generated_reuse_ratio"),
    )


def prompt_tps(row):
    return first_number(
        row.get("server_prompt_tps"),
        row.get("last_prompt_tps"),
        row.get("cache_prompt_tps"),
    )


def prompt_tokens(row):
    pcache = row.get("prompt_cache") if isinstance(row.get("prompt_cache"), dict) else {}
    return first_number(
        row.get("server_prompt_tokens"),
        row.get("last_prompt_tokens"),
        row.get("cache_prompt_tokens"),
        row.get("prompt_tokens"),
        pcache.get("key_tokens"),
        pcache.get("cache_len"),
    )


def ttft(row):
    return first_number(
        row.get("server_ttft_s"),
        row.get("last_ttft_s"),
        row.get("client_ttft_s"),
        row.get("second_elapsed_s"),
    )


def decode_tps(row):
    return first_number(row.get("server_decode_tps"), row.get("last_decode_tps"))


def derive_prompt_tps(row):
    explicit = prompt_tps(row)
    if explicit and explicit > 0:
        return explicit
    tokens = prompt_tokens(row)
    t = ttft(row)
    if tokens and t and t > 0:
        return tokens / t
    return None


def fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def median(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


def collect(paths):
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for lineno, obj in iter_json_lines(p):
            row = unwrap_probe_row(obj)
            if not isinstance(row, dict):
                continue
            if not any(k in row for k in ("name", "label", "server_ttft_s", "last_ttft_s", "cache_action", "miss_reason")):
                continue
            row = dict(row)
            row["_source"] = str(p)
            row["_line"] = lineno
            rows.append(row)
    return rows


def default_paths():
    paths = []
    for pattern in DEFAULT_PATTERNS:
        paths.extend(glob.glob(pattern))
    return sorted(dict.fromkeys(paths))


def print_markdown(rows):
    if not rows:
        print("No probe rows found.")
        return

    by_source_name = defaultdict(list)
    for row in rows:
        by_source_name[(row["_source"], row_name(row, "row"))].append(row)

    print("# M3 Probe Analysis\n")
    print("| Source | Case | Rows | TTFT s | Prompt tok/s | Decode tok/s | Cache reason | Gen reuse | Reprocess tok |")
    print("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for (source, name), group in sorted(by_source_name.items()):
        reasons = Counter(str(cache_reason(r) or "-") for r in group)
        reason = reasons.most_common(1)[0][0]
        reprocess = median([first_number(r.get("cache_would_reprocess_tokens"), r.get("would_reprocess_tokens")) for r in group])
        print(
            f"| {Path(source).name} | {name} | {len(group)} | "
            f"{fmt(median([ttft(r) for r in group]))} | "
            f"{fmt(median([derive_prompt_tps(r) for r in group]))} | "
            f"{fmt(median([decode_tps(r) for r in group]))} | "
            f"{reason} | {fmt(median([generated_reuse(r) for r in group]), 3)} | "
            f"{fmt(reprocess, 0)} |"
        )

    all_reasons = Counter(str(cache_reason(r) or "-") for r in rows)
    print("\n## Cache Reasons\n")
    for reason, count in all_reasons.most_common():
        print(f"- `{reason}`: {count}")

    long_rows = [r for r in rows if "long" in row_name(r, "").lower()]
    cold = [r for r in long_rows if "cold" in row_name(r, "").lower()]
    if cold:
        best = max(cold, key=lambda r: derive_prompt_tps(r) or -1)
        print("\n## Best Cold Long-Prompt Row\n")
        print(
            f"- `{Path(best['_source']).name}` `{row_name(best, 'row')}`: "
            f"{fmt(derive_prompt_tps(best))} prompt tok/s, "
            f"TTFT {fmt(ttft(best))}s, prompt tokens {fmt(prompt_tokens(best), 0)}"
        )

    bad = [
        r for r in rows
        if cache_reason(r) in {
            "previous_assistant_start_mismatch",
            "previous_assistant_partial_mismatch",
            "history_prefix_mismatch",
        }
    ]
    if bad:
        print("\n## Follow-Up Cache Action Items\n")
        print("- Assistant-boundary misses mean the client likely omitted hidden reasoning; test `MLX_M3_VISIBLE_TRANSCRIPT_PREWARM=1`.")
        print("- History-prefix mismatches mean message/template text changed before the assistant response; inspect the raw client transcript.")


def print_json(rows):
    print(json.dumps(rows, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Probe logs or overnight events.jsonl files")
    parser.add_argument("--json", action="store_true", help="Emit normalized rows as JSON")
    args = parser.parse_args()

    paths = args.paths or default_paths()
    rows = collect(paths)
    if args.json:
        print_json(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
