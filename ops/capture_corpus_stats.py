#!/usr/bin/env python3
"""Walk the capture-only corpus and report how it is growing.

Reads ONLY the .npy headers inside each .npz (numpy.lib.format), so it is cheap
to run against a large corpus and never loads a hidden-state array into RAM.

Reports, per capture-only request dir (prompt_*.npz + decode_*.npz written by
m3_capture) and also legacy eagle dirs (prompt_*.npz + round_*.npz):
  - requests captured
  - prompt positions and decode positions (== drafter teacher-forced pairs)
  - estimated total training pairs (what ops/eagle3_finetune would build)
  - on-disk size (GB)

Usage:
  ops/capture_corpus_stats.py [DUMP_DIR]     # default: $MLX_M3_EAGLE3_DUMP_DIR
"""
import os
import sys
import glob
import zipfile

import numpy as np
from numpy.lib import format as _npyfmt


def _npy_shape(zf, name):
    """(shape, dtype) of a member .npy from its header alone (no data read)."""
    with zf.open(name) as f:
        major, _minor = _npyfmt.read_magic(f)
        reader = getattr(_npyfmt, f"read_array_header_{major}_0")
        shape, _fortran, dtype = reader(f)
        return shape, dtype


def _npz_arrays(path):
    """{array_name: (shape, dtype)} for every array in an .npz, header-only."""
    out = {}
    try:
        with zipfile.ZipFile(path) as zf:
            for zi in zf.namelist():
                if zi.endswith(".npy"):
                    out[zi[:-4]] = _npy_shape(zf, zi)
    except Exception as e:
        print(f"  ! unreadable {os.path.basename(path)}: {e}", file=sys.stderr)
    return out


def _dir_size(path):
    total = 0
    for f in os.listdir(path):
        try:
            total += os.path.getsize(os.path.join(path, f))
        except OSError:
            pass
    return total


def scan_request(req_dir):
    """(n_prompt, n_decode_pairs, n_rounds, bytes) for one request dir, or None
    if it holds no capture files."""
    prompts = sorted(glob.glob(os.path.join(req_dir, "prompt_*.npz")))
    decodes = sorted(glob.glob(os.path.join(req_dir, "decode_*.npz")))
    rounds = sorted(glob.glob(os.path.join(req_dir, "round_*.npz")))
    if not (prompts or decodes or rounds):
        return None

    n_prompt = 0
    for p in prompts[:1]:  # build_request_sequence uses prompt_*[0]
        arrs = _npz_arrays(p)
        if "prompt_hidden" in arrs:
            shp = arrs["prompt_hidden"][0]
            n_prompt = int(shp[1]) if len(shp) >= 2 else int(shp[0])
        elif "prompt_ids" in arrs:
            n_prompt = int(np.prod(arrs["prompt_ids"][0]))

    n_decode = 0
    for d in decodes:
        arrs = _npz_arrays(d)
        if "hidden" in arrs:
            n_decode += int(arrs["hidden"][0][0])  # (T, D) -> T labeled pairs

    for r in rounds:
        arrs = _npz_arrays(r)
        if "verify_hidden" in arrs:
            shp = arrs["verify_hidden"][0]
            n_decode += int(shp[1]) if len(shp) >= 2 else int(shp[0])

    return n_prompt, n_decode, len(rounds), _dir_size(req_dir)


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "MLX_M3_EAGLE3_DUMP_DIR", ""
    ).strip()
    if not dump or not os.path.isdir(dump):
        print(f"dump dir not found: {dump!r}\n"
              f"usage: {sys.argv[0]} [DUMP_DIR]", file=sys.stderr)
        return 2

    # Per-request subdirs (capture-only + eagle e3dump layout) AND a flat dump
    # dir (legacy eagle dumps written straight into DUMP_DIR).
    candidates = [
        d for d in sorted(glob.glob(os.path.join(dump, "*")))
        if os.path.isdir(d)
    ]
    candidates.append(dump)  # flat files at the root, if any

    n_req = tot_prompt = tot_decode = tot_rounds = tot_bytes = 0
    speculative = capture_only = 0
    for d in candidates:
        res = scan_request(d)
        if res is None:
            continue
        n_prompt, n_decode, n_rounds, nbytes = res
        n_req += 1
        tot_prompt += n_prompt
        tot_decode += n_decode
        tot_rounds += n_rounds
        tot_bytes += nbytes
        if n_rounds:
            speculative += 1
        else:
            capture_only += 1

    # build_request_sequence emits (n_prompt + n_decode) labeled positions per
    # request then teacher-forcing drops the final unlabeled one.
    pairs = max(0, tot_prompt + tot_decode - n_req)
    gb = tot_bytes / (1 << 30)

    print(f"corpus:            {dump}")
    print(f"requests captured: {n_req}  "
          f"(capture-only {capture_only}, speculative {speculative})")
    print(f"prompt positions:  {tot_prompt:,}")
    print(f"decode positions:  {tot_decode:,}"
          + (f"  ({tot_rounds:,} speculative rounds)" if tot_rounds else ""))
    print(f"training pairs:    ~{pairs:,}")
    print(f"on-disk size:      {gb:.2f} GB"
          + (f"  ({gb / n_req * 1024:.1f} MB/request)" if n_req else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
